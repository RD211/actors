from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum, auto
import inspect
import itertools
import os
import shutil
import time
import tempfile
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import warnings

import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler
from accelerate import Accelerator, InitProcessGroupKwargs
from transformers import PreTrainedTokenizerBase
from datasets import Dataset as HFDataset, DatasetDict
from accelerate.utils import DeepSpeedPlugin, DistributedType

# ----- project-local helpers -------------------------------------------------
from actors.actors.base import TrainableLLMActor
from actors.utils.deepspeed import _OptimizerProxy, prepare_deepspeed, prepare_deepspeed_reference, offload_model_and_optimizer, reload_model_and_optimizer, log_memory_usage
from actors.utils.logger import init_logger, colorize, Palette, VERBOSE, NORMAL, QUIET
from actors.utils.ipc_utils import gather_and_stream_state_dict
from actors.environments.env_base import Environment
from actors.environments.types import EnvironmentOutput, ActorOutput
from actors.losses.base_loss import BaseRLLoss
from actors.utils.softmax import _selective_softmax
from actors.utils.tracker import start_step_profiling, log_step_profiling, _step_profiler, gpu_profiler
from actors.utils.wandb import is_wandb_active
from actors.utils.train_utils import disable_dropout_in_model, free_memory

# Import PEFT
from peft import PeftConfig, get_peft_model
import shutil
import tempfile

def is_peft_model(model):
    """Check if a model is a PEFT model."""
    return hasattr(model, 'peft_config') and hasattr(model, 'base_model')

# ═════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=False)
class InitializedTrainableLLMActor:
    """Immutable record that bundles everything the trainer needs for one actor."""
    name: str
    actor: TrainableLLMActor
    model: torch.nn.Module
    tokenizer: PreTrainedTokenizerBase
    loss_fn: BaseRLLoss
    optim: torch.optim.Optimizer
    sched: Optional[torch.optim.lr_scheduler.LRScheduler]
    accel: Accelerator 
    model_config: Dict[str, Any]
    reference_model: Optional[torch.nn.Module] = None



class SaveStrategy(Enum):
    """Allowed checkpointing modes."""
    NONE   = auto()  # never save
    STEPS  = auto()  # checkpoint_every_n only
    FINAL  = auto()  # one model save at the very end
    ALL    = auto()  # both periodic + final


class EvalStrategy(Enum):
    """Allowed evaluation modes."""
    NONE   = auto()  # never evaluate
    STEPS  = auto()  # evaluate every eval_every_n steps
    FINAL  = auto()  # evaluate only at the end
    ALL    = auto()  # evaluate both periodically and at the end

# ═════════════════════════════════════════════════════════════════════════════
# Stateless helper functions
# ═════════════════════════════════════════════════════════════════════════════
def expand_batch(batch: Dict[str, List[Any]], group_size: int) -> Dict[str, List[Any]]:
    """Tile every list value in *batch* `group_size` times."""
    if not isinstance(batch, dict):
        raise ValueError("batch must be a dictionary")
    return {k: [item for item in v for _ in range(group_size)] for k, v in batch.items()}


def split_for_grad_accum(seq: Sequence[Any], steps: int) -> List[Sequence[Any]]:
    """Evenly split *seq* into *steps* chunks; drops remainder."""
    stride = len(seq) // steps
    return [seq[i * stride : (i + 1) * stride] for i in range(steps)]


def norm_advantages(rewards: List[float], group_size: int) -> List[float]:
    """Normalise rewards per sampling group (µ=0, σ=1)."""
    out: List[float] = []
    for i in range(0, len(rewards), group_size):
        grp = rewards[i : i + group_size]
        µ = sum(grp) / len(grp)
        σ = (sum((x - µ) ** 2 for x in grp) / len(grp)) ** 0.5 + 1e-8
        out.extend([(r - µ) / σ for r in grp])
    return out


def default_advantage_calculator(
    rewards: List[float], 
    group_size: int, 
    ended_in_eos: Optional[List[bool]] = None,
    std_normalization: bool = True
) -> List[float]:
    out: List[float] = []
    for i in range(0, len(rewards), group_size):
        grp = rewards[i : i + group_size]
        µ = sum(grp) / len(grp)
        
        if std_normalization:
            σ = (sum((x - µ) ** 2 for x in grp) / len(grp)) ** 0.5 + 1e-8
            out.extend([(r - µ) / σ for r in grp])
        else:
            out.extend([r - µ for r in grp])
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Main trainer
# ═════════════════════════════════════════════════════════════════════════════
class Trainer:
    # --------------------------------------------------------------------- #
    # construction
    # --------------------------------------------------------------------- #
    def __init__(
        self,
        env: Environment,
        *,
        data: Union[List[Dict[str, Any]], Dataset, HFDataset, DatasetDict],
        batch_size: int = 8,
        group_size: int = 8,
        grad_accumulation_steps: int = 1,
        reference_batch_size: int = 1,
        num_iterations: int = 1,
        max_grad_norm: float = 1.0,
        gradient_checkpointing: bool = True,
        # advantage calculation
        advantage_calculator: Optional[Callable[..., List[float]]] = None,
        std_normalization: bool = True,
        # logging
        use_wandb: bool = True,
        log_every_n: int = 1,
        use_dashboard: bool = True,
        # evaluation
        eval_data: Optional[Union[
            List[Dict[str, Any]], 
            Dataset, 
            HFDataset, 
            DatasetDict,
            Dict[str, Union[List[Dict[str, Any]], Dataset, HFDataset, DatasetDict]]
        ]] = None,
        eval_every_n: Optional[int] = None,
        eval_strategy: EvalStrategy = EvalStrategy.NONE,
    ):
        # ─── trainer-level bookkeeping ────────────────────────────────
        self.env   = env
        self.max_grad_norm  = max_grad_norm
        self.num_iterations = num_iterations
        self.gradient_checkpointing = gradient_checkpointing
        self.use_wandb = use_wandb
        self.log_every_n = log_every_n
        self.use_dashboard = use_dashboard
        self._step = 0
        self._logical_step = 0

        # ─── advantage calculation setup ──────────────────────────────
        if advantage_calculator is not None:
            self.advantage_calculator = advantage_calculator
        else:
            # Use default calculator with configurable std normalization
            self.advantage_calculator = lambda rewards, group_size, ended_in_eos=None: default_advantage_calculator(
                rewards, group_size, ended_in_eos, std_normalization
            )

        # ─── LoRA tracking for first updates ──────────────────────────
        self._first_lora_update = {}  # Track which actors need first LoRA update

        # ─── data / RNG setup ──────────────────────────────
        self._data = self._normalise_hf_splits(data)
        self._data_state = {"epoch": 0, "step_in_epoch": 0, "current_generator_seed": 0}
        self._rng = torch.Generator()
        self.group_size  = group_size
        self.batch_size  = batch_size
        self.grad_accumulation_steps = grad_accumulation_steps
        self.reference_batch_size    = reference_batch_size
        self._build_dataloader()
        
        # ─── evaluation setup ──────────────────────────────
        if eval_data is not None:
            if isinstance(eval_data, dict):
                # Multiple named eval datasets
                self.eval_datasets = {
                    name: self._normalise_hf_splits(data) 
                    for name, data in eval_data.items()
                }
            else:
                # Single eval dataset - give it a default name
                self.eval_datasets = {"eval": self._normalise_hf_splits(eval_data)}
        else:
            self.eval_datasets = {}
        
        self.eval_every_n = eval_every_n
        self.eval_strategy = eval_strategy
        


        # ═══════════════════════════════════════════════════════════════
        # 1.  one DeepSpeedPlugin per actor – default config first
        # ═══════════════════════════════════════════════════════════════
        self.ds_plugins: Dict[str, DeepSpeedPlugin] = {
            name: DeepSpeedPlugin() for name in env.get_trainable_actors().keys()
        }

        # tweak each plugin’s config **after** creation (no base template)
        for name, plug in self.ds_plugins.items():
            cfg = plug.deepspeed_config                             # default “auto” dict
            cfg["max_grad_norm"] = max_grad_norm
            cfg["train_batch_size"] = batch_size
            cfg["gradient_accumulation_steps"] = grad_accumulation_steps
            cfg["train_micro_batch_size_per_gpu"] = (
                batch_size // grad_accumulation_steps // torch.cuda.device_count()
            )
            if gradient_checkpointing:
                activation_config = {
                    "partition_activations": True,
                    "contiguous_memory_optimization": True,
                    "number_checkpoints": 1,
                    "synchronize_checkpoint_boundary": False,
                }
                
                # Add CPU offloading for activations if enabled for this actor
                actor = env.get_trainable_actors().get(name)
                if actor.offload_activations_to_cpu:
                    activation_config.update({
                        "cpu_checkpointing": True,
                        "contiguous_memory_optimization": False,  # Disable when using CPU checkpointing
                        "synchronize_checkpoint_boundary": True,  # Better for CPU offloading
                    })
                
                cfg.setdefault("activation_checkpointing", activation_config)

        # ─── validate offloading compatibility ────────────────────────
        # Check if any actors have offloading enabled
        actors_with_offloading = {
            name: actor for name, actor in env.get_trainable_actors().items()
            if actor.offload_optimizer or actor.offload_model
        }
        
        if actors_with_offloading:
            # Check ZeRO stage from the first plugin (all should be the same)
            first_plugin = next(iter(self.ds_plugins.values()))
            zero_config = first_plugin.deepspeed_config.get("zero_optimization", {})
            zero_stage = zero_config.get("stage", 2)  # Default is usually stage 2
            
            if zero_stage != 3:
                actor_names = ", ".join(actors_with_offloading.keys())
                warning_msg = (
                    f"⚠️  Offloading is only supported with DeepSpeed ZeRO Stage 3, "
                    f"but current stage is {zero_stage}. "
                    f"Offloading for actors [{actor_names}] will be disabled and have no effect."
                )
                self.logger.warning(colorize(warning_msg, Palette.WARNING))
                
                # Disable offloading since it won't work
                for actor in actors_with_offloading.values():
                    actor.offload_optimizer = False
                    actor.offload_model = False

        # ═══════════════════════════════════════════════════════════════
        # 2.  one Accelerator per actor
        # ═══════════════════════════════════════════════════════════════
        first_actor = next(iter(self.ds_plugins))
        self.accelerators: Dict[str, Accelerator] = {
            first_actor: Accelerator(
                mixed_precision="bf16",
                deepspeed_plugin=self.ds_plugins[first_actor],
                kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(hours=10))],
            )
        }
        for name in self.ds_plugins:
            if name != first_actor:
                self.accelerators[name] = Accelerator()  # shares AcceleratorState

        # main handle used for cross-actor ops (metrics gather etc.)
        self.main_accel = self.accelerators[first_actor]
        self.number_of_devices = self.main_accel.num_processes
        self.rank              = self.main_accel.process_index
        self.logger            = init_logger(f"trainer{self.rank}")

        # ─── basic sanity checks ───────────────────────────
        if batch_size % group_size:
            raise ValueError("batch_size must be a divisible by group_size")
        if (batch_size // grad_accumulation_steps) % self.number_of_devices:
            raise ValueError("batch_size/grad_accumulation_steps must be divisible by world size")
        if batch_size % (reference_batch_size * self.number_of_devices):
            raise ValueError("batch_size must be divisible by reference_batch_size*world_size")

        # ═══════════════════════════════════════════════════════════════
        # 3.  build TrainableLLMActor registry
        # ═══════════════════════════════════════════════════════════════
        self.actors: Dict[str, InitializedTrainableLLMActor] = {}

        for name, actor_obj in env.get_trainable_actors().items():
            accel = self.accelerators[name]

            # ── create & optionally checkpoint-enable model ────────────
            model = actor_obj.model_factory()  # Use the convenient property
            
            # Apply PEFT configuration if available
            if actor_obj.training_config.peft_config is not None:
                if self.main_accel.is_main_process:
                    self.logger.info(colorize(f"🔧 Applying PEFT configuration to actor '{name}'", Palette.INFO))
                model = get_peft_model(model, actor_obj.training_config.peft_config)
                
                # Track this actor for first LoRA update
                self._first_lora_update[name] = True
                
                # Warn about PEFT + offloading incompatibility
                if (actor_obj.offload_optimizer or actor_obj.offload_model) and self.main_accel.is_main_process:
                    self.logger.warning(colorize(
                        f"⚠️  Actor '{name}' has both PEFT and offloading enabled. "
                        f"PEFT models don't work well with DeepSpeed offloading due to parameter structure differences. "
                        f"Offloading will be automatically disabled for this actor.",
                        Palette.WARNING
                    ))

            if gradient_checkpointing:
                # Enable gradient checkpointing
                if is_peft_model(model):
                    # For PEFT models, apply to base model
                    model.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                    model.base_model.config.use_cache = False
                    model.base_model.enable_input_require_grads()
                else:
                    # For regular models
                    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                    model.config.use_cache = False
                    model.enable_input_require_grads()
            disable_dropout_in_model(model)

            model_cfg = model.config.to_dict()

            # ── loss / reference model (if any) ────────────────────────
            loss_fn = actor_obj.training_config.loss_factory()
            beta    = getattr(loss_fn, "beta", 0.0)
            
            # Handle reference model based on PEFT configuration
            if beta == 0.0:
                # If beta is 0.0, the reference model is not needed
                ref_model = None
            elif is_peft_model(model):
                # If PEFT is used, the reference model is not needed since the adapter can be disabled
                # to revert to the initial model.
                ref_model = None
                if self.main_accel.is_main_process:
                    self.logger.info(colorize(f"📚 Using adapter disabling for reference model with actor '{name}'", Palette.INFO))
            else:
                # For deepspeed, fsdp or non-distributed models, create a reference model from scratch
                ref_model = (
                    actor_obj.training_config.reference_model_factory().eval()
                    if actor_obj.training_config.reference_model_factory and beta != 0.0
                    else None
                )
                if ref_model is not None and self.main_accel.is_main_process:
                    self.logger.info(colorize(f"📚 Created separate reference model for actor '{name}'", Palette.INFO))

            # ── optimiser / scheduler ─────────────────────────────────
            optim = actor_obj.training_config.optim_factory(model.parameters())
            if actor_obj.offload_optimizer or actor_obj.offload_model:
                optim = _OptimizerProxy(optim)
            # ── wrap with this actor’s accelerator ────────────────────
            model, optim = accel.prepare(model, optim)
            
            # Log activation offloading status
            if actor_obj.offload_activations_to_cpu and self.main_accel.is_main_process:
                self.logger.info(colorize(f"💾 Enabled CPU activation offloading for training model '{name}'", Palette.INFO))
            
            if ref_model is not None:
                ref_model.requires_grad_(False)  # no gradients for reference model
                ref_model.config.use_cache = False  # disable cache for reference model

                ref_model = ref_model.to(dtype=model.dtype)
                # Use specialized DeepSpeed config for reference model with CPU offloading
                if actor_obj.offload_reference_to_cpu:
                    if self.main_accel.is_main_process:
                        self.logger.info(colorize(f"🔄 Using CPU-offloaded DeepSpeed config for reference model '{name}'", Palette.INFO))
                    ref_model = prepare_deepspeed_reference(ref_model, accel, use_cpu_offload=True)
                else:
                    ref_model = prepare_deepspeed(ref_model, accel)
                disable_dropout_in_model(ref_model)

            # ── register ──────────────────────────────────────────────
            self.actors[name] = InitializedTrainableLLMActor(
                name           = name,
                actor          = actor_obj,
                model          = model,
                tokenizer      = actor_obj.tokenizer,  # Use the convenient property
                loss_fn        = loss_fn,
                optim          = optim,
                sched          = None,
                reference_model= ref_model,
                accel          = accel,
                model_config   = model_cfg,
            )
            # We offload the model and optimizer if requested
            if actor_obj.offload_optimizer or actor_obj.offload_model:
                if self.main_accel.is_main_process:
                    self.logger.info(colorize(f"🔄 Offloading model and optimizer for actor '{name}'", Palette.INFO))
                offload_model_and_optimizer(self.actors[name].model, self.actors[name].optim, 
                                           offload_optimizer=actor_obj.offload_optimizer, 
                                           offload_model=actor_obj.offload_model)


        # Initialize the lora adapters for all actors
        self._setup_loras()
    # ------------------------------------------------------------------ #
    # data helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalise_hf_splits(
        data: Union[List[Dict[str, Any]], Dataset, HFDataset, DatasetDict]
    ):
        """Return a single split no matter what HuggingFace object is given."""
        if isinstance(data, DatasetDict):
            return data.get("train", next(iter(data.values())))
        return data

    def _build_dataloader(self):
        """Create a DataLoader with the current RNG seed & generator state."""
        self._rng.manual_seed(self._data_state["current_generator_seed"])
        sampler = RandomSampler(self._data, generator=self._rng)

        def collate_fn(batch):
            if not batch:
                return {}
            keys = batch[0].keys()
            return {k: [d[k] for d in batch] for k in keys}

        self._dataloader = DataLoader(
            self._data,
            batch_size=self.batch_size // self.group_size,
            sampler=sampler,
            collate_fn=collate_fn,
            drop_last=True,
        )
        
    # --------------------------------------------------------------------- #
    # (tiny) utilities
    # --------------------------------------------------------------------- #
    @staticmethod
    def _pad(tok: PreTrainedTokenizerBase, seqs: List[List[int]]) -> Dict[str, torch.Tensor]:
        return tok.pad({"input_ids": seqs}, padding="longest", return_tensors="pt")

    def _clip_gradients(
        self,
        ta: InitializedTrainableLLMActor,
        clip_to: float | None = None,
    ) -> float:
        ta.accel.gradient_state._set_sync_gradients(True)
        max_norm = clip_to if clip_to is not None else torch.finfo(torch.float32).max

        # Gradient clipping
        _grad_norm = ta.accel.clip_grad_norm_(
            ta.model.parameters(),
            max_norm,
        )

        if (
            ta.accel.distributed_type == DistributedType.DEEPSPEED
        ):
            grad_norm = ta.model.get_global_grad_norm()
            # In some cases the grad norm may not return a float
            if hasattr(grad_norm, "item"):
                grad_norm = grad_norm.item()
        else:
            grad_norm = _grad_norm

        return grad_norm

    def _calculate_advantages(
        self, 
        rewards: List[float], 
        group_size: int, 
        ended_in_eos: Optional[List[bool]] = None
    ) -> List[float]:
        try:
            sig = inspect.signature(self.advantage_calculator)
            params = list(sig.parameters.keys())
            
            kwargs = {'rewards': rewards}
            
            if 'group_size' in params:
                kwargs['group_size'] = group_size
            if 'ended_in_eos' in params and ended_in_eos is not None:
                kwargs['ended_in_eos'] = ended_in_eos
                
            return self.advantage_calculator(**kwargs)
            
        except Exception as e:
            try:
                return self.advantage_calculator(rewards)
            except:
                try:
                    return self.advantage_calculator(rewards, group_size)
                except:
                    return default_advantage_calculator(rewards, group_size, ended_in_eos, True)



    # --------------------------------------------------------------------- #
    # log-probabilities 
    # --------------------------------------------------------------------- #

    def _get_logps(
        self,
        model: torch.nn.Module,
        ids: List[List[int]],
        tokenizer: PreTrainedTokenizerBase,
        temperature: float = 1.0,
        batch_size: int = 1,
    ) -> List[List[float]]:
        total = len(ids)
        world = self.number_of_devices
        per_rank = (total + world - 1) // world
        start, end = self.rank * per_rank, min((self.rank + 1) * per_rank, total)
        ids_local = ids[start:end]
        local_logps: List[List[float]] = []
        for i in range(0, len(ids_local), batch_size):
            batch_ids = ids_local[i : i + batch_size]
            lengths = [len(seq) for seq in batch_ids]

            padded = tokenizer.pad({"input_ids": batch_ids}, padding="longest", return_tensors="pt")
            input_ids = padded["input_ids"].to(model.device)

            L = input_ids.size(1)
            attn_mask = padded["attention_mask"]

            logits = model(input_ids=input_ids, attention_mask=attn_mask).logits / temperature # (B,L,V)
            logits = logits[:, :-1, :]  # remove last token logits
            lp = _selective_softmax(logits, input_ids[:, 1:])  # (B,L-1)

            for row, ln in zip(lp, lengths):
                local_logps.append(row[: ln - 1].cpu().tolist())
            del padded, input_ids, attn_mask, logits, lp  # free memory
        gathered = self.main_accel.gather_for_metrics(local_logps)
        free_memory()
        return gathered

    def _log_configuration_to_wandb(self, wandb) -> None:
        """Log important configuration parameters to WandB using nested dictionaries."""
        if not is_wandb_active():
            return
            
        config = {}
        
        # ═══ Trainer Configuration ═══
        config["trainer"] = {
            "batch_size": self.batch_size,
            "group_size": self.group_size,
            "grad_accumulation_steps": self.grad_accumulation_steps,
            "num_iterations": self.num_iterations,
            "reference_batch_size": self.reference_batch_size,
            "max_grad_norm": self.max_grad_norm,
            "gradient_checkpointing": self.gradient_checkpointing,
        }
        
        # ═══ Evaluation Configuration ═══
        config["eval"] = {
            "num_datasets": len(self.eval_datasets),
            "size_per_dataset": {name: len(data) for name, data in self.eval_datasets.items()},
            "dataset_names": list(self.eval_datasets.keys()) if self.eval_datasets else [],
            "eval_every_n": self.eval_every_n
        }

        # ═══ Data Configuration ═══
        config["data"] = {
            "train_size": len(self._data)
        }
        
        # ═══ System Configuration ═══
        config["system"] = {
            "num_devices": self.number_of_devices,
            "num_nodes": self.main_accel.num_processes // torch.cuda.device_count(),
        }
        
        # ═══ Per-Actor Configuration ═══
        config["actors"] = {}
        for name, ta in self.actors.items():
            actor_config = {
                "name": name,
                "actor_type": type(ta.actor).__name__,
                "has_reference_model": ta.reference_model is not None,
                "is_peft_model": is_peft_model(ta.model),
                "uses_adapter_for_ref": (ta.reference_model is None and is_peft_model(ta.model))
            }
            
            # PEFT configuration
            if ta.actor.training_config.peft_config is not None:
                peft_config = ta.actor.training_config.peft_config
                actor_config["peft"] = {
                    "peft_type": peft_config.peft_type.value if hasattr(peft_config, 'peft_type') else str(type(peft_config).__name__),
                    "task_type": peft_config.task_type.value if hasattr(peft_config, 'task_type') else None,
                }
                # Add LoRA-specific parameters if available
                if hasattr(peft_config, 'r'):
                    actor_config["peft"]["rank"] = peft_config.r
                if hasattr(peft_config, 'lora_alpha'):
                    actor_config["peft"]["alpha"] = peft_config.lora_alpha
                if hasattr(peft_config, 'lora_dropout'):
                    actor_config["peft"]["dropout"] = peft_config.lora_dropout
            
            # Model path from actor (accessible via ta.actor.model_path)
            if hasattr(ta.actor, 'model_path'):
                actor_config["model_path"] = ta.actor.model_path
            
            # Loss function configuration
            loss_config = {"type": type(ta.loss_fn).__name__}
            if hasattr(ta.loss_fn, 'beta'):
                loss_config["beta"] = ta.loss_fn.beta
            if hasattr(ta.loss_fn, 'temperature'):
                loss_config["temperature"] = ta.loss_fn.temperature
            actor_config["loss"] = loss_config
            
            # Optimizer configuration
            optimizer_config = {"type": type(ta.optim).__name__}
            if hasattr(ta.optim, 'param_groups') and ta.optim.param_groups:
                param_group = ta.optim.param_groups[0]
                if 'lr' in param_group:
                    optimizer_config["lr"] = param_group['lr']
            # Also include learning rate from actor if available
            if hasattr(ta.actor, 'current_learning_rate'):
                optimizer_config["configured_lr"] = ta.actor.current_learning_rate
            actor_config["optimizer"] = optimizer_config

            # Scheduler info (if exists)
            if ta.sched:
                actor_config["scheduler"] = {"type": type(ta.sched).__name__}
            
            config["actors"][name] = actor_config
        
        # ═══ Environment Configuration ═══
        config["environment"] = {
            "type": type(self.env).__name__
        }
        
        # Recursively filter out None values to keep config clean
        def filter_none_values(d):
            if isinstance(d, dict):
                return {k: filter_none_values(v) for k, v in d.items() if v is not None}
            return d
        
        config = filter_none_values(config)
        
        # Log to WandB
        wandb.config.update(config)

    def evaluate(self, is_final: bool = False) -> Optional[Dict[str, Any]]:
        """
        Run evaluation on all eval datasets if available.
        Returns evaluation metrics or None if no eval data.
        """
        if not self.eval_datasets:
            return None
            
        if self.main_accel.is_main_process:
            self.logger.normal(colorize("🔍 Starting evaluation...", Palette.INFO))
        
        all_eval_metrics = {}
        
        # Iterate through each eval dataset
        for eval_name, eval_data in self.eval_datasets.items():
            if self.main_accel.is_main_process:
                self.logger.normal(colorize(f"📋 Evaluating on '{eval_name}' dataset...", Palette.INFO))
            
            # Create eval batch with all eval data (no sampling, use entire dataset)
            eval_batch = {}
            for key in eval_data[0].keys():
                eval_batch[key] = [item[key] for item in eval_data]
            
            # Run environment on eval data (no gradient accumulation needed for eval)
            env_out = self.env(eval_batch)
            
            # Collect metrics for each actor
            eval_metrics = {}
            
            for actor_name, actor_output in env_out.actors.items():
                if actor_name not in self.actors:
                    continue
                    
                ta = self.actors[actor_name]
                
                # Basic reward statistics
                rewards = actor_output.rewards
                reward_mean = sum(rewards) / len(rewards) if rewards else 0.0
                reward_std = (sum((r - reward_mean) ** 2 for r in rewards) / len(rewards)) ** 0.5 if len(rewards) > 1 else 0.0
                
                # Completion length statistics
                completion_lens = [len(ta.tokenizer.decode(ids, skip_special_tokens=False)) for ids in actor_output.input_ids]
                completion_len_mean = sum(completion_lens) / len(completion_lens) if completion_lens else 0.0
                
                actor_metrics = {
                    "reward_mean": reward_mean,
                    "reward_std": reward_std,
                    "completion_len_mean": completion_len_mean,
                }
                
                # Add reward component statistics
                if actor_output.reward_components:
                    for comp_name, comp_rewards in actor_output.reward_components.items():
                        comp_mean = sum(comp_rewards) / len(comp_rewards) if comp_rewards else 0.0
                        comp_std = (sum((r - comp_mean) ** 2 for r in comp_rewards) / len(comp_rewards)) ** 0.5 if len(comp_rewards) > 1 else 0.0
                        actor_metrics[f"{comp_name}_mean"] = comp_mean
                        actor_metrics[f"{comp_name}_std"] = comp_std
                
                eval_metrics[actor_name] = actor_metrics
                
            # Log to console
            if self.main_accel.is_main_process:
                self.logger.quiet(colorize(f"📊 Evaluation Results for '{eval_name}':", Palette.BOLD))
                for actor_name, metrics in eval_metrics.items():
                    self.logger.quiet(colorize(f"   🎭 {actor_name}:", Palette.CYAN))
                    for metric_name, value in metrics.items():
                        self.logger.quiet(colorize(f"      • {metric_name}: {value:.3f}", Palette.CYAN))
                self.logger.quiet("")
                
            # Log to wandb
            if self.use_wandb and is_wandb_active() and self.main_accel.is_main_process:
                import wandb
                
                # Log eval metrics under eval/ section
                for actor_name, metrics in eval_metrics.items():
                    for metric_name, value in metrics.items():
                        wandb.log({f"eval/{eval_name}_{actor_name}_{metric_name}": value}, step=self._step)
                
                # Log eval completions table for each actor under eval/ section
                for actor_name, actor_output in env_out.actors.items():
                    if actor_name not in self.actors:
                        continue
                        
                    ta = self.actors[actor_name]
                    
                    # Prepare table columns
                    columns = list(eval_batch.keys()) + ["completion", "total_reward"]
                    if actor_output.reward_components:
                        columns.extend(actor_output.reward_components.keys())
                    
                    table = wandb.Table(columns=columns)
                    
                    # Add each completion as a row
                    for i in range(len(actor_output.input_ids)):
                        row = [eval_batch[k][i] for k in eval_batch.keys()]
                        completion = ta.tokenizer.decode(actor_output.input_ids[i], skip_special_tokens=False)
                        row.append(completion)
                        row.append(actor_output.rewards[i])
                        
                        if actor_output.reward_components:
                            for comp_name in actor_output.reward_components.keys():
                                row.append(actor_output.reward_components[comp_name][i])
                        
                        table.add_data(*row)
                    
                    # Add _final suffix for final evaluation table names, put under eval/ section
                    table_suffix = "_final" if is_final else f"_step_{self._logical_step}"
                    wandb.log({f"eval_completions/{eval_name}_{actor_name}_{table_suffix}": table}, step=self._step)
            
            # Store metrics for this eval dataset
            all_eval_metrics[eval_name] = eval_metrics
        
        if self.main_accel.is_main_process:
            self.logger.normal(colorize("✅ Evaluation completed", Palette.INFO))
            
        return all_eval_metrics

    # ═════════════════════════════════════════════════════════════════════
    # public API
    # ═════════════════════════════════════════════════════════════════════
    def train(
        self,
        epochs: int = 1,
        max_steps: Optional[int] = None,
        *,
        checkpoint_every_n: Optional[int] = 10,
        checkpoint_path: str = "checkpoints",
        max_checkpoints_to_keep: Optional[int] = 3,
        save_strategy: SaveStrategy = SaveStrategy.ALL,
    ):
        """
        Main training loop.
        Call exactly once; afterwards you can resume with `load_checkpoint()`
        and call `train()` again for the remaining epochs/steps.
        """
        if self.use_wandb and is_wandb_active() and self.main_accel.is_main_process:
            import wandb
            self._log_configuration_to_wandb(wandb)


        # unique folder for this training run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_checkpoint_path = os.path.join(checkpoint_path, f"run_{timestamp}")
        if self.main_accel.is_main_process:
            os.makedirs(run_checkpoint_path, exist_ok=True)

        start_time = time.time()
        total_steps = 0

        # total steps for ETA
        steps_per_epoch = len(self._dataloader)
        total_expected_steps = (
            max_steps * self.num_iterations if max_steps is not None else epochs * steps_per_epoch * self.num_iterations
        )

        # create schedulers now that we know total steps
        for name, ta in self.actors.items():
            sched = ta.actor.training_config.scheduler_factory(ta.optim, total_expected_steps)
            sched = ta.accel.prepare(sched)
            self.actors[name].sched = sched

        # fast-forward dataloader if we resumed mid-epoch
        dataloader = self._dataloader
        if self._data_state["step_in_epoch"]:
            dataloader = itertools.islice(
                dataloader, self._data_state["step_in_epoch"], None
            )

        for epoch in range(self._data_state["epoch"], epochs):
            for raw_batch in dataloader:
                # ensure lists, not tensors
                for k, v in raw_batch.items():
                    if isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
                        raw_batch[k] = [t.tolist() for t in v]

                if max_steps is not None and total_steps >= max_steps:
                    if self.main_accel.is_main_process:
                        self.logger.normal("Max steps reached, stopping training.")
                    return

                metrics = self.train_step(raw_batch)

                # ─── update sampler bookkeeping ───────────────────────────
                self._data_state["step_in_epoch"] += 1
                total_steps += 1

                # ─── checkpointing ────────────────────────────────────────
                if (
                    save_strategy in {SaveStrategy.STEPS, SaveStrategy.ALL}
                    and checkpoint_every_n
                    and self._logical_step % checkpoint_every_n == 0
                ):
                    path = os.path.join(run_checkpoint_path, f"step_{self._logical_step}")
                    if self.main_accel.is_main_process:
                        self.logger.quiet(
                            colorize(f"💾 Checkpoint saved: step_{self._logical_step}", Palette.VERB)
                        )
                    self.save_checkpoint(path)

                    # prune old checkpoints
                    if self.main_accel.is_main_process and max_checkpoints_to_keep is not None:
                        checkpoints = [
                            d for d in os.listdir(run_checkpoint_path)
                            if d.startswith("step_") and os.path.isdir(os.path.join(run_checkpoint_path, d))
                        ]
                        if len(checkpoints) > max_checkpoints_to_keep:
                            checkpoints.sort(key=lambda x: int(x.split("_")[1]))
                            for c in checkpoints[:-max_checkpoints_to_keep]:
                                shutil.rmtree(os.path.join(run_checkpoint_path, c))

                # ─── console / WandB logging (unchanged) ─────────────────
                if self.main_accel.is_main_process and self._logical_step % self.log_every_n == 0:
                    progress = (total_steps * self.num_iterations / total_expected_steps) * 100
                    eta_str = "N/A"
                    if total_steps:
                        elapsed = time.time() - start_time
                        eta_seconds = elapsed / total_steps * (total_expected_steps - total_steps)
                        eta_str = str(timedelta(seconds=int(eta_seconds)))

                    fractional_epoch = epoch + (self._data_state["step_in_epoch"] / steps_per_epoch)
                    header = (
                        f"STEP {self._logical_step:,}/{max_steps:,} ({progress:.1f}%) • ETA: {eta_str}"
                        if max_steps
                        else f"STEP {self._logical_step:,} • EPOCH {fractional_epoch:.2f}/{epochs} "
                             f"({progress:.1f}%) • ETA: {eta_str}"
                    )
                    if self.logger.isEnabledFor(QUIET):
                        self.logger.quiet(colorize(header, Palette.BOLD))

                        for actor_name, actor_metrics_list in metrics.items():
                            if not actor_metrics_list or not any(actor_metrics_list):
                                continue
                            self.logger.quiet(colorize(f"   🎭 {actor_name}:", Palette.CYAN))
                            for iteration_idx, actor_metrics in enumerate(actor_metrics_list):
                                if not actor_metrics:
                                    continue
                                indent = "      "
                                if self.num_iterations > 1:
                                    self.logger.quiet(
                                        colorize(f"      📊 Iter {iteration_idx+1}:", Palette.YELLOW)
                                    )
                                    indent = "         "
                                # Include reward component metrics in static set
                                static = {"completion_len", "reward_mean", "reward_std"}
                                for metric_name in actor_metrics.keys():
                                    if metric_name.endswith("_mean") or metric_name.endswith("_std"):
                                        static.add(metric_name)
                                        
                                for m, v in actor_metrics.items():
                                    if iteration_idx and m in static:
                                        continue
                                    if m == "learning_rate":
                                        self.logger.quiet(
                                            colorize(f"{indent}• {m}: {v:.2e}", Palette.CYAN)
                                        )
                                    else:
                                        self.logger.quiet(
                                            colorize(f"{indent}• {m}: {v:.3f}", Palette.CYAN)
                                        )
                        self.logger.quiet("")

                if self.use_wandb and is_wandb_active() and self.main_accel.is_main_process:
                    import wandb
                    # Build static metrics set dynamically to include reward components
                    static = {"completion_len", "reward_mean", "reward_std"}
                    
                    # Add reward component metrics to static set
                    for actor_metrics_list in metrics.values():
                        if actor_metrics_list:
                            for metric_name in actor_metrics_list[0].keys():
                                if metric_name.endswith("_mean") or metric_name.endswith("_std"):
                                    static.add(metric_name)
                    
                    for iteration_idx in range(self.num_iterations):
                        if iteration_idx > 0:
                            self._step += 1
                        for actor_name, actor_metrics_list in metrics.items():
                            if iteration_idx >= len(actor_metrics_list):
                                continue
                            for m, v in actor_metrics_list[iteration_idx].items():
                                if iteration_idx and m in static:
                                    continue
                                wandb.log({f"{actor_name}/{m}": v}, step=self._step)
                            # We also log the learning rate for each actor
                            learning_rate = self.actors[actor_name].sched.get_last_lr()[0]
                            wandb.log({f"{actor_name}/learning_rate": learning_rate}, step=self._step)

                # ─── evaluation ───────────────────────────────────────────
                if (
                    self.eval_datasets
                    and self.eval_strategy in {EvalStrategy.STEPS, EvalStrategy.ALL}
                    and self.eval_every_n is not None 
                    and self._logical_step % self.eval_every_n == 0
                ):
                    self.evaluate(is_final=False)

            # ─── epoch boundary ───────────────────────────────────────────
            self._data_state.update(
                epoch=self._data_state["epoch"] + 1,
                step_in_epoch=0,
                current_generator_seed=self._data_state["current_generator_seed"] + 1,
            )
            self._build_dataloader()
            dataloader = self._dataloader   # fresh iterator for next epoch
            
        # ─── final evaluation ─────────────────────────────────────────
        if (
            self.eval_datasets
            and self.eval_strategy in {EvalStrategy.FINAL, EvalStrategy.ALL}
        ):
            if self.main_accel.is_main_process:
                self.logger.normal(colorize("🎯 Running final evaluation...", Palette.INFO))
            self.evaluate(is_final=True)
            
        # ─── final model save ───────────────────────────────────────────
        if save_strategy in {SaveStrategy.FINAL, SaveStrategy.ALL}:
            final_path = os.path.join(run_checkpoint_path, "final_models")
            self.save_pretrained(final_path)
            if self.main_accel.is_main_process:
                self.logger.quiet(colorize("💾 Models saved", Palette.VERB))



    @gpu_profiler(name="train_step", use_wandb=True)
    def train_step(self, raw_batch: Dict[str, List[Any]]) -> Dict[str, Dict[str, float]]:
        """
        One environment interaction + optimisation step.

        Returns a nested dict of averaged metrics:
        `{actor_name: {"loss": float, "kl": float, ...}, ...}`
        """
        self._logical_step += 1
        self._step += 1  # Increment WandB step counter
        
        # Start profiling for this step
        start_step_profiling()
        
        # Log sampling stage in normal mode
        self.logger.normal(colorize("🎲 Sampling...", Palette.INFO))
        
        batch = expand_batch(raw_batch, self.group_size)
        
        # Track environment execution time
        with _step_profiler.track("environment"):
            env_out = self.env(batch)

        for actor_name, act_out in env_out.actors.items():
            for i, ended_in_eos in enumerate(act_out.ended_in_eos):
                if ended_in_eos:
                    # We get eos and check if the txt at this position has an EOS token
                    eos_token_id = self.actors[actor_name].tokenizer.eos_token_id
                    if act_out.input_ids[i][-1] != eos_token_id:
                        # If not, we append the EOS token to the input_ids
                        act_out.input_ids[i].append(eos_token_id)
                        act_out.attention_mask[i].append(1)
                    
        
        # Ensure we have the new EnvironmentOutput format
        if not isinstance(env_out, EnvironmentOutput):
            raise TypeError(f"Environment must return EnvironmentOutput, got {type(env_out)}")

        if self.use_wandb and is_wandb_active() and self.main_accel.is_main_process and self._logical_step % self.log_every_n == 0:
            import wandb
            for name, ta in self.actors.items():
                if name in env_out.actors:
                    actor_output = env_out.actors[name]
                    batch_keys = list(batch.keys())
                    completions_ids = actor_output.input_ids
                    total_rewards = actor_output.rewards
                    advantages = self._calculate_advantages(total_rewards, self.group_size, actor_output.ended_in_eos)

                    columns = batch_keys + ["completion", "total_reward", "advantage"]
                    if actor_output.reward_components:
                        component_names = list(actor_output.reward_components.keys())
                        columns.extend(component_names)
                    
                    table = wandb.Table(columns=columns)

                    for i in range(len(completions_ids)):
                        row = [batch[k][i] for k in batch_keys]
                        row.append(ta.tokenizer.decode(completions_ids[i], skip_special_tokens=False))
                        row.append(total_rewards[i])
                        row.append(advantages[i])
                        
                        if actor_output.reward_components:
                            for comp_name in component_names:
                                row.append(actor_output.reward_components[comp_name][i])
                        
                        table.add_data(*row)
                    
                    wandb.log({f"completions/{name}_step_{self._logical_step}": table}, step=self._step)

        # warn about unexpected actor keys (once per step, rank==0 only)
        if self.rank == 0:
            for k in env_out.actors.keys() - self.actors.keys():
                logger_method = self.logger.verbose if self.logger.isEnabledFor(VERBOSE) else self.logger.quiet
                logger_method(colorize(f"env produced data for unknown actor '{k}'",
                                             Palette.WARNING))

        base_metrics: Dict[str, List[float]] = {
            "loss":           [],
            "kl":             [],
            "grad_norm":      [],
            "completion_len": [],
            "reward_mean":    [],
            "reward_std":     [],
            "learning_rate":  [],
        }

        def make_template(actor_name: str) -> Dict[str, List[float]]:
            tpl = dict(base_metrics)
            actor_out = env_out.actors.get(actor_name)
            if actor_out and actor_out.reward_components:
                for comp in actor_out.reward_components:
                    tpl[f"{comp}_mean"] = []
                    tpl[f"{comp}_std"]  = []
            return tpl

        metrics: Dict[str, List[Dict[str, List[float]]]] = {
            name: [deepcopy(make_template(name)) for _ in range(self.num_iterations)]
            for name in self.actors
        }

        for name, ta in self.actors.items():
            if name not in env_out.actors:
                continue
            self._process_actor_step(name, ta, env_out.actors[name], metrics[name])

        # aggregate over grad-steps and (internally) over ranks
        out: Dict[str, List[Dict[str, float]]] = {}
        for name, bucket_list in metrics.items():
            ta = self.actors[name]
            beta = getattr(ta.loss_fn, "beta", 0.0)
            
            iter_stats: List[Dict[str, float]] = []
            for bucket in bucket_list:
                stats = {}
                for k, v in bucket.items():
                    if k == "kl" and beta == 0.0:
                        # Skip KL metric when beta is 0.0
                        continue
                    stats[k] = (sum(v) / len(v) if v else float("nan"))
                iter_stats.append(stats)
            out[name] = iter_stats

        # Log all profiling metrics - tracker handles everything automatically
        if self.main_accel.is_main_process:
            log_step_profiling(self._step, self.main_accel, use_wandb=self.use_wandb)

        return out

    # ------------------------------------------------------------------ #
    # internal – per-actor handling
    # ------------------------------------------------------------------ #
    def _process_actor_step(
        self,
        name: str,
        ta: InitializedTrainableLLMActor,
        actor_output: ActorOutput,
        buckets: List[Dict[str, List[float]]]) -> None:
        
        # Reload states before training if offloading is enabled
        if ta.actor.offload_optimizer or ta.actor.offload_model:
            with _step_profiler.track("reload_states", actor_name=name):
                reload_model_and_optimizer(
                    ta.model, ta.optim,
                    reload_optimizer=ta.actor.offload_optimizer,
                    reload_model=ta.actor.offload_model
                )
        total_rewards = actor_output.rewards
        advantages = self._calculate_advantages(total_rewards, self.group_size, actor_output.ended_in_eos)

        ids_list = actor_output.input_ids
        mask_list = actor_output.attention_mask

        # We assert they are the same exact lengths.
        assert len(ids_list) == len(mask_list) == len(total_rewards) == len(advantages), \
            f"Actor '{name}' output lengths mismatch: " \
            f"ids={len(ids_list)}, mask={len(mask_list)}, rewards={len(total_rewards)}, " \
            f"advantages={len(advantages)}"
        # All ids_list entries must have same length as mask_list entries
        assert all(len(ids) == len(mask) for ids, mask in zip(ids_list, mask_list)), \
            f"Actor '{name}' input_ids and attention_mask lengths mismatch: " \
            f"ids={len(ids_list)}, mask={len(mask_list)}"

        # compute reference log-ps once (if any)
        old_lp: Optional[Sequence[Sequence[float]]] = None
        ref_lp: Optional[Sequence[Sequence[float]]] = None

        with torch.no_grad():
            with _step_profiler.track("get_logps", actor_name=name):
                old_lp = self._get_logps(ta.model, ids_list, ta.tokenizer, 
                                        temperature=ta.loss_fn.temperature, batch_size=self.reference_batch_size) if self.num_iterations > 1 else None
                
                # Handle reference logits based on model type
                if ta.reference_model is not None:
                    # Traditional separate reference model
                    ref_lp = self._get_logps(ta.reference_model, ids_list, ta.tokenizer, temperature=ta.loss_fn.temperature, batch_size=self.reference_batch_size)
                elif is_peft_model(ta.model):
                    # PEFT model - disable adapter to get reference logits
                    with ta.model.disable_adapter():
                        ref_lp = self._get_logps(ta.model, ids_list, ta.tokenizer, temperature=ta.loss_fn.temperature, batch_size=self.reference_batch_size)
                else:
                    # No reference model needed
                    ref_lp = None
                
        # iterate over grad-accumulation micro-batches
        for it in range(self.num_iterations):
            # Log backprop iteration in normal mode
            if self.main_accel.is_main_process and self.logger.isEnabledFor(NORMAL):
                if self.num_iterations > 1:
                    self.logger.normal(colorize(f"🔄 Backwards iter {it+1}/{self.num_iterations} for actor '{name}'", Palette.INFO))
                else:
                    self.logger.normal(colorize(f"🔄 Backwards for actor '{name}'", Palette.INFO))
            
            # Track backward pass
            with _step_profiler.track("backward", actor_name=name):
                for adv_slice, id_slice, m_slice, old_slice, ref_slice in zip(
                    split_for_grad_accum(advantages, self.grad_accumulation_steps),
                    split_for_grad_accum(ids_list,   self.grad_accumulation_steps),
                    split_for_grad_accum(mask_list,  self.grad_accumulation_steps),
                    split_for_grad_accum(old_lp or [None]*len(ids_list), self.grad_accumulation_steps),
                    split_for_grad_accum(ref_lp or [None]*len(ids_list), self.grad_accumulation_steps),
                ):
                    self._backward_one_slice(
                        ta, id_slice, m_slice, adv_slice, ref_slice, old_slice, buckets[it]
                    )
                    free_memory()
            grad_norm = self._clip_gradients(
                ta,
                clip_to=self.max_grad_norm
            )
            buckets[it]["grad_norm"].append(grad_norm)
            free_memory()
            # Track optimizer step
            with _step_profiler.track("optim_step", no_memory_measurement=True, actor_name=name):
                self._optim_step(ta)

            free_memory()

            # total rewards / completion stats identical across iterations
            b = buckets[it]
            if not b["reward_mean"]:
                b["reward_mean"].append(sum(total_rewards) / len(total_rewards))
                b["reward_std"].append(torch.tensor(total_rewards).float().std(unbiased=False).item())
                
                # Add reward component statistics
                if actor_output.reward_components:
                    for comp_name, comp_rewards in actor_output.reward_components.items():
                        mean_key = f"{comp_name}_mean"
                        std_key = f"{comp_name}_std"
                        if mean_key in b:
                            b[mean_key].append(sum(comp_rewards) / len(comp_rewards))
                            b[std_key].append(torch.tensor(comp_rewards).float().std(unbiased=False).item())
            b["learning_rate"].append(ta.sched.get_last_lr()[0])

        
        # Offload states after training is complete for this actor
        if ta.actor.offload_optimizer:
            with _step_profiler.track("offload_optimizer", actor_name=name):
                offload_model_and_optimizer(
                    ta.model, ta.optim, 
                    offload_optimizer=True,
                    offload_model=False
                )

        # Track actor weight update
        self._update_actor_weights(ta)
        ta.actor.sleep()

    # ------------------------------------------------------------------ #
    def _backward_one_slice(
        self,
        ta: InitializedTrainableLLMActor,
        ids: List[List[int]],
        masks: List[List[int]],
        advantages: List[float],
        ref_lp_slice: Optional[List[List[float]]],
        old_lp_slice: List[List[float]],
        bucket: Dict[str, List[float]],
    ) -> None:
        tok, dev = ta.tokenizer, ta.model.device
        padded = self._pad(tok, ids)
        ids_pt, attention_mask = padded["input_ids"].to(dev), padded["attention_mask"].to(dev)


        max_len = ids_pt.size(1) - 1
        def to_tensor(slice_):
            t = torch.zeros(len(slice_), max_len, dtype=torch.float32, device=dev)
            for i,row in enumerate(slice_):
                n = min(len(row), max_len)
                if n: t[i,:n] = torch.tensor(row[:n], dtype=torch.float32, device=dev)
            return t
        ref_lp = to_tensor(ref_lp_slice) if any(ref_lp_slice) else None
        old_lp = to_tensor(old_lp_slice) if any(old_lp_slice) else None
        loss_attention_mask = to_tensor([x[1:] for x in masks]) if masks else None

        adv_pt  = torch.tensor(advantages, dtype=torch.float32, device=dev)

        loss, stats = ta.loss_fn(
            policy       = ta.accel.unwrap_model(ta.model),
            input_ids    = ids_pt,
            attention_mask=attention_mask,
            loss_attention_mask=loss_attention_mask,
            advantages   = adv_pt,
            ref_logps    = ref_lp,
            old_logps    = old_lp,
        )
        ta.accel.backward(loss)

        bucket["loss"].append(loss.item())
        if "kl" in stats and getattr(ta.loss_fn, "beta", 0.0) != 0.0:
            bucket["kl"].append(stats["kl"])
        bucket["completion_len"].append(attention_mask[:,1:].sum(-1).float().mean().item())


    # ------------------------------------------------------------------ #
    def _optim_step(self, ta: InitializedTrainableLLMActor) -> None:
        ta.optim.step()
        ta.sched.step()
        ta.optim.zero_grad()
        ta.accel.wait_for_everyone()

    # ------------------------------------------------------------------ #
    def _update_actor_weights(self, ta: InitializedTrainableLLMActor) -> None:
        
        with _step_profiler.track("update_weights", actor_name=ta.name):

            # On each node, the local main process will orchestrate the update.
            if self.main_accel.is_local_main_process:
                ta.actor.start_weight_update()

            # This callback will be executed on the local main process of each node for each batch.
            def stream_batch_callback(batch_state_dict):
                if self.main_accel.is_local_main_process:
                    ta.actor.update_weights_batch(batch_state_dict)

            gather_and_stream_state_dict(
                ta.accel, self.logger, 
                ta.model, stream_batch_callback, 
                tie_word_embeddings=ta.model_config['tie_word_embeddings'], 
                lora_only=is_peft_model(ta.model)
            )

        # We offload the model before finalizing the weight update.
        if ta.actor.offload_model:
            with _step_profiler.track("offload_model", actor_name=ta.name):
                offload_info = offload_model_and_optimizer(
                    ta.model, ta.optim,
                    offload_optimizer=False,
                    offload_model=True
                )
                if self.main_accel.is_main_process and self.logger.isEnabledFor(NORMAL):
                    if offload_info["model_offloaded"]:
                        self.logger.normal(colorize(f"💤 Offloaded model", Palette.INFO))

        if self.main_accel.is_local_main_process:
            # For LoRA models, use update_lora_weights instead of finalize_weight_update
            if is_peft_model(ta.model):
                ta.actor.update_lora_weights()
            else:
                ta.actor.finalize_weight_update()

    def _setup_loras(self) -> str:
        import tempfile
        
        # Create a temporary directory for the LoRA adapter
        if self.main_accel.is_local_main_process:
            temp_dir = tempfile.mkdtemp(prefix="lora_adapter_")
        
        # Sync the path.
        temp_dir = self.main_accel.gather_for_metrics([(self.rank,temp_dir)] if self.main_accel.is_local_main_process else [(self.rank,None)])
        # Get the largest rank's temp_dir that is is not None and less than our rank
        temp_dir = max((d for d in temp_dir if d is not None and d[0] <= self.rank), key=lambda x: x[0])[1]
        self.save_pretrained(temp_dir)

        multiple_actors = len(self.actors) > 1
        # Initialize LoRA in the vLLM actor if it's a vLLMActor
        if self.main_accel.is_local_main_process:
            for name, ta in self.actors.items():
                if ta.actor.has_peft_config:
                    print(os.listdir(os.path.join(temp_dir, name)) if multiple_actors else os.listdir(temp_dir))
                    ta.actor.create_lora_if_not_present(os.path.join(temp_dir, name) if multiple_actors else temp_dir)
            
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        return temp_dir

    # ------------------------------------------------------------------ #
    # checkpointing
    # ------------------------------------------------------------------ #
    def save_checkpoint(self, path: str):
        """
        Save every actor's DeepSpeed engine plus the trainer's
        bookkeeping (steps, RNG, sampler position…).

        Layout:
            <path>/
                trainer_state.pt
                <actor_0>/  (DeepSpeed files)
                <actor_1>/
                …
        """
        self.main_accel.wait_for_everyone()

        if self.main_accel.is_main_process:
            os.makedirs(path, exist_ok=True)

        # 1️⃣ model/optim/sched for each actor
        for name, ta in self.actors.items():
            subdir = os.path.join(path, name)
            if ta.accel.is_main_process:
                os.makedirs(subdir, exist_ok=True)
            ta.accel.save_state(output_dir=subdir)

        self.main_accel.wait_for_everyone()

        # 2️⃣ trainer-level bookkeeping (once)
        if self.main_accel.is_main_process:
            torch.save(
                {
                    "step":           self._step,
                    "logical_step":   self._logical_step,
                    "data_state":     self._data_state,
                    "rng_state":      self._rng.get_state(),
                },
                os.path.join(path, "trainer_state.pt"),
            )
        self.main_accel.wait_for_everyone()

    # ------------------------------------------------------------------
    def load_checkpoint(self, path: str):
        """
        Restore trainer bookkeeping **and** each actor's DeepSpeed engine.
        Afterwards the vLLM weights are pushed to the actors as usual.
        """
        # 1️⃣ trainer bookkeeping --------------------------------------
        state = torch.load(os.path.join(path, "trainer_state.pt"), map_location="cpu")
        self._step          = state["step"]
        self._logical_step  = state["logical_step"]
        self._data_state    = state["data_state"]

        self._rng = torch.Generator()
        self._rng.set_state(state["rng_state"])
        self._build_dataloader()

        # 2️⃣ per-actor engine state -----------------------------------
        for name, ta in self.actors.items():
            subdir = os.path.join(path, name)

            # make sure the right plugin is active for *this* accelerator
            # ta.accel.state.select_deepspeed_plugin(name)
            ta.accel.load_state(subdir)

        self.main_accel.wait_for_everyone()

        # 3️⃣ push weights into the serving actors ---------------------
        if self.main_accel.is_main_process:
            self.logger.normal(colorize("🔄 Updating actor weights from checkpoint…", Palette.INFO))

        for ta in self.actors.values():
            self._update_actor_weights(ta)
            if self.main_accel.is_local_main_process:
                ta.actor.sleep()
                if self.main_accel.is_main_process:
                    self.logger.normal(
                        colorize(f"😴 Actor '{ta.name}' put to sleep after resume", Palette.INFO)
                    )

    def save_pretrained(self, output_dir: str):
        """
        Save all actors' models **and** tokenizers so they can later be
        re-loaded with `from_pretrained`.

        • If there is a single actor, files are written directly to
          ``output_dir`` (mirrors HF default).
        • If there are >1 actors, each actor gets its own sub-folder,
          ``output_dir/<actor_name>/``.

        Only the model and tokenizer are saved - no optimiser / scheduler
        state (just like HF `Trainer.save_model`).
        """
        os.makedirs(output_dir, exist_ok=True)
        multi = len(self.actors) > 1

        for name, ta in self.actors.items():
            # reload if offloaded
            if ta.actor.offload_model:
                reload_model_and_optimizer(ta.model, ta.optim, reload_model=True, reload_optimizer=False)
            tgt = os.path.join(output_dir, name) if multi else output_dir
            os.makedirs(tgt, exist_ok=True)
            
            state_dict = ta.accel.get_state_dict(ta.model)


            ta.accel.unwrap_model(ta.model, keep_torch_compile=False).save_pretrained(
                tgt, state_dict=state_dict, safe_serialization=True,
            )

            if ta.tokenizer is not None:
                ta.tokenizer.save_pretrained(tgt)
            else:
                warnings.warn(f"Actor '{name}' has no tokenizer - skipped.")
            
            # offload.
            if ta.actor.offload_model:
                offload_model_and_optimizer(
                    ta.model, ta.optim,
                    offload_optimizer=False,
                    offload_model=True
                )
    # ------------------------------------------------------------------ #
    # Hugging Face Hub upload
    # ------------------------------------------------------------------ #
    def push_to_hub(
        self,
        repo_map: Union[str, Dict[str, str]],
        *,
        private: bool = False,
        commit_message: str | None = None,
        **push_kwargs,
    ):
        """
        Upload model(s) & tokenizer(s) to the Hugging Face Hub.

        Parameters
        ----------
        repo_map
            • str  - repository ID for a **single** actor setup.
            • dict - {actor_name: repo_id} for multi-actor setups.
        private
            Create / push to a private repository.
        commit_message
            Git commit message.
        **push_kwargs
            Forwarded to `model.push_to_hub` / `tokenizer.push_to_hub`.
        """
        if commit_message is None:
            commit_message = "Upload model trained with Actors"

        # ─── single-actor convenience ───────────────────────────────────
        if isinstance(repo_map, str):
            if len(self.actors) != 1:
                raise ValueError(
                    "repo_map is a string but multiple actors exist. "
                    "Provide a dict mapping each actor to a repo ID."
                )
            name, ta = next(iter(self.actors.items()))
            self._push_single_actor(
                ta,
                repo_map,
                private=private,
                commit_message=commit_message,
                **push_kwargs,
            )
            return

        # ─── multi-actor case ───────────────────────────────────────────
        if not isinstance(repo_map, dict):
            raise TypeError("repo_map must be a str or a dict[str, str].")

        for name, ta in self.actors.items():
            if name not in repo_map:
                raise KeyError(f"No repo ID supplied for actor '{name}'.")
            self._push_single_actor(
                ta,
                repo_map[name],
                private=private,
                commit_message=commit_message,
                **push_kwargs,
            )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _push_single_actor(
        ta: TrainableLLMActor,
        repo_id: str,
        *,
        private: bool,
        commit_message: str,
        **push_kwargs,
    ):
        # 1️⃣ model
        unwrapped_model = ta.accel.unwrap_model(ta.model)
        unwrapped_model.push_to_hub(
            repo_id,
            private=private,
            commit_message=commit_message,
            **push_kwargs,
        )
        # 2️⃣ tokenizer (if present)
        if ta.tokenizer is not None:
            ta.tokenizer.push_to_hub(
                repo_id,
                private=private,
                commit_message=commit_message,
                **push_kwargs,
            )

def is_peft_model(model):
    """Check if a model is a PEFT model."""
    return hasattr(model, 'peft_config') and hasattr(model, 'base_model')

# ═════════════════════════════════════════════════════════════════════════════