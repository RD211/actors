from dataclasses import dataclass
from datetime import datetime, timedelta
import os
import shutil
import tempfile
import time
import warnings
import deepspeed

from peft import get_peft_model
import torch

from torch.utils.data import Dataset, DataLoader, RandomSampler
from accelerate import Accelerator, InitProcessGroupKwargs
from transformers import PreTrainedTokenizerBase
from datasets import Dataset as HFDataset, DatasetDict
from accelerate.utils import DeepSpeedPlugin, DistributedType
from actors.actors.base import TrainableLLMActor
from actors.environments.env_base import Environment
from actors.environments.types import EnvironmentOutput, GroupedEnvironmentOutput
from actors.losses.base_loss import BaseRLLoss
from typing import Dict, Any, List, Optional, Union
from enum import Enum, auto
from transformers import AutoConfig

from actors.utils.deepspeed import (
    offload_model_and_optimizer,
    prepare_deepspeed,
    reload_model_and_optimizer,
)
from actors.utils.logger import Palette, colorize, init_logger
from actors.utils.ipc_utils import gather_and_stream_state_dict
from actors.utils.tracker import (
    start_step_profiling,
    log_step_profiling,
    _step_profiler,
    gpu_profiler,
)
from actors.utils.train_utils import disable_dropout_in_model, free_memory
from actors.utils.wandb import is_wandb_active
from actors.environments.types import ActorOutput
from actors.utils.get_logps import _chunked_logp
from peft import prepare_model_for_kbit_training

# ═════════════════════════════════════════════════════════════════════════════
# Utility functions
# ═════════════════════════════════════════════════════════════════════════════


def is_peft_model(model: torch.nn.Module) -> bool:
    """Check if a model is a PEFT model."""
    return hasattr(model, "peft_config") and hasattr(model, "base_model")


# ═════════════════════════════════════════════════════════════════════════════
# Utility dataclasses
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class TrainingMetrics:
    """
    Unified container for training step results.

    Structure:
    - substep_metrics: Dict[actor_name -> List[Dict[metric_name -> value]]] (varies per iteration)
    - step_metrics: Dict[actor_name -> Dict[metric_name -> value]] (constant per step)
    - completions: Dict[actor_name -> Dict[column_name -> List[values]]] (once per step)
    """

    substep_metrics: Dict[str, List[Dict[str, float]]] = (
        None  # actor -> [substep_metrics]
    )
    step_metrics: Dict[str, Dict[str, float]] = None  # actor -> step_metrics
    completions: Dict[str, Dict[str, List[Any]]] = None  # actor -> completion_data

    def __post_init__(self):
        if self.substep_metrics is None:
            self.substep_metrics = {}
        if self.step_metrics is None:
            self.step_metrics = {}
        if self.completions is None:
            self.completions = {}

    def add_substep_metric(
        self, actor_name: str, substep_idx: int, metric_name: str, value: float
    ):
        """Add a metric that varies per iteration (loss, kl, grad_norm, learning_rate)."""
        if actor_name not in self.substep_metrics:
            self.substep_metrics[actor_name] = []

        # Ensure we have enough substeps
        while len(self.substep_metrics[actor_name]) <= substep_idx:
            self.substep_metrics[actor_name].append({})

        self.substep_metrics[actor_name][substep_idx][metric_name] = value

    def add_step_metric(self, actor_name: str, metric_name: str, value: float):
        """Add a metric that's constant per step (reward_mean, reward_std, completion_len, etc.)."""
        if actor_name not in self.step_metrics:
            self.step_metrics[actor_name] = {}

        self.step_metrics[actor_name][metric_name] = value

    def add_actor_rewards(self, actor_name: str, rewards: List[float]):
        """Helper to add reward statistics for an actor."""
        if rewards:
            mean_value = sum(rewards) / len(rewards)
            std_value = (
                torch.tensor(rewards).float().std(unbiased=False).item()
                if len(rewards) > 1
                else 0.0
            )
            self.add_step_metric(actor_name, "reward_mean", mean_value)
            self.add_step_metric(actor_name, "reward_std", std_value)

    def add_actor_reward_component(
        self, actor_name: str, component_name: str, rewards: List[float]
    ):
        """Helper to add reward component statistics for an actor."""
        if rewards:
            mean_value = sum(rewards) / len(rewards)
            std_value = (
                torch.tensor(rewards).float().std(unbiased=False).item()
                if len(rewards) > 1
                else 0.0
            )
            self.add_step_metric(actor_name, f"{component_name}_mean", mean_value)
            self.add_step_metric(actor_name, f"{component_name}_std", std_value)

    def add_completion_data(self, actor_name: str, data: Dict[str, List[Any]]):
        """Add completion data for an actor."""
        self.completions[actor_name] = data

    def get_combined_metrics(self) -> Dict[str, List[Dict[str, float]]]:
        """Combine step and substep metrics for logging."""
        result = {}

        for actor_name in set(self.substep_metrics.keys()) | set(
            self.step_metrics.keys()
        ):
            actor_substeps = self.substep_metrics.get(actor_name, [])
            actor_step_metrics = self.step_metrics.get(actor_name, {})

            # Ensure we have at least one substep
            if not actor_substeps:
                actor_substeps = [{}]

            # Add step metrics only to the first substep
            combined_substeps = []
            for idx, substep_metrics in enumerate(actor_substeps):
                combined = dict(substep_metrics)
                if idx == 0:  # Add step metrics only to first substep
                    combined.update(actor_step_metrics)
                combined_substeps.append(combined)

            result[actor_name] = combined_substeps

        return result


@dataclass
class EvaluationMetrics:
    """
    Container for evaluation step results.

    Structure:
    - metrics: Dict[actor_name -> Dict[metric_name -> value]] (evaluation metrics)
    - completions: Dict[actor_name -> Dict[column_name -> List[values]]] (completion data)
    """

    metrics: Dict[str, Dict[str, float]] = None
    completions: Dict[str, Dict[str, List[Any]]] = None

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}
        if self.completions is None:
            self.completions = {}

    def add_actor_metrics(self, actor_name: str, metrics: Dict[str, float]):
        """Add evaluation metrics for an actor."""
        self.metrics[actor_name] = metrics

    def add_completion_data(self, actor_name: str, data: Dict[str, List[Any]]):
        """Add completion data for an actor."""
        self.completions[actor_name] = data


@dataclass
class InitializedTrainableLLMActor:
    """
    The initialized state of a TrainableLLMActor.
    The TrainableLLMActor contains mostly factories and methods to create these objects.
    This dataclass holds the initialized objects that are ready to be used for training.
    """

    name: str
    actor: TrainableLLMActor
    model: torch.nn.Module
    tokenizer: PreTrainedTokenizerBase
    loss_fn: BaseRLLoss
    optim: torch.optim.Optimizer
    accel: Accelerator
    model_config: Dict[str, Any]  # used for stuff like tie_embeddings, etc.
    reference_model: Optional[torch.nn.Module] = None
    sched: Optional[torch.optim.lr_scheduler.LRScheduler] = None
    beta: float = 0.0001


class SaveStrategy(Enum):
    NONE = auto()  # never save
    STEPS = auto()  # checkpoint_every_n only
    FINAL = auto()  # one model save at the very end
    ALL = auto()  # both periodic + final


class EvalStrategy(Enum):
    NONE = auto()  # never evaluate
    STEPS = auto()  # evaluate every eval_every_n steps
    FINAL = auto()  # evaluate only at the end
    ALL = auto()  # evaluate both periodically and at the end


# ═══════════════════════════════════════════════════════════════════════
# Trainer configuration
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class TrainerCfg:

    # Training
    epochs: int = 1
    batch_size: int = 8
    max_steps: Optional[int] = None
    grad_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True
    num_iterations: int = 1
    group_size: int = 1 

    # Logging
    log_every_n: int = 1
    use_wandb: bool = True

    # Eval
    eval_every_n: int = 1000
    eval_strategy: EvalStrategy = EvalStrategy.ALL

    # Checkpointing
    save_strategy: SaveStrategy = SaveStrategy.ALL
    checkpoint_every_n: int = 1000
    max_checkpoints_to_keep: int = 3
    checkpoint_path: str = "checkpoints"


    # TODO: Implement this.
    sync_ref_model: bool = False
    sync_ref_model_every_n: int = 1000

# ═══════════════════════════════════════════════════════════════════════
# Base RLTrainer class
# ═══════════════════════════════════════════════════════════════════════


class BaseRLTrainer:
    def __init__(
        self,
        cfg: TrainerCfg,
        env: Environment,
    ):
        self.cfg = cfg
        self.env = env
        self.logger = init_logger("trainer")
        self._step = 0
        self._substep = 0  # Will be incremented before first use

        # Get actors from environment if not provided
        trainable_actors = env.get_trainable_actors()
        if not trainable_actors:
            raise ValueError(
                "No trainable actors found in the environment. "
                "Please ensure the environment has registered actors."
            )
        self.actors = self._setup_actors(trainable_actors)
        self._setup_loras()

    # ═══════════════════════════════════════════════════════════════════════
    # Actor setup methods
    # ═══════════════════════════════════════════════════════════════════════

    def _setup_actors(
        self, trainable_actors: Dict[str, TrainableLLMActor]
    ) -> Dict[str, InitializedTrainableLLMActor]:
        """
        Initializes the trainable actors and returns a dictionary of initialized actors.
        """
        self.ds_plugins: Dict[str, DeepSpeedPlugin] = {
            name: DeepSpeedPlugin() for name in trainable_actors.keys()
        }

        for name, plug in self.ds_plugins.items():
            cfg = plug.deepspeed_config
            cfg["max_grad_norm"] = self.cfg.max_grad_norm
            cfg["train_batch_size"] = self.cfg.batch_size
            cfg["gradient_accumulation_steps"] = self.cfg.grad_accumulation_steps
            cfg["train_micro_batch_size_per_gpu"] = (
                self.cfg.batch_size
                // self.cfg.grad_accumulation_steps
                // torch.cuda.device_count()
            )

        actors_with_offloading = {
            name: actor
            for name, actor in trainable_actors.items()
            if actor.offload_optimizer or actor.offload_model
        }

        if actors_with_offloading:
            first_plugin = next(iter(self.ds_plugins.values()))
            zero_config = first_plugin.deepspeed_config.get("zero_optimization", {})
            zero_stage = zero_config.get("stage", 2)

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

        accelerators: Dict[str, Accelerator] = {
            actor: Accelerator(
                mixed_precision="bf16",
                deepspeed_plugin=self.ds_plugins[actor],
                kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(hours=10))],
            )
            for actor in self.ds_plugins
        }

        actors: Dict[str, InitializedTrainableLLMActor] = {}

        for name, actor_obj in trainable_actors.items():
            accel = accelerators[name]
            model = actor_obj.model_factory().train()

            self.logger.normal(
                colorize(
                    f"🔧 Initializing actor '{name}'",
                    Palette.INFO,
                )
            )

            # We check if the model has a quantization_config
            if hasattr(model.config, "quantization_config"):
                if model.config.quantization_config.bnb_4bit_quant_storage != torch.bfloat16:
                    raise ValueError(
                        f"Expected bnb_4bit_quant_storage to be torch.bfloat16, but got {model.config.quantization_config.bnb_4bit_quant_storage}, consider making a custom model factory."
                    )
                prepare_model_for_kbit_training(model)
                
            # Apply PEFT configuration if available
            if actor_obj.training_config.peft_config is not None:
                model = get_peft_model(model, actor_obj.training_config.peft_config).train()

            if self.cfg.gradient_checkpointing:
                if is_peft_model(model):

                    model.base_model.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={
                            "use_reentrant": True,
                        }
                    )
                    model.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={
                            "use_reentrant": True,
                        }
                    )

                    model.config.use_cache = False
                    model.base_model.config.use_cache = False

                    model.enable_input_require_grads()
                    model.base_model.enable_input_require_grads()
                else:
                    model.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={
                            "use_reentrant": True,
                        }
                    )
                    model.config.use_cache = False
                    model.enable_input_require_grads()

            disable_dropout_in_model(model)
            model.train()
            model_cfg = model.config.to_dict()

            optim = actor_obj.training_config.optim_factory(model.parameters())
            model, optim = accel.prepare(model, optim)

            loss_fn = actor_obj.training_config.loss_factory()
            beta = getattr(loss_fn, "beta", 0.0)

            if (
                beta == 0.0
                or is_peft_model(model)
                or not actor_obj.training_config.reference_model_factory
            ):
                ref_model = None
            else:
                ref_model = actor_obj.training_config.reference_model_factory().eval()

                ref_model.requires_grad_(False)
                ref_model.config.use_cache = False

                ref_model = ref_model.to(dtype=model.dtype)
                ref_model = prepare_deepspeed(ref_model, accel)
                disable_dropout_in_model(ref_model)

            sched = actor_obj.training_config.scheduler_factory(
                optim, self.total_expected_steps
            )
            sched = accel.prepare(sched)

            actors[name] = InitializedTrainableLLMActor(
                name=name,
                actor=actor_obj,
                model=model,
                tokenizer=actor_obj.tokenizer,
                loss_fn=loss_fn,
                optim=optim,
                sched=sched,
                reference_model=ref_model,
                accel=accel,
                model_config=model_cfg,
                beta=beta,
            )

            # We offload the model and optimizer if requested
            if actor_obj.offload_optimizer or actor_obj.offload_model:
                offload_model_and_optimizer(
                    actors[name].model,
                    actors[name].optim,
                    offload_optimizer=actor_obj.offload_optimizer,
                    offload_model=actor_obj.offload_model,
                )

        return actors

    def _setup_loras(self) -> str:
        #TODO: If not loras. there is no need to do all of this.
        os.makedirs(self.cfg.checkpoint_path, exist_ok=True)

        if self.accel.is_main_process:
            temp_dir = tempfile.mkdtemp(
                prefix="lora_adapter_", dir=self.cfg.checkpoint_path
            )

        temp_dir = self.accel.gather_for_metrics(
            [(self.rank, temp_dir)]
            if self.accel.is_main_process
            else [(self.rank, None)]
        )

        temp_dir = [d for _, d in temp_dir if d is not None][0]

        self.save_pretrained(temp_dir)

        multiple_actors = len(self.actors) > 1
        if self.accel.is_local_main_process:
            for name, ta in self.actors.items():
                if ta.actor.has_peft_config:
                    ta.actor.create_lora_if_not_present(
                        os.path.join(temp_dir, name) if multiple_actors else temp_dir
                    )
        self.accel.wait_for_everyone()
        if self.accel.is_main_process:
            shutil.rmtree(temp_dir, ignore_errors=True)

        return temp_dir

    # ═══════════════════════════════════════════════════════════════════════
    # Training parts
    # ═══════════════════════════════════════════════════════════════════════

    def _update_actor_weights(self, ta: InitializedTrainableLLMActor) -> None:

        with _step_profiler.track("update_weights", actor_name=ta.name):

            if ta.accel.is_local_main_process:
                ta.actor.start_weight_update()

            def stream_batch_callback(batch_state_dict):
                if ta.accel.is_local_main_process:
                    ta.actor.update_weights_batch(batch_state_dict)

            gather_and_stream_state_dict(
                ta.accel,
                self.logger,
                ta.model,
                stream_batch_callback,
                tie_word_embeddings=ta.model_config["tie_word_embeddings"],
                lora_only=is_peft_model(ta.model),
            )

        if ta.actor.offload_model:
            with _step_profiler.track("offload_model", actor_name=ta.name):
                offload_info = offload_model_and_optimizer(
                    ta.model, ta.optim, offload_optimizer=False, offload_model=True
                )
                if self.accel.is_main_process:
                    if offload_info["model_offloaded"]:
                        self.logger.normal(
                            colorize(f"💤 Offloaded model", Palette.INFO)
                        )

        if self.accel.is_local_main_process:
            if is_peft_model(ta.model):
                ta.actor.update_lora_weights()
            else:
                ta.actor.finalize_weight_update()

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

        if ta.accel.distributed_type == DistributedType.DEEPSPEED:
            grad_norm = ta.model.get_global_grad_norm()
            # In some cases the grad norm may not return a float
            if hasattr(grad_norm, "item"):
                grad_norm = grad_norm.item()
        else:
            grad_norm = _grad_norm

        return grad_norm

    def _optim_step(self, ta: InitializedTrainableLLMActor) -> None:
        ta.optim.step()
        ta.sched.step()
        ta.optim.zero_grad()
        ta.accel.wait_for_everyone()

    def train(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_checkpoint_path = os.path.join(self.cfg.checkpoint_path, f"run_{timestamp}")
        if self.accel.is_main_process:
            os.makedirs(run_checkpoint_path, exist_ok=True)

        start_time = time.time()
        total_steps = 0
        
        dataloader = self.env.get_dataloader(self.batch_size // self.group_size)
        if dataloader is None:
            raise ValueError("Environment has no training data")
        
        steps_per_epoch = len(dataloader)
        
        while True:
            if self.cfg.max_steps is not None and self._step >= self.cfg.max_steps:
                self.logger.normal(
                    colorize(
                        f"🏁 Reached max steps {self.cfg.max_steps:,}, stopping training.",
                        Palette.INFO,
                    )
                )
                break
                
            if self.cfg.max_steps is None:
                current_epoch = self.env.get_data_state()["epoch"]
                if current_epoch >= self.cfg.epochs:
                    break
                    
                if self.env.batches_left(self.batch_size // self.group_size) == 0:
                    break

            self._step += 1

            start_step_profiling()
            with _step_profiler.track(
                "full_train_step", no_memory_measurement=True
            ):
                try:
                    env_output = self.env(batch_size=self.batch_size // self.group_size, group_size=self.group_size)
                    result = self.train_step(env_output)
                except StopIteration:
                    break
                    
            log_step_profiling(
                self._substep, self.accel, use_wandb=self.cfg.use_wandb
            )

            metrics = result.get_combined_metrics()
            
            if self._step % self.log_every_n == 0:
                self.log_training_metrics(metrics)
                for actor_name, completion_data in result.completions.items():
                    if completion_data:
                        enhanced_completion_data = {}
                        
                        if env_output.problems:
                            completion_length = len(next(iter(completion_data.values())))
                            
                            for key in env_output.problems[0].keys():
                                enhanced_completion_data[key] = []
                                for problem in env_output.problems:
                                    for _ in range(completion_length // len(env_output.problems)):
                                        enhanced_completion_data[key].append(problem[key])
                        
                        enhanced_completion_data.update(completion_data)
                        
                        self.log_completions(
                            enhanced_completion_data, prefix=f"completions/{actor_name}"
                        )

            if (
                self.cfg.save_strategy in {SaveStrategy.STEPS, SaveStrategy.ALL}
                and self.cfg.checkpoint_every_n
                and self._step % self.cfg.checkpoint_every_n == 0
            ):
                path = os.path.join(run_checkpoint_path, f"step_{self._step}")
                if self.accel.is_main_process:
                    self.logger.quiet(
                        colorize(
                            f"💾 Checkpoint saved: step_{self._step}",
                            Palette.VERB,
                        )
                    )
                self.save_checkpoint(path)

                if (
                    self.accel.is_main_process
                    and self.cfg.max_checkpoints_to_keep is not None
                ):
                    checkpoints = [
                        d
                        for d in os.listdir(run_checkpoint_path)
                        if d.startswith("step_")
                        and os.path.isdir(os.path.join(run_checkpoint_path, d))
                    ]
                    if len(checkpoints) > self.cfg.max_checkpoints_to_keep:
                        checkpoints.sort(key=lambda x: int(x.split("_")[1]))
                        for c in checkpoints[: -self.cfg.max_checkpoints_to_keep]:
                            shutil.rmtree(os.path.join(run_checkpoint_path, c))

            if self.accel.is_main_process and self._step % self.log_every_n == 0:
                progress = (
                    total_steps * self.num_iterations / self.total_expected_steps
                ) * 100
                eta_str = "N/A"
                if total_steps:
                    elapsed = time.time() - start_time
                    eta_seconds = (
                        elapsed
                        / total_steps
                        * (self.total_expected_steps - total_steps)
                    )
                    eta_str = str(timedelta(seconds=int(eta_seconds)))

                current_data_state = self.env.get_data_state()
                fractional_epoch = current_data_state["epoch"] + (current_data_state["step_in_epoch"] / steps_per_epoch)
                header = (
                    f"STEP {self._step:,}/{self.cfg.max_steps:,} ({progress:.1f}%) • ETA: {eta_str}"
                    if self.cfg.max_steps
                    else f"STEP {self._step:,} • EPOCH {fractional_epoch:.2f}/{self.cfg.epochs} "
                    f"({progress:.1f}%) • ETA: {eta_str}"
                )
                self.logger.quiet(colorize(header, Palette.BOLD))

            if (
                self.env.eval_datasets
                and self.eval_strategy in {EvalStrategy.STEPS, EvalStrategy.ALL}
                and self.eval_every_n is not None
                and self._step % self.eval_every_n == 0
            ):
                self.evaluate(is_final=False)
            
            total_steps += 1

        if self.env.eval_datasets and self.eval_strategy in {
            EvalStrategy.FINAL,
            EvalStrategy.ALL,
        }:
            if self.accel.is_main_process:
                self.logger.normal(
                    colorize("🎯 Running final evaluation...", Palette.INFO)
                )
            self.evaluate(is_final=True)

        if self.cfg.save_strategy in {SaveStrategy.FINAL, SaveStrategy.ALL}:
            final_path = os.path.join(run_checkpoint_path, "final_models")
            self.save_pretrained(final_path)
            if self.accel.is_main_process:
                self.logger.quiet(colorize("💾 Models saved", Palette.VERB))

    def train_step(self, env_output: GroupedEnvironmentOutput) -> TrainingMetrics:
        """Train step that takes GroupedEnvironmentOutput directly."""
        raise NotImplementedError("train_step must be implemented in subclasses.")

    # ═══════════════════════════════════════════════════════════════════════
    # Eval
    # ═══════════════════════════════════════════════════════════════════════

    def evaluate(self, is_final: bool = False) -> Optional[Dict[str, Any]]:
        if not self.env.eval_datasets:
            return None

        if self.accel.is_main_process:
            self.logger.normal(colorize("🔍 Starting evaluation...", Palette.INFO))

        # Use environment's eval method to get all eval results
        eval_outputs = self.env.eval(group_size=1)
        
        all_eval_metrics = {}
        for eval_name, env_output in eval_outputs.items():
            result = self.eval_step(env_output)
            self.log_evaluation_metrics(eval_name, result, env_output, is_final)
            all_eval_metrics[eval_name] = result.metrics

        if self.accel.is_main_process:
            self.logger.normal(colorize("✅ Evaluation completed", Palette.INFO))

        return all_eval_metrics

    def eval_step(self, env_output: GroupedEnvironmentOutput) -> EvaluationMetrics:
        """Generic eval step implementation."""
        result = EvaluationMetrics()

        flat_output = env_output.to_environment_output()

        for actor_name, actor_output in flat_output.actors.items():
            if actor_name not in self.actors:
                continue

            ta = self.actors[actor_name]

            metrics = self._compute_actor_eval_metrics(actor_output)
            result.add_actor_metrics(actor_name, metrics)

            completion_data = self._build_completion_data(ta, actor_output, is_eval=True)
            result.add_completion_data(actor_name, completion_data)

        return result

    def _compute_actor_eval_metrics(self, actor_output: "ActorOutput") -> Dict[str, float]:
        """Compute evaluation metrics for an actor output."""
        rewards = actor_output.rewards
        reward_mean = sum(rewards) / len(rewards) if rewards else 0.0
        reward_std = (
            (sum((r - reward_mean) ** 2 for r in rewards) / len(rewards)) ** 0.5
            if len(rewards) > 1
            else 0.0
        )

        completion_lens = [len(ids) for ids in actor_output.input_ids]
        completion_len_mean = (
            sum(completion_lens) / len(completion_lens) if completion_lens else 0.0
        )

        metrics = {
            "reward_mean": reward_mean,
            "reward_std": reward_std,
            "completion_len_mean": completion_len_mean,
        }

        if actor_output.reward_components:
            for comp_name, comp_rewards in actor_output.reward_components.items():
                comp_mean = (
                    sum(comp_rewards) / len(comp_rewards) if comp_rewards else 0.0
                )
                comp_std = (
                    (
                        sum((r - comp_mean) ** 2 for r in comp_rewards)
                        / len(comp_rewards)
                    )
                    ** 0.5
                    if len(comp_rewards) > 1
                    else 0.0
                )
                metrics[f"{comp_name}_mean"] = comp_mean
                metrics[f"{comp_name}_std"] = comp_std

        return metrics

    def _build_completion_data(
        self,
        ta: InitializedTrainableLLMActor,
        actor_output: "ActorOutput",
        is_eval: bool = False,
    ) -> Dict[str, List[Any]]:
        """Build completion data for logging. Works for both training and eval."""
        data = {}
        
        data["completion"] = [
            ta.tokenizer.decode(completion_ids, skip_special_tokens=False)
            for completion_ids in actor_output.input_ids
        ]

        data["total_reward"] = actor_output.rewards

        # Add advantages only for training
        if not is_eval and hasattr(self, "_calculate_advantages"):
            advantages = self._calculate_advantages(
                actor_output.rewards,
                self.group_size,
                actor_output.ended_in_eos,
            )
            data["advantage"] = advantages

        # Add reward components if available
        if actor_output.reward_components:
            for comp_name, comp_values in actor_output.reward_components.items():
                data[comp_name] = comp_values

        return data

    # ═══════════════════════════════════════════════════════════════════════
    # Checkpointing & Uploading methods
    # ═══════════════════════════════════════════════════════════════════════

    def save_checkpoint(self, path: str):
        self.accel.wait_for_everyone()

        if self.accel.is_main_process:
            os.makedirs(path, exist_ok=True)

        for name, ta in self.actors.items():
            subdir = os.path.join(path, name)
            if ta.accel.is_main_process:
                os.makedirs(subdir, exist_ok=True)
            ta.accel.save_state(output_dir=subdir)

        self.accel.wait_for_everyone()

        if self.accel.is_main_process:
            torch.save(
                {
                    "step": self._step,
                    "substep": self._substep,
                    "env_data_state": self.env.get_data_state(),
                    "env_rng_state": self.env.get_rng_state(),
                },
                os.path.join(path, "trainer_state.pt"),
            )
        self.accel.wait_for_everyone()

    def load_checkpoint(self, path: str):
        state = torch.load(os.path.join(path, "trainer_state.pt"), map_location="cpu")
        self._step = state["step"]
        self._substep = state["substep"]
        
        # Restore environment state
        self.env.set_data_state(state["env_data_state"], self.batch_size // self.group_size)
        self.env.set_rng_state(state["env_rng_state"])

        for name, ta in self.actors.items():
            subdir = os.path.join(path, name)
            ta.accel.load_state(subdir)

        self.accel.wait_for_everyone()

        if self.accel.is_main_process:
            self.logger.normal(
                colorize("🔄 Updating actor weights from checkpoint…", Palette.INFO)
            )

        for ta in self.actors.values():
            self._update_actor_weights(ta)
            if self.accel.is_local_main_process:
                ta.actor.sleep()
                if self.accel.is_main_process:
                    self.logger.normal(
                        colorize(
                            f"😴 Actor '{ta.name}' put to sleep after resume",
                            Palette.INFO,
                        )
                    )

    def save_pretrained(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        multi = len(self.actors) > 1

        for name, ta in self.actors.items():
            # reload if offloaded
            if ta.actor.offload_model:
                self.logger.normal(
                    colorize(f"🔄 Reloading model for actor LORA", Palette.INFO)
                )
                reload_model_and_optimizer(
                    ta.model, ta.optim, reload_model=True, reload_optimizer=False
                )
                self.logger.normal(
                    colorize(f"🔄 Model reloaded for actor LORA", Palette.INFO)
                )
            tgt = os.path.join(output_dir, name) if multi else output_dir
            os.makedirs(tgt, exist_ok=True)

            state_dict = ta.accel.get_state_dict(ta.model)
            self.logger.normal(colorize(f"Gathered a state dict", Palette.INFO))

            ta.accel.unwrap_model(ta.model, keep_torch_compile=False).save_pretrained(
                tgt,
                state_dict=state_dict,
                safe_serialization=True,
            )
            self.logger.normal(colorize(f"Model saved to {tgt}", Palette.INFO))

            if ta.tokenizer is not None:
                ta.tokenizer.save_pretrained(tgt)
            else:
                warnings.warn(f"Actor '{name}' has no tokenizer - skipped.")

            # offload.
            if ta.actor.offload_model:
                offload_model_and_optimizer(
                    ta.model, ta.optim, offload_optimizer=False, offload_model=True
                )
                self.logger.normal(
                    colorize(f"💤 Offloaded model for actor '{name}'", Palette.INFO)
                )

    def push_to_hub(
        self,
        repo_map: Union[str, Dict[str, str]],
        *,
        private: bool = False,
        commit_message: str | None = None,
        **push_kwargs,
    ):
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

    def _push_single_actor(
        self,
        ta: TrainableLLMActor,
        repo_id: str,
        *,
        private: bool,
        commit_message: str,
        **push_kwargs,
    ):
        unwrapped_model = ta.accel.unwrap_model(ta.model)
        unwrapped_model.push_to_hub(
            repo_id,
            private=private,
            commit_message=commit_message,
            **push_kwargs,
        )
        if ta.tokenizer is not None:
            ta.tokenizer.push_to_hub(
                repo_id,
                private=private,
                commit_message=commit_message,
                **push_kwargs,
            )
    # ═══════════════════════════════════════════════════════════════════════
    # Others
    # ═══════════════════════════════════════════════════════════════════════
    
    @torch.no_grad()
    def _get_logps(
        self,
        model: torch.nn.Module,
        ids: List[List[int]],
        tokenizer: PreTrainedTokenizerBase,
        temperature: float = 1.0,
        batch_size: int = 4,
        max_fused: int = 1 << 15,
    ) -> List[List[float]]:
        """
        Super memory efficient logp computation and actually faster than standard.
        """
        total   = len(ids)
        world   = self.number_of_devices
        per_rank = (total + world - 1) // world
        start, end = self.rank * per_rank, min((self.rank + 1) * per_rank, total)
        ids_local = ids[start:end]

        local_out: List[List[float]] = []

        for i in range(0, len(ids_local), batch_size):
            batch_ids = ids_local[i : i + batch_size]
            lengths = [len(seq) for seq in batch_ids]
            enc = tokenizer.pad({"input_ids": batch_ids},
                                padding=True,
                                return_tensors="pt")
            input_ids = enc.input_ids.to(model.device)         # (B,L)
            attn_mask = enc.attention_mask.to(model.device)
            L = input_ids.shape[1]                        # sequence length
            hidden = model.model(                          # type: ignore[attr-defined]
                input_ids=input_ids,
                attention_mask=attn_mask,
                use_cache=False,
            ).last_hidden_state                            # (B,L,H)

            hidden = hidden[:, :-1]                            # drop last step
            target = input_ids[:, 1:]                          # predict t from t−1

            non_pad  = torch.tensor(
                [
                    [1] * (l-1) + [0] * (L - l)
                    for l in lengths
                ]
            )
            # Flatten
            h_flat = hidden.reshape(-1, hidden.shape[-1])  # (N,L-1,H)
            tgt_flat = target.reshape(-1)            # (N,)
            non_pad   = non_pad.reshape(-1).bool()  # (N,)
            h_flat   = h_flat[non_pad] / temperature # (N,H)
            tgt_flat = tgt_flat[non_pad]          # (N,)

            with deepspeed.zero.GatheredParameters(
                [model.lm_head.weight, model.lm_head.bias],
                modifier_rank=None,
            ):
                lp_flat = _chunked_logp(
                    h_flat, model.lm_head, tgt_flat, max_fused=max_fused
                ).cpu()

            pos = 0
            for l in lengths:
                row_len = l - 1
                local_out.append(lp_flat[pos : pos + row_len].tolist())
                pos += row_len

            del enc, input_ids, attn_mask, hidden, target
            del non_pad, h_flat, tgt_flat, lp_flat

        gathered = self.accel.gather_for_metrics(local_out)
        return gathered
    
    @property
    def accel(self) -> Accelerator:
        return next(iter(self.actors.values())).accel

    @property
    def batch_size(self) -> int:
        return self.cfg.batch_size

    @property
    def use_wandb(self) -> bool:
        return self.cfg.use_wandb

    @property
    def log_every_n(self) -> int:
        return self.cfg.log_every_n

    @property
    def num_iterations(self) -> int:
        return self.cfg.num_iterations

    @property
    def total_expected_steps(self) -> int:
        if self.cfg.max_steps is not None:
            return self.cfg.max_steps * self.cfg.num_iterations

        dataloader = self.env.get_dataloader(self.batch_size // self.group_size)
        if dataloader is None:
            return 0
            
        steps_per_epoch = len(dataloader)
        total_expected_steps = (
            self.cfg.max_steps
            if self.cfg.max_steps is not None
            else self.cfg.epochs * steps_per_epoch
        ) * self.cfg.num_iterations
        return total_expected_steps

    @property
    def group_size(self) -> int:
        return getattr(self.cfg, 'group_size', 1)

    @property
    def grad_accumulation_steps(self) -> int:
        return self.cfg.grad_accumulation_steps

    @property
    def max_grad_norm(self) -> float:
        return self.cfg.max_grad_norm

    @property
    def number_of_devices(self) -> int:
        return self.accel.num_processes

    @property
    def rank(self) -> int:
        return self.accel.process_index

    @property
    def reference_batch_size(self) -> int:
        return self.cfg.reference_batch_size

    @property
    def eval_strategy(self) -> EvalStrategy:
        return self.cfg.eval_strategy

    @property
    def eval_every_n(self) -> int:
        return self.cfg.eval_every_n

    def log_completions(self, data: Dict[str, List[Any]], prefix: str = "completions"):
        """
        Log arbitrary data as wandb tables.

        Args:
            data: Dictionary mapping column names to lists of values
            prefix: Custom prefix for wandb logging key
        """
        if self.use_wandb and is_wandb_active() and self.accel.is_main_process:
            import wandb

            if not data:
                return

            data_length = None
            for values in data.values():
                if isinstance(values, list) and values:
                    data_length = len(values)
                    break

            if data_length is None:
                return

            for key, values in data.items():
                if not isinstance(values, list) or len(values) != data_length:
                    self.logger.warning(
                        f"Skipping completions logging: inconsistent data length for key '{key}'"
                    )
                    return

            columns = list(data.keys())
            table = wandb.Table(columns=columns)

            for i in range(data_length):
                row = [data[col][i] for col in columns]
                table.add_data(*row)

            wandb.log(
                {f"{prefix}/step_{self._step}": table},
                step=self._substep,
            )

    def log_training_metrics(self, metrics: Dict[str, List[Dict[str, float]]]):
        if self.accel.is_main_process and self._step % self.log_every_n == 0:
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
                            colorize(
                                f"      📊 Iter {iteration_idx+1}:",
                                Palette.YELLOW,
                            )
                        )
                        indent = "         "

                    static_metrics = set()
                    for metric_name in actor_metrics.keys():
                        if any(
                            suffix in metric_name
                            for suffix in ["_mean", "_std", "completion_len"]
                        ):
                            static_metrics.add(metric_name)

                    for m, v in actor_metrics.items():
                        if iteration_idx > 0 and m in static_metrics:
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

        if self.use_wandb and is_wandb_active() and self.accel.is_main_process:
            import wandb

            static_metrics = set()
            for actor_metrics_list in metrics.values():
                if actor_metrics_list:
                    for metric_name in actor_metrics_list[0].keys():
                        if any(
                            suffix in metric_name
                            for suffix in ["_mean", "_std", "completion_len"]
                        ):
                            static_metrics.add(metric_name)

            for iteration_idx in range(self.num_iterations):
                self._substep += 1
                for actor_name, actor_metrics_list in metrics.items():
                    if iteration_idx >= len(actor_metrics_list):
                        continue
                    for m, v in actor_metrics_list[iteration_idx].items():
                        if iteration_idx > 0 and m in static_metrics:
                            continue
                        wandb.log({f"{actor_name}/{m}": v}, step=self._substep)
                    if actor_name in self.actors:
                        learning_rate = self.actors[actor_name].sched.get_last_lr()[0]
                        wandb.log(
                            {f"{actor_name}/learning_rate": learning_rate},
                            step=self._substep,
                        )

    def log_evaluation_metrics(
        self, eval_name: str, metrics: "EvaluationMetrics", env_output: GroupedEnvironmentOutput, is_final: bool = False
    ):
        if self.accel.is_main_process:
            self.logger.quiet(
                colorize(f"📊 Evaluation Results for '{eval_name}':", Palette.BOLD)
            )
            for actor_name, actor_metrics in metrics.metrics.items():
                self.logger.quiet(colorize(f"   🎭 {actor_name}:", Palette.CYAN))
                for metric_name, value in actor_metrics.items():
                    self.logger.quiet(
                        colorize(f"      • {metric_name}: {value:.3f}", Palette.CYAN)
                    )
            self.logger.quiet("")

        if self.use_wandb and is_wandb_active() and self.accel.is_main_process:
            import wandb

            for actor_name, actor_metrics in metrics.metrics.items():
                for metric_name, value in actor_metrics.items():
                    wandb.log(
                        {f"eval/{eval_name}/{actor_name}/{metric_name}": value},
                        step=self._substep,
                    )

            for actor_name, completion_data in metrics.completions.items():
                if completion_data:
                    enhanced_completion_data = {}
                    
                    if env_output.problems:
                        completion_length = len(next(iter(completion_data.values())))
                        
                        for key in env_output.problems[0].keys():
                            enhanced_completion_data[key] = []
                            for problem in env_output.problems:
                                for _ in range(completion_length // len(env_output.problems)):
                                    enhanced_completion_data[key].append(problem[key])
                    
                    enhanced_completion_data.update(completion_data)
                    
                    columns = list(enhanced_completion_data.keys())
                    table = wandb.Table(columns=columns)

                    data_length = len(next(iter(enhanced_completion_data.values())))
                    for i in range(data_length):
                        row = [enhanced_completion_data[col][i] for col in columns]
                        table.add_data(*row)

                    table_suffix = "_final" if is_final else f"_step_{self._step}"
                    wandb.log(
                        {
                            f"eval_completions/{eval_name}/{actor_name}/{table_suffix}": table
                        },
                        step=self._substep,
                    )
