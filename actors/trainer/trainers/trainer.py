from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import deepspeed                                    # noqa: F401 – used implicitly by Accelerate
import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from transformers import PreTrainedTokenizerBase

# ----- project-local helpers -------------------------------------------------
from actors.utils.logger import init_logger, colorize, Palette
from actors.utils.ipc_utils import gather_and_stream_state_dict
from actors.trainer.environments.env_base import Environment
from actors.trainer.losses.base_loss import BaseRLLoss
from actors.utils.tracker import gpu_profiler

# ═════════════════════════════════════════════════════════════════════════════
# Pure-data container (no helper methods)
# ═════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class TrainableLLMActor:
    """Immutable record that bundles everything the trainer needs for one actor."""
    name: str
    actor: object                                  # Env-side proxy that owns the rollout workers
    model: torch.nn.Module
    tokenizer: PreTrainedTokenizerBase
    loss_fn: BaseRLLoss
    optim: torch.optim.Optimizer
    sched: torch.optim.lr_scheduler._LRScheduler
    reference_model: Optional[torch.nn.Module] = None


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
        grad_accumulation_steps: int = 1,
        batch_size: int = 4,
        group_size: int = 4,
        reference_batch_size: int = 1,
        num_iterations: int = 1,
        max_grad_norm: float = 1.0,
        temperature: float = 1.0,
        gradient_checkpointing: bool = False,
    ):
        self.env = env
        self.max_grad_norm = max_grad_norm
        self.num_iterations = num_iterations
        self.temperature = temperature
        self.gradient_checkpointing = gradient_checkpointing
        self._step = 0
        # ----- distributed / mixed precision ---------------------------------
        self.accel = Accelerator(
            mixed_precision="bf16",
            kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(hours=10))],
        )
        # make DeepSpeed match user args
        ds_cfg = self.accel.state.deepspeed_plugin.deepspeed_config
        ds_cfg['max_grad_norm'] = max_grad_norm
        ds_cfg["train_batch_size"] = batch_size
        ds_cfg["gradient_accumulation_steps"] = grad_accumulation_steps
        ds_cfg["train_micro_batch_size_per_gpu"] = (
            batch_size // self.accel.num_processes // grad_accumulation_steps
        )
        if self.gradient_checkpointing:
            ds_cfg.setdefault("activation_checkpointing", {
                "partition_activations": True,
                "contiguous_memory_optimization": True,
                "number_checkpoints": 1,
                "synchronize_checkpoint_boundary": False,
            })

        # ----- sanity checks --------------------------------------------------
        self.group_size = group_size
        self.batch_size = batch_size
        self.grad_accumulation_steps = grad_accumulation_steps
        self.reference_batch_size = reference_batch_size
        self.number_of_devices = self.accel.num_processes
        self.rank = self.accel.process_index
        self.logger = init_logger(f"trainer{self.rank}")

        if batch_size % group_size:
            raise ValueError("batch_size must be divisible by group_size")
        if (batch_size // grad_accumulation_steps) % self.number_of_devices:
            raise ValueError("batch_size/grad_accumulation_steps must be divisible by world size")
        if batch_size % (reference_batch_size * self.number_of_devices):
            raise ValueError("batch_size must be divisible by reference_batch_size*world_size")

        # ----- build TrainableLLMActor registry ------------------------------
        self.actors: Dict[str, TrainableLLMActor] = {}
        for name, (actor, spec) in env.get_actor_specs().items():
            model = spec.model_factory()
            if self.gradient_checkpointing:
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )

                model.config.use_cache = False            # :contentReference[oaicite:3]{index=3}
                # LoRA / PEFT or frozen-layer edge-case
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
                    model.config.gradient_checkpointing = True

            model = model.train()

            ref_model = spec.reference_model_factory().eval() if spec.reference_model_factory else None

            optim = spec.optim_factory(model.parameters())
            sched = spec.scheduler_factory(optim)

            # put model/optim onto devices via Accelerate/DeepSpeed
            model, optim, sched = self.accel.prepare(model, optim, sched)
            if ref_model is not None and ds_cfg['zero_optimization']['stage'] == 3:
                ref_model = self.accel.prepare(ref_model)
            else:
                ref_model = ref_model.to(self.accel.device) if ref_model is not None else None

            self.actors[name] = TrainableLLMActor(
                name=name,
                actor=actor,
                model=model,
                tokenizer=spec.tokenizer,          # 🤗 tokenizer
                loss_fn=spec.loss_factory(),
                optim=optim,
                sched=sched,
                reference_model=ref_model,
            )

    # --------------------------------------------------------------------- #
    # (tiny) utilities
    # --------------------------------------------------------------------- #
    @staticmethod
    def _pad(tok: PreTrainedTokenizerBase, seqs: List[List[int]]) -> Dict[str, torch.Tensor]:
        return tok.pad({"input_ids": seqs}, padding="longest", return_tensors="pt")

    @staticmethod
    def _grad_global_norm(model: torch.nn.Module) -> float:
        norms = [p.grad.detach().norm(2) for p in model.parameters() if p.grad is not None]
        if not norms:
            return 0.0
        return torch.linalg.vector_norm(torch.stack(norms), 2).item()


    # --------------------------------------------------------------------- #
    # log-probabilities 
    # --------------------------------------------------------------------- #
    @gpu_profiler(name='get_logps')
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
            input_ids = padded["input_ids"].to(self.accel.device)

            L = input_ids.size(1)
            rng = torch.arange(L, device=self.accel.device).unsqueeze(0)
            att_mask = (rng < torch.tensor(lengths, device=self.accel.device).unsqueeze(1)).long()

            with torch.no_grad():
                logits = model(input_ids=input_ids, attention_mask=att_mask).logits / temperature

            lp = torch.log_softmax(logits, -1)[:, :-1]                       # (B,L-1,V)
            tgt = input_ids[:, 1:].unsqueeze(-1)                             # (B,L-1,1)
            lp = lp.gather(-1, tgt).squeeze(-1) * att_mask[:, 1:]            # (B,L-1)

            for row, ln in zip(lp, lengths):
                local_logps.append(row[: ln - 1].cpu().tolist())

        gathered = self.accel.gather_for_metrics(local_logps)
        return gathered[:total]

    # ═════════════════════════════════════════════════════════════════════
    # public API
    # ═════════════════════════════════════════════════════════════════════
    def train_step(self, raw_batch: Dict[str, List[Any]]) -> Dict[str, Dict[str, float]]:
        """
        One environment interaction + optimisation step.

        Returns a nested dict of averaged metrics:
        `{actor_name: {"loss": float, "kl": float, ...}, ...}`
        """
        self._step += 1
        batch = expand_batch(raw_batch, self.group_size)
        env_out = self.env(batch)

        # warn about unexpected actor keys (once per step, rank==0 only)
        if self.rank == 0:
            for k in env_out.keys() - self.actors.keys():
                self.logger.warning(colorize(f"env produced data for unknown actor '{k}'",
                                             Palette.WARNING))

        # collect metrics
        metrics = {name: [dict(loss=[], kl=[], grad_norm=[],
                            completion_len=[], reward_mean=[], reward_std=[])
                        for _ in range(self.num_iterations)]
                for name in self.actors}

        for name, ta in self.actors.items():
            if name not in env_out:
                continue
            self._process_actor_step(name, ta, env_out[name], metrics[name])

        # aggregate over grad-steps and (internally) over ranks
        out: Dict[str, Dict[str, List[float]]] = {}
        for name, arr in metrics.items():
            out[name] = [{k: (sum(v) / len(v)) for k, v in d.items() if v} for d in arr]
        return out

    # ------------------------------------------------------------------ #
    # internal – per-actor handling
    # ------------------------------------------------------------------ #
    def _process_actor_step(
        self,
        name: str,
        ta: TrainableLLMActor,
        items: Dict[str, Any],
        buckets: List[Dict[str, List[float]]]) -> None:
        
        rewards = items["rewards"]
        advantages = norm_advantages(rewards, self.group_size)

        ids_list = items["input_ids"]
        mask_list = items["attention_mask"]

        # compute reference log-ps once (if any)
        old_lp = self._get_logps(ta.model, ids_list, ta.tokenizer, 
                                 temperature=self.temperature, batch_size=self.reference_batch_size) if self.num_iterations > 1 else None
        ref_lp = (self._get_logps(ta.reference_model, ids_list, ta.tokenizer, temperature=self.temperature, batch_size=self.reference_batch_size)
                if ta.reference_model is not None else None)

        # iterate over grad-accumulation micro-batches
        for it in range(self.num_iterations):
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

            self._optim_step(ta)

            # rewards / completion stats identical across iterations
            b = buckets[it]
            if not b["reward_mean"]:
                b["reward_mean"].append(sum(rewards) / len(rewards))
                b["reward_std"].append(torch.tensor(rewards).float().std(unbiased=False).item())
        self._update_actor_weights(ta)

    # ------------------------------------------------------------------ #
    @gpu_profiler(name='backward_one_slice')
    def _backward_one_slice(
        self,
        ta: TrainableLLMActor,
        ids: List[List[int]],
        masks: List[List[int]],
        advantages: List[float],
        ref_lp_slice: Optional[Sequence[Sequence[float]]],
        old_lp_slice: Sequence[Sequence[float]],
        bucket: Dict[str, List[float]],
    ) -> None:
        tok, dev = ta.tokenizer, ta.model.device
        ids_pt = self._pad(tok, ids)["input_ids"].to(dev)
        msk_pt = torch.tensor(
            [m + [0]*(ids_pt.size(1)-len(m)) for m in masks],
            dtype=torch.long, device=dev
        )

        max_len = ids_pt.size(1) - 1
        def to_tensor(slice_):
            t = torch.zeros(len(slice_), max_len, dtype=torch.float32, device=dev)
            for i,row in enumerate(slice_):
                n = min(len(row), max_len)
                if n: t[i,:n] = torch.tensor(row[:n], dtype=torch.float32, device=dev)
            return t
        ref_lp = to_tensor(ref_lp_slice) if any(ref_lp_slice) else None
        old_lp = to_tensor(old_lp_slice) if any(old_lp_slice) else None

        adv_pt  = torch.tensor(advantages, dtype=torch.float32, device=dev)

        loss, stats = ta.loss_fn(
            policy       = ta.model,
            input_ids    = ids_pt,
            attention_mask=msk_pt,
            advantages   = adv_pt,
            ref_logps    = ref_lp,
            old_logps    = old_lp,
        )
        self.accel.backward(loss)

        bucket["loss"].append(loss.item())
        if "kl" in stats:
            bucket["kl"].append(stats["kl"])
        bucket["completion_len"].append(msk_pt[:,1:].sum(-1).float().mean().item())


    # ------------------------------------------------------------------ #
    @gpu_profiler(name='optim_step', no_memory_measurement=True)
    def _optim_step(self, ta: TrainableLLMActor) -> None:
        ta.optim.step()
        ta.sched.step()
        ta.optim.zero_grad()
        self.accel.wait_for_everyone()

    # ------------------------------------------------------------------ #
    @gpu_profiler(name='update_actor_weights')
    def _update_actor_weights(self, ta: TrainableLLMActor) -> None:
        # On each node, the local main process will orchestrate the update.
        if self.accel.is_local_main_process:
            ta.actor.start_weight_update()

        # This callback will be executed on the local main process of each node for each batch.
        def stream_batch_callback(batch_state_dict):
            if self.accel.is_local_main_process:
                ta.actor.update_weights_batch(batch_state_dict)

        gather_and_stream_state_dict(
            self.accel, self.logger, ta.model, stream_batch_callback
        )

        if self.accel.is_local_main_process:
            ta.actor.finalize_weight_update()


    # ------------------------------------------------------------------ #
    # checkpointing
    # ------------------------------------------------------------------ #
    def save_checkpoint(self, path: str):
        """Save all models, optimizers, schedulers, and state to the given directory."""
        self.accel.wait_for_everyone()
        self.accel.save_state(output_dir=path) 
        if self.accel.is_main_process:
            state = {"step": self._step}
            with open(os.path.join(path, "trainer_state.json"), "w") as f:
                json.dump(state, f)
        self.accel.wait_for_everyone()

    def load_checkpoint(self, path: str):
        """Load model, optimizer, scheduler states from the checkpoint directory."""
        self.accel.load_state(path)
        state_path = os.path.join(path, "trainer_state.json")
        if os.path.isfile(state_path):
            with open(state_path, "r") as f:
                state = json.load(f)
            self._step = state.get("step", 0)
