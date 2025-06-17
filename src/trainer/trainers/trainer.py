from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

import deepspeed                                    # noqa: F401 – used implicitly by Accelerate
import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from transformers import PreTrainedTokenizerBase

# ----- project-local helpers -------------------------------------------------
from src.utils.logger import init_logger, colorize, Palette
from src.utils.shm_utils import gather_incremental_state_dict_to_cpu
from src.trainer.environments.env_base import Environment
from src.trainer.losses.base_loss import BaseRLLoss


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
        max_grad_norm: float = 1.0,
    ):
        self.env = env
        self.max_grad_norm = max_grad_norm
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


        # ----- sanity checks --------------------------------------------------
        self.group_size = group_size
        self.batch_size = batch_size
        self.grad_accumulation_steps = grad_accumulation_steps
        self.reference_batch_size = reference_batch_size
        self.number_of_devices = self.accel.num_processes
        self.rank = self.accel.process_index  # convenience alias
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
            model = spec.model_factory().train()
            ref_model = spec.reference_model_factory().eval() if spec.reference_model_factory else None

            optim = spec.optim_factory(model.parameters())
            sched = spec.scheduler_factory(optim)

            # put model/optim onto devices via Accelerate/DeepSpeed
            model, optim, sched = self.accel.prepare(model, optim, sched)
            if ref_model is not None:
                ref_model = self.accel.prepare(ref_model)

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
    # reference log-probabilities – distributed, order-preserving
    # --------------------------------------------------------------------- #
    def _ref_logps(
        self,
        model: torch.nn.Module,
        ids: List[List[int]],
        tokenizer: PreTrainedTokenizerBase,
        temperature: float = 1.0,
    ) -> List[List[float]]:
        total = len(ids)
        world = self.number_of_devices
        per_rank = (total + world - 1) // world
        start, end = self.rank * per_rank, min((self.rank + 1) * per_rank, total)
        ids_local = ids[start:end]

        local_logps: List[List[float]] = []
        for i in range(0, len(ids_local), self.reference_batch_size):
            batch_ids = ids_local[i : i + self.reference_batch_size]
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
        batch = expand_batch(raw_batch, self.group_size)
        env_out = self.env(batch)

        # warn about unexpected actor keys (once per step, rank==0 only)
        if self.rank == 0:
            for k in env_out.keys() - self.actors.keys():
                self.logger.warning(colorize(f"env produced data for unknown actor '{k}'",
                                             Palette.WARNING))

        # collect metrics
        metrics: Dict[str, Dict[str, List[float]]] = {
            name: {"loss": [], "kl": []} for name in self.actors
        }

        for name, ta in self.actors.items():
            if name not in env_out:
                continue
            self._process_actor_step(name, ta, env_out[name], metrics[name])

        # aggregate over grad-steps and (internally) over ranks
        out: Dict[str, Dict[str, float]] = {}
        for name, m in metrics.items():
            out[name] = {k: (sum(v) / max(1, len(v))) for k, v in m.items() if v}

        return out

    # ------------------------------------------------------------------ #
    # internal – per-actor handling
    # ------------------------------------------------------------------ #
    def _process_actor_step(
        self,
        name: str,
        ta: TrainableLLMActor,
        items: Dict[str, Any],
        bucket: Dict[str, List[float]],
    ) -> None:
        rewards = items["rewards"]
        advantages = norm_advantages(rewards, self.group_size)

        ids_list = items["input_ids"]
        mask_list = items["attention_mask"]

        # compute reference log-ps once (if any)
        reference_lp: Optional[List[List[float]]] = None
        if ta.reference_model is not None:
            reference_lp = self._ref_logps(ta.reference_model, ids_list, ta.tokenizer)

        # iterate over grad-accumulation micro-batches
        for r_slice, id_slice, m_slice, ref_slice in zip(
            split_for_grad_accum(advantages, self.grad_accumulation_steps),
            split_for_grad_accum(ids_list, self.grad_accumulation_steps),
            split_for_grad_accum(mask_list, self.grad_accumulation_steps),
            split_for_grad_accum(reference_lp or [None] * len(ids_list), self.grad_accumulation_steps),
        ):
            self._backward_one_slice(ta, id_slice, m_slice, r_slice, ref_slice, bucket)

        # optimisation step + weight sync
        self._optim_step(ta)
        self._update_actor_weights(ta)

        # Log some metrics
        bucket.setdefault("reward_mean", []).append(
            sum(rewards) / max(1, len(rewards))
        )
        bucket.setdefault("reward_std", []).append(
            torch.tensor(rewards, dtype=torch.float32, device="cpu").std(unbiased=False).item()
        )
        # TODO: Make this work.
        # bucket.setdefault("grad_norm", []).append(
        #     self._grad_global_norm(ta.model)
        # )

    # ------------------------------------------------------------------ #
    def _backward_one_slice(
        self,
        ta: TrainableLLMActor,
        ids: List[List[int]],
        masks: List[List[int]],
        advantages: List[float],
        ref_lp_slice: Optional[Sequence[Sequence[float]]],
        bucket: Dict[str, List[float]],
    ) -> None:
        tok, dev = ta.tokenizer, ta.model.device

        ids_pt = self._pad(tok, ids)["input_ids"].to(dev)          # (B, L)
        msk_pt = torch.tensor([m + [0] * (ids_pt.size(1) - len(m)) for m in masks],
                              dtype=torch.long, device=dev)
        # -------- reference log-ps -----------------------------------------
        if any(ref_lp_slice):
            max_len = ids_pt.size(1) - 1
            ref_lp = torch.zeros(len(ref_lp_slice), max_len, dtype=torch.float32, device=dev)
            for i, row in enumerate(ref_lp_slice):
                n = min(len(row), max_len)
                if n > 0:
                    ref_lp[i, :n] = torch.tensor(row[:n], dtype=torch.float32, device=dev)
        else:
            ref_lp = None

        # -------- advantages ----------------------------------------------
        adv_pt = torch.tensor(advantages,
                            dtype=torch.float32, device=dev)      # (B,)

        # -------- loss -----------------------------------------------------
        loss, stats = ta.loss_fn(
            policy=ta.model,
            input_ids=ids_pt,
            attention_mask=msk_pt,
            advantages=adv_pt,
            ref_logps=ref_lp,
        )

        self.accel.backward(loss)

        # -------- metrics --------------------------------------------------
        bucket.setdefault("loss", []).append(loss.item())
        if "kl" in stats:
            bucket.setdefault("kl", []).append(stats["kl"])

        comp_len = msk_pt[:, 1:].sum(-1).float().mean().item()      # avg predicted tokens
        bucket.setdefault("completion_len", []).append(comp_len)


    # ------------------------------------------------------------------ #
    def _optim_step(self, ta: TrainableLLMActor) -> None:
        ta.optim.step()
        ta.sched.step()
        ta.optim.zero_grad()
        self.accel.wait_for_everyone()

    # ------------------------------------------------------------------ #
    def _update_actor_weights(self, ta: TrainableLLMActor) -> None:
        state = gather_incremental_state_dict_to_cpu(self.accel, self.logger, ta.model)
        if self.accel.is_main_process:
            ta.actor.update_weights(state)
