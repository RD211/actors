from __future__ import annotations

import inspect
from collections.abc import Sequence
from typing import Any

import torch

from actors.actors.base import TrainableLLMActor
from actors.environments.env_base import Environment
from actors.environments.types import ActorOutput, GroupedEnvironmentOutput
from actors.trainers.base_trainer import (
    ActorTrainState,
    BaseRLTrainer,
    TrainingMetrics,
    is_peft_model,
)
from actors.trainers.grpo_config import GRPOTrainerCfg
from actors.utils.deepspeed import (
    offload_model_and_optimizer,
    reload_model_and_optimizer,
)
from actors.utils.logger import Palette, colorize
from actors.utils.tracker import _step_profiler
from actors.utils.train_utils import _ForwardRedirection, free_memory

# ═════════════════════════════════════════════════════════════════════════════
# Helper functions
# ═════════════════════════════════════════════════════════════════════════════


def split_for_grad_accum(seq: Sequence[Any], steps: int) -> list[Sequence[Any]]:
    stride = len(seq) // steps
    return [seq[i * stride : (i + 1) * stride] for i in range(steps)]


def default_advantage_calculator(
    rewards: list[float],
    group_size: int,
    ended_in_eos: list[bool] | None = None,
    std_normalization: bool = True,
) -> list[float]:
    out: list[float] = []
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
# GRPO trainer
# ═════════════════════════════════════════════════════════════════════════════
class GRPOTrainer(BaseRLTrainer):
    def __init__(
        self,
        cfg: GRPOTrainerCfg,
        env: Environment,
        actors: list[TrainableLLMActor],
    ):
        self.cfg: GRPOTrainerCfg = cfg

        if self.cfg.batch_size % self.cfg.group_size:
            raise ValueError("batch_size must be a divisible by group_size")

        super().__init__(
            cfg,
            env=env,
            actors=actors,
        )

        self.env = env
        self._forward_redirection = _ForwardRedirection()

    def _calculate_advantages(
        self,
        actor_name: str,
        rewards: list[float],
        group_size: int,
        ended_in_eos: list[bool] | None = None,
    ) -> list[float]:
        actor_obj = self.actor_objects[actor_name]
        advantage_calculator = actor_obj.training_config.advantage_calculator
        std_normalization = actor_obj.training_config.std_normalization

        if advantage_calculator is not None:
            try:
                sig = inspect.signature(advantage_calculator)
                params = list(sig.parameters.keys())

                kwargs = {"rewards": rewards}

                if "group_size" in params:
                    kwargs["group_size"] = group_size
                if "ended_in_eos" in params and ended_in_eos is not None:
                    kwargs["ended_in_eos"] = ended_in_eos

                return advantage_calculator(**kwargs)

            except Exception:
                try:
                    return advantage_calculator(rewards)
                except:
                    try:
                        return advantage_calculator(rewards, group_size)
                    except:
                        return default_advantage_calculator(
                            rewards, group_size, ended_in_eos, std_normalization
                        )
        else:
            return default_advantage_calculator(
                rewards, group_size, ended_in_eos, std_normalization
            )

    def train_step(self, env_output: GroupedEnvironmentOutput) -> TrainingMetrics:
        result = TrainingMetrics()

        for actor_name, _ in env_output.groups.items():
            if actor_name not in self.actors:
                continue

            ta = self.actors[actor_name]

            flat_output = env_output.to_environment_output().actors[actor_name]
            completion_data = self._build_completion_data(
                ta, flat_output, actor_name, is_eval=False
            )
            result.add_completion_data(actor_name, completion_data)

            self._process_actor_step(actor_name, ta, flat_output, result)

        return result

    def _process_actor_step(
        self,
        name: str,
        ta: ActorTrainState,
        actor_output: ActorOutput,
        result: TrainingMetrics,
    ) -> None:
        actor_obj = self.actor_objects[name]

        if (
            actor_obj.training_config.offload_optimizer
            or actor_obj.training_config.offload_model
        ):
            reload_model_and_optimizer(
                ta.model,
                ta.optim,
                reload_optimizer=actor_obj.training_config.offload_optimizer,
                reload_model=actor_obj.training_config.offload_model,
            )
        total_rewards = actor_output.rewards
        advantages = self._calculate_advantages(
            name, total_rewards, self.group_size, actor_output.ended_in_eos
        )

        ids_list = actor_output.input_ids
        mask_list = actor_output.attention_mask

        assert (
            len(ids_list) == len(mask_list) == len(total_rewards) == len(advantages)
        ), (
            f"Actor '{name}' output lengths mismatch: "
            f"ids={len(ids_list)}, mask={len(mask_list)}, rewards={len(total_rewards)}, "
            f"advantages={len(advantages)}"
        )
        assert all(
            len(ids) == len(mask)
            for ids, mask in zip(ids_list, mask_list, strict=False)
        ), (
            f"Actor '{name}' input_ids and attention_mask lengths mismatch: "
            f"ids={len(ids_list)}, mask={len(mask_list)}"
        )

        old_lp: Sequence[Sequence[float]] | None = None
        ref_lp: Sequence[Sequence[float]] | None = None

        with _step_profiler.track("get_logps", actor_name=name):
            old_lp = (
                self._get_logps(
                    (
                        self.accel.unwrap_model(ta.model).base_model.model
                        if is_peft_model(ta.model)
                        else self.accel.unwrap_model(ta.model)
                    ),
                    ids_list,
                    ta.tokenizer,
                    temperature=ta.loss_fn.temperature,
                    batch_size=actor_obj.training_config.reference_batch_size,
                )
                if self.num_iterations > 1
                else None
            )
            if ta.ref_model is not None:
                ref_lp = self._get_logps(
                    ta.ref_model,
                    ids_list,
                    ta.tokenizer,
                    temperature=ta.loss_fn.temperature,
                    batch_size=actor_obj.training_config.reference_batch_size,
                )
            elif is_peft_model(ta.model) and actor_obj.training_config.beta != 0.0:
                with ta.model.disable_adapter():
                    ref_lp = self._get_logps(
                        self.accel.unwrap_model(ta.model).base_model.model,
                        ids_list,
                        ta.tokenizer,
                        temperature=ta.loss_fn.temperature,
                        batch_size=actor_obj.training_config.reference_batch_size,
                    )
            else:
                ref_lp = None

        for substep_idx in range(self.num_iterations):
            if self.accel.is_main_process:
                if self.num_iterations > 1:
                    self.logger.normal(
                        colorize(
                            f"🔄 Backwards iter {substep_idx + 1}/{self.num_iterations} for actor '{name}'",
                            Palette.INFO,
                        )
                    )
                else:
                    self.logger.normal(
                        colorize(f"🔄 Backwards for actor '{name}'", Palette.INFO)
                    )

            for adv_slice, id_slice, m_slice, old_slice, ref_slice in zip(
                split_for_grad_accum(advantages, self.grad_accumulation_steps),
                split_for_grad_accum(ids_list, self.grad_accumulation_steps),
                split_for_grad_accum(mask_list, self.grad_accumulation_steps),
                split_for_grad_accum(
                    old_lp or [None] * len(ids_list), self.grad_accumulation_steps
                ),
                split_for_grad_accum(
                    ref_lp or [None] * len(ids_list), self.grad_accumulation_steps
                ),
                strict=False,
            ):
                self._backward_one_slice(
                    ta,
                    id_slice,
                    m_slice,
                    adv_slice,
                    ref_slice,
                    old_slice,
                    result,
                    substep_idx,
                    name,
                )
                free_memory()

            grad_norm = self._clip_gradients(
                ta, clip_to=actor_obj.training_config.max_grad_norm
            )
            result.add_substep_metric(name, substep_idx, "grad_norm", grad_norm)

            self._optim_step(ta)

            if substep_idx == 0:
                result.add_actor_rewards(name, total_rewards)

                # Add reward component statistics
                if actor_output.reward_components:
                    for (
                        comp_name,
                        comp_rewards,
                    ) in actor_output.reward_components.items():
                        result.add_actor_reward_component(name, comp_name, comp_rewards)

            result.add_substep_metric(
                name, substep_idx, "learning_rate", ta.sched.get_last_lr()[0]
            )

        # Offload states after training is complete for this actor
        if actor_obj.training_config.offload_optimizer:
            offload_model_and_optimizer(
                ta.model, ta.optim, offload_optimizer=True, offload_model=False
            )

        # Track actor weight update
        self._update_actor_weights(ta, name)

        if actor_obj.training_config.offload_model:
            offload_model_and_optimizer(
                ta.model, ta.optim, offload_optimizer=False, offload_model=True
            )

    def _backward_one_slice(
        self,
        ta: ActorTrainState,
        ids: list[list[int]],
        masks: list[list[int]],
        advantages: list[float],
        ref_lp_slice: list[list[float]] | None,
        old_lp_slice: list[list[float]],
        result: TrainingMetrics,
        substep_idx: int,
        actor_name: str,
    ) -> None:
        tok, dev = ta.tokenizer, ta.model.device
        padded = tok.pad({"input_ids": ids}, padding="longest", return_tensors="pt")
        ids_pt, attention_mask = (
            padded["input_ids"].to(dev),
            padded["attention_mask"].to(dev),
        )

        max_len = ids_pt.size(1) - 1

        def to_tensor(slice_):
            t = torch.zeros(len(slice_), max_len, dtype=torch.float32, device=dev)
            for i, row in enumerate(slice_):
                n = min(len(row), max_len)
                if n:
                    t[i, :n] = torch.tensor(row[:n], dtype=torch.float32, device=dev)
            return t

        ref_lp = to_tensor(ref_lp_slice) if any(ref_lp_slice) else None
        old_lp = to_tensor(old_lp_slice) if any(old_lp_slice) else None
        loss_attention_mask = to_tensor([x[1:] for x in masks]) if masks else None

        adv_pt = torch.tensor(advantages, dtype=torch.float32, device=dev)
        unwrapped_model = ta.accel.unwrap_model(ta.model)
        with _step_profiler.track("loss_fn", actor_name=actor_name):
            loss, stats = self._forward_redirection(
                ta.model,
                unwrapped_model,
                ta.loss_fn.forward,
                # ---- everything the loss expects --------------------
                unwrapped_model,
                ids_pt,
                attention_mask,
                loss_attention_mask,
                adv_pt,
                ref_lp,
                old_lp,
            )

        ta.accel.backward(loss)

        result.add_substep_metric(actor_name, substep_idx, "loss", loss.item())
        if "kl" in stats and getattr(ta.loss_fn, "beta", 0.0) != 0.0:
            result.add_substep_metric(actor_name, substep_idx, "kl", stats["kl"])
        result.add_step_metric(
            actor_name,
            "completion_len",
            attention_mask[:, 1:].sum(-1).float().mean().item(),
        )
