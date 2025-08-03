from __future__ import annotations

import inspect
from collections.abc import Sequence
from typing import Any

import torch

from actors.actors.base import TrainableLLMActor
from actors.environments.env_base import Environment
from actors.environments.types import (
    ActorOutput,
    EnvironmentOutput,
)
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
from actors.utils.train_utils import _ForwardRedirection, free_memory_if_needed

# ═════════════════════════════════════════════════════════════════════════════
# Helper functions
# ═════════════════════════════════════════════════════════════════════════════


def split_for_grad_accum(seq: Sequence[Any], steps: int) -> list[Sequence[Any]]:
    stride = len(seq) // steps
    return [seq[i * stride : (i + 1) * stride] for i in range(steps)]


def default_advantage_calculator(
    rewards: list[float],
    ended_in_eos: list[bool] | None = None,
    std_normalization: bool = True,
) -> list[float]:
    µ = sum(rewards) / len(rewards)

    if std_normalization:
        σ = (sum((x - µ) ** 2 for x in rewards) / len(rewards)) ** 0.5 + 1e-8
        return [(r - µ) / σ for r in rewards]
    else:
        return [r - µ for r in rewards]


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

                if "ended_in_eos" in params and ended_in_eos is not None:
                    kwargs["ended_in_eos"] = ended_in_eos

                return advantage_calculator(**kwargs)

            except Exception:
                try:
                    return advantage_calculator(rewards)
                except:
                    try:
                        return advantage_calculator(rewards, ended_in_eos)
                    except:
                        return default_advantage_calculator(
                            rewards, ended_in_eos, std_normalization
                        )
        else:
            return default_advantage_calculator(
                rewards, ended_in_eos, std_normalization
            )

    def train_step(self, env_output: EnvironmentOutput) -> TrainingMetrics:
        result = TrainingMetrics()

        for actor_name in self.actors:
            actor_part = env_output.get_actor_output(actor_name=actor_name)

            ta = self.actors[actor_name]

            completion_data = self._build_completion_data(
                ta, actor_part, actor_name, is_eval=False
            )
            result.add_completion_data(actor_name, completion_data)

            self._process_actor_step(actor_name, ta, actor_part, result)

        return result

    def _process_actor_step(
        self,
        name: str,
        ta: ActorTrainState,
        actor_output: list[dict[str, ActorOutput]],
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

        flattened_actor_outputs = [
            ao for group in actor_output for ao in group.values()
        ]
        advantage_groups = []
        for ac_out in flattened_actor_outputs:
            total_rewards = ac_out.rewards
            advantages = self._calculate_advantages(
                name, total_rewards, ac_out.ended_in_eos
            )
            advantage_groups.append(advantages)
        advantages = [adv for group in advantage_groups for adv in group]
        combined_actor_output = flattened_actor_outputs[0]
        for ao in flattened_actor_outputs[1:]:
            combined_actor_output += ao

        ids_list = combined_actor_output.input_ids
        mask_list = combined_actor_output.attention_mask
        total_rewards = combined_actor_output.rewards

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
        # If the len of ids_list and all entries is not equal to the batch size, we make a warning.
        if len(ids_list) != actor_obj.training_config.batch_size:
            num_processes = self.accel.num_processes
            if len(ids_list) % num_processes != 0:
                # Closest divisible by num_processes
                closest_divisible = (len(ids_list) // num_processes) * num_processes
                self.logger.warning(
                    f"Actor '{name}' has {len(ids_list)} input_ids, "
                    f"which is not divisible by the number of processes ({num_processes}). "
                    f"Using {closest_divisible} instead."
                )
                ids_list = ids_list[:closest_divisible]
                mask_list = mask_list[:closest_divisible]
                total_rewards = total_rewards[:closest_divisible]
                advantages = advantages[:closest_divisible]
            else:
                self.logger.warning(
                    f"Actor '{name}' has {len(ids_list)} input_ids, "
                    f"but expected {actor_obj.training_config.batch_size}. We will scale the batch size."
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
            grad_accumulation_steps = actor_obj.training_config.grad_accumulation_steps
            for adv_slice, id_slice, m_slice, old_slice, ref_slice in zip(
                split_for_grad_accum(advantages, grad_accumulation_steps),
                split_for_grad_accum(ids_list, grad_accumulation_steps),
                split_for_grad_accum(mask_list, grad_accumulation_steps),
                split_for_grad_accum(
                    old_lp or [None] * len(ids_list), grad_accumulation_steps
                ),
                split_for_grad_accum(
                    ref_lp or [None] * len(ids_list), grad_accumulation_steps
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
                free_memory_if_needed()

            grad_norm = self._clip_gradients(
                ta, clip_to=actor_obj.training_config.max_grad_norm
            )
            result.add_substep_metric(name, substep_idx, "grad_norm", grad_norm)

            self._optim_step(ta)

            if substep_idx == 0:
                result.add_actor_rewards(name, total_rewards)

                # Add reward component statistics
                if combined_actor_output.reward_components:
                    for (
                        comp_name,
                        comp_rewards,
                    ) in combined_actor_output.reward_components.items():
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
