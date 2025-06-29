from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch
from actors.trainers.base_trainer import (
    BaseRLTrainer,
    EvaluationMetrics,
    InitializedTrainableLLMActor,
    TrainerCfg,
    TrainingMetrics,
    is_peft_model,
)
from actors.environments.types import GroupedEnvironmentOutput, GroupedEnvironmentOutput
from actors.utils.deepspeed import (
    offload_model_and_optimizer,
    reload_model_and_optimizer,
)
from actors.utils.logger import colorize, Palette
from actors.environments.env_base import Environment
from actors.environments.types import ActorOutput
from actors.utils.tracker import _step_profiler
from actors.utils.train_utils import _ForwardRedirection, free_memory


# ═════════════════════════════════════════════════════════════════════════════
# Configuration dataclass
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class GRPOTrainerCfg(TrainerCfg):
    reference_batch_size: int = 1
    advantage_calculator: Optional[Callable[..., List[float]]] = None
    std_normalization: bool = True


# ═════════════════════════════════════════════════════════════════════════════
# Helper functions
# ═════════════════════════════════════════════════════════════════════════════


def expand_batch(batch: Dict[str, List[Any]], group_size: int) -> Dict[str, List[Any]]:
    if not isinstance(batch, dict):
        raise ValueError("batch must be a dictionary")
    return {
        k: [item for item in v for _ in range(group_size)] for k, v in batch.items()
    }


def split_for_grad_accum(seq: Sequence[Any], steps: int) -> List[Sequence[Any]]:
    stride = len(seq) // steps
    return [seq[i * stride : (i + 1) * stride] for i in range(steps)]


def norm_advantages(rewards: List[float], group_size: int) -> List[float]:
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
    std_normalization: bool = True,
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
# GRPO trainer
# ═════════════════════════════════════════════════════════════════════════════
class GRPOTrainer(BaseRLTrainer):

    def __init__(
        self,
        cfg: GRPOTrainerCfg,
        env: Environment,
    ):
        self.cfg: GRPOTrainerCfg = cfg

        if self.cfg.batch_size % self.cfg.group_size:
            raise ValueError("batch_size must be a divisible by group_size")

        super().__init__(
            cfg,
            env=env,
        )

        self.env = env
        self._forward_redirection = _ForwardRedirection()

        if self.cfg.advantage_calculator is not None:
            self.advantage_calculator = self.cfg.advantage_calculator
        else:
            self.advantage_calculator = lambda rewards, group_size, ended_in_eos=None: default_advantage_calculator(
                rewards, group_size, ended_in_eos, self.cfg.std_normalization
            )

    def _calculate_advantages(
        self,
        rewards: List[float],
        group_size: int,
        ended_in_eos: Optional[List[bool]] = None,
    ) -> List[float]:
        try:
            sig = inspect.signature(self.advantage_calculator)
            params = list(sig.parameters.keys())

            kwargs = {"rewards": rewards}

            if "group_size" in params:
                kwargs["group_size"] = group_size
            if "ended_in_eos" in params and ended_in_eos is not None:
                kwargs["ended_in_eos"] = ended_in_eos

            return self.advantage_calculator(**kwargs)

        except Exception as e:
            try:
                return self.advantage_calculator(rewards)
            except:
                try:
                    return self.advantage_calculator(rewards, group_size)
                except:
                    return default_advantage_calculator(
                        rewards, group_size, ended_in_eos, True
                    )

    def eval_step(self, env_output: GroupedEnvironmentOutput) -> EvaluationMetrics:

        result = EvaluationMetrics()

        flat_output = env_output.to_environment_output()

        for actor_name, actor_output in flat_output.actors.items():
            if actor_name not in self.actors:
                continue

            ta = self.actors[actor_name]

            metrics = self._compute_actor_eval_metrics(actor_output)
            result.add_actor_metrics(actor_name, metrics)

            completion_data = self._build_eval_completion_data(
                actor_name, ta, actor_output, {}
            )
            result.add_completion_data(actor_name, completion_data)

        return result

    def _build_eval_completion_data(
        self,
        actor_name: str,
        ta: InitializedTrainableLLMActor,
        actor_output: ActorOutput,
        eval_batch: Dict[str, List[Any]],
    ) -> Dict[str, List[Any]]:
        data = {}

        for k in eval_batch.keys():
            data[k] = eval_batch[k]

        data["completion"] = [
            ta.tokenizer.decode(completion_ids, skip_special_tokens=False)
            for completion_ids in actor_output.input_ids
        ]

        data["total_reward"] = actor_output.rewards

        if actor_output.reward_components:
            for comp_name, comp_values in actor_output.reward_components.items():
                data[comp_name] = comp_values

        return data

    def _compute_actor_eval_metrics(
        self, actor_output: ActorOutput
    ) -> Dict[str, float]:
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

    def train_step(self, env_output: GroupedEnvironmentOutput) -> TrainingMetrics:

        result = TrainingMetrics()

        for actor_name, actor_groups in env_output.groups.items():
            if actor_name not in self.actors:
                continue

            ta = self.actors[actor_name]

            if actor_groups and actor_groups[0]:
                first_output = actor_groups[0][0]  # First group, first output
                completion_data = self._build_completion_data(
                    actor_name, ta, first_output, {}
                )
                result.add_completion_data(actor_name, completion_data)

            flat_output = env_output.to_environment_output().actors[actor_name]
            self._process_actor_step(actor_name, ta, flat_output, result)

        return result

    # ------------------------------------------------------------------ #
    # internal – per-actor handling
    # ------------------------------------------------------------------ #

    def _process_actor_step(
        self,
        name: str,
        ta: InitializedTrainableLLMActor,
        actor_output: ActorOutput,
        result: TrainingMetrics,
    ) -> None:

        if ta.actor.offload_optimizer or ta.actor.offload_model:
            with _step_profiler.track("reload_states", actor_name=name):
                reload_model_and_optimizer(
                    ta.model,
                    ta.optim,
                    reload_optimizer=ta.actor.offload_optimizer,
                    reload_model=ta.actor.offload_model,
                )
        total_rewards = actor_output.rewards
        advantages = self._calculate_advantages(
            total_rewards, self.group_size, actor_output.ended_in_eos
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
        assert all(len(ids) == len(mask) for ids, mask in zip(ids_list, mask_list)), (
            f"Actor '{name}' input_ids and attention_mask lengths mismatch: "
            f"ids={len(ids_list)}, mask={len(mask_list)}"
        )

        old_lp: Optional[Sequence[Sequence[float]]] = None
        ref_lp: Optional[Sequence[Sequence[float]]] = None

        with _step_profiler.track("get_logps", actor_name=name):
            old_lp = (
                self._get_logps(
                    ta.model,
                    ids_list,
                    ta.tokenizer,
                    temperature=ta.loss_fn.temperature,
                    batch_size=self.reference_batch_size,
                )
                if self.num_iterations > 1
                else None
            )
            if ta.reference_model is not None:
                ref_lp = self._get_logps(
                    ta.reference_model,
                    ids_list,
                    ta.tokenizer,
                    temperature=ta.loss_fn.temperature,
                    batch_size=self.reference_batch_size,
                )
            elif is_peft_model(ta.model):  # TODO: Beta here check too.
                with ta.model.disable_adapter():
                    ref_lp = self._get_logps(
                        ta.model,
                        ids_list,
                        ta.tokenizer,
                        temperature=ta.loss_fn.temperature,
                        batch_size=self.reference_batch_size,
                    )
            else:
                ref_lp = None

        for substep_idx in range(self.num_iterations):
            if self.accel.is_main_process:
                if self.num_iterations > 1:
                    self.logger.normal(
                        colorize(
                            f"🔄 Backwards iter {substep_idx+1}/{self.num_iterations} for actor '{name}'",
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

            grad_norm = self._clip_gradients(ta, clip_to=self.max_grad_norm)
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
        if ta.actor.offload_optimizer:
            with _step_profiler.track("offload_optimizer", actor_name=name):
                offload_model_and_optimizer(
                    ta.model, ta.optim, offload_optimizer=True, offload_model=False
                )

        # Track actor weight update
        self._update_actor_weights(ta)
        ta.actor.sleep()

    def _backward_one_slice(
        self,
        ta: InitializedTrainableLLMActor,
        ids: List[List[int]],
        masks: List[List[int]],
        advantages: List[float],
        ref_lp_slice: Optional[List[List[float]]],
        old_lp_slice: List[List[float]],
        result: TrainingMetrics,
        substep_idx: int,
        actor_name: str,
    ) -> None:
        tok, dev = ta.tokenizer, ta.model.device
        padded = tok.pad({"input_ids": ids}, padding="longest", return_tensors="pt")
        ids_pt, attention_mask = padded["input_ids"].to(dev), padded[
            "attention_mask"
        ].to(dev)

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
        # policy_for_loss = (
        #     unwrapped_model.base_model.model  # PEFT: use the inner model
        #     if is_peft_model(ta.model)
        #     else unwrapped_model              # vanilla model
        # )

        with _step_profiler.track("loss_fn", actor_name=ta.name):
            # TODO: Higher VRAM usage here which is undesirable.
            # loss, stats = self._forward_redirection(
            #     ta.model,              # wrapper  (DeepSpeedEngine/FSDP/…)
            #     unwrapped_model,       # original (torch.nn.Module)
            #     ta.loss_fn.forward,    # *bound* method of the loss object
            #     # ---- everything the loss expects --------------------
            #     policy_for_loss,       # ← first positional arg: `policy`
            #     ids_pt,
            #     attention_mask,
            #     loss_attention_mask,
            #     adv_pt,
            #     ref_lp,
            #     old_lp,
            # )
            loss, stats = ta.loss_fn(
                policy=(
                    unwrapped_model.base_model.model
                    if is_peft_model(ta.model)
                    else unwrapped_model
                ),
                input_ids=ids_pt,
                attention_mask=attention_mask,
                loss_attention_mask=loss_attention_mask,
                advantages=adv_pt,
                ref_logps=ref_lp,
                old_logps=old_lp,
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

    @property
    def group_size(self) -> int:
        return self.cfg.group_size

    def _build_completion_data(
        self,
        actor_name: str,
        ta: InitializedTrainableLLMActor,
        actor_output,
        batch: Dict[str, List[Any]],
    ) -> Dict[str, List[Any]]:
        batch_keys = list(batch.keys())
        completions_ids = actor_output.input_ids
        total_rewards = actor_output.rewards

        data = {}

        for k in batch_keys:
            data[k] = batch[k]

        data["completion"] = [
            ta.tokenizer.decode(completion_ids, skip_special_tokens=False)
            for completion_ids in completions_ids
        ]

        data["total_reward"] = total_rewards

        if hasattr(self, "_calculate_advantages"):
            advantages = self._calculate_advantages(
                total_rewards,
                getattr(self.cfg, "group_size", 1),
                actor_output.ended_in_eos,
            )
            data["advantage"] = advantages

        if actor_output.reward_components:
            for comp_name, comp_values in actor_output.reward_components.items():
                data[comp_name] = comp_values

        return data
