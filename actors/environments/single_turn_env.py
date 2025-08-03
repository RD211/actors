from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from datasets import Dataset as HFDataset
from datasets import DatasetDict
from vllm import SamplingParams

from actors.actors.base import TrainableLLMActor
from actors.environments.env_base import Environment
from actors.environments.types import EnvironmentOutput
from actors.rewards import RewardFunction


class SingleTurnEnvironment(Environment):
    def __init__(
        self,
        actor: TrainableLLMActor,
        sampling_params: SamplingParams,
        reward_functions: Sequence[RewardFunction | Callable],
        prompt_column: str = "text",
        mask_prompt_for_loss: bool = True,
        train_data: HFDataset | DatasetDict | None = None,
        eval_data: (
            HFDataset | DatasetDict | dict[str, HFDataset | DatasetDict] | None
        ) = None,
    ):
        super().__init__(train_data=train_data, eval_data=eval_data)

        if not reward_functions:
            raise ValueError("At least one reward function must be provided")

        self.actor = actor
        self.tokenizer = self.actor.training_config.tokenizer_factory()
        self.prompt_column = prompt_column
        self.mask_prompt_for_loss = mask_prompt_for_loss
        self.actor.training_config.loss_temp = sampling_params.temperature
        self.sampling_params = sampling_params

        self.reward_functions: list[RewardFunction] = []
        for rf in reward_functions:
            if isinstance(rf, RewardFunction):
                self.reward_functions.append(rf)
            elif callable(rf):
                name = getattr(rf, "__name__", "reward")
                self.reward_functions.append(
                    RewardFunction(name=name, weight=1.0, func=rf)
                )
            else:
                raise ValueError(f"Unsupported reward function type: {type(rf)}")

        names = [r.name for r in self.reward_functions]
        if len(names) != len(set(names)):
            raise ValueError(f"Reward function names must be unique: {names}")

    async def generate(self, batch: dict[str, Any]) -> EnvironmentOutput:
        prompts: list[str] = batch[self.prompt_column]

        self.actor.wake()
        gens = await self.actor.agenerate(prompts, self.sampling_params)
        completions = [g.outputs[0].text for g in gens]
        self.actor.sleep()

        env_out = EnvironmentOutput()

        for idx, (prompt, completion) in enumerate(
            zip(prompts, completions, strict=False)
        ):
            full_text = prompt + completion

            enc = self.tokenizer(
                full_text, return_tensors="pt", add_special_tokens=False
            )
            ids = enc.input_ids.squeeze(0).tolist()
            mask = enc.attention_mask.squeeze(0).tolist()

            if self.mask_prompt_for_loss:
                prompt_ids_len = len(
                    self.tokenizer(prompt, add_special_tokens=False).input_ids
                )
                for j in range(prompt_ids_len):
                    mask[j] = 0

            reward_components: dict[str, float] = {}
            total_reward = 0.0
            for rf in self.reward_functions:
                val = rf.compute_reward(
                    prompt=prompt,
                    completion=completion,
                    actor_name=self.actor.name,
                    **{
                        k: v[idx]
                        for k, v in batch.items()
                        if k != self.prompt_column and isinstance(v, list)
                    },
                )
                reward_components[rf.name] = val
                total_reward += rf.weight * val

            env_out.add_entry(
                problem_idx=idx,
                actor_name=self.actor.name,
                group_name="completion",
                input_ids=ids,
                attention_mask=mask,
                rewards=total_reward,
                reward_components=reward_components,
            )

        return env_out
