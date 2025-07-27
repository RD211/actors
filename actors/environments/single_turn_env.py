from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from datasets import Dataset as HFDataset
from datasets import DatasetDict
from vllm import SamplingParams

from actors.actors.base import TrainableLLMActor
from actors.environments.env_base import Environment
from actors.environments.types import ActorOutput, EnvironmentOutput
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
        super().__init__(
            train_data=train_data,
            eval_data=eval_data,
        )

        if not reward_functions:
            raise ValueError("At least one reward function must be provided")

        self.actor = actor
        self.tokenizer = self.actor.training_config.tokenizer_factory()
        self.prompt_column = prompt_column
        self.mask_prompt_for_loss = mask_prompt_for_loss

        # Set actor.loss_temp to sampling_params.temperature
        self.actor.training_config.loss_temp = sampling_params.temperature

        self.sampling_params = sampling_params

        self.reward_functions: list[RewardFunction] = []
        for rf in reward_functions:
            if isinstance(rf, RewardFunction):
                self.reward_functions.append(rf)
            elif callable(rf):
                func_name = rf.__name__ if hasattr(rf, "__name__") else "reward_func"
                self.reward_functions.append(
                    RewardFunction(name=func_name, weight=1.0, func=rf)
                )
            else:
                raise ValueError(
                    f"Reward function must be RewardFunction or callable, "
                    f"got {type(rf)}"
                )

        names = [rf.name for rf in self.reward_functions]
        if len(names) != len(set(names)):
            raise ValueError(f"Reward function names must be unique, got: {names}")

    async def generate(self, batch: dict[str, Any]) -> EnvironmentOutput:
        self.actor.wake()

        prompts = batch[self.prompt_column]
        generations = await self.actor.agenerate(
            prompts, sampling_params=self.sampling_params
        )

        completions = [gen.outputs[0].text for gen in generations]

        generated_texts = []
        for prompt, completion in zip(prompts, completions, strict=False):
            generated_texts.append(prompt + completion)

        input_ids_list = []
        attention_mask_list = []

        for text in generated_texts:
            tokenized = self.tokenizer(
                text, return_tensors="pt", padding=False, truncation=False
            )
            input_ids_list.append(tokenized.input_ids.squeeze(0).tolist())
            attention_mask_list.append(tokenized.attention_mask.squeeze(0).tolist())

        if self.mask_prompt_for_loss:
            modified_attention_mask_list = []

            for i, prompt in enumerate(prompts):
                prompt_tokens = (
                    self.tokenizer(
                        prompt, return_tensors="pt", padding=False, truncation=False
                    )
                    .input_ids.squeeze(0)
                    .tolist()
                )

                prompt_length = len(prompt_tokens)
                current_mask = attention_mask_list[i].copy()

                for j in range(len(current_mask)):
                    if j < prompt_length and current_mask[j] == 1:
                        current_mask[j] = 0

                modified_attention_mask_list.append(current_mask)

            attention_mask_list = modified_attention_mask_list

        rewards_by_function = {}
        total_rewards = []

        for i, (prompt, completion) in enumerate(
            zip(prompts, completions, strict=False)
        ):
            entry_rewards = {}
            total_reward = 0.0

            for reward_func in self.reward_functions:
                reward_value = reward_func.compute_reward(
                    prompt=prompt,
                    completion=completion,
                    actor_name=self.actor.name,
                    **{
                        k: v[i]
                        for k, v in batch.items()
                        if k != self.prompt_column and type(v) is list
                    },
                )

                entry_rewards[reward_func.name] = reward_value
                total_reward += reward_func.weight * reward_value

            for name, value in entry_rewards.items():
                if name not in rewards_by_function:
                    rewards_by_function[name] = []
                rewards_by_function[name].append(value)

            total_rewards.append(total_reward)

        actor_output = ActorOutput(
            input_ids=input_ids_list,
            attention_mask=attention_mask_list,
            rewards=total_rewards,
            reward_components=rewards_by_function,
        )

        return EnvironmentOutput(
            actors={self.actor.name: actor_output},
        )
