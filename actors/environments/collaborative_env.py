from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from datasets import Dataset as HFDataset
from datasets import DatasetDict
from vllm import RequestOutput, SamplingParams

from actors.actors.base import TrainableLLMActor
from actors.environments.actors_schedule_dsl import sample_schedule
from actors.environments.env_base import Environment
from actors.environments.masking import mask_turns_and_encode
from actors.environments.types import ActorOutput, EnvironmentOutput
from actors.rewards import (
    ConversationRewardFunction,
)


@dataclass
class CollaborativeActorConfig:
    actor: TrainableLLMActor
    system_prompt: str
    sampling_params: SamplingParams


class CollaborativeEnvironment(Environment):
    def __init__(
        self,
        *,
        actor_cfgs: Sequence[CollaborativeActorConfig],
        round_spec: str,
        reward_functions: Sequence[ConversationRewardFunction | Callable],
        run_concurrently: bool = True,
        prompt_column: str = "text",
        mask_other_agents_for_loss: bool = False,
        train_data: HFDataset | DatasetDict | None = None,
        prefill_name: bool = False,  # Show name of other agents to the current agent?
        eval_data: (
            HFDataset | DatasetDict | Mapping[str, HFDataset | DatasetDict] | None
        ) = None,
    ) -> None:
        if not actor_cfgs:
            raise ValueError("Provide at least one CollaborativeActorConfig")
        super().__init__(train_data=train_data, eval_data=eval_data)

        # Store actors and build a lookup by name
        self.actor_cfgs: list[CollaborativeActorConfig] = list(actor_cfgs)
        self.actor_by_name: dict[str, CollaborativeActorConfig] = {
            cfg.actor.name: cfg for cfg in self.actor_cfgs
        }
        self.all_names: list[str] = list(self.actor_by_name)
        if len(self.all_names) != len(self.actor_cfgs):
            raise ValueError(
                "Actor names must be unique, found duplicates: "
                f"{[name for name in self.all_names if self.all_names.count(name) > 1]}"
            )

        self.schedule_dsl_spec = round_spec

        # Execution control
        self.run_concurrently = run_concurrently
        self.prompt_column = prompt_column
        self.mask_other_agents_for_loss = mask_other_agents_for_loss

        # Propagate actor sampling temperatures into training config
        for cfg in self.actor_cfgs:
            cfg.actor.training_config.loss_temp = cfg.sampling_params.temperature

        # Build reward functions
        self.reward_functions: list[ConversationRewardFunction] = []
        for rf in reward_functions:
            if isinstance(rf, ConversationRewardFunction):
                self.reward_functions.append(rf)
            elif callable(rf):
                # Wrap bare callables into ConversationRewardFunction objects
                self.reward_functions.append(
                    ConversationRewardFunction(
                        name=getattr(rf, "__name__", "reward"), weight=1.0, func=rf
                    )
                )
            else:
                raise ValueError(f"Unsupported reward-function type: {type(rf)}")

        # Ensure unique reward names
        names = [r.name for r in self.reward_functions]
        if len(names) != len(set(names)):
            raise ValueError(f"Reward function names must be unique: {names}")

        self.prefill_name = prefill_name

    async def generate(self, batch: Mapping[str, Any]) -> EnvironmentOutput:
        problems = batch[self.prompt_column]
        batch_size = len(problems)

        per_row_turns: list[list[str]] = [
            sample_schedule(self.schedule_dsl_spec, self.all_names)
            for _ in range(batch_size)
        ]

        dialogs: list[list[dict[str, Any]]] = [[] for _ in range(batch_size)]

        max_turns = max(len(ts) for ts in per_row_turns)

        for turn_idx in range(max_turns):
            rows_with_turn: list[int] = [
                i for i, turns in enumerate(per_row_turns) if turn_idx < len(turns)
            ]
            if not rows_with_turn:
                continue

            actor_for_row: dict[int, str] = {
                i: per_row_turns[i][turn_idx] for i in rows_with_turn
            }

            rows_by_actor: dict[str, list[int]] = defaultdict(list)
            for row_idx, name in actor_for_row.items():
                rows_by_actor[name].append(row_idx)

            order: Iterable[str] = rows_by_actor.keys()

            async def _chat_one_actor(name: str, row_indices: list[int]) -> None:
                cfg = self.actor_by_name[name]
                actor_prompts: list[str] = []
                tokenizer = cfg.actor.training_config.tokenizer_factory()

                cfg.actor.wake()
                for idx in row_indices:
                    conv_msgs: list[dict[str, str]] = []
                    conv_msgs.append(
                        {
                            "role": "system",
                            "content": cfg.system_prompt + problems[idx],
                        }
                    )

                    for m in dialogs[idx]:
                        role = "user" if m["author"] != name else "assistant"
                        content = m["content"]
                        if self.prefill_name and role == "user":
                            content = f"{m['author']} says: {content}"
                        conv_msgs.append({"role": role, "content": content})

                    prompt = tokenizer.apply_chat_template(
                        conv_msgs,
                        add_generation_prompt=True,
                        tokenize=False,
                    )
                    actor_prompts.append(prompt)

                outs: list[RequestOutput] = await cfg.actor.agenerate(
                    prompts=actor_prompts,
                    sampling_params=cfg.sampling_params,
                )
                completions = [o.outputs[0].text for o in outs]
                for idx, comp in zip(row_indices, completions, strict=False):
                    dialogs[idx].append(
                        {
                            "content": comp,
                            "author": name,
                        }
                    )
                cfg.actor.sleep()

            if self.run_concurrently:
                await asyncio.gather(
                    *[
                        _chat_one_actor(name, rows_by_actor[name])
                        for name in list(order)
                    ]
                )
            else:
                for name in order:
                    await _chat_one_actor(name, rows_by_actor[name])

        # ------------------------------------------------------------------------
        n = len(dialogs)
        actors_out: dict[str, ActorOutput] = {}

        actors_tok: dict[str, dict[str, list[list[int]]]] = {}
        conversations_by_actor: dict[str, list[list[dict[str, str]]]] = {}

        for cfg in self.actor_cfgs:
            tok = cfg.actor.training_config.tokenizer_factory()
            ids_batch, mask_batch = [], []
            convs: list[list[dict[str, str]]] = []

            for i, row_msgs in enumerate(dialogs):
                converted: list[dict[str, str]] = [
                    {"role": "system", "content": cfg.system_prompt + problems[i]}
                ]
                for m in row_msgs:
                    role = "assistant" if m.get("author") == cfg.actor.name else "user"
                    content = m.get("content", "")
                    if self.prefill_name and role == "user":
                        content = f"{m['author']} says: {content}"
                    converted.append({"role": role, "content": content})

                convs.append(converted)

                ids, msk = mask_turns_and_encode(
                    tok,
                    converted,
                    mask_non_assistant_turns=self.mask_other_agents_for_loss,
                )
                ids_batch.append(ids)
                mask_batch.append(msk)

            actors_tok[cfg.actor.name] = {
                "input_ids": ids_batch,
                "attention_mask": mask_batch,
            }
            conversations_by_actor[cfg.actor.name] = convs

        # ------------------------------------------------------------------------
        A = len([cfg for cfg in self.actor_cfgs if cfg.actor.is_actually_trainable])
        conversations_flat = [
            conv
            for cfg in self.actor_cfgs
            for conv in conversations_by_actor[cfg.actor.name]
            if cfg.actor.is_actually_trainable
        ]
        actor_names_flat = [
            cfg.actor.name
            for cfg in self.actor_cfgs
            for _ in range(n)
            if cfg.actor.is_actually_trainable
        ]

        reward_components_by_actor: dict[str, dict[str, list[float]]] = {
            cfg.actor.name: {}
            for cfg in self.actor_cfgs
            if cfg.actor.is_actually_trainable
        }

        for rf in self.reward_functions:
            vals = rf.compute_rewards(
                conversations=conversations_flat,
                actor_names=actor_names_flat,
                **{k: v * A for k, v in batch.items() if isinstance(v, list)},
            )
            if len(vals) != A * n:
                raise ValueError(
                    f"Reward '{rf.name}' returned {len(vals)} items; expected {A * n}."
                )

            for a, cfg in enumerate(
                [cfg for cfg in self.actor_cfgs if cfg.actor.is_actually_trainable]
            ):
                reward_components_by_actor[cfg.actor.name][rf.name] = [
                    float(x) for x in vals[a * n : (a + 1) * n]
                ]

        for cfg in [cfg for cfg in self.actor_cfgs if cfg.actor.is_actually_trainable]:
            comps = reward_components_by_actor[cfg.actor.name]
            totals = [
                sum(rf.weight * comps[rf.name][i] for rf in self.reward_functions)
                for i in range(n)
            ]
            actors_out[cfg.actor.name] = ActorOutput(
                input_ids=actors_tok[cfg.actor.name]["input_ids"],
                attention_mask=actors_tok[cfg.actor.name]["attention_mask"],
                rewards=totals,
                reward_components=comps,
            )

        return EnvironmentOutput(actors=actors_out)
