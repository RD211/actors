from __future__ import annotations

import asyncio
import re
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from datasets import Dataset as HFDataset
from datasets import DatasetDict
from jinja2 import Template
from vllm import SamplingParams

from actors.actors.base import TrainableLLMActor
from actors.environments.actors_schedule_dsl import sample_schedule
from actors.environments.env_base import Environment
from actors.environments.masking import mask_turns_and_encode
from actors.environments.types import EnvironmentOutput
from actors.rewards import (
    ConversationRewardFunction,
)


def looks_like_jinja2_template(s):
    return bool(re.search(r"({{.*?}}|{%.*?%}|{#.*?#})", s))


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
        mask_other_agents_for_loss: bool = True,
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
        n = len(problems)

        def prompt_for(cfg: CollaborativeActorConfig, idx: int) -> str:
            if looks_like_jinja2_template(cfg.system_prompt):
                tmpl = Template(cfg.system_prompt)
                entry = {k: batch[k][idx] for k in batch}
                return tmpl.render(**entry)
            return cfg.system_prompt + problems[idx]

        per_row_turns = [
            sample_schedule(self.schedule_dsl_spec, self.all_names) for _ in range(n)
        ]
        dialogs: list[list[dict[str, str]]] = [[] for _ in range(n)]
        max_turns = max(len(ts) for ts in per_row_turns)

        async def _chat_one_actor(name: str, row_indices: list[int]) -> None:
            cfg = self.actor_by_name[name]
            tok = cfg.actor.training_config.tokenizer_factory()
            prompts = []

            cfg.actor.wake()
            for ridx in row_indices:
                conv_msgs = [{"role": "system", "content": prompt_for(cfg, ridx)}]
                for m in dialogs[ridx]:
                    role = "assistant" if m["author"] == name else "user"
                    content = m["content"]
                    if self.prefill_name and role == "user":
                        content = f"{m['author']} says: {content}"
                    conv_msgs.append({"role": role, "content": content})
                prompts.append(
                    tok.apply_chat_template(
                        conv_msgs, add_generation_prompt=True, tokenize=False
                    )
                )

            outs = await cfg.actor.agenerate(prompts, cfg.sampling_params)
            completions = [o.outputs[0].text for o in outs]
            for ridx, comp in zip(row_indices, completions, strict=False):
                dialogs[ridx].append({"content": comp, "author": name})
            cfg.actor.sleep()

        for turn_idx in range(max_turns):
            row_has_turn = [
                i for i, ts in enumerate(per_row_turns) if turn_idx < len(ts)
            ]
            rows_by_actor: dict[str, list[int]] = defaultdict(list)
            for ridx in row_has_turn:
                rows_by_actor[per_row_turns[ridx][turn_idx]].append(ridx)

            coros = [
                _chat_one_actor(name, rows_by_actor[name]) for name in rows_by_actor
            ]
            await (
                asyncio.gather(*coros)
                if self.run_concurrently
                else asyncio.gather(*list(coros))
            )

        actors_tok: dict[str, dict[str, list[list[int]]]] = {}
        conversations_by_actor: dict[str, list[list[dict[str, str]]]] = {}

        for cfg in self.actor_cfgs:
            tok = cfg.actor.training_config.tokenizer_factory()
            ids_batch, mask_batch, conv_batch = [], [], []

            for idx in range(n):
                conv_msgs = [{"role": "system", "content": prompt_for(cfg, idx)}]
                for m in dialogs[idx]:
                    role = "assistant" if m["author"] == cfg.actor.name else "user"
                    content = m["content"]
                    if self.prefill_name and role == "user":
                        content = f"{m['author']} says: {content}"
                    conv_msgs.append({"role": role, "content": content})

                conv_batch.append(conv_msgs)
                ids, msk = mask_turns_and_encode(
                    tok,
                    conv_msgs,
                    mask_non_assistant_turns=self.mask_other_agents_for_loss,
                )
                ids_batch.append(ids)
                mask_batch.append(msk)

            actors_tok[cfg.actor.name] = {
                "input_ids": ids_batch,
                "attention_mask": mask_batch,
            }
            conversations_by_actor[cfg.actor.name] = conv_batch

        # ---------------------------------------------------------------
        # Compute rewards
        # ---------------------------------------------------------------
        trainable_cfgs = [
            cfg for cfg in self.actor_cfgs if cfg.actor.is_actually_trainable
        ]
        A = len(trainable_cfgs)

        conversations_flat = [
            conv
            for cfg in trainable_cfgs
            for conv in conversations_by_actor[cfg.actor.name]
        ]
        actor_names_flat = [cfg.actor.name for cfg in trainable_cfgs for _ in range(n)]

        reward_components_by_actor: dict[str, dict[str, list[float]]] = {
            cfg.actor.name: {} for cfg in trainable_cfgs
        }

        for rf in self.reward_functions:
            vals = rf.compute_rewards(
                conversations=conversations_flat,
                actor_names=actor_names_flat,
                **{k: v * A for k, v in batch.items() if isinstance(v, list)},
            )
            if len(vals) != A * n:
                raise ValueError(
                    f"Reward '{rf.name}' returned {len(vals)}, expected {A * n}"
                )
            for a, cfg in enumerate(trainable_cfgs):
                reward_components_by_actor[cfg.actor.name][rf.name] = [
                    float(x) for x in vals[a * n : (a + 1) * n]
                ]

        # ---------------------------------------------------------------
        # Assemble EnvironmentOutput
        # ---------------------------------------------------------------
        env_out = EnvironmentOutput()

        for cfg in trainable_cfgs:
            comps = reward_components_by_actor[cfg.actor.name]
            totals = [
                sum(rf.weight * comps[rf.name][i] for rf in self.reward_functions)
                for i in range(n)
            ]

            for idx in range(n):
                env_out.add_entry(
                    problem_idx=idx,
                    actor_name=cfg.actor.name,
                    group_name="conversation",
                    input_ids=actors_tok[cfg.actor.name]["input_ids"][idx],
                    attention_mask=actors_tok[cfg.actor.name]["attention_mask"][idx],
                    rewards=totals[idx],
                    reward_components={k: v[idx] for k, v in comps.items()},
                )

        return env_out
