from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from datasets import Dataset as HFDataset
from datasets import DatasetDict
from jinja2 import Template
from vllm import RequestOutput, SamplingParams

from actors.actors.base import TrainableLLMActor
from actors.environments.env_base import Environment
from actors.environments.types import EnvironmentOutput
from actors.rewards import (
    RewardFunction,
)

SAMPLER_PROMPT = r"""
You are tasked with helping a user with a task.
You will try to be very creative and helpful with your answers.
You must put your hidden thinking like such:
<think>
... your hidden thinking here ...
</think>
You will then provide a response to the user normally after the thinking has been completed.
You must always think before you answer and never leak your hidden thinking to the user.

Here is the task you must help with:
{{ problem }}
"""

COMBINER_PROMPT = r"""
You are tasked with combining the answers of multiple agents.

Here is the task you must help with:
{{ problem }}
Here are the answers you must combine:
{% for answer in answers %}
[answer]
{{ answer }}
[/answer]
{% endfor %}
You must combine these answers into a single coherent response.
Obviously, if the answers are conflicting, you must try to resolve the conflict and pick the best one.
You can also be creative and suggest a new answer based on the provided answers.
Please put your final answer in a boxed format like this: \\boxed{your_answer}.
"""


# ════════════════════════════════════════════════════════════════════════
# Config objects
# ════════════════════════════════════════════════════════════════════════
@dataclass
class ParallelActorConfig:
    actor: TrainableLLMActor
    sampling_params: SamplingParams
    problem_prompt: str | None = None
    num_samples: int = 1


@dataclass
class CombinerActorConfig:
    actor: TrainableLLMActor
    sampling_params: SamplingParams
    problem_prompt: str | None = None


# ════════════════════════════════════════════════════════════════════════
# Helper
# ════════════════════════════════════════════════════════════════════════
def _wrap_reward_fns(
    fns: Sequence[RewardFunction | Callable],
) -> list[RewardFunction]:
    wrapped: list[RewardFunction] = []
    for rf in fns:
        if isinstance(rf, RewardFunction):
            wrapped.append(rf)
        elif callable(rf):
            wrapped.append(
                RewardFunction(
                    name=getattr(rf, "__name__", "reward"), weight=1.0, func=rf
                )
            )
        else:
            raise ValueError(f"Unsupported reward function type: {type(rf)}")
    names = [r.name for r in wrapped]
    if len(names) != len(set(names)):
        raise ValueError(f"Reward function names must be unique: {names}")
    return wrapped


# ════════════════════════════════════════════════════════════════════════
# Environment
# ════════════════════════════════════════════════════════════════════════
class ParallelEnvironment(Environment):
    """
    B user problems -> K sampler actors -> combiner actor.

    For every problem:
      • sampler drafts get their own rewards *plus* the problem-level combiner reward
      • combiner answer gets its own reward

    Generated entries:
      group_name="generation"  - each sampler draft
      group_name="combiner"    - final combined answer
    """

    def __init__(
        self,
        *,
        sampler_cfgs: Sequence[ParallelActorConfig],
        final_combiner: CombinerActorConfig,
        generate_reward_functions: Sequence[RewardFunction | Callable],
        combiner_reward_functions: Sequence[RewardFunction | Callable],
        run_concurrently: bool = True,
        prompt_column: str = "text",
        train_data: HFDataset | DatasetDict | None = None,
        eval_data: (
            HFDataset | DatasetDict | Mapping[str, HFDataset | DatasetDict] | None
        ) = None,
    ):
        if not sampler_cfgs:
            raise ValueError("Provide at least one ParallelActorConfig")
        super().__init__(train_data=train_data, eval_data=eval_data)

        # Actors --------------------------------------------------------
        self.sampler_cfgs = list(sampler_cfgs)
        self.sampler_by_name = {cfg.actor.name: cfg for cfg in self.sampler_cfgs}
        self.final_combiner = final_combiner
        self.prompt_column = prompt_column
        self.run_concurrently = run_concurrently

        # pass sampling temp -> loss temp
        for cfg in self.sampler_cfgs:
            cfg.actor.training_config.loss_temp = cfg.sampling_params.temperature
        self.final_combiner.actor.training_config.loss_temp = (
            self.final_combiner.sampling_params.temperature
        )

        # Rewards -------------------------------------------------------
        self.gen_rewards = _wrap_reward_fns(generate_reward_functions)
        self.comb_rewards = _wrap_reward_fns(combiner_reward_functions)

    # ------------------------------------------------------------------ #
    # generate
    # ------------------------------------------------------------------ #
    async def generate(self, batch: Mapping[str, Any]) -> EnvironmentOutput:
        problems = batch[self.prompt_column]  # batch size B
        B = len(problems)

        # ---------- 1. build prompts -----------------------------------
        def _render(cfg: ParallelActorConfig, idx: int) -> str:
            tmpl = Template(cfg.problem_prompt or SAMPLER_PROMPT)
            entry = {k: batch[k][idx] for k in batch}
            entry["problem"] = problems[idx]
            return [{"role": "user", "content": tmpl.render(**entry)}]

        prompts_by_actor: dict[str, list[dict[str, str]]] = defaultdict(list)
        for cfg in self.sampler_cfgs:
            for idx in range(B):
                prompts_by_actor[cfg.actor.name].extend(
                    [_render(cfg, idx)] * cfg.num_samples
                )

        # ---------- 2. run sampler actors ------------------------------
        async def _chat(cfg: ParallelActorConfig) -> list[str]:
            outs: list[RequestOutput] = await cfg.actor.achat(
                prompts_by_actor[cfg.actor.name], cfg.sampling_params
            )
            return [o.outputs[0].text for o in outs]

        if self.run_concurrently:
            tasks = {
                cfg.actor.name: asyncio.create_task(_chat(cfg))
                for cfg in self.sampler_cfgs
            }

            results = await asyncio.gather(*tasks.values())

            flat_drafts = dict(zip(tasks.keys(), results, strict=True))
        else:
            flat_drafts = {
                cfg.actor.name: await _chat(cfg) for cfg in self.sampler_cfgs
            }

        # reshape -> drafts[actor][row][sample]
        drafts: dict[str, list[list[str]]] = {}
        for cfg in self.sampler_cfgs:
            L = cfg.num_samples
            flat = flat_drafts[cfg.actor.name]
            drafts[cfg.actor.name] = [flat[i * L : (i + 1) * L] for i in range(B)]

        # ---------- 3. combiner ----------------------------------------
        comb_prompts = []
        for idx in range(B):
            all_drafts = [
                d for cfg in self.sampler_cfgs for d in drafts[cfg.actor.name][idx]
            ]

            # We remove the thinking part from the drafts
            all_drafts = [
                d.split("</think>")[-1].strip() for d in all_drafts if "</think>" in d
            ]
            comb_prompts.append(
                [
                    {
                        "role": "user",
                        "content": Template(
                            self.final_combiner.problem_prompt or COMBINER_PROMPT
                        ).render(
                            problem=problems[idx],
                            answers=all_drafts,
                            **{
                                k: v[idx]
                                for k, v in batch.items()
                                if k != self.prompt_column
                            },
                        ),
                    }
                ]
            )

        comb_outs = await self.final_combiner.actor.achat(
            comb_prompts, self.final_combiner.sampling_params
        )
        comb_answers = [o.outputs[0].text for o in comb_outs]

        # ---------- 4. rewards - drafts --------------------------------
        draft_convs, draft_actor_names = [], []
        for cfg in self.sampler_cfgs:
            for idx in range(B):
                user_msg = {"role": "user", "content": _render(cfg, idx)}
                for dr in drafts[cfg.actor.name][idx]:
                    draft_convs.append([user_msg, {"role": "assistant", "content": dr}])
                    draft_actor_names.append(cfg.actor.name)

        def _expand(col: list[Any]) -> list[Any]:
            expanded: list[Any] = []
            for cfg in self.sampler_cfgs:  # same actor order
                for v in col:  # iterate over problems
                    expanded.extend([v] * cfg.num_samples)
            return expanded

        draft_reward: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        if self.gen_rewards:
            extra_cols = {
                k: _expand(v) for k, v in batch.items() if isinstance(v, list)
            }
            vals_all = {}  # cache so we slice only once per RF
            for rf in self.gen_rewards:
                vals = rf.compute_rewards(
                    prompts=[c[0]["content"] for c in draft_convs],
                    completions=[c[1]["content"] for c in draft_convs],
                    actor_names=draft_actor_names,
                    **extra_cols,
                )
                vals_all[rf.name] = vals

            # slice rewards back to per-actor buckets -------------------------
            cursor = {cfg.actor.name: 0 for cfg in self.sampler_cfgs}
            for rf in self.gen_rewards:
                for cfg in self.sampler_cfgs:
                    N = B * cfg.num_samples
                    start = cursor[cfg.actor.name]
                    end = start + N
                    draft_reward[cfg.actor.name][rf.name] = vals_all[rf.name][start:end]
                    cursor[cfg.actor.name] += N

        # ---------- 5. rewards - combiner ------------------------------

        comb_reward: dict[str, list[float]] = defaultdict(list)
        if self.comb_rewards:
            for rf in self.comb_rewards:
                comb_reward[rf.name] = rf.compute_rewards(
                    prompts=[c[0]["content"] for c in comb_prompts],
                    completions=comb_answers,
                    actor_names=[self.final_combiner.actor.name] * B,
                    **batch,
                )

        # pre-compute weighted combiner total per problem
        comb_weights = {rf.name: rf.weight for rf in self.comb_rewards}
        comb_total = [
            sum(comb_weights[n] * comb_reward[n][i] for n in comb_reward)
            for i in range(B)
        ]

        # ---------- 6. EnvironmentOutput -------------------------------
        env_out = EnvironmentOutput()

        # sampler drafts
        for cfg in self.sampler_cfgs:
            if not cfg.actor.is_actually_trainable:
                continue
            tok = cfg.actor.training_config.tokenizer_factory()
            gen_w = {rf.name: rf.weight for rf in self.gen_rewards}

            for idx in range(B):
                for s, draft in enumerate(drafts[cfg.actor.name][idx]):
                    flat_idx = idx * cfg.num_samples + s
                    comp_vals = {
                        k: draft_reward[cfg.actor.name][k][flat_idx]
                        for k in draft_reward[cfg.actor.name]
                    }
                    # merge combiner components (prefixed)
                    for k in comb_reward:
                        comp_vals[f"combiner::{k}"] = comb_reward[k][idx]

                    total = sum(
                        gen_w[n] * v
                        for n, v in comp_vals.items()
                        if not n.startswith("combiner::")
                    )
                    total += comb_total[idx]  # propagate combiner signal

                    env_out.add_entry(
                        problem_idx=idx,
                        actor_name=cfg.actor.name,
                        group_name="generation",
                        tokenizer=tok,
                        messages=[
                            {
                                "role": "user",
                                "content": _render(cfg, idx)[0]["content"],
                            },
                            {"role": "assistant", "content": draft},
                        ],
                        rewards=total,
                        reward_components=comp_vals,
                    )

        # combiner
        if self.final_combiner.actor.is_actually_trainable:
            tok_c = self.final_combiner.actor.training_config.tokenizer_factory()
            comb_w = {rf.name: rf.weight for rf in self.comb_rewards}
            for idx in range(B):
                comp_vals = {k: comb_reward[k][idx] for k in comb_reward}
                total = sum(comb_w[n] * v for n, v in comp_vals.items())

                env_out.add_entry(
                    problem_idx=idx,
                    actor_name=self.final_combiner.actor.name,
                    group_name="combiner",
                    tokenizer=tok_c,
                    messages=[
                        {"role": "user", "content": comb_prompts[idx][0]["content"]},
                        {"role": "assistant", "content": comb_answers[idx]},
                    ],
                    rewards=total,
                    reward_components=comp_vals,
                )

        return env_out
