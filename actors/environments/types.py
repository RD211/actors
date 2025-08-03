from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase

from actors.environments.masking import mask_turns_and_encode


# ════════════════════════════════════════════════════════════════════════
# Actor-level payload
# ════════════════════════════════════════════════════════════════════════
@dataclass
class ActorOutput:
    input_ids: list[list[int]]
    rewards: list[float]
    reward_components: dict[str, list[float]]
    attention_mask: list[list[int]] | None = None
    ended_in_eos: list[bool] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.attention_mask is None:
            self.attention_mask = [[1] * len(seq) for seq in self.input_ids]
        if self.ended_in_eos is None:
            self.ended_in_eos = [True] * len(self.input_ids)

        n = len(self.input_ids)
        if (
            n == 0
            or len(self.attention_mask) != n
            or len(self.rewards) != n
            or len(self.ended_in_eos) != n
        ):
            raise ValueError("length mismatch")
        if any(len(seq) == 0 for seq in self.input_ids):
            raise ValueError("empty sequence")
        for v in self.reward_components.values():
            if len(v) != n:
                raise ValueError("reward component length mismatch")

    def get_reward_stats(self) -> dict[str, dict[str, float]]:
        def stats(vals: list[float]) -> dict[str, float]:
            t = torch.tensor(vals, dtype=torch.float32)
            return {
                "mean": t.mean().item(),
                "std": t.std(unbiased=False).item(),
                "min": t.min().item(),
                "max": t.max().item(),
            }

        out = {"primary": stats(self.rewards)}
        out.update({k: stats(v) for k, v in self.reward_components.items()})
        return out

    def __iadd__(self, other: ActorOutput) -> ActorOutput:
        if not isinstance(other, ActorOutput):
            raise ValueError("Can only merge with another ActorOutput")

        prev_len = len(self.input_ids)
        new_len = len(other.input_ids)

        for k, v in self.reward_components.items():
            if k not in other.reward_components:
                v.extend([None] * new_len)

        for k, v in other.reward_components.items():
            if k in self.reward_components:
                self.reward_components[k].extend(v)
            else:
                self.reward_components[k] = ([None] * prev_len) + list(v)

        self.input_ids.extend(other.input_ids)
        self.attention_mask.extend(other.attention_mask)
        self.rewards.extend(other.rewards)
        self.ended_in_eos.extend(other.ended_in_eos)

        return self


# ════════════════════════════════════════════════════════════════════════
# Environment-level container
# ════════════════════════════════════════════════════════════════════════
@dataclass
class EnvironmentOutput:
    # problem_idx -> actor_name -> group_name -> ActorOutput
    outputs: list[dict[str, dict[str, ActorOutput]]] = field(default_factory=list)
    problems: list[dict[str, Any]] = field(default_factory=list)

    def _ensure_problem_idx(self, idx: int) -> None:
        while len(self.outputs) <= idx:
            self.outputs.append(defaultdict(dict))

    @staticmethod
    def _to_list_floats(x: float | list[float]) -> list[float]:
        return [float(x)] if not isinstance(x, list) else [float(v) for v in x]

    @staticmethod
    def _compile_patterns(rx: str | list[str] | None) -> list[re.Pattern]:
        if rx is None:
            return []
        pats = [rx] if isinstance(rx, str) else list(rx)
        return [re.compile(p) for p in pats]

    def _encode_text_with_mask(
        self,
        tok: PreTrainedTokenizerBase,
        text: str,
        mask_regex: str | list[str] | None = None,
        mask_spans: list[tuple[int, int]] | None = None,
    ) -> tuple[list[int], list[int]]:
        enc = tok.encode_plus(
            text, add_special_tokens=False, return_offsets_mapping=True
        )
        ids = enc["input_ids"]
        offs = enc["offset_mapping"]
        mask = [1] * len(ids)
        for pat in self._compile_patterns(mask_regex):
            for m in pat.finditer(text):
                sc, ec = m.span()
                for ti, (ts, te) in enumerate(offs):
                    if ts < ec and te > sc:
                        mask[ti] = 0
        if mask_spans:
            T = len(ids)
            for s, e in mask_spans:
                for ti in range(max(0, s), min(T, e)):
                    mask[ti] = 0
        return ids, mask

    def add_entry(
        self,
        *,
        problem_idx: int,
        actor_name: str,
        group_name: str,
        tokenizer: PreTrainedTokenizerBase | None = None,
        text: str | None = None,
        input_ids: list[int] | None = None,
        messages: list[dict[str, str]] | None = None,
        rewards: float | list[float],
        reward_components: dict[str, float | list[float]],
        attention_mask: list[int] | None = None,
        mask_non_actor_turns: bool = False,
        mask_regex: str | list[str] | None = None,
        mask_spans: list[tuple[int, int]] | None = None,
        ended_in_eos: bool | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        opts = [text is not None, input_ids is not None, messages is not None]
        if sum(opts) != 1:
            raise ValueError("provide exactly one of text, input_ids, messages")
        if (text is not None or messages is not None) and tokenizer is None:
            raise ValueError("tokenizer required for text or messages")

        # build ids / mask
        if text is not None:
            ids, loss_mask = self._encode_text_with_mask(
                tokenizer, text, mask_regex, mask_spans
            )
        elif messages is not None:
            ids, loss_mask = mask_turns_and_encode(
                tokenizer, messages, mask_non_actor_turns
            )
        else:
            ids = input_ids
            loss_mask = attention_mask or [1] * len(ids)

        if not ids:
            raise ValueError("empty ids")
        if len(ids) != len(loss_mask):
            raise ValueError("ids mask length mismatch")

        rewards_list = self._to_list_floats(rewards)
        rc_norm = {k: self._to_list_floats(v) for k, v in reward_components.items()}

        for v in rc_norm.values():
            if len(v) != len(rewards_list):
                raise ValueError("reward component length mismatch")

        self._ensure_problem_idx(problem_idx)
        new_ao = ActorOutput(
            input_ids=[ids],
            attention_mask=[loss_mask],
            rewards=rewards_list,
            reward_components=rc_norm,
            ended_in_eos=[ended_in_eos if ended_in_eos is not None else True],
            metadata=metadata or {},
        )

        problem_bucket = self.outputs[problem_idx]
        actor_bucket = problem_bucket.setdefault(actor_name, {})
        if group_name in actor_bucket:
            actor_bucket[group_name] += new_ao
        else:
            actor_bucket[group_name] = new_ao

    def get_actor_output(
        self,
        actor_name: str,
        *,
        problem_indices: list[int] | None = None,
        groups: list[str] | None = None,
    ) -> list[dict[str, ActorOutput]]:
        result: list[dict[str, ActorOutput]] = []
        for idx, pb in enumerate(self.outputs):
            if problem_indices is not None and idx not in problem_indices:
                continue
            group_outs: dict[str, ActorOutput] = {}
            if actor_name in pb:
                for g, ao in pb[actor_name].items():
                    if groups is None or g in groups:
                        group_outs[g] = ao
            result.append(group_outs)
        return result

    def reward_stats(self, actor_name: str) -> dict[str, dict[str, float]]:
        agg: dict[str, list[float]] = {}
        prim: list[float] = []
        for probs in self.get_actor_output(actor_name):
            for ao in probs:
                prim.extend(ao.rewards)
                for k, v in ao.reward_components.items():
                    agg.setdefault(k, []).extend(v)

        def stats(vals: list[float]) -> dict[str, float]:
            if not vals:
                return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
            t = torch.tensor(vals, dtype=torch.float32)
            return {
                "mean": t.mean().item(),
                "std": t.std(unbiased=False).item(),
                "min": t.min().item(),
                "max": t.max().item(),
            }

        out = {"primary": stats(prim)}
        out.update({k: stats(v) for k, v in agg.items()})
        return out

    # merge in-place
    def __iadd__(self, other: EnvironmentOutput) -> EnvironmentOutput:
        for p_idx, pb in enumerate(other.outputs):
            self._ensure_problem_idx(p_idx)
            for actor, gb in pb.items():
                for grp, ao in gb.items():
                    bucket = self.outputs[p_idx][actor]
                    if grp in bucket:
                        bucket[grp] += ao
                    else:
                        bucket[grp] = ao
        self.problems.extend(other.problems)
        return self

    # out-of-place +
    def __add__(self, other: EnvironmentOutput) -> EnvironmentOutput:
        merged = EnvironmentOutput()
        merged += self
        merged += other
        return merged

    @staticmethod
    def combine_and_group(
        env_outputs: list[EnvironmentOutput],
        problem_groups: list[int],
        problems: list[dict[str, Any]],
    ) -> EnvironmentOutput:
        pre_combined_out = EnvironmentOutput()
        for eo in env_outputs:
            pre_combined_out += eo

        # We now have a single EnvironmentOutput with all problems.
        # We need to group them by problem_groups.
        combined_out = EnvironmentOutput()
        for problem_idx, problem_out in enumerate(pre_combined_out.outputs):
            corresponding_group = problem_groups[problem_idx]
            for actor_name, group_outs in problem_out.items():
                for group_name, actor_output in group_outs.items():
                    for i in range(len(actor_output.input_ids)):
                        combined_out.add_entry(
                            problem_idx=corresponding_group,
                            actor_name=actor_name,
                            group_name=group_name,
                            input_ids=actor_output.input_ids[i],
                            attention_mask=actor_output.attention_mask[i],
                            rewards=actor_output.rewards[i],
                            reward_components={
                                k: v[i]
                                for k, v in actor_output.reward_components.items()
                            },
                            ended_in_eos=actor_output.ended_in_eos[i],
                            metadata=actor_output.metadata,
                        )
            if corresponding_group >= len(combined_out.problems):
                combined_out.problems.append(problems[corresponding_group])
        return combined_out
