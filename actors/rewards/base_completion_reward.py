from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass
class RewardFunction:
    name: str
    weight: float
    func: Callable[..., float | list[float]]
    batched: bool = False

    def __post_init__(self):
        if not callable(self.func):
            raise ValueError(f"Reward function '{self.name}' must be callable")

        # We overwrite the func to ensure we pass it on only the parameters it wants.
        func = self.func

        def wrapped_func(**kwargs: Any) -> float | list[float]:
            signature = inspect.signature(func)
            filtered_kwargs = {
                k: v for k, v in kwargs.items() if k in signature.parameters.keys()
            }
            return func(**filtered_kwargs)

        self.func = wrapped_func

    def compute_reward(
        self, prompt: str, completion: str, actor_name: str, **entry_data: Any
    ) -> float:
        if self.batched:
            params = {
                "prompt": [prompt],
                "completion": [completion],
                "actor_name": [actor_name],
                **{
                    k: ([v] if not isinstance(v, list | tuple) else v)
                    for k, v in entry_data.items()
                },
            }
            out = self.func(**params)
            if not isinstance(out, Sequence) or isinstance(out, str | bytes):
                raise TypeError(
                    f"Batched reward '{self.name}' must return a sequence of floats; got {type(out)!r}"
                )
            if len(out) != 1:
                raise ValueError(
                    f"Batched reward '{self.name}' returned {len(out)} items for a single input; expected 1."
                )
            return float(out[0])

        params = {
            "prompt": prompt,
            "completion": completion,
            "actor_name": actor_name,
            **dict(entry_data.items()),
        }
        return float(self.func(**params))

    def compute_rewards(
        self,
        prompts: Sequence[str],
        completions: Sequence[str],
        actor_names: Sequence[str],
        **entry_data: Any,
    ) -> list[float]:
        if not self.batched:
            if not (len(prompts) == len(completions) == len(actor_names)):
                raise ValueError(
                    "prompts, completions, and actor_names must have equal length"
                )
            return [
                self.compute_reward(
                    p,
                    c,
                    a,
                    **{
                        k: (v[i] if isinstance(v, list | tuple) else v)
                        for k, v in entry_data.items()
                    },
                )
                for i, (p, c, a) in enumerate(
                    zip(prompts, completions, actor_names, strict=False)
                )
            ]

        if not (len(prompts) == len(completions) == len(actor_names)):
            raise ValueError(
                "prompts, completions, and actor_names must have equal length"
            )

        params = {
            "prompt": list(prompts),
            "completion": list(completions),
            "actor_name": list(actor_names),
            **entry_data,
        }
        out = self.func(**params)

        if not isinstance(out, Sequence) or isinstance(out, str | bytes):
            raise TypeError(
                f"Batched reward '{self.name}' must return a sequence of floats; got {type(out)!r}"
            )
        if len(out) != len(prompts):
            raise ValueError(
                f"Batched reward '{self.name}' returned {len(out)} items; expected {len(prompts)}."
            )
        return [float(x) for x in out]


def reward_function(
    name: str | None = None, weight: float = 1.0, *, batched: bool = False
) -> Callable[[Callable[..., Any]], RewardFunction]:
    def decorator(func: Callable[..., Any]) -> RewardFunction:
        reward_name = name if name is not None else func.__name__
        return RewardFunction(
            name=reward_name, weight=weight, func=func, batched=batched
        )

    return decorator
