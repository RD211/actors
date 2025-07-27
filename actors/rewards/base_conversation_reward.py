from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

Message = dict[str, Any]


@dataclass
class ConversationRewardFunction:
    name: str
    weight: float
    func: Callable[..., float | list[float]]
    batched: bool = False

    def __post_init__(self) -> None:
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
        self, conversation: list[Message], actor_name: str, **entry_data: Any
    ) -> float:
        if self.batched:
            params = {
                "conversation": [conversation],
                "actor_name": [actor_name],
                **{
                    k: ([v] if not isinstance(v, list | tuple) else v)
                    for k, v in entry_data.items()
                },
            }
            out = self.func(**params)
            if not isinstance(out, Sequence) or isinstance(out, str | bytes):
                raise TypeError(
                    f"Batched conversation reward '{self.name}' must return a sequence of floats."
                )
            if len(out) != 1:
                raise ValueError(
                    f"Batched conversation reward '{self.name}' returned {len(out)} items; expected 1."
                )
            return float(out[0])

        params = {
            "conversation": conversation,
            "actor_name": actor_name,
            **dict(entry_data.items()),
        }
        return float(self.func(**params))

    def compute_rewards(
        self,
        conversations: Sequence[list[Message]],
        actor_names: Sequence[str],
        **entry_data: Any,
    ) -> list[float]:
        if not self.batched:
            if len(conversations) != len(actor_names):
                raise ValueError("conversations and actor_names must have equal length")
            return [
                self.compute_reward(
                    conv,
                    name,
                    **{
                        k: (v[i] if isinstance(v, list | tuple) else v)
                        for k, v in entry_data.items()
                    },
                )
                for i, (conv, name) in enumerate(
                    zip(conversations, actor_names, strict=True)
                )
            ]

        if len(conversations) != len(actor_names):
            raise ValueError("conversations and actor_names must have equal length")

        params = {
            "conversation": list(conversations),
            "actor_name": list(actor_names),
            **entry_data,
        }
        out = self.func(**params)

        if not isinstance(out, Sequence) or isinstance(out, str | bytes):
            raise TypeError(
                f"Batched conversation reward '{self.name}' must return a sequence of floats."
            )
        if len(out) != len(conversations):
            raise ValueError(
                f"Batched conversation reward '{self.name}' returned {len(out)} items; "
                f"expected {len(conversations)}."
            )
        return [float(x) for x in out]


def conversation_reward_function(
    name: str | None = None, weight: float = 1.0, *, batched: bool = False
) -> Callable[[Callable[..., Any]], ConversationRewardFunction]:
    def decorator(func: Callable[..., Any]) -> ConversationRewardFunction:
        reward_name = name if name is not None else func.__name__
        return ConversationRewardFunction(
            name=reward_name, weight=weight, func=func, batched=batched
        )

    return decorator
