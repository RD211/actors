"""
Base reward function definitions for the actors library.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class RewardFunction:
    """
    Configuration for a reward function in the single-turn environment.

    This class encapsulates a reward function along with its metadata including
    weight and name for use in weighted reward aggregation.

    Attributes:
        name: Human-readable name for this reward component
        weight: Weight to apply when aggregating rewards (default: 1.0)
        func: The actual reward function to call
    """

    name: str
    weight: float
    func: Callable[..., float]

    def __post_init__(self):
        """Validate the reward function signature."""
        if not callable(self.func):
            raise ValueError(f"Reward function '{self.name}' must be callable")

    def compute_reward(
        self, prompt: str, completion: str, actor_name: str, **entry_data: Any
    ) -> float:
        """
        Compute the reward for a given text, completion, and entry data.

        Args:
            text: The input text/prompt
            completion: The generated completion
            actor_name: Name of the actor that generated the completion
            **entry_data: Additional data from the batch entry

        Returns:
            The computed reward value
        """
        # Get the function signature to determine which parameters it accepts
        sig = inspect.signature(self.func)
        func_params = {}

        # Always provide text, completion, and actor_name if the function accepts them
        if "prompt" in sig.parameters:
            func_params["prompt"] = prompt
        if "completion" in sig.parameters:
            func_params["completion"] = completion
        if "actor_name" in sig.parameters:
            func_params["actor_name"] = actor_name

        # Add any additional parameters the function accepts from entry_data
        for param_name in sig.parameters:
            if param_name in entry_data and param_name not in func_params:
                func_params[param_name] = entry_data[param_name]

        try:
            return float(self.func(**func_params))
        except Exception as e:
            raise RuntimeError(f"Error computing reward '{self.name}': {str(e)}") from e


def reward_function(
    name: str | None = None, weight: float = 1.0
) -> Callable[[Callable[..., float]], RewardFunction]:
    """
    Decorator to create a RewardFunction from a simple function.

    This provides a convenient way to create reward functions without manually
    instantiating RewardFunction objects.

    Args:
        name: Name for the reward function (defaults to function name)
        weight: Weight for the reward function (default: 1.0)

    Returns:
        Decorator that converts a function to a RewardFunction

    Example:
        @reward_function(name="length_penalty", weight=0.5)
        def length_penalty(text: str, completion: str) -> float:
            return -len(completion) / 1000
    """

    def decorator(func: Callable[..., float]) -> RewardFunction:
        reward_name = name if name is not None else func.__name__
        return RewardFunction(name=reward_name, weight=weight, func=func)

    return decorator


class BaseRewardFunction(ABC):
    """
    Abstract base class for reward functions that need more complex state.

    For simple stateless reward functions, use the RewardFunction dataclass
    or the @reward_function decorator instead.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of this reward function."""
        pass

    @property
    def weight(self) -> float:
        """The weight of this reward function (default: 1.0)."""
        return 1.0

    @abstractmethod
    def compute_reward(
        self, prompt: str, completion: str, actor_name: str, **entry_data: Any
    ) -> float:
        """
        Compute the reward for a given text, completion, and entry data.

        Args:
            text: The input text/prompt
            completion: The generated completion
            actor_name: Name of the actor that generated the completion
            **entry_data: Additional data from the batch entry

        Returns:
            The computed reward value
        """
        pass

    def to_reward_function(self) -> RewardFunction:
        """Convert this BaseRewardFunction to a RewardFunction."""
        return RewardFunction(
            name=self.name, weight=self.weight, func=self.compute_reward
        )
