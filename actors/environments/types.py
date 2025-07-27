"""
Type definitions for environment outputs with support for multiple reward types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class ActorOutput:
    """
    Type-safe output for a single actor from an environment step.

    Attributes:
        input_ids: List of token sequences for generated text
        attention_mask: Attention masks corresponding to input_ids
        rewards: Primary reward values (for backward compatibility)
        reward_components: Optional dictionary of named reward components
        ended_in_eos: Optional list indicating if each sequence ended with an EOS token. If not provided, it is assumed all sequences ended in EOS.
        metadata: Optional metadata about the generation
    """

    input_ids: list[list[int]]
    rewards: list[float]
    attention_mask: list[list[int]] | None = None
    reward_components: dict[str, list[float]] | None = None
    ended_in_eos: list[bool] = None
    metadata: dict[str, Any] | None = field(default_factory=dict)

    def __post_init__(self):
        """Validate that all lists have consistent lengths."""
        if not self.attention_mask:
            self.attention_mask = [[1] * len(seq) for seq in self.input_ids]

        lengths = [len(self.input_ids), len(self.attention_mask), len(self.rewards)]
        if self.reward_components:
            for _, values in self.reward_components.items():
                lengths.append(len(values))

        if not all(length == lengths[0] for length in lengths):
            raise ValueError(
                f"Inconsistent lengths in ActorOutput: "
                f"input_ids={len(self.input_ids)}, attention_mask={len(self.attention_mask)}, "
                f"rewards={len(self.rewards)}"
                + (
                    f", reward_components={[(name, len(values)) for name, values in self.reward_components.items()]}"
                    if self.reward_components
                    else ""
                )
            )
        # verify that if ended_in_eos is provided, it matches the length of input_ids
        if self.ended_in_eos is not None and len(self.ended_in_eos) != len(
            self.input_ids
        ):
            raise ValueError(
                f"ended_in_eos length {len(self.ended_in_eos)} does not match input_ids length {len(self.input_ids)}"
            )
        if self.ended_in_eos is None:
            self.ended_in_eos = [True] * len(self.input_ids)

        # We must also make sure that there is no empty sequence in input_ids or attention_mask
        if any(len(seq) == 0 for seq in self.input_ids):
            raise ValueError("input_ids contains an empty sequence")
        if any(len(seq) == 0 for seq in self.attention_mask):
            raise ValueError("attention_mask contains an empty sequence")

    def get_total_reward(self, weights: dict[str, float] | None = None) -> list[float]:
        """
        Compute total reward as weighted sum of components.

        Args:
            weights: Dictionary mapping reward component names to weights.
                    If None, uses only the primary rewards.

        Returns:
            List of total reward values
        """
        if weights is None or self.reward_components is None:
            return self.rewards.copy()

        total_rewards = []
        for i in range(len(self.rewards)):
            total = self.rewards[i]
            for component_name, weight in weights.items():
                if component_name in self.reward_components:
                    total += weight * self.reward_components[component_name][i]
            total_rewards.append(total)

        return total_rewards

    def get_reward_stats(self) -> dict[str, dict[str, float]]:
        """
        Get statistics for all reward types.

        Returns:
            Dictionary mapping reward names to their statistics (mean, std, min, max)
        """

        def compute_stats(values: list[float]) -> dict[str, float]:
            if not values:
                return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

            tensor_vals = torch.tensor(values, dtype=torch.float32)
            return {
                "mean": tensor_vals.mean().item(),
                "std": tensor_vals.std(unbiased=False).item(),
                "min": tensor_vals.min().item(),
                "max": tensor_vals.max().item(),
            }

        stats = {"primary": compute_stats(self.rewards)}

        if self.reward_components:
            for name, values in self.reward_components.items():
                stats[name] = compute_stats(values)

        return stats


@dataclass
class EnvironmentOutput:
    """
    Type-safe output from an environment step containing outputs for all actors.

    Attributes:
        actors: Dictionary mapping actor names to their outputs
        global_metadata: Optional metadata about the environment step
    """

    actors: dict[str, ActorOutput]
    global_metadata: dict[str, Any] | None = field(default_factory=dict)

    def __post_init__(self):
        """Validate that all actor outputs have consistent structure."""

        # We need to check that all actors have the same lengths for input_ids, attention_mask, and rewards
        if not self.actors:
            raise ValueError("EnvironmentOutput must contain at least one actor output")
        lengths = None
        for actor_name, actor_output in self.actors.items():
            if lengths is None:
                lengths = {
                    "input_ids": len(actor_output.input_ids),
                    "attention_mask": len(actor_output.attention_mask),
                    "rewards": len(actor_output.rewards),
                }
            elif (
                len(actor_output.input_ids) != lengths["input_ids"]
                or len(actor_output.attention_mask) != lengths["attention_mask"]
                or len(actor_output.rewards) != lengths["rewards"]
            ):
                raise ValueError(
                    f"Inconsistent lengths in actor '{actor_name}': "
                    f"input_ids={len(actor_output.input_ids)}, "
                    f"attention_mask={len(actor_output.attention_mask)}, "
                    f"rewards={len(actor_output.rewards)}"
                )

    def to_dict(self) -> dict[str, dict[str, Any]]:
        """
        Convert to dictionary format for backward compatibility.

        Returns:
            Dictionary in the format expected by the current trainer
        """
        result = {}
        for actor_name, actor_output in self.actors.items():
            result[actor_name] = {
                "input_ids": actor_output.input_ids,
                "attention_mask": actor_output.attention_mask,
                "rewards": actor_output.rewards,
            }
            if actor_output.reward_components:
                result[actor_name]["reward_components"] = actor_output.reward_components
            if actor_output.metadata:
                result[actor_name]["metadata"] = actor_output.metadata

        return result

    @classmethod
    def from_dict(cls, data: dict[str, dict[str, Any]]) -> EnvironmentOutput:
        """
        Create EnvironmentOutput from dictionary format.

        Args:
            data: Dictionary in the format returned by current environments

        Returns:
            EnvironmentOutput instance
        """
        actors = {}
        for actor_name, actor_data in data.items():
            # Extract required fields
            input_ids = actor_data["input_ids"]
            attention_mask = actor_data["attention_mask"]
            rewards = actor_data["rewards"]

            # Extract optional fields
            reward_components = actor_data.get("reward_components")
            metadata = actor_data.get("metadata", {})

            actors[actor_name] = ActorOutput(
                input_ids=input_ids,
                attention_mask=attention_mask,
                rewards=rewards,
                reward_components=reward_components,
                metadata=metadata,
            )

        return cls(actors=actors)


@dataclass
class GroupedEnvironmentOutput:
    """
    Environment output organized by problems and groups.

    This organizes the outputs by problems (unique inputs) and groups (multiple generations per problem).

    Attributes:
        problems: List of unique problem inputs
        groups: Dictionary mapping actor names to their grouped outputs
                Format: {actor_name: [[group1_for_problem1], [group2_for_problem1], ...]}
        group_size: Number of generations per problem
    """

    problems: list[dict[str, Any]]
    groups: dict[str, list[list[ActorOutput]]]
    group_size: int

    @classmethod
    def from_environment_output(
        cls,
        env_output: EnvironmentOutput,
        original_batch: list[dict[str, Any]],
        group_size: int,
    ) -> GroupedEnvironmentOutput:
        """
        Create GroupedEnvironmentOutput from regular EnvironmentOutput.

        Args:
            env_output: Original environment output
            original_batch: The original problems before group expansion
            group_size: Number of generations per problem
        """
        problems = original_batch
        groups = {}

        for actor_name, actor_output in env_output.actors.items():
            actor_groups = []

            # Reshape the flat actor output into groups
            total_outputs = len(actor_output.input_ids)
            num_problems = total_outputs // group_size

            for problem_idx in range(num_problems):
                group_outputs = []
                for group_idx in range(group_size):
                    flat_idx = problem_idx * group_size + group_idx
                    if flat_idx < total_outputs:
                        group_output = ActorOutput(
                            input_ids=[actor_output.input_ids[flat_idx]],
                            attention_mask=[actor_output.attention_mask[flat_idx]],
                            rewards=[actor_output.rewards[flat_idx]],
                            reward_components=(
                                {
                                    comp_name: [comp_values[flat_idx]]
                                    for comp_name, comp_values in actor_output.reward_components.items()
                                }
                                if actor_output.reward_components
                                else None
                            ),
                            ended_in_eos=(
                                [actor_output.ended_in_eos[flat_idx]]
                                if actor_output.ended_in_eos
                                else None
                            ),
                            metadata=actor_output.metadata,
                        )
                        group_outputs.append(group_output)

                actor_groups.append(group_outputs)

            groups[actor_name] = actor_groups

        return cls(
            problems=problems,
            groups=groups,
            group_size=group_size,
        )

    def to_environment_output(self) -> EnvironmentOutput:
        """Convert back to regular EnvironmentOutput by flattening groups."""
        actors = {}

        for actor_name, actor_groups in self.groups.items():
            # Flatten the grouped outputs back to a single actor output
            all_input_ids = []
            all_attention_mask = []
            all_rewards = []
            all_reward_components = {}
            all_ended_in_eos = []

            for problem_groups in actor_groups:
                for group_output in problem_groups:
                    all_input_ids.extend(group_output.input_ids)
                    all_attention_mask.extend(group_output.attention_mask)
                    all_rewards.extend(group_output.rewards)
                    all_ended_in_eos.extend(
                        group_output.ended_in_eos
                        or [True] * len(group_output.input_ids)
                    )

                    if group_output.reward_components:
                        for (
                            comp_name,
                            comp_values,
                        ) in group_output.reward_components.items():
                            if comp_name not in all_reward_components:
                                all_reward_components[comp_name] = []
                            all_reward_components[comp_name].extend(comp_values)

            actors[actor_name] = ActorOutput(
                input_ids=all_input_ids,
                attention_mask=all_attention_mask,
                rewards=all_rewards,
                reward_components=(
                    all_reward_components if all_reward_components else None
                ),
                ended_in_eos=all_ended_in_eos,
                metadata={},
            )

        return EnvironmentOutput(actors=actors)

    def __add__(self, other: GroupedEnvironmentOutput) -> GroupedEnvironmentOutput:
        if self.group_size != other.group_size:
            raise ValueError(
                f"Cannot add GroupedEnvironmentOutput with different group sizes: "
                f"{self.group_size} != {other.group_size}"
            )

        if set(self.groups.keys()) != set(other.groups.keys()):
            raise ValueError(
                "Cannot add GroupedEnvironmentOutput with different actor sets: "
                f"{set(self.groups.keys())} != {set(other.groups.keys())}"
            )

        combined_problems = self.problems + other.problems
        combined_groups = {}

        for actor_name in self.groups:
            combined_groups[actor_name] = (
                self.groups[actor_name] + other.groups[actor_name]
            )

        return GroupedEnvironmentOutput(
            problems=combined_problems,
            groups=combined_groups,
            group_size=self.group_size,
        )


# Type aliases for convenience
RewardComponents = dict[str, list[float]]
ActorOutputDict = dict[str, ActorOutput]
