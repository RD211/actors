"""
Type definitions for environment outputs with support for multiple reward types.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
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
    input_ids: List[List[int]]
    attention_mask: List[List[int]]
    rewards: List[float]
    reward_components: Optional[Dict[str, List[float]]] = None
    ended_in_eos: List[bool] = None
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate that all lists have consistent lengths."""
        lengths = [len(self.input_ids), len(self.attention_mask), len(self.rewards)]
        if self.reward_components:
            for name, values in self.reward_components.items():
                lengths.append(len(values))
        
        if not all(length == lengths[0] for length in lengths):
            raise ValueError(
                f"Inconsistent lengths in ActorOutput: "
                f"input_ids={len(self.input_ids)}, attention_mask={len(self.attention_mask)}, "
                f"rewards={len(self.rewards)}"
                + (f", reward_components={[(name, len(values)) for name, values in self.reward_components.items()]}" 
                   if self.reward_components else "")
            )
        # verify that if ended_in_eos is provided, it matches the length of input_ids
        if self.ended_in_eos is not None and len(self.ended_in_eos) != len(self.input_ids):
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
    
    def get_total_reward(self, weights: Optional[Dict[str, float]] = None) -> List[float]:
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
    
    def get_reward_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for all reward types.
        
        Returns:
            Dictionary mapping reward names to their statistics (mean, std, min, max)
        """
        def compute_stats(values: List[float]) -> Dict[str, float]:
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
    actors: Dict[str, ActorOutput]
    global_metadata: Optional[Dict[str, Any]] = field(default_factory=dict)

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
            else:
                if (len(actor_output.input_ids) != lengths["input_ids"] or
                    len(actor_output.attention_mask) != lengths["attention_mask"] or
                    len(actor_output.rewards) != lengths["rewards"]):
                    raise ValueError(
                        f"Inconsistent lengths in actor '{actor_name}': "
                        f"input_ids={len(actor_output.input_ids)}, "
                        f"attention_mask={len(actor_output.attention_mask)}, "
                        f"rewards={len(actor_output.rewards)}"
                    )
    
    def to_dict(self) -> Dict[str, Dict[str, Any]]:
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
    def from_dict(cls, data: Dict[str, Dict[str, Any]]) -> EnvironmentOutput:
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
                metadata=metadata
            )
        
        return cls(actors=actors)


# Type aliases for convenience
RewardComponents = Dict[str, List[float]]
ActorOutputDict = Dict[str, ActorOutput]
