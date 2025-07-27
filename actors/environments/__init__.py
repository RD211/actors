"""
Environment modules for the actors library.
"""

from .actors_schedule_dsl import sample_schedule
from .collaborative_env import (
    CollaborativeActorConfig,
    CollaborativeEnvironment,
)
from .env_base import Environment
from .masking import mask_turns_and_encode
from .single_turn_env import RewardFunction, SingleTurnEnvironment
from .types import (
    ActorOutput,
    ActorOutputDict,
    EnvironmentOutput,
    GroupedEnvironmentOutput,
    RewardComponents,
)

__all__ = [
    # Base classes
    "Environment",
    # Type definitions
    "EnvironmentOutput",
    "ActorOutput",
    "RewardComponents",
    "ActorOutputDict",
    "GroupedEnvironmentOutput",
    # Single turn environment
    "SingleTurnEnvironment",
    "RewardFunction",
    # Collaborative environment
    "CollaborativeEnvironment",
    "CollaborativeActorConfig",
    # DSL for actor schedules
    "sample_schedule",
    # Masking utility
    "mask_turns_and_encode",
]
