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
from .parallel_environment import (
    CombinerActorConfig,
    ParallelActorConfig,
    ParallelEnvironment,
)
from .single_turn_env import RewardFunction, SingleTurnEnvironment
from .types import (
    ActorOutput,
    EnvironmentOutput,
)

__all__ = [
    # Base classes
    "Environment",
    # Type definitions
    "EnvironmentOutput",
    "ActorOutput",
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
    # Parallel environment
    "ParallelEnvironment",
    "ParallelActorConfig",
    "CombinerActorConfig",
]
