__version__ = "0.1.2"

from .actors import LLMActor, OpenAIActor, TrainableLLMActor, vLLMActor
from .environments import (
    ActorOutput,
    CollaborativeActorConfig,
    CollaborativeEnvironment,
    CombinerActorConfig,
    Environment,
    EnvironmentOutput,
    ParallelActorConfig,
    ParallelEnvironment,
    RewardFunction,
    SingleTurnEnvironment,
    mask_turns_and_encode,
)
from .rewards import conversation_reward_function, reward_function
from .trainers import (
    ActorTrainCfg,
    BaseRLTrainer,
    GRPOTrainer,
    GRPOTrainerCfg,
    TrainerCfg,
)
from .trainers.base_config import EvalStrategy, SaveStrategy

__all__ = [
    # Package info
    "__version__",
    # Actors
    "LLMActor",
    "OpenAIActor",
    "TrainableLLMActor",
    "vLLMActor",
    # Rewards
    "conversation_reward_function",
    "reward_function",
    # Trainers
    "GRPOTrainer",
    "BaseRLTrainer",
    # Configurations
    "ActorTrainCfg",
    "GRPOTrainerCfg",
    "TrainerCfg",
    "EvalStrategy",
    "SaveStrategy",
    # Environments
    "Environment",
    "SingleTurnEnvironment",
    "CollaborativeEnvironment",
    "CollaborativeActorConfig",
    "RewardFunction",
    "ParallelEnvironment",
    "ParallelActorConfig",
    "CombinerActorConfig",
    # Types
    "ActorOutput",
    "EnvironmentOutput",
    # Utility functions
    "mask_turns_and_encode",
]
