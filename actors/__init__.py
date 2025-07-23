__version__ = "0.1.0"

from .actors import LLMActor, OpenAIActor, TrainableLLMActor, vLLMActor
from .environments import (
    ActorOutput,
    ActorOutputDict,
    Environment,
    EnvironmentOutput,
    GroupedEnvironmentOutput,
    RewardComponents,
    RewardFunction,
    SingleTurnEnvironment,
)
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
    "RewardFunction",
    # Types
    "ActorOutput",
    "ActorOutputDict",
    "EnvironmentOutput",
    "GroupedEnvironmentOutput",
    "RewardComponents",
]
