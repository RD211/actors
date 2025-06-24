from __future__ import annotations
import asyncio
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers import PreTrainedTokenizer

import abc, dataclasses
from typing import Dict, Union

from actors import TrainableLLMActor
from actors.environments.types import EnvironmentOutput

from dataclasses import dataclass
from typing import Callable, Iterable


from actors.losses.base_loss import BaseRLLoss


class Environment(abc.ABC):

    def __init__(self) -> None:
        self._reg: Dict[str, TrainableLLMActor] = {}

    def register(self, actor: TrainableLLMActor) -> None:
        if actor.name in self._reg:
            raise ValueError(f"duplicate actor {actor.name}")
        self._reg[actor.name] = actor


    # ------------------------------------------------------------------
    def get_trainable_actors(self) -> Dict[str, TrainableLLMActor]:
        return self._reg
    # ------------------------------------------------------------------
    
    def __call__(self, batch) -> Union[Dict[str, dict], EnvironmentOutput]:
        """
        Synchronous wrapper that runs the async acall method.
        This allows the trainer to use sync interface while environments use async internally.
        """
        return asyncio.run(self.generate(batch))
    
    @abc.abstractmethod
    async def generate(self, batch) -> Union[Dict[str, dict], EnvironmentOutput]:
        """
        Async version that should use actor async methods for better performance.
        This is now the primary method that subclasses must implement.
        """
