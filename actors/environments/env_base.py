from __future__ import annotations
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
    @abc.abstractmethod
    def __call__(self, batch) -> Union[Dict[str, dict], EnvironmentOutput]:
        """
        """
