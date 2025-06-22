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


@dataclass
class ActorSpec:
    actor_name: str
    model_factory: Callable[[], nn.Module]
    tokenizer: PreTrainedTokenizer
    loss_factory: Callable[[], BaseRLLoss]
    optim_factory: Callable[[Iterable[nn.Parameter]], Optimizer]
    scheduler_factory: Callable[[Optimizer], LRScheduler]
    reference_model_factory: Callable[[], nn.Module] | None = None

@dataclasses.dataclass
class _Entry:
    actor: TrainableLLMActor
    spec: ActorSpec


class Environment(abc.ABC):

    def __init__(self) -> None:
        self._reg: Dict[str, _Entry] = {}

    def register(self, actor: TrainableLLMActor, spec: ActorSpec) -> None:
        if actor.name != spec.actor_name:
            raise ValueError("actor.name and spec.actor_name must match")
        if actor.name in self._reg:
            raise ValueError(f"duplicate actor {actor.name}")
        self._reg[actor.name] = _Entry(actor=actor, spec=spec)

    # ------------------------------------------------------------------
    def get_actor_specs(self) -> Dict[str, tuple[TrainableLLMActor, ActorSpec]]:
        return {k: (v.actor, v.spec) for k, v in self._reg.items()}

    # ------------------------------------------------------------------
    @abc.abstractmethod
    def __call__(self, batch) -> Union[Dict[str, dict], EnvironmentOutput]:
        """
        Process a batch and return environment outputs.
        
        Can return either:
        1. Legacy format: Dict[str, dict] with structure:
           { actor_name : { "rewards": List[float],
                           "input_ids": List[List[int]], 
                           "attention_mask": List[List[int]],
                           "reward_components": Optional[Dict[str, List[float]]],
                           "metadata": Optional[Dict[str, Any]] } }
        
        2. New typed format: EnvironmentOutput instance
        
        The new format provides better type safety and support for multiple reward types.
        """
