from __future__ import annotations
import abc
from typing import Dict, Sequence
import torch


class LLMActor(abc.ABC):
    def __init__(self, name: str):
        self.name = name

    @abc.abstractmethod
    def generate(self, prompts: Sequence[str], **kwargs): ...
    @abc.abstractmethod
    def chat(self, dialogs: Sequence[list], **kwargs): ...


class TrainableLLMActor(LLMActor):
    @abc.abstractmethod
    def start_weight_update(self):
        ...

    @abc.abstractmethod
    def update_weights_batch(self, state_dict: Dict[str, torch.Tensor]):
        ...

    @abc.abstractmethod
    def finalize_weight_update(self):
        ...
