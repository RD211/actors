from __future__ import annotations

import abc
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import torch
from torch import nn
from transformers import PreTrainedTokenizer

if TYPE_CHECKING:
    from actors.trainers.base_config import ActorTrainCfg, ActorTrainState


class LLMActor(abc.ABC):
    def __init__(self, name: str, model_path: str | None = None):
        self.name = name
        self.model_path = model_path

    @abc.abstractmethod
    def generate(self, prompts: Sequence[str], **kwargs): ...
    @abc.abstractmethod
    def chat(self, dialogs: Sequence[list], **kwargs): ...
    @abc.abstractmethod
    async def agenerate(self, prompts: Sequence[str], **kwargs): ...
    @abc.abstractmethod
    async def achat(self, dialogs: Sequence[list], **kwargs): ...


class TrainableLLMActor(LLMActor):
    @abc.abstractmethod
    def sleep(self, level: int = 1) -> None: ...
    @abc.abstractmethod
    def wake(self) -> None: ...

    @abc.abstractmethod
    def start_weight_update(self): ...
    @abc.abstractmethod
    def update_weights_batch(self, state_dict: dict[str, torch.Tensor]): ...
    @abc.abstractmethod
    def finalize_weight_update(self): ...

    # ═══════════════════════════════════════════════════════════════
    # LoRA/PEFT Support Methods
    # ═══════════════════════════════════════════════════════════════
    @abc.abstractmethod
    def update_lora_weights(self): ...
    @abc.abstractmethod
    def create_lora_if_not_present(self, lora_path: str): ...

    def __init__(
        self,
        name: str,
        model_path: str,
        training_config: ActorTrainCfg | None = None,
    ):
        """
        Initialize a trainable LLM actor with configuration options.

        Args:
            name: Actor name
            model_path: Path to the model
            training_config: ActorTrainCfg instance for training configuration
        """
        super().__init__(name, model_path)

        # Import here to avoid circular imports
        from actors.trainers.base_config import ActorTrainCfg

        # Use provided config or create a default one
        if training_config is not None:
            self.training_config = training_config
        else:
            self.training_config = ActorTrainCfg()

        # Create default factories if we have a model path
        self.training_config.create_default_factories(model_path)

        # Initialize train state as None - will be set during training setup
        self.train_state: ActorTrainState | None = None

    # ═══════════════════════════════════════════════════════════════
    # Convenience Properties - Delegates to ActorTrainCfg
    # ═══════════════════════════════════════════════════════════════

    @property
    def has_peft_config(self) -> bool:
        """Check if PEFT configuration is set."""
        return self.training_config.has_peft_config

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        """Get the tokenizer instance."""
        return self.training_config.tokenizer_factory()

    @property
    def model_factory(self) -> Callable[[], nn.Module]:
        """Get the model factory function."""
        return self.training_config.model_factory

    @property
    def current_learning_rate(self) -> float:
        """Get the current learning rate."""
        return self.training_config.learning_rate

    def get_training_summary(self) -> dict:
        """Get a summary of current training configuration."""
        return self.training_config.get_training_summary()

    # ═══════════════════════════════════════════════════════════════
    # Training State Access
    # ═══════════════════════════════════════════════════════════════

    @property
    def is_training_initialized(self) -> bool:
        """Check if training state has been initialized."""
        return self.train_state is not None
