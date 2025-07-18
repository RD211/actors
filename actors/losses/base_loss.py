from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from actors.trainers.base_config import ActorTrainCfg


class BaseRLLoss(abc.ABC):
    """Every loss must return (scalar_loss, metrics:dict[str,float])."""

    beta: float = 0.4
    temperature: float = 1.0

    def __init__(self, config: ActorTrainCfg):
        """Initialize loss with actor training configuration.

        Args:
            config: Full actor training configuration
        """
        self.beta = config.beta
        self.temperature = config.loss_temp

    @abc.abstractmethod
    def forward(
        self,
        policy,  # nn.Module (on device, requires_grad)
        input_ids: torch.LongTensor,  # (B, L)
        attention_mask: torch.LongTensor,  # (B, L)
        loss_attention_mask: torch.LongTensor,  # (B, L-1)
        advantages: torch.Tensor,  # (B,)
        ref_logps: torch.Tensor | None = None,  # (B,L-1)
        old_logps: torch.Tensor | None = None,  # (B,L-1)
        **kw,
    ) -> tuple[torch.Tensor, dict[str, float]]: ...

    def __call__(
        self,
        policy,  # nn.Module (on device, requires_grad)
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        loss_attention_mask: torch.LongTensor,
        advantages: torch.Tensor,  # shape (B,) or (B,L-1)
        ref_logps: torch.Tensor | None = None,  # shape (B,L-1)
        old_logps: torch.Tensor | None = None,  # shape (B,L-1)
        **kw,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        return self.forward(
            policy,
            input_ids,
            attention_mask,
            loss_attention_mask,
            advantages,
            ref_logps,
            old_logps,
            **kw,
        )
