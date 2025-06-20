from __future__ import annotations
import abc, torch
from typing import Dict, List, Optional, Sequence

class BaseRLLoss(abc.ABC):
    """Every loss must return (scalar_loss, metrics:dict[str,float])."""

    @abc.abstractmethod
    def forward(
        self,
        policy,                    # nn.Module (on device, requires_grad)
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        advantages: torch.Tensor,  # shape (B,) or (B,L-1)
        ref_logps: Optional[torch.Tensor] = None,  # shape (B,L-1)
        old_logps: Optional[torch.Tensor] = None,  # shape (B,L-1)
        **kw,
    ) -> tuple[torch.Tensor, Dict[str, float]]: ...

    def __call__(
        self,
        policy,                    # nn.Module (on device, requires_grad)
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        advantages: torch.Tensor,  # shape (B,) or (B,L-1)
        ref_logps: Optional[torch.Tensor] = None,  # shape (B,L-1)
        old_logps: Optional[torch.Tensor] = None,  # shape (B,L-1)
        **kw,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        return self.forward(policy, input_ids, attention_mask, advantages, ref_logps, old_logps, **kw)