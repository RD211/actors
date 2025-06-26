from __future__ import annotations

from typing import Dict, Literal, Optional, Tuple

import deepspeed
import torch
from torch import Tensor, nn
from liger_kernel.chunked_loss import LigerFusedLinearGRPOLoss

from .base_loss import BaseRLLoss


AllowedLoss = Literal["grpo", "bnpo", "dr_grpo"]


class LigerLoss(BaseRLLoss):
    """
    Liger token-level GRPO / BNPO / DR-GRPO loss.

    Parameters
    ----------
    beta : float
    loss_type : {"grpo", "bnpo", "dr_grpo"}
    use_ref_model : bool
    temperature : float
    """

    def __init__(
        self,
        *,
        beta: float,
        temperature: float = 1.0,
        loss_type: AllowedLoss = "bnpo",
    ) -> None:
        super().__init__()

        if loss_type not in ("grpo", "bnpo", "dr_grpo"):
            raise ValueError(f"invalid loss_type '{loss_type}'")

        self.core: LigerFusedLinearGRPOLoss = LigerFusedLinearGRPOLoss(
            beta=beta,
            use_ref_model=beta > 0.0,
            loss_type=loss_type,
            temperature=temperature,
        )
        self.beta: float = beta
        self.loss_type: AllowedLoss = loss_type

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        policy: nn.Module,
        input_ids: Tensor, # (B, L)
        attention_mask: Tensor, # (B, L)
        loss_attention_mask: Tensor, # (B, L-1)
        advantages: Tensor, # (B,)
        ref_logps: Optional[Tensor] = None, # (B, L-1)
        old_logps: Optional[Tensor] = None, # (B, L-1)
        **_: Dict,
    ) -> Tuple[Tensor, Dict[str, float]]:
        #TODO: Fix for zero3
        hidden: Tensor = policy.model(
            input_ids
        ).last_hidden_state[:, :-1, :]

        tgt_ids: Tensor = input_ids[:, 1:]
        mask: Tensor = attention_mask[:, 1:]
        gp = deepspeed.zero.GatheredParameters(
            [policy.lm_head.weight, policy.lm_head.bias],
            modifier_rank=None,
        )
        gp.__enter__()
        loss, metrics = self.core(
            _input=hidden,
            lin_weight=policy.lm_head.weight,
            bias=policy.lm_head.bias,
            selected_token_ids=tgt_ids,
            attention_mask=mask * loss_attention_mask,
            advantages=advantages,
            ref_per_token_logps=ref_logps,
            old_per_token_logps=old_logps,
        )
        gp.__exit__(None, None, None)

        if self.beta > 0.0:
            kl = metrics[0]
        else:
            kl = torch.tensor(0.0, device=loss.device, dtype=loss.dtype)

        return loss, {"kl": kl.item()}
