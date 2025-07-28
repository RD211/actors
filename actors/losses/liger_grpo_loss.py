from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
from liger_kernel.chunked_loss import LigerFusedLinearGRPOLoss
from torch import Tensor, nn

from .base_loss import BaseRLLoss

if TYPE_CHECKING:
    from actors.trainers.base_config import ActorTrainCfg

AllowedLoss = Literal["grpo", "bnpo", "dr_grpo"]


class LigerGRPOLoss(BaseRLLoss):
    def __init__(
        self,
        config: ActorTrainCfg,
        loss_type: AllowedLoss = "grpo",
    ) -> None:
        super().__init__(config=config)

        if loss_type not in ("grpo", "bnpo", "dr_grpo"):
            raise ValueError(f"invalid loss_type '{loss_type}'")

        self.loss: LigerFusedLinearGRPOLoss = LigerFusedLinearGRPOLoss(
            beta=self.beta,
            use_ref_model=self.beta > 0.0,
            loss_type=loss_type,
            temperature=self.temperature,
        )
        self.loss_type: AllowedLoss = loss_type
        self.is_lora = config.peft_config is not None

    def forward(
        self,
        policy: nn.Module,
        input_ids: Tensor,  # (B, L)
        attention_mask: Tensor,  # (B, L)
        loss_attention_mask: Tensor,  # (B, L-1)
        advantages: Tensor,  # (B,)
        ref_logps: Tensor | None = None,  # (B, L-1)
        old_logps: Tensor | None = None,  # (B, L-1)
        **_: dict,
    ) -> tuple[Tensor, dict[str, float]]:
        model_to_infer = policy.base_model.model.model if self.is_lora else policy.model
        hidden: Tensor = model_to_infer(
            input_ids, attention_mask=attention_mask
        ).last_hidden_state[:, :-1, :]

        tgt_ids: Tensor = input_ids[:, 1:]
        mask: Tensor = attention_mask[:, 1:]

        loss, metrics = self.loss(
            _input=hidden,
            lin_weight=policy.lm_head.weight,
            bias=policy.lm_head.bias,
            selected_token_ids=tgt_ids,
            attention_mask=mask * loss_attention_mask,
            advantages=advantages,
            ref_per_token_logps=ref_logps,
            old_per_token_logps=old_logps,
        )

        if self.beta > 0.0:
            kl = metrics[0]
        else:
            kl = torch.tensor(0.0, device=loss.device, dtype=loss.dtype)

        return loss, {"kl": kl.item()}
