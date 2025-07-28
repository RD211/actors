from typing import TYPE_CHECKING, Any, Literal

import torch

from actors.utils.softmax import _selective_softmax

from .base_loss import BaseRLLoss

if TYPE_CHECKING:
    from actors.trainers.base_config import ActorTrainCfg

AllowedLoss = Literal["grpo", "bnpo", "dr_grpo"]


class GRPOLoss(BaseRLLoss):
    def __init__(
        self,
        config: "ActorTrainCfg",
        eps_low: float = 0.2,
        eps_high: float = 0.2,
        loss_type: AllowedLoss = "grpo",
        delta: float | None = None,
        max_completion_length: int | None = None,
        gspo: bool = False,
    ):
        super().__init__(config=config)

        self.eps_l = eps_low
        self.eps_h = eps_high
        self.loss_type = loss_type
        self.delta = delta
        self.max_completion_length = max_completion_length
        self.gspo = gspo

    def forward(
        self,
        policy,
        input_ids: torch.Tensor,  # shape (B, L)
        attention_mask: torch.Tensor,  # shape (B, L)
        loss_attention_mask: torch.Tensor,  # shape (B, L-1)
        advantages: torch.Tensor,  # shape (B,)
        ref_logps: torch.Tensor | None = None,  # shape (B, L-1)
        old_logps: torch.Tensor | None = None,  # shape (B, L-1)
        **kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        logits = (
            policy(input_ids, attention_mask=attention_mask).logits / self.temperature
        )
        new_lp = _selective_softmax(logits[:, :-1, :], input_ids[:, 1:])
        if old_logps is None:
            old_logps = new_lp.detach()

        mask = attention_mask[:, 1:].to(new_lp.dtype) * loss_attention_mask

        # From https://github.com/huggingface/trl/blob/03034317d0be0c259c315f5ffad71be138c17d2c/trl/trainer/grpo_trainer.py#L1793
        ratio = new_lp - old_logps
        if not self.gspo:
            log_importance_weights = ratio
        else:
            log_importance_weights = (ratio * mask).sum(-1) / mask.sum(-1).clamp(
                min=1.0
            )
            log_importance_weights = log_importance_weights.unsqueeze(-1)

        ratio = torch.exp(log_importance_weights)

        if self.delta is not None:
            ratio = torch.clamp(ratio, max=self.delta)
        ratio_clipped = torch.clamp(ratio, 1 - self.eps_l, 1 + self.eps_h)

        if advantages.dim() == 1:
            adv = advantages[:, None].expand_as(new_lp)
        else:
            adv = advantages
        adv = adv.to(new_lp.dtype)

        per_tok = -torch.min(ratio * adv, ratio_clipped * adv)
        kl = None
        if self.beta != 0.0 and ref_logps is not None:
            kl = torch.exp(ref_logps - new_lp) - (ref_logps - new_lp) - 1
            per_tok = per_tok + self.beta * kl

        if self.loss_type == "grpo":
            loss = ((per_tok * mask).sum(-1) / mask.sum(-1).clamp(min=1)).mean()
        elif self.loss_type == "bnpo":
            loss = (per_tok * mask).sum() / mask.sum().clamp(min=1)
        elif self.loss_type == "dr_grpo":
            if self.max_completion_length is None:
                raise ValueError("max_completion_length required for dr_grpo")
            loss = (per_tok * mask).sum() / (
                input_ids.size(0) * self.max_completion_length
            )
        else:
            raise ValueError(f"unknown loss_type {self.loss_type}")

        metrics = {}
        if kl is not None:
            metrics["kl"] = (
                ((kl * mask).sum() / mask.sum()).item()
                if kl.shape[1] != 1
                else kl.mean().item()
            )

        return loss, metrics
