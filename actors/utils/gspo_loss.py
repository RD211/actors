# ═══════════════════════════════════════════════════════════════
# Forked from Liger-Kernel
# Only added a couple of lines for GSPO.
# Will remove when this becomes part of Liger-Kernel.
# ═══════════════════════════════════════════════════════════════


import torch
from liger_kernel.chunked_loss.fused_linear_ppo import LigerFusedLinearPPOBase


def k3_loss_fn(log_p, log_q):
    return torch.exp(log_p - log_q) - (log_p - log_q) - 1.0


def clip_coef_fn(coef, epsilon_low, epsilon_high):
    return torch.clamp(coef, 1 - epsilon_low, 1 + epsilon_high)


class LigerFusedLinearGSPOFunction(LigerFusedLinearPPOBase):
    @staticmethod
    def ppo_loss_fn(
        log_probs,
        selected_token_ids,
        attention_mask,
        advantages,
        full_attention_mask,
        ref_per_token_logps=None,
        old_per_token_logps=None,
        ref_log_probs=None,
        epsilon_low=0.2,
        epsilon_high=0.2,
        beta=0.04,
        loss_type="bnpo",
        max_completion_length=None,
        **kwargs,
    ):
        """GRPO Loss Function matching GRPOTrainer implementation."""
        per_token_logps = log_probs.gather(
            dim=-1, index=selected_token_ids.unsqueeze(-1)
        ).squeeze(-1)

        if ref_per_token_logps is None:
            if ref_log_probs is not None:
                with torch.no_grad():
                    ref_per_token_logps = ref_log_probs.gather(
                        dim=-1, index=selected_token_ids.unsqueeze(-1)
                    ).squeeze(-1)
            else:
                ref_per_token_logps = per_token_logps.detach()

        old_per_token_logps = (
            old_per_token_logps
            if old_per_token_logps is not None
            else per_token_logps.detach()
        )

        # --
        log_ratio = per_token_logps - old_per_token_logps
        coef_1 = torch.exp(
            (
                (log_ratio * attention_mask).sum(-1)
                / attention_mask.sum(-1).clamp(min=1.0)
            ).unsqueeze(-1)
        )
        # --

        coef_2 = clip_coef_fn(coef_1, epsilon_low, epsilon_high)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if beta != 0.0:
            kl_div = k3_loss_fn(ref_per_token_logps, per_token_logps)
            per_token_loss = per_token_loss + beta * kl_div

        if loss_type == "grpo":
            loss = (
                (per_token_loss * attention_mask).sum(-1)
                / torch.clamp(attention_mask.sum(-1), min=1.0)
            ).sum() / full_attention_mask.shape[0]
        elif loss_type == "bnpo":
            loss = (per_token_loss * attention_mask).sum() / torch.clamp(
                full_attention_mask.sum(), min=1.0
            )
        elif loss_type == "dr_grpo":
            if max_completion_length is None:
                raise ValueError(
                    "max_completion_length must be provided for loss_type 'dr_grpo'"
                )
            loss = (per_token_loss * attention_mask).sum() / (
                full_attention_mask.shape[0] * max_completion_length
            )
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        metrics = []
        if beta != 0.0:
            if kl_div.shape[1] != 1:
                metrics.append(
                    (kl_div * attention_mask).sum()
                    / torch.clamp(full_attention_mask.sum(), min=1.0)
                )
            else:
                metrics.append(kl_div.mean())
        is_clipped = ((coef_1 < 1 - epsilon_low) & (advantages.unsqueeze(1) < 0)) | (
            (coef_1 > 1 + epsilon_high) & (advantages.unsqueeze(1) > 0)
        )
        metrics.append(is_clipped.float().mean())
        return loss, metrics

    @classmethod
    def forward(
        cls,
        ctx,
        _input,
        weight,
        selected_token_ids,
        attention_mask,
        advantages,
        bias=None,
        ref_per_token_logps=None,
        old_per_token_logps=None,
        ref_input=None,
        ref_weight=None,
        ref_bias=None,
        beta=0.04,
        epsilon_low=0.2,
        epsilon_high=0.2,
        loss_type="bnpo",
        max_completion_length=None,
        temperature=1.0,
        compiled=True,
        use_ref_model=True,
        chunk_size=1,
    ):
        return super().forward(
            cls=cls,
            ctx=ctx,
            _input=_input,
            weight=weight,
            selected_token_ids=selected_token_ids,
            attention_mask=attention_mask,
            advantages=advantages,
            bias=bias,
            ref_per_token_logps=ref_per_token_logps,
            old_per_token_logps=old_per_token_logps,
            ref_input=ref_input,
            ref_weight=ref_weight,
            ref_bias=ref_bias,
            beta=beta,
            epsilon_low=epsilon_low,
            epsilon_high=epsilon_high,
            loss_type=loss_type,
            max_completion_length=max_completion_length,
            temperature=temperature,
            compiled=compiled,
            use_ref_model=use_ref_model,
            chunk_size=chunk_size,
        )

    @staticmethod
    def backward(ctx, grad_output, *grad_metrics):
        grads = LigerFusedLinearPPOBase.backward(ctx, grad_output)
        return (
            *grads[:6],
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class LigerFusedLinearGSPOLoss(torch.nn.Module):
    def __init__(
        self,
        beta: float = 0.04,
        compiled: bool = True,
        use_ref_model: bool = True,
        chunk_size: int = 1,
        epsilon_low: float = 0.2,
        epsilon_high: float = 0.2,
        loss_type: str = "bnpo",
        max_completion_length: int | None = None,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.beta = beta
        self.compiled = compiled
        self.use_ref_model = use_ref_model
        self.chunk_size = chunk_size
        self.epsilon_low = epsilon_low
        self.epsilon_high = epsilon_high
        self.loss_type = loss_type
        self.max_completion_length = max_completion_length
        self.temperature = temperature

    def forward(
        self,
        _input,
        lin_weight,
        selected_token_ids,
        attention_mask,
        advantages,
        bias=None,
        ref_per_token_logps=None,
        old_per_token_logps=None,
        ref_input=None,
        ref_weight=None,
        ref_bias=None,
    ):
        return LigerFusedLinearGSPOFunction.apply(
            _input,
            lin_weight,
            selected_token_ids,
            attention_mask,
            advantages,
            bias,
            ref_per_token_logps,
            old_per_token_logps,
            ref_input,
            ref_weight,
            ref_bias,
            self.beta,
            self.epsilon_low,
            self.epsilon_high,
            self.loss_type,
            self.max_completion_length,
            self.temperature,
            self.compiled,
            self.use_ref_model,
            self.chunk_size,
        )
