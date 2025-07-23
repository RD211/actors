# ═══════════════════════════════════════════════════════════════════════
#
# Memory-efficient per-token log-probabilities without materialising
# the (B × L × V) logits tensor.
#
#   • Similar or better speed than the naive reference
#   • Peak VRAM ≈ 20 × smaller
#   • Numerically tighter than _selective_softmax
#   * Downside: Does not compute gradients. Altough if combined with liger_grpo_loss,
#     then it does not matter.
#
#   Based on (and heavily trimmed from) the Cross-Entropy kernel in
#   https://github.com/linkedin/Liger-Kernel
# ═══════════════════════════════════════════════════════════════════════

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _logp_kernel(
    X_ptr,
    X_stride,  # logits
    Y_ptr,
    Y_stride,  # targets
    out_ptr,
    out_stride,  # log-p output
    n_vocab: tl.constexpr,
    BLOCK: tl.constexpr = 128,
):
    tok = tl.program_id(0).to(tl.int64)
    tgt = tl.load(Y_ptr + tok * Y_stride, mask=True)

    row = X_ptr + tok * X_stride
    m = float("-inf")
    d = 0.0
    for i in range(0, n_vocab, BLOCK):
        off = tl.arange(0, BLOCK) + i
        x = tl.load(row + off, mask=off < n_vocab, other=float("-inf")).to(tl.float32)
        bmax = tl.max(x)
        m_new = tl.maximum(m, bmax)
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new))
        m = m_new

    tgt_val = tl.load(row + tgt).to(tl.float32)
    tl.store(out_ptr + tok * out_stride, (tgt_val - m) - tl.log(d))


def chunked_logp(
    hidden: torch.Tensor,  # (BT, H)
    lm_head: torch.nn.Linear,
    target: torch.Tensor,  # (BT,)
    max_fused: int,
    temperature: float = 1.0,
    block: int = 128,
) -> torch.Tensor:  # (BT,) fp32
    BT, H = hidden.shape
    V = lm_head.weight.shape[0]
    out = torch.empty(BT, dtype=torch.float32, device=hidden.device)

    inc = (V + H - 1) // H
    chunk = min(BT, (max_fused + inc - 1) // inc)

    W, b = lm_head.weight, getattr(lm_head, "bias", None)

    for start in range(0, BT, chunk):
        end = min(start + chunk, BT)
        logits = (hidden[start:end] @ W.T).float()
        if b is not None:
            logits += b
        logits = logits / temperature
        _logp_kernel[(end - start,)](
            logits,
            logits.stride(0),
            target[start:end],
            target.stride(0),
            out[start:end],
            out.stride(0),
            n_vocab=V,
            BLOCK=min(block, triton.next_power_of_2(V)),
        )

    return out
