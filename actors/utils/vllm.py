import re
from collections import OrderedDict

import torch


def fp8_quantize_state_dict(sd):
    from vllm import _custom_ops as ops

    out = OrderedDict()
    for k, v in sd.items():
        if (
            v.ndim == 2
            and "embed" not in k
            and "embedding" not in k
            and "lm_head" not in k
            and "bias" not in k
            and "norm" not in k
        ):
            q, s = ops.scaled_fp8_quant(v.cuda(), scale=None)
            out[k] = q.T
            out[k.replace(".weight", ".weight_scale")] = s
        else:
            out[k] = v
    return out


_QKV_PAT = re.compile(r"\.self_attn\.(q|k|v)_proj\.(weight|bias)$")


def merge_qkv(state_dict):
    out_sd, cache = OrderedDict(), {}
    for k, v in state_dict.items():
        m = _QKV_PAT.search(k)
        if m is None:
            out_sd[k] = v
            continue
        prefix, typ, what = k[: m.start()], m.group(1), m.group(2)
        bucket = cache.setdefault((prefix, what), {})
        bucket[typ] = v
        if len(bucket) == 3:
            out_sd[f"{prefix}.self_attn.qkv_proj.{what}"] = torch.cat(
                [bucket["q"], bucket["k"], bucket["v"]], 0
            )
            del cache[(prefix, what)]
    return out_sd


_GU = re.compile(r"\.mlp\.(gate|up)_proj\.(weight|bias)$")


def merge_gate_and_up_proj(sd):
    out, buf = OrderedDict(), {}
    for k, v in sd.items():
        m = _GU.search(k)
        if m is None:
            out[k] = v
            continue
        pre, part, typ = k[: m.start()], m.group(1), m.group(2)
        b = buf.setdefault((pre, typ), {})
        b[part] = v
        if len(b) == 2:
            fused = torch.cat([b["gate"], b["up"]], 0)
            out[f"{pre}.mlp.gate_up_proj.{typ}"] = fused
            del buf[(pre, typ)]
    assert not buf
    return out


def to_vllm_state_dict(state_dict):
    state_dict = merge_qkv(state_dict)
    state_dict = merge_gate_and_up_proj(state_dict)
    return state_dict


def to_vllm_lora_state_dict(state_dict):
    """
    Convert LoRA state dict to vLLM format by grouping q/k/v and gate/up projections into lists.

    For LoRA parameters, vLLM expects:
    - qkv_proj.lora_A/lora_B as lists: [q_tensor, k_tensor, v_tensor]
    - gate_up_proj.lora_A/lora_B as lists: [gate_tensor, up_tensor]

    Args:
        state_dict: Dictionary of LoRA parameter names to tensors

    Returns:
        OrderedDict with vLLM-compatible LoRA parameter groupings
    """
    out_sd = OrderedDict()
    qkv_cache = {}  # (prefix, lora_type) -> {'q': tensor, 'k': tensor, 'v': tensor}
    gate_up_cache = {}  # (prefix, lora_type) -> {'gate': tensor, 'up': tensor}

    # Patterns for LoRA parameters - updated to handle adapter names like '.default.'
    qkv_lora_pat = re.compile(
        r"(.+)\.self_attn\.(q|k|v)_proj\.(lora_[AB])\.([^.]+)\.weight$"
    )
    gate_up_lora_pat = re.compile(
        r"(.+)\.mlp\.(gate|up)_proj\.(lora_[AB])\.([^.]+)\.weight$"
    )

    for k, v in state_dict.items():
        # Check for q/k/v LoRA parameters
        qkv_match = qkv_lora_pat.search(k)
        if qkv_match:
            prefix, proj_type, lora_type, adapter_name = qkv_match.groups()
            cache_key = (prefix, lora_type, adapter_name)
            bucket = qkv_cache.setdefault(cache_key, {})
            bucket[proj_type] = v

            # If we have all three (q, k, v), create the list
            if len(bucket) == 3:
                qkv_list = [bucket["q"], bucket["k"], bucket["v"]]
                out_sd[
                    f"{prefix}.self_attn.qkv_proj.{lora_type}.{adapter_name}.weight"
                ] = qkv_list
                del qkv_cache[cache_key]
            continue

        # Check for gate/up LoRA parameters
        gate_up_match = gate_up_lora_pat.search(k)
        if gate_up_match:
            prefix, proj_type, lora_type, adapter_name = gate_up_match.groups()
            cache_key = (prefix, lora_type, adapter_name)
            bucket = gate_up_cache.setdefault(cache_key, {})
            bucket[proj_type] = v

            # If we have both (gate, up), create the list
            if len(bucket) == 2:
                gate_up_list = [bucket["gate"], bucket["up"]]
                out_sd[
                    f"{prefix}.mlp.gate_up_proj.{lora_type}.{adapter_name}.weight"
                ] = gate_up_list
                del gate_up_cache[cache_key]
            continue

        # For all other parameters, keep as-is
        out_sd[k] = v

    # Ensure all cached items were processed
    assert not qkv_cache, f"Incomplete q/k/v LoRA groups: {qkv_cache.keys()}"
    assert not gate_up_cache, f"Incomplete gate/up LoRA groups: {gate_up_cache.keys()}"

    return out_sd
