import re
from typing import OrderedDict
import torch
from collections import OrderedDict
from vllm import _custom_ops as ops


def fp8_quantize_state_dict(sd):
    out = OrderedDict()
    for k, v in sd.items():
        if (
            v.ndim == 2
            and "embed" not in k
            and "embedding" not in k
            and not "lm_head" in k
            and not "bias" in k
            and not "norm" in k
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
