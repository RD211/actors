#!/usr/bin/env python
import gc, time, torch, bitsandbytes as bnb
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin
from transformers import AutoModelForCausalLM, AutoTokenizer
from deepspeed.runtime.zero.offload_config import OffloadStateTypeEnum
from deepspeed.runtime.zero.offload_config import (
    OffloadDeviceEnum,
    OffloadStateTypeEnum,
)
def mb(x): return x / 1024**2
def cuda_mb(k="allocated"):
    f = torch.cuda.memory_allocated if k == "allocated" else torch.cuda.memory_reserved
    return mb(f())
def log(tag): 
    print(f"[{time.strftime('%X')}] {tag:<10} A:{cuda_mb():.0f} R:{cuda_mb('reserved'):.0f}")
    time.sleep(10)

# ---------- ZeRO-3 optimizer state move (works for bitsandbytes) -------------
def zero_tensors(zopt):
    for n in dir(zopt):
        if n.endswith("_groups_flat"):
            for t in getattr(zopt, n, []):
                if torch.is_tensor(t):
                    yield t
    inner = getattr(zopt, "optimizer", zopt)
    for st in inner.state.values():
        for v in st.values():
            if torch.is_tensor(v):
                yield v

def move_zero_tensors(zopt, device):
    moved = 0
    for t in zero_tensors(zopt):
        if t.device != device:
            moved += t.numel() * t.element_size()
            t.data = t.data.to(device, non_blocking=False)
    return moved

# ---------- DeepSpeed engine-level state offload / reload --------------------
def offload_engine_states(engine, include):
    engine.offload_states(include=include,
                          device="cpu",
                          pin_memory=True,
                          non_blocking=False)

def reload_engine_states(engine):
    engine.reload_states(non_blocking=False)

def main():
    plug = DeepSpeedPlugin()
    plug.deepspeed_config.update({
        "zero_optimization": {"stage": 3},
        "train_batch_size": 1,
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1,
        "zero_allow_untested_optimizer": True
    })
    acc = Accelerator(deepspeed_plugin=plug)
    dev = acc.device

    name = "Qwen/Qwen2.5-0.5B-Instruct"
    log("init")
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, trust_remote_code=True)
    tok   = AutoTokenizer.from_pretrained(name)
    if not tok.pad_token: tok.pad_token = tok.eos_token
    model.resize_token_embeddings(len(tok))
    log("model_load")

    opt = bnb.optim.AdamW32bit(model.parameters(), lr=1e-4)
    import deepspeed.ops.adam.fused_adam as _fused

    class _OptimizerProxy:
        def __init__(self, real_opt):
            object.__setattr__(self, "_real", real_opt)

        # lie when someone asks for __class__
        def __getattribute__(self, name):
            if name == "__class__":
                return _fused.FusedAdam
            return getattr(object.__getattribute__(self, "_real"), name)

        # delegate setattr too
        def __setattr__(self, name, value):
            setattr(self._real, name, value)

    opt = _OptimizerProxy(opt)



    model, opt = acc.prepare(model, opt)
    batch = tok("test", return_tensors="pt").to(dev)

    model.train(); log("start")
    loss = model(**batch, labels=batch["input_ids"]).loss
    acc.backward(loss); opt.step(); opt.zero_grad(); acc.wait_for_everyone()
    log("step1")

    # -------- Offload ZeRO-3 optimizer shards --------------------------------
    moved = move_zero_tensors(opt, torch.device("cpu"))
    for _ in range(10):
        gc.collect(); torch.cuda.empty_cache()
    log("optim_off"); print(f"moved {mb(moved):.1f} MB optim")

    # -------- Offload FP16 params + gradients + contiguous grad buf ----------
    model.optimizer.offload_states(
        include=[
            OffloadStateTypeEnum.optim_states,
            OffloadStateTypeEnum.contiguous_grad_buffer,
            OffloadStateTypeEnum.hp_params,
            OffloadStateTypeEnum.lp_params,
            OffloadStateTypeEnum.lp_grads,
        ],
        device=OffloadDeviceEnum.cpu,
        pin_memory=True,
        non_blocking=True,
    )

    for _ in range(10):
        gc.collect(); torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    log("eng_off")
    log("after offloading even the last one.")

    # -------- Reload everything back to GPU ----------------------------------
    reload_engine_states(model.optimizer)
    move_zero_tensors(opt, dev)
    gc.collect(); torch.cuda.empty_cache()
    log("reload")

    # Second training step to verify
    model.train()
    loss = model(**batch, labels=batch["input_ids"]).loss
    acc.backward(loss); opt.step(); opt.zero_grad(); acc.wait_for_everyone()
    log("step2")

if __name__ == "__main__":
    main()
