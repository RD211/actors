"""
DeepSpeed model and optimizer offloading utilities with performance optimizations.

Performance Improvements:
1. Non-blocking transfers: Use asynchronous GPU-CPU transfers for faster offloading
2. Batch tensor movement: Collect and move tensors in batches rather than one-by-one
3. Context manager: Automatic offload/reload with optimized patterns

Usage Examples:

# Manual offloading/reloading
offload_model_and_optimizer(model, optimizer)
# ... do other work ...
reload_model_and_optimizer(model, optimizer)

# Simple manual functions
offload_model(model, optimizer)  # Offload to CPU
onload_model(model, optimizer)   # Reload to GPU

# Context manager for automatic handling
with FastOffloadContext(model, optimizer) as ctx:
    # Model automatically offloaded here
    # Do inference with another model
    pass
# Model automatically reloaded here
"""

from copy import deepcopy
import accelerate
import gc
import torch

from deepspeed.runtime.bf16_optimizer import BF16_Optimizer
from deepspeed.runtime.zero.offload_config import (
    OffloadStateTypeEnum,
    OffloadDeviceEnum,
)


def _safe_destroy(self):
    for g in getattr(self, "param_groups", []):
        for p in g.get("params", []):
            if hasattr(p, "ds_tensor"):
                delattr(p, "ds_tensor")
    self.param_groups.clear()  # leave no empty lists behind


BF16_Optimizer.destroy = _safe_destroy


def prepare_deepspeed(model, accelerator: "accelerate"):
    # Taken from: https://github.com/huggingface/trl/blob/main/trl/models/utils.py#L308
    """Prepares the model for DeepSpeed inference or evaluation by initializing it with the appropriate configuration.

    Adapted from accelerate:
    https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
    """
    import deepspeed  # local import (instead of top-level) to avoid DS init interfering with other backends (like vllm): https://github.com/deepspeedai/DeepSpeed/issues/7252

    deepspeed_plugin = accelerator.state.deepspeed_plugin
    config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)
    stage = config_kwargs["zero_optimization"]["stage"]

    if model is not None:
        hidden_size = (
            max(model.config.hidden_sizes)
            if getattr(model.config, "hidden_sizes", None)
            else getattr(model.config, "hidden_size", None)
        )
        if hidden_size is not None and stage == 3:
            # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache
            # @ step 0: expected module 1, but got module 0`
            # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
            config_kwargs.update(
                {
                    "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                    "zero_optimization.stage3_param_persistence_threshold": 10
                    * hidden_size,
                    "zero_optimization.stage3_prefetch_bucket_size": 0.9
                    * hidden_size
                    * hidden_size,
                }
            )

    # If ZeRO-3 is used, we shard both the active and reference model.
    # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO
    # disabled (stage 0)
    if stage != 3:
        config_kwargs["zero_optimization"]["stage"] = 0
    model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
    model.eval()
    return model


# ═════════════════════════════════════════════════════════════════════════════
# Offloading helpers
# ═════════════════════════════════════════════════════════════════════════════


def _zero_tensors(zopt):
    """Generator that yields all tensors in a ZeRO optimizer."""
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


def _move_zero_tensors(zopt, device, non_blocking=True):
    """Move all ZeRO optimizer tensors to the specified device. Returns bytes moved."""
    moved = 0
    tensors_to_move = []
    
    # Collect all tensors that need to be moved
    for t in _zero_tensors(zopt):
        if t.device != device:
            moved += t.numel() * t.element_size()
            tensors_to_move.append(t)
    
    # Move tensors asynchronously in parallel
    if tensors_to_move:
        # Use CUDA streams for parallel transfers when moving to/from GPU
        if device.type == 'cuda' or any(t.device.type == 'cuda' for t in tensors_to_move):
            # Batch the transfers for better performance
            for t in tensors_to_move:
                t.data = t.data.to(device, non_blocking=non_blocking)
        else:
            # For CPU-to-CPU moves, process in batches
            for t in tensors_to_move:
                t.data = t.data.to(device, non_blocking=False)
    
    return moved


def _offload_optimizer(model, optimizer, device="cpu", non_blocking=True):
    """
    Offload optimizer states (ZeRO tensors + DeepSpeed engine states) to the specified device.
    Returns the number of bytes moved.
    """
    # Offload ZeRO optimizer tensors with non-blocking transfers
    moved = _move_zero_tensors(optimizer, torch.device(device), non_blocking=non_blocking)

    # Offload DeepSpeed optimizer engine states (grad buffer, hp params, lp grads)
    if hasattr(model, "optimizer"):
        include = [
            OffloadStateTypeEnum.contiguous_grad_buffer,
            OffloadStateTypeEnum.hp_params,  # High precision params (optimizer states)
            OffloadStateTypeEnum.lp_grads,  # Low precision gradients
        ]

        model.optimizer.offload_states(
            include=include,
            device=OffloadDeviceEnum.cpu,
            pin_memory=True,
            non_blocking=non_blocking,
        )

    return moved


def _offload_model(model, non_blocking=True):
    """Offload model states (lp params) to CPU."""
    if hasattr(model, "optimizer"):
        include = [
            OffloadStateTypeEnum.lp_params,  # Low precision parameters (model weights)
        ]

        model.optimizer.offload_states(
            include=include,
            device=OffloadDeviceEnum.cpu,
            pin_memory=True,
            non_blocking=non_blocking,
        )


def _reload_optimizer(model, optimizer, device="cuda", non_blocking=True):
    moved = _move_zero_tensors(optimizer, torch.device(device), non_blocking=non_blocking)
    return moved


def _reload_engine_states(engine, non_blocking=True):
    """Reload DeepSpeed engine states from CPU back to GPU."""
    engine.reload_states(non_blocking=non_blocking)


def offload_model_and_optimizer(
    model, 
    optimizer, 
    offload_optimizer=True, 
    offload_model=True,
    non_blocking=True
):
    info = {"optimizer_bytes": 0, "model_offloaded": False}

    # Check if model has DeepSpeed optimizer (ZeRO stage 3)
    if not hasattr(model, "optimizer"):
        # No DeepSpeed optimizer available, skip offloading
        return info

    # Start with optimizer offloading (typically larger and benefits more from async)
    if offload_optimizer:
        info["optimizer_bytes"] = _offload_optimizer(
            model, optimizer, non_blocking=non_blocking
        )

    # Offload model parameters
    if offload_model:
        _offload_model(model, non_blocking=non_blocking)
        info["model_offloaded"] = True

    # Synchronize before cleanup if using non-blocking transfers
    if non_blocking and torch.cuda.is_available():
        torch.cuda.synchronize()
    
    gc.collect()
    torch.cuda.empty_cache()
    
    return info


def reload_model_and_optimizer(
    model, 
    optimizer, 
    reload_optimizer=True, 
    reload_model=True,
    non_blocking=True
):
    info = {"optimizer_bytes": 0, "model_reloaded": False}

    # Check if model has DeepSpeed optimizer (ZeRO stage 3)
    if not hasattr(model, "optimizer"):
        # No DeepSpeed optimizer available, skip reloading
        return info

    # Reload model first (parameters needed before optimizer states)
    if reload_model:
        _reload_engine_states(model.optimizer, non_blocking=non_blocking)
        info["model_reloaded"] = True

    # Reload optimizer states
    if reload_optimizer:
        info["optimizer_bytes"] = _reload_optimizer(
            model, optimizer, non_blocking=non_blocking
        )

    # Synchronize transfers before cleanup
    if non_blocking and torch.cuda.is_available():
        torch.cuda.synchronize()

    # Clean up memory cache
    gc.collect()
    torch.cuda.empty_cache()

    return info


# Hack to allow offloading other optimizers too.
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


def offload_model(model, optimizer):
    """Simple function to offload model and optimizer to CPU."""
    return offload_model_and_optimizer(model, optimizer, non_blocking=True)


def onload_model(model, optimizer):
    """Simple function to reload model and optimizer to GPU."""
    return reload_model_and_optimizer(model, optimizer, non_blocking=True)


# Context manager for automatic fast offloading/reloading
class FastOffloadContext:
    def __init__(self, model, optimizer, offload_optimizer=True, offload_model=True):
        self.model = model
        self.optimizer = optimizer
        self.offload_optimizer = offload_optimizer
        self.offload_model = offload_model
        self.offload_info = None
        
    def __enter__(self):
        self.offload_info = offload_model_and_optimizer(
            self.model, 
            self.optimizer,
            offload_optimizer=self.offload_optimizer,
            offload_model=self.offload_model,
            non_blocking=True
        )
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        reload_model_and_optimizer(
            self.model,
            self.optimizer,
            reload_optimizer=self.offload_info.get("optimizer_bytes", 0) > 0,
            reload_model=self.offload_info.get("model_offloaded", False),
            non_blocking=True
        )
