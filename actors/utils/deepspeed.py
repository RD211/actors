import gc
from collections.abc import Container
from copy import deepcopy

import accelerate
import deepspeed
import torch
from deepspeed.runtime.bf16_optimizer import BF16_Optimizer
from deepspeed.runtime.zero.offload_config import (
    OffloadDeviceEnum,
    OffloadStateTypeEnum,
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

    for v in inner.__dict__[
        "_DeepSpeedZeroOptimizer_Stage3__param_id_to_grad_partition"
    ].values():
        if torch.is_tensor(v):
            yield v
    keys = inner.__dict__.get("ipg_buckets", {}).keys()
    for k in keys:
        yield inner.__dict__["ipg_buckets"][k].buffer


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
        for t in tensors_to_move:
            t.data = t.data.to(device, non_blocking=non_blocking)

    return moved


def _offload_optimizer(model, optimizer, device="cpu", non_blocking=True):
    """
    Offload optimizer states (ZeRO tensors + DeepSpeed engine states) to the specified device.
    Returns the number of bytes moved.
    """

    # Offload DeepSpeed optimizer engine states
    include = [
        OffloadStateTypeEnum.contiguous_grad_buffer,
        OffloadStateTypeEnum.hp_params,  # High precision params
        OffloadStateTypeEnum.lp_grads,  # Low precision gradients
        OffloadStateTypeEnum.optim_states,  # Optimizer states
    ]

    model.optimizer.offload_states(
        include=include,
        device=OffloadDeviceEnum.cpu,
        pin_memory=True,
        non_blocking=non_blocking,
    )

    moved = _move_zero_tensors(
        optimizer, torch.device(device), non_blocking=non_blocking
    )

    return moved


def _offload_model(optimizer, non_blocking=True):
    """Offload model states (lp params) to CPU."""
    optimizer.offload_states(
        include=[
            OffloadStateTypeEnum.lp_params,
        ],
        device=OffloadDeviceEnum.cpu,
        pin_memory=True,
        non_blocking=non_blocking,
    )


def _reload_optimizer(optimizer, device="cuda", non_blocking=True):
    moved = _move_zero_tensors(
        optimizer, torch.device(device), non_blocking=non_blocking
    )
    return moved


def _reload_model(engine, non_blocking=True):
    copy_of_engine_reload_states = engine.offloaded_states
    if OffloadStateTypeEnum.lp_params not in copy_of_engine_reload_states:
        return
    engine.offloaded_states = set([OffloadStateTypeEnum.lp_params])
    engine.reload_states(non_blocking=non_blocking)
    engine.offloaded_states = set(copy_of_engine_reload_states) - set(
        [OffloadStateTypeEnum.lp_params]
    )


def _reload_engine_states(engine, non_blocking=True):
    """Reload DeepSpeed engine states from CPU back to GPU."""
    _copy_of_engine_reload_states = engine.offloaded_states
    engine_states = set(
        [
            OffloadStateTypeEnum.contiguous_grad_buffer,
            OffloadStateTypeEnum.hp_params,
            OffloadStateTypeEnum.lp_grads,
        ]
    )
    if not any(state in _copy_of_engine_reload_states for state in engine_states):
        return
    engine.offloaded_states = [
        st for st in engine_states if st in _copy_of_engine_reload_states
    ]
    engine.reload_states(non_blocking=non_blocking)
    engine.offloaded_states = set(_copy_of_engine_reload_states) - set(engine_states)


def offload_model_and_optimizer(
    model, optimizer, offload_optimizer=False, offload_model=False, non_blocking=True
):
    info = {"optimizer_bytes": 0, "model_offloaded": False}

    if not _validate_offloading_config(model):
        return info

    if offload_optimizer:
        info["optimizer_bytes"] = _offload_optimizer(
            model, optimizer, non_blocking=non_blocking
        )

    if offload_model:
        _offload_model(model.optimizer, non_blocking=non_blocking)
        info["model_offloaded"] = True

    if non_blocking and torch.cuda.is_available():
        torch.cuda.synchronize()

    gc.collect()
    torch.cuda.empty_cache()

    return info


def reload_model_and_optimizer(
    model, optimizer, reload_optimizer=True, reload_model=True, non_blocking=True
):
    info = {"optimizer_bytes": 0, "model_reloaded": False}

    if not hasattr(model, "optimizer"):
        # No DeepSpeed optimizer available, skip reloading
        return info

    # Reload model first (parameters needed before optimizer states)
    if reload_model:
        _reload_model(model.optimizer, non_blocking=non_blocking)
        info["model_reloaded"] = True

    # Reload optimizer states
    if reload_optimizer:
        info["optimizer_bytes"] = _reload_optimizer(
            optimizer, non_blocking=non_blocking
        )
        # reload DeepSpeed engine states
        _reload_engine_states(model.optimizer, non_blocking=non_blocking)

    # Synchronize transfers before cleanup
    if non_blocking and torch.cuda.is_available():
        torch.cuda.synchronize()

    # Clean up memory cache
    gc.collect()
    torch.cuda.empty_cache()

    return info


def prepare_deepspeed_reference(model, accelerator: "accelerate", use_cpu_offload=True):
    deepspeed_plugin = accelerator.state.deepspeed_plugin
    config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)

    if use_cpu_offload:
        config_kwargs.update(
            {
                "zero_optimization": {
                    "stage": 3,
                    "offload_param": {
                        "device": "cpu",
                        "pin_memory": True,
                        "buffer_count": 1,
                        "buffer_size": 1e8,
                        "max_in_cpu": 1e9,
                    },
                    "offload_optimizer": {
                        "device": "cpu",
                        "pin_memory": True,
                    },
                    "stage3_param_persistence_threshold": 0,
                    "stage3_prefetch_bucket_size": 1e6,
                    "stage3_max_live_parameters": 1e6,
                    "stage3_max_reuse_distance": 0,
                    "reduce_bucket_size": 1e6,
                    "contiguous_gradients": False,
                    "overlap_comm": False,
                    "sub_group_size": 1e6,
                },
            }
        )

        if model is not None:
            hidden_size = (
                max(model.config.hidden_sizes)
                if getattr(model.config, "hidden_sizes", None)
                else getattr(model.config, "hidden_size", None)
            )
            if hidden_size is not None:
                config_kwargs["zero_optimization"].update(
                    {
                        "stage3_param_persistence_threshold": 0,
                        "stage3_prefetch_bucket_size": min(hidden_size * 10, 1e6),
                        "stage3_max_live_parameters": min(hidden_size * 100, 1e6),
                    }
                )
    else:
        stage = config_kwargs["zero_optimization"]["stage"]
        if stage != 3:
            config_kwargs["zero_optimization"]["stage"] = 0

    model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
    model.eval()
    return model


# Hack to allow offloading other optimizers too.


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
            non_blocking=True,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        reload_model_and_optimizer(
            self.model,
            self.optimizer,
            reload_optimizer=self.offload_info.get("optimizer_bytes", 0) > 0,
            reload_model=self.offload_info.get("model_offloaded", False),
            non_blocking=True,
        )


def _validate_offloading_config(model):
    """
    Validate that the DeepSpeed model is properly configured for offloading.
    Returns True if offloading should be attempted, False otherwise.
    """
    if not hasattr(model, "optimizer"):
        return False

    # Check if the model has ZeRO stage 3 with proper offloading config
    if not hasattr(model.optimizer, "offload_states"):
        return False

    return True


from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.zero.offload_states import (
    offload_adam_states,
    reload_adam_states,
)

################
# Patch
#################
from deepspeed.runtime.zero.utils import get_mapping_to_flat_buffer


def patched_offload_states(
    self,
    include: Container[OffloadStateTypeEnum] = None,
    device: OffloadDeviceEnum = OffloadDeviceEnum.cpu,
    pin_memory: bool = True,
    non_blocking: bool = False,
):
    device = device.value
    self.empty_partition_cache()

    def needs_offload(target):
        return target not in self.offloaded_states and (
            include is None or target in include
        )

    # HP param
    if needs_offload(OffloadStateTypeEnum.hp_params):
        if pin_memory:
            if not hasattr(self, "hp_params_pin_buffers"):
                self.hp_params_pin_buffers = [
                    get_accelerator().pin_memory(torch.empty_like(t, device=device))
                    for t in self.fp32_partitioned_groups_flat
                ]

            for src_tensor, dest_buf in zip(
                self.fp32_partitioned_groups_flat,
                self.hp_params_pin_buffers,
                strict=False,
            ):
                dest_buf.copy_(src_tensor, non_blocking=non_blocking)
                src_tensor.data = dest_buf
        else:
            for buf in self.fp32_partitioned_groups_flat:
                buf.data = buf.data.to(device, non_blocking=non_blocking)
        self.offloaded_states.add(OffloadStateTypeEnum.hp_params)

    if needs_offload(OffloadStateTypeEnum.lp_params):
        mapping = get_mapping_to_flat_buffer(
            [p.ds_tensor for p in self.module.parameters()]
        )
        required = max(offset + numel for _, offset, numel in mapping)
        buffer_size = self.lp_param_buffer.numel()
        if required > buffer_size or hasattr(self, "lora_mode"):
            # We are in LoRA mode.
            self.lora_mode = True

            if required > buffer_size:
                self.lp_param_buffer = torch.empty(
                    required,
                    dtype=self.lp_param_buffer.dtype,
                    device=self.lp_param_buffer.device,
                )

            lora_params = self._get_parameter_partitions()

            parameters_to_offload = [p.ds_tensor for p in self.module.parameters()]
            # We remove all parameters in parameters_to_offload that are in lora_params
            parameters_to_offload = [
                p
                for p in parameters_to_offload
                if all(p is not lora_param for lora_param in lora_params)
            ]
            # We now add them back at the beginning of the list
            parameters_to_offload = lora_params + parameters_to_offload

            for tensor, offset, tensor_numel in get_mapping_to_flat_buffer(
                parameters_to_offload
            ):
                self.lp_param_buffer.narrow(0, offset, tensor_numel).copy_(tensor.data)

            if pin_memory:
                if not hasattr(self, "lp_param_contiguous_pin_buffer"):
                    self.lp_param_contiguous_pin_buffer = get_accelerator().pin_memory(
                        torch.empty_like(self.lp_param_buffer, device=device)
                    )
                self.lp_param_contiguous_pin_buffer.copy_(
                    self.lp_param_buffer, non_blocking=non_blocking
                )
                cpu_buffer = self.lp_param_contiguous_pin_buffer
            else:
                cpu_buffer = self.lp_param_buffer.to(device, non_blocking=non_blocking)

            self.lp_param_buffer.data = cpu_buffer
            for tensor, offset, tensor_numel in get_mapping_to_flat_buffer(
                parameters_to_offload
            ):
                tensor.data = cpu_buffer.narrow(0, offset, tensor_numel)
        else:
            if pin_memory:
                if not hasattr(self, "lp_param_contiguous_pin_buffer"):
                    self.lp_param_contiguous_pin_buffer = get_accelerator().pin_memory(
                        torch.empty_like(self.lp_param_buffer, device=device)
                    )
                self.lp_param_contiguous_pin_buffer.copy_(
                    self.lp_param_buffer, non_blocking=non_blocking
                )
                cpu_buffer = self.lp_param_contiguous_pin_buffer
            else:
                cpu_buffer = self.lp_param_buffer.to(device, non_blocking=non_blocking)

            self.lp_param_buffer.data = cpu_buffer
            for tensor, offset, tensor_numel in get_mapping_to_flat_buffer(
                [p.ds_tensor for p in self.module.parameters()]
            ):
                tensor.data = cpu_buffer.narrow(0, offset, tensor_numel)

        self.fp16_partitioned_groups_flat.clear()
        self.offloaded_states.add(OffloadStateTypeEnum.lp_params)

    # LP grad
    if needs_offload(OffloadStateTypeEnum.lp_grads):
        if pin_memory:
            if not hasattr(self, "lp_grad_partitions_flat_pin_buffers"):
                self.lp_grad_partitions_flat_pin_buffers = get_accelerator().pin_memory(
                    torch.empty_like(self.grad_partitions_flat_buffer, device=device)
                )
            self.lp_grad_partitions_flat_pin_buffers.copy_(
                self.grad_partitions_flat_buffer, non_blocking=non_blocking
            )
            self.grad_partitions_flat_buffer.data = (
                self.lp_grad_partitions_flat_pin_buffers
            )
        else:
            self.grad_partitions_flat_buffer.data = (
                self.grad_partitions_flat_buffer.data.to(device)
            )
        self.averaged_gradients = {}

        self.__param_id_to_grad_partition = {}

        self.offloaded_states.add(OffloadStateTypeEnum.lp_grads)

    # contiguous bucket
    if needs_offload(OffloadStateTypeEnum.contiguous_grad_buffer):
        if (
            hasattr(self, "_DeepSpeedZeroOptimizer_Stage3__ipg_bucket_flat_buffer")
            and self._DeepSpeedZeroOptimizer_Stage3__ipg_bucket_flat_buffer is not None
        ):
            # Record properties like shape, strides, etc. as a meta tensor
            self.grad_buffer_meta = (
                self._DeepSpeedZeroOptimizer_Stage3__ipg_bucket_flat_buffer.to("meta")
            )
            self._DeepSpeedZeroOptimizer_Stage3__ipg_bucket_flat_buffer = None
            self.offloaded_states.add(OffloadStateTypeEnum.contiguous_grad_buffer)

    # Adam
    if needs_offload(OffloadStateTypeEnum.optim_states):
        offload_adam_states(
            self.optimizer, device, pin_memory=pin_memory, non_blocking=non_blocking
        )
        self.offloaded_states.add(OffloadStateTypeEnum.optim_states)

    gc.collect()
    get_accelerator().empty_cache()


import itertools


def patched_reload_states(self, non_blocking: bool = False):
    device = get_accelerator().current_device_name()

    # HP param
    if OffloadStateTypeEnum.hp_params in self.offloaded_states:
        if hasattr(self, "hp_params_pin_buffers"):
            for src, dest in zip(
                self.hp_params_pin_buffers,
                self.fp32_partitioned_groups_flat,
                strict=False,
            ):
                dest.data = src.to(device, non_blocking=non_blocking)
        else:
            for buf in self.fp32_partitioned_groups_flat:
                buf.data = buf.data.to(device, non_blocking=non_blocking)
        self.offloaded_states.remove(OffloadStateTypeEnum.hp_params)

    # LP Param
    if OffloadStateTypeEnum.lp_params in self.offloaded_states:
        cpu_buffer = (
            self.lp_param_contiguous_pin_buffer
            if hasattr(self, "lp_param_contiguous_pin_buffer")
            else self.lp_param_buffer
        )
        self.lp_param_buffer.data = cpu_buffer.data.to(
            device, non_blocking=non_blocking
        )
        self._set_fp16_partitioned_groups_flat()

        if hasattr(self, "lora_mode") and self.lora_mode:
            lora_params = self._get_parameter_partitions()
            parameters_to_offload = [p.ds_tensor for p in self.module.parameters()]
            parameters_to_offload = [
                p
                for p in parameters_to_offload
                if all(p is not lora_param for lora_param in lora_params)
            ]
            parameters_to_offload = lora_params + parameters_to_offload
            for tensor, offset, tensor_numel in get_mapping_to_flat_buffer(
                parameters_to_offload
            ):
                tensor.data = self.lp_param_buffer.narrow(0, offset, tensor_numel)
        else:
            parameter_partitions = self._get_parameter_partitions()
            for tensor, offset, tensor_numel in get_mapping_to_flat_buffer(
                parameter_partitions
            ):
                tensor.data = self.lp_param_buffer.narrow(0, offset, tensor_numel)

        self.offloaded_states.remove(OffloadStateTypeEnum.lp_params)

    # LP grad
    if OffloadStateTypeEnum.lp_grads in self.offloaded_states:
        if hasattr(self, "lp_grad_partitions_flat_pin_buffers"):
            self.grad_partitions_flat_buffer.data = (
                self.lp_grad_partitions_flat_pin_buffers.to(
                    device, non_blocking=non_blocking
                )
            )
        else:
            self.grad_partitions_flat_buffer.data = (
                self.grad_partitions_flat_buffer.data.to(
                    device, non_blocking=non_blocking
                )
            )
        self.averaged_gradients = {}

        offset = 0
        all_params = list(itertools.chain.from_iterable(self.fp16_groups))
        for param in all_params:
            self.__param_id_to_grad_partition[param.ds_id] = (
                self.grad_partitions_flat_buffer.narrow(
                    0, offset, param.partition_numel()
                )
            )
            offset += param.partition_numel()

        self.offloaded_states.remove(OffloadStateTypeEnum.lp_grads)

    # contiguous bucket
    if OffloadStateTypeEnum.contiguous_grad_buffer in self.offloaded_states:
        self._DeepSpeedZeroOptimizer_Stage3__ipg_bucket_flat_buffer = torch.empty_like(
            self.grad_buffer_meta, device=device
        )
        # self.__ipg_bucket_flat_buffer.data = self.__ipg_bucket_flat_buffer.data.to(device)
        self.offloaded_states.remove(OffloadStateTypeEnum.contiguous_grad_buffer)

    # Adam
    if OffloadStateTypeEnum.optim_states in self.offloaded_states:
        reload_adam_states(self.optimizer, device, non_blocking=non_blocking)
        self.offloaded_states.remove(OffloadStateTypeEnum.optim_states)

    if non_blocking:
        get_accelerator().synchronize()


from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3

DeepSpeedZeroOptimizer_Stage3.offload_states = patched_offload_states
DeepSpeedZeroOptimizer_Stage3.reload_states = patched_reload_states
