import torch
import ray

from actors.utils.logger import Palette, colorize, logger
from actors.utils.vllm import (
    fp8_quantize_state_dict,
    to_vllm_lora_state_dict,
    to_vllm_state_dict,
)


class ColocateWorkerExtension:
    def report_device_id(self) -> str:
        from vllm.platforms import current_platform

        self.device_uuid = current_platform.get_device_uuid(self.device.index)
        return self.device_uuid

    def init_cpu_cache(self, gpu_group: list[int]):
        """Initializes a CPU cache for storing weight batches."""
        self.cpu_cache = {}
        self.gpu_group = gpu_group

    def receive_and_cache_weights(self, ipc_handles_batch: dict):
        """
        Receives a batch of IPC handles and caches the corresponding tensors
        on the CPU for updating the model weights later.
        """
        if not hasattr(self, "cpu_cache"):
            self.init_cpu_cache()

        if not hasattr(self, "device_uuid"):
            self.report_device_id()

        if self.device_uuid not in ipc_handles_batch: #and not (hasattr(self, "initialized_lora") and self.initialized_lora):
            return

        handles = ipc_handles_batch[self.device_uuid] if self.device_uuid in ipc_handles_batch else ipc_handles_batch[list(ipc_handles_batch.keys())[0]]
        device_id = self.device.index

        # Use a dedicated stream to allow for asynchronous H2D transfers.
        for name, handle in handles.items():
            func, args = handle
            list_args = list(args)
            # The 6th argument to ipc_open is the device ID. We override it
            # to ensure the tensor is created on the correct local device.
            list_args[6] = device_id
            tensor = func(*list_args)
            # Asynchronously copy the tensor to CPU to avoid blocking.
            # The tensor must be contiguous for non-blocking transfer.
            self.cpu_cache[name] = tensor.contiguous().to(
                device="cpu", non_blocking=True
            )


    def load_weights_from_cache(self):
        """
        Loads the complete set of weights from the CPU cache into the model
        on the GPU.
        """
        torch.cuda.synchronize()

        if not hasattr(self, "cpu_cache") or not self.cpu_cache:
            return
        

        # If vllm has any fp8 weights, we do something special.
        if any(
            "weight_scale" in k for k in self.model_runner.model.state_dict().keys()
        ):

            # This currently only works on Qwen2 models and no tensor parallelism.
            # Also fails if the gpu does not support fp8 :)
            # Very experimental and hacky.
            # TODO: Make this not hacky.
            self.cpu_cache = to_vllm_state_dict(self.cpu_cache)
            self.cpu_cache = fp8_quantize_state_dict(self.cpu_cache)
            self.model_runner.model.load_state_dict(self.cpu_cache)
            # We get the model class.
            # from inspect import unwrap
            # model_class = unwrap(self.model_runner.model.model.__class__)
            # big_class = self.model_runner.model.__class__
            # big_class.Qwen2Model = model_class
            # print(self.model_runner.vllm_config)
            # #copy of vllm_config
            # v_config = self.model_runner.vllm_config
            # level_before = v_config.compilation_config.level
            # v_config.compilation_config.level = 1
            # new_model = big_class(
            #     vllm_config=self.model_runner.vllm_config,
            # )
            # to_convert_weights = list(self.cpu_cache.items())
            # # move everything to gpu
            # for k, v in to_convert_weights:
            #     self.cpu_cache[k] = v.to(device="cuda", non_blocking=True)
            # torch.cuda.empty_cache()
            # torch.cuda.synchronize()
            # self.model_runner.model.load_weights(weights=list(self.cpu_cache.items()))

            # state_dict = new_model.state_dict()
            # with open("fp8_weights.txt", "w") as f:
            #     for k, v in state_dict.items():
            #         f.write(f"{k}: {v.shape} {v.dtype} {v.device}\n")

            # with open("model_weights.txt", "w") as f:
            #     for k, v in self.model_runner.model.state_dict().items():
            #         f.write(f"{k}: {v.shape} {v.dtype} {v.device}\n")
            # print("A"*100)
            # self.model_runner.model.load_state_dict(state_dict)
            # print("A"*100)
            # v_config.compilation_config.level = level_before

            # # Output all keys and shapes to file.
            # with open("fp8_weights.txt", "w") as f:
            #     for k, v in self.cpu_cache.items():
            #         f.write(f"{k}: {v.shape} {v.dtype} {v.device}\n")

            # with open("model_weights.txt", "w") as f:
            #     for k, v in self.model_runner.model.state_dict().items():
            #         f.write(f"{k}: {v.shape} {v.dtype} {v.device}\n")
        else:
            weights = list(self.cpu_cache.items())
            self.model_runner.model.load_weights(weights=weights)
        torch.cuda.synchronize()

        # Clean up the cache to free memory
        self.cpu_cache = {}

    def _create_lora_if_not_present(self, lora_path: str):
        from vllm.lora.request import LoRARequest

        self.model_runner.add_lora(
            lora_request=LoRARequest(
                lora_name="lora",
                lora_int_id=1,
                lora_local_path=lora_path,
            )
        )
        self.initialized_lora = True

    def update_lora_weights(self):
        """
        Updates the LoRA weights in the model.
        This method assumes that the LoRA weights are already cached in `self.cpu_cache`.
        """
        if not hasattr(self, "cpu_cache") or not self.cpu_cache:
            logger.warning(
                colorize("No LoRA weights in CPU cache to update.", Palette.WARNING)
            )
            return
        if not hasattr(self, "initialized_lora") or not self.initialized_lora:
            logger.warning(
                colorize(
                    "LoRA is not initialized. Call `_create_lora_if_not_present` first.",
                    Palette.WARNING,
                )
            )
            return

        self.cpu_cache = to_vllm_lora_state_dict(self.cpu_cache)

        # The hackiest shit possible :)
        adapter_manager = self.model_runner.lora_manager._adapter_manager
        lora_A_keys = [k for k in self.cpu_cache.keys() if "lora_A" in k]
        lora_B_keys = [k for k in self.cpu_cache.keys() if "lora_B" in k]
        loras = adapter_manager.modules

        def _copy_lora_from_cpu(
            src: torch.Tensor, 
            dst: torch.Tensor,
            tp_idx: int,
            tp_world_size: int
        ):
            if src.shape == dst.shape:
                dst.data.copy_(src)
                return

            shard_dim = None
            for dim, (full, local) in enumerate(zip(src.shape, dst.shape)):
                if full != local:
                    if full // tp_world_size == local and full % tp_world_size == 0:
                        shard_dim = dim
                        break
            if shard_dim is None:
                raise RuntimeError(
                    f"Shape mismatch {src.shape} → {dst.shape} does not "
                    "look like tensor-parallel sharding."
                )
            slice_size = src.shape[shard_dim] // tp_world_size
            start      = tp_idx * slice_size
            end        = start + slice_size

            sl = [slice(None)] * src.ndim
            sl[shard_dim] = slice(start, end)
            dst.data.copy_(src[tuple(sl)])

        tp_idx = self.device.index
        tp_world_size = len(self.gpu_group)
        for lora_key in loras.keys():
            lora_a_key = next(
                k for k in lora_A_keys if lora_key.split("model.layers")[1] in k
            )
            lora_b_key = next(
                k for k in lora_B_keys if lora_key.split("model.layers")[1] in k
            )
            if not isinstance(self.cpu_cache[lora_a_key], list):
                self.cpu_cache[lora_a_key] = [self.cpu_cache[lora_a_key]]
            if not isinstance(self.cpu_cache[lora_b_key], list):
                self.cpu_cache[lora_b_key] = [self.cpu_cache[lora_b_key]]

            for i, _ in enumerate(loras[lora_key].lora_a_stacked):
                # LoRA‑A
                if lora_a_key:
                    cpu_tensor = self.cpu_cache[lora_a_key][i].unsqueeze(0).unsqueeze(0)
                    gpu_tensor = loras[lora_key].lora_a_stacked[i]
                    _copy_lora_from_cpu(cpu_tensor, gpu_tensor, tp_idx, tp_world_size)

                # LoRA‑B
                if lora_b_key:
                    cpu_tensor = self.cpu_cache[lora_b_key][i].unsqueeze(0).unsqueeze(0)
                    gpu_tensor = loras[lora_key].lora_b_stacked[i]
                    _copy_lora_from_cpu(cpu_tensor, gpu_tensor, tp_idx, tp_world_size)
                
        self.cpu_cache = {}
