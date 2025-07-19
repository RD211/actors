import torch

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

    def init_cpu_cache(self):
        """Initializes a CPU cache for storing weight batches."""
        self.cpu_cache = {}

    def receive_and_cache_weights(self, ipc_handles_batch: dict):
        """
        Receives a batch of IPC handles and caches the corresponding tensors
        on the CPU for updating the model weights later.
        """
        if not hasattr(self, "cpu_cache"):
            self.init_cpu_cache()

        if not hasattr(self, "device_uuid"):
            self.report_device_id()

        # The ipc_handles_batch is a dictionary mapping device_uuid to
        # another dictionary of tensor_name: ipc_handle.
        if self.device_uuid not in ipc_handles_batch:
            return

        handles = ipc_handles_batch[self.device_uuid]
        device_id = self.device.index

        # Use a dedicated stream to allow for asynchronous H2D transfers.
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
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

        # Wait for all asynchronous transfers to complete.
        stream.synchronize()

    def load_weights_from_cache(self):
        """
        Loads the complete set of weights from the CPU cache into the model
        on the GPU.
        """
        if not hasattr(self, "cpu_cache") or not self.cpu_cache:
            logger.warning(
                colorize(
                    "No weights in CPU cache to load. Ensure that `receive_and_cache_weights` was called.",
                    Palette.WARNING,
                )
            )
            return

        # If vllm has any fp8 weights, we do something special.
        if any(
            "weight_scale" in k for k in self.model_runner.model.state_dict().keys()
        ):
            self.cpu_cache = to_vllm_state_dict(self.cpu_cache)
            self.cpu_cache = fp8_quantize_state_dict(self.cpu_cache)
            self.model_runner.model.load_state_dict(self.cpu_cache)
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
                if lora_a_key:
                    self.cpu_cache[lora_a_key][i] = (
                        self.cpu_cache[lora_a_key][i].unsqueeze(0).unsqueeze(0)
                    )
                    loras[lora_key].lora_a_stacked[i].data.copy_(
                        self.cpu_cache[lora_a_key][i]
                    )
                if lora_b_key:
                    self.cpu_cache[lora_b_key][i] = (
                        self.cpu_cache[lora_b_key][i].unsqueeze(0).unsqueeze(0)
                    )
                    loras[lora_key].lora_b_stacked[i].data.copy_(
                        self.cpu_cache[lora_b_key][i]
                    )
        # Clean up the cache to free memory
        self.cpu_cache = {}
