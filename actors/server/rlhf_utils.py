# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch


class ColocateWorkerExtension:
    """
    The class for vLLM's worker to inherit from, in the colocate setting.
    By defining an extension class, the code can work no matter what is
    the underlying worker class. This way, the code can be compatible
    with both vLLM V0 and V1.
    NOTE: we define this class in a separate module, and the main module
    should pass the full qualified name as `worker_extension_cls` argument.
    """

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
            print(
                "No weights in CPU cache to load. Ensure that `receive_and_cache_weights` was called."
            )
            return

        # The model_runner's load_weights method handles moving the tensors
        # from CPU to the correct GPU device.
        weights = list(self.cpu_cache.items())
        self.model_runner.model.load_weights(weights=weights)
        torch.cuda.synchronize()

        # Clean up the cache to free memory
        self.cpu_cache = {}