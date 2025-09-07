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

    def init_cpu_cache(self, gpu_group: list[int]):
        self.cpu_cache = {}
        self.gpu_group = gpu_group

    def receive_and_cache_weights(self, ipc_handles_batch: dict):
        if not hasattr(self, "cpu_cache"):
            self.init_cpu_cache()

        if not hasattr(self, "device_uuid"):
            self.report_device_id()

        if self.device_uuid not in ipc_handles_batch:
            return

        handles = (
            ipc_handles_batch[self.device_uuid]
            if self.device_uuid in ipc_handles_batch
            else ipc_handles_batch[list(ipc_handles_batch.keys())[0]]
        )
        device_id = self.device.index

        for name, handle in handles.items():
            func, args = handle
            list_args = list(args)

            list_args[6] = device_id
            tensor = func(*list_args)

            self.cpu_cache[name] = tensor.contiguous().to(
                device="cpu", non_blocking=True
            )
        torch.cuda.synchronize()

    def load_weights_from_cache(self):
        torch.cuda.synchronize()

        if not hasattr(self, "cpu_cache") or not self.cpu_cache:
            return

        # If vllm has any fp8 weights, we do something special.
        if any(
            "weight_scale" in k for k in self.model_runner.model.state_dict().keys()
        ):
            # This currently only works on Qwen2 models and no tensor parallelism.
            # TODO: Make this actually work on all gpus.
            self.cpu_cache = to_vllm_state_dict(self.cpu_cache)
            self.cpu_cache = fp8_quantize_state_dict(self.cpu_cache)
            self.model_runner.model.load_state_dict(self.cpu_cache)

        else:
            weights = list(self.cpu_cache.items())
            self.model_runner.model.load_weights(weights=weights)
        torch.cuda.synchronize()

        # Clean up the cache to free memory
        self.cpu_cache = {}

    def _create_lora_if_not_present(self, lora_path: str, identifier):
        from vllm.lora.request import LoRARequest

        adapter_id = identifier
        adapter_name = f"adapter_{adapter_id}"

        self.adapter_name = adapter_name
        self.adapter_id = adapter_id

        self.model_runner.add_lora(
            lora_request=LoRARequest(
                lora_name=f"lora_{adapter_name}",
                lora_int_id=adapter_id,
                lora_local_path=lora_path,
            )
        )
        self.initialized_lora = True

    def update_lora_weights(self):
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

        adapter_manager = self.model_runner.lora_manager._adapter_manager
        lora_A_keys = [k for k in self.cpu_cache.keys() if "lora_A" in k]
        lora_B_keys = [k for k in self.cpu_cache.keys() if "lora_B" in k]
        loras = adapter_manager.modules

        # TP info: compute local rank as index within gpu_group
        try:
            tp_idx = self.gpu_group.index(self.device.index)
        except ValueError:
            tp_idx = 0

        # Determine adapter slot (0-based) from adapter_id
        adapter_slot = max(int(getattr(self, "adapter_id", 1)) - 1, 0)

        def _copy_A_padded(cpu_A: torch.Tensor, gpu_A: torch.Tensor):
            # cpu_A: [r_small, in_full]
            # gpu_A slice: [1, 1, Rcap, in_local]; full tensor: [max_loras, 1, Rcap, in_local]
            r_cap = gpu_A.shape[2]
            in_local = gpu_A.shape[3]
            in_full = int(cpu_A.shape[1])

            src = torch.zeros(
                (1, 1, r_cap, in_full), dtype=gpu_A.dtype, device=gpu_A.device
            )
            r_small = cpu_A.shape[0]
            src[0, 0, :r_small, :in_full] = cpu_A.to(src.dtype)
            start = int(tp_idx) * in_local
            end = start + in_local

            if src.shape == gpu_A.shape:
                gpu_A.data.copy_(src)
                return

            gpu_A.data.copy_(src[:, :, :, start:end])

        def _copy_B_padded(cpu_B: torch.Tensor, gpu_B: torch.Tensor):
            # cpu_B: [out_full, r_small]
            # gpu_B slice: [1, 1, out_local, Rcap]; full tensor: [max_loras, 1, out_local, Rcap]
            r_cap = gpu_B.shape[3]
            out_local = gpu_B.shape[2]
            out_full = int(cpu_B.shape[0])

            src = torch.zeros(
                (1, 1, out_full, r_cap), dtype=gpu_B.dtype, device=gpu_B.device
            )
            r_small = cpu_B.shape[1]
            src[0, 0, :out_full, :r_small] = cpu_B.to(src.dtype)
            start = int(tp_idx) * out_local
            end = start + out_local
            if src.shape == gpu_B.shape:
                gpu_B.data.copy_(src)
                return
            gpu_B.data.copy_(src[:, :, start:end, :])

        for lora_key in loras.keys():
            lora_a_key = [
                k for k in lora_A_keys if lora_key.strip("model.").strip("layers.") in k
            ]
            lora_b_key = [
                k for k in lora_B_keys if lora_key.strip("model.").strip("layers.") in k
            ]
            if not lora_a_key and not lora_b_key:
                continue
            lora_a_key = lora_a_key[0] if lora_a_key else None
            lora_b_key = lora_b_key[0] if lora_b_key else None

            # Ensure list grouping
            if lora_a_key and not isinstance(self.cpu_cache[lora_a_key], list):
                self.cpu_cache[lora_a_key] = [self.cpu_cache[lora_a_key]]
            if lora_b_key and not isinstance(self.cpu_cache[lora_b_key], list):
                self.cpu_cache[lora_b_key] = [self.cpu_cache[lora_b_key]]

            for i, _ in enumerate(loras[lora_key].lora_a_stacked):
                # LoRA‑A
                if lora_a_key:
                    cpu_A = self.cpu_cache[lora_a_key][i]  # [r_small, in_full]
                    gpu_A = loras[lora_key].lora_a_stacked[i][
                        adapter_slot : adapter_slot + 1, :, :, :
                    ]
                    _copy_A_padded(cpu_A, gpu_A)

                # LoRA‑B
                if lora_b_key:
                    cpu_B = self.cpu_cache[lora_b_key][i]  # [out_full, r_small]
                    gpu_B = loras[lora_key].lora_b_stacked[i][
                        adapter_slot : adapter_slot + 1, :, :, :
                    ]
                    _copy_B_padded(cpu_B, gpu_B)

        # Clear CPU cache after updates
        self.cpu_cache = {}
