from __future__ import annotations
import atexit
from typing import Dict, List, Sequence
import torch
from vllm import SamplingParams, RequestOutput
from actors.inference.pool import ModelPool
from torch.multiprocessing.reductions import reduce_tensor
from vllm.platforms import current_platform
from .base import TrainableLLMActor


class vLLMActor(TrainableLLMActor):
    def __init__(
        self,
        *,
        name: str,
        model_path: str,
        gpu_groups: List[List[int]] | int | None = None,
        use_v1_engine: bool = True,
        engine_kwargs: Dict[str, any] | None = None,
        insomnia: bool = False, # If true all sleep calls will be ignored
    ):
        super().__init__(name, model_path)
        self.pool = ModelPool()
        if name not in self.pool.list_models():
            self.pool.load_model(
                name=name,
                model_path=model_path,
                gpu_groups=gpu_groups,
                use_v1_engine=use_v1_engine,
                engine_kwargs=engine_kwargs,
            )
        # Register cleanup function for this actor
        atexit.register(self._cleanup)
        self.name = name
        self.insomnia = insomnia

        self._sleep_level = 0

        self.sleep(level=1)

    def _cleanup(self):
        """Clean up resources when the program exits."""
        try:
            if hasattr(self, 'pool') and self.pool is not None:
                # Try to unload this model if it exists
                if self.name in self.pool.list_models():
                    self.pool.unload_model(self.name)
        except Exception:
            # Silently ignore cleanup errors to avoid segfaults
            pass

    def __del__(self):
        """Destructor for additional cleanup safety."""
        self._cleanup()

    def sleep(self, level: int = 1):
        if self.insomnia:
            return
        self.pool.sleep(self.name, level)
        self._sleep_level = level

    def wake(self):
        self.pool.wake(self.name)
        self._sleep_level = 0

    def finalize_weight_update(self):
        self.pool.finalize_update(self.name)
        self._sleep_level = 0

    def generate(
        self, prompts: Sequence[str], sampling_params: SamplingParams | None = None
    ) -> List[RequestOutput]:
        self._handle_sleep_state()
        sampling = sampling_params or SamplingParams()
        return self.pool.generate(self.name, list(prompts), sampling)

    def chat(
        self, dialogs: Sequence[list], sampling_params: SamplingParams | None = None
    ) -> List[RequestOutput]:
        self._handle_sleep_state()
        sampling = sampling_params or SamplingParams()
        return self.pool.chat(self.name, list(dialogs), sampling)

    def start_weight_update(self):
        self.pool.start_update(self.name)

    def update_weights_batch(self, state_dict: Dict[str, torch.Tensor]):
        if not state_dict:
            return

        tensors_by_device: Dict[torch.device, Dict[str, torch.Tensor]] = {}
        for name, tensor in state_dict.items():
            device = tensor.device
            if device not in tensors_by_device:
                tensors_by_device[device] = {}
            tensors_by_device[device][name] = tensor

        all_ipc_handles = {}
        for device, tensors in tensors_by_device.items():
            if device.type == "cuda":
                device_uuid = current_platform.get_device_uuid(device.index)

                ipc_handles = {
                    name: reduce_tensor(p.detach()) for name, p in tensors.items()
                }

                all_ipc_handles[device_uuid] = ipc_handles

        if not all_ipc_handles:
            return

        self.pool.update_weights_batch(self.name, all_ipc_handles)

    def _handle_sleep_state(self):
        if self._sleep_level == 1:
            self.wake()
        elif self._sleep_level == 2:
            raise RuntimeError(
                f"Model {self.name} is sleeping at level 2. While attempting to generate or chat."
            )
