from __future__ import annotations
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
    ):
        super().__init__(name)
        self.pool = ModelPool()
        if name not in self.pool.list_models():
            self.pool.load_model(
                name=name,
                model_path=model_path,
                gpu_groups=gpu_groups,
                use_v1_engine=use_v1_engine,
                engine_kwargs=engine_kwargs,
            )

    def generate(
        self, prompts: Sequence[str], sampling_params: SamplingParams | None = None
    ) -> List[RequestOutput]:
        sampling = sampling_params or SamplingParams()
        return self.pool.generate(self.name, list(prompts), sampling)

    def chat(
        self, dialogs: Sequence[list], sampling_params: SamplingParams | None = None
    ) -> List[RequestOutput]:
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

    def finalize_weight_update(self):
        self.pool.finalize_update(self.name)

    def sleep(self, level: int = 1):
        self.pool.sleep(self.name, level)
        
    def wake(self):
        self.pool.wake(self.name)
