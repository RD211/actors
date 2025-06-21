from __future__ import annotations
import abc, logging, random, time
from typing import Dict, List, Sequence
import openai, torch
from vllm import SamplingParams, RequestOutput
from actors.utils.logger import init_logger
from actors.server.pool import ModelPool
from torch.multiprocessing.reductions import reduce_tensor
from vllm.platforms import current_platform


logger = init_logger(__name__, level=logging.INFO)


class LLMActor(abc.ABC):
    def __init__(self, name: str):
        self.name = name

    @abc.abstractmethod
    def generate(self, prompts: Sequence[str], **kwargs): ...
    @abc.abstractmethod
    def chat(self, dialogs: Sequence[list], **kwargs): ...


class TrainableLLMActor(LLMActor):
    @abc.abstractmethod
    def start_weight_update(self):
        ...

    @abc.abstractmethod
    def update_weights_batch(self, state_dict: Dict[str, torch.Tensor]):
        ...

    @abc.abstractmethod
    def finalize_weight_update(self):
        ...


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

        # Group tensors by device to handle multi-GPU training scenarios
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
            # This can happen if the batch is empty or contains only non-CUDA tensors
            return

        self.pool.update_weights_batch(self.name, all_ipc_handles)

    def finalize_weight_update(self):
        self.pool.finalize_update(self.name)

    def sleep(self, level: int = 1):
        self.pool.sleep(self.name, level)
        
    def wake(self):
        self.pool.wake(self.name)


class OpenAIActor(LLMActor):
    def __init__(
        self,
        *,
        name: str,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        retries: int = 5,
        backoff_start: float = 1.0,
        backoff_cap: float = 30.0,
    ):
        super().__init__(name)
        openai.api_key = api_key
        openai.base_url = base_url
        self.retries = retries
        self.backoff_start = backoff_start
        self.backoff_cap = backoff_cap

    def _retry(self, fn, *args, **kwargs):
        backoff = self.backoff_start
        for attempt in range(1, self.retries + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                logger.warning(f"[retry {attempt}] OpenAI error: {e}")
                if attempt == self.retries:
                    raise
                time.sleep(backoff + random.uniform(0, 1))
                backoff = min(backoff * 2, self.backoff_cap)

    def generate(self, prompts: Sequence[str], **params):
        return [
            self._retry(openai.Completion.create, model=self.name, prompt=p, **params)
            for p in prompts
        ]

    def chat(self, dialogs: Sequence[list], **params):
        return [
            self._retry(
                openai.ChatCompletion.create, model=self.name, messages=d, **params
            )
            for d in dialogs
        ]
