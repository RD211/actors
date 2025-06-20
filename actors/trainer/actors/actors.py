from __future__ import annotations
import abc, contextlib, logging, random, time
from typing import Dict, List, Sequence
import openai, torch
from multiprocessing import managers
from vllm import SamplingParams, RequestOutput
from actors.utils.shm_utils import create_shared_state_dict, get_shareable_version
from actors.utils.logger import init_logger

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
    def update_weights(self, state_dict: Dict[str, torch.Tensor]): ...


class vLLMActor(TrainableLLMActor):
    def __init__(
        self,
        *,
        name: str,
        model_path: str,
        server_host: str = "localhost",
        server_port: int = 6000,
        gpu_groups: List[List[int]] | int | None = None,
        use_v1_engine: bool = True,
        engine_kwargs: Dict[str, any] | None = None,
    ):
        super().__init__(name)
        self.pool = self._connect(server_host, server_port, auth="secret")
        if name not in self.pool.list_models():
            self.pool.load_model(
                name=name,
                model_path=model_path,
                gpu_groups=gpu_groups,
                use_v1_engine=use_v1_engine,
                engine_kwargs=engine_kwargs,
            )

    @staticmethod
    def _connect(host: str, port: int, auth: str | None):
        class _Mgr(managers.BaseManager): ...

        _Mgr.register("ModelPool")
        m = _Mgr(address=(host, port), authkey=(auth.encode() if auth else None))
        m.connect()
        return m.ModelPool()

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

    def update_weights(self, state_dict: Dict[str, torch.Tensor]):
        meta = create_shared_state_dict(state_dict)
        try:
            self.pool.update_weights(self.name, get_shareable_version(meta))
        finally:
            for blob in meta.values():
                with contextlib.suppress(Exception):
                    blob["_shm_obj"].close()
                    blob["_shm_obj"].unlink()

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
