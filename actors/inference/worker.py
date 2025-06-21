from __future__ import annotations
import os, traceback, torch
from typing import Any, Dict, List
import ray
from vllm import LLM, SamplingParams
from actors.utils.logger import should_use_tqdm


@ray.remote
class ModelWorker:
    """A Ray actor that runs a vLLM engine."""

    def __init__(
        self,
        model_name: str,
        model_path: str,
        gpus: List[int],
        use_v1_engine: bool,
        engine_kwargs: Dict[str, Any],
    ) -> None:
        os.environ["VLLM_USE_V1"] = "1" if use_v1_engine else "0"
        os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
        os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(map(str, gpus))

        # This is for the new weight update mechanism
        engine_kwargs[
            "worker_extension_cls"
        ] = "actors.inference.rlhf_utils.ColocateWorkerExtension"

        self.engine = LLM(
            model=model_path,
            tensor_parallel_size=len(gpus),
            trust_remote_code=True,
            enable_sleep_mode=True,
            distributed_executor_backend="external_launcher",
            **engine_kwargs,
        )
        self.is_sleeping: bool = False
        self.sleep_level: int = 0
        self.model_name = model_name

    def ready(self):
        return True

    def sleep(self, level: int = 1) -> None:
        if not self.is_sleeping:
            self.engine.sleep(level=level)
            self.is_sleeping, self.sleep_level = True, level

    def wake(self) -> None:
        if self.is_sleeping:
            self.engine.wake_up()
            self.is_sleeping, self.sleep_level = False, 0

    def start_update(self) -> tuple[str, str | None]:
        try:
            self.engine.collective_rpc("init_cpu_cache")
            return "OK", None
        except Exception:
            return "ERROR", traceback.format_exc()

    def update_weights_batch(self, ipc_handles: dict) -> tuple[str, str | None]:
        try:
            self.engine.collective_rpc(
                "receive_and_cache_weights", args=(ipc_handles,)
            )
            return "OK", None
        except Exception:
            return "ERROR", traceback.format_exc()

    def finalize_update(self) -> tuple[str, str | None]:
        try:
            self.engine.collective_rpc("load_weights_from_cache")
            torch.cuda.empty_cache()
            return "OK", None
        except Exception:
            return "ERROR", traceback.format_exc()

    def generate(self, shard: list, sampling_params: SamplingParams) -> list:
        if self.is_sleeping:
            raise RuntimeError(f"asleep level {self.sleep_level}")
        if not shard:
            return []

        indices, inputs = zip(*shard)
        outputs = self.engine.generate(list(inputs), sampling_params, use_tqdm=should_use_tqdm())
        return list(zip(indices, outputs))

    def chat(self, shard: list, sampling_params: SamplingParams) -> list:
        if self.is_sleeping:
            raise RuntimeError(f"asleep level {self.sleep_level}")
        if not shard:
            return []

        indices, inputs = zip(*shard)
        outputs = self.engine.chat(list(inputs), sampling_params, use_tqdm=should_use_tqdm())
        return list(zip(indices, outputs))
