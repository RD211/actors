from __future__ import annotations

import os
import traceback
from typing import Any

import ray
import torch
from vllm import LLM, SamplingParams

from actors.utils.logger import should_use_tqdm


# Sentinel value for default LoRA behavior (use LoRA if enabled, otherwise None)
class DefaultLoRA:
    """Sentinel class representing default LoRA behavior based on model configuration."""

    def __repr__(self):
        return "DefaultLoRA"


DEFAULT_LORA = DefaultLoRA()

_CLEAN_VARS = (
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "GROUP_RANK",
    "GROUP_WORLD_SIZE",
    "TORCHELASTIC_",
    "ACCELERATE_",
)


@ray.remote
class ModelWorker:
    """A Ray actor that runs a vLLM engine."""

    def __init__(
        self,
        model_name: str,
        model_path: str,
        gpus: list[int],
        use_v1_engine: bool,
        engine_kwargs: dict[str, Any],
    ) -> None:
        for k in list(os.environ):
            if k in _CLEAN_VARS or any(
                k.startswith(p) for p in _CLEAN_VARS if p.endswith("_")
            ):
                os.environ.pop(k, None)
        os.environ["VLLM_USE_V1"] = "1" if use_v1_engine else "0"
        os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
        self.gpu_group = gpus

        # This is for the new weight update mechanism
        engine_kwargs["worker_extension_cls"] = (
            "actors.inference.rlhf_utils.ColocateWorkerExtension"
        )

        self.engine = LLM(
            model=model_path,
            tensor_parallel_size=len(gpus),
            trust_remote_code=True,
            enable_sleep_mode=True,
            **engine_kwargs,
        )
        self.is_sleeping: bool = False
        self.sleep_level: int = 0
        self.model_name = model_name
        self.lora_enabled = engine_kwargs.get("enable_lora", False)

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
            self.engine.collective_rpc("init_cpu_cache", args=(self.gpu_group,))
            return "OK", None
        except Exception:
            return "ERROR", traceback.format_exc()

    def update_weights_batch(self, ipc_handles: dict) -> tuple[str, str | None]:
        try:
            self.engine.collective_rpc("receive_and_cache_weights", args=(ipc_handles,))
            return "OK", None
        except Exception:
            return "ERROR", traceback.format_exc()

    def update_lora_weights(self) -> tuple[str, str | None]:
        try:
            self.engine.collective_rpc("update_lora_weights")
            torch.cuda.empty_cache()
            return "OK", None
        except Exception:
            return "ERROR", traceback.format_exc()

    def create_lora_if_not_present(self, lora_path: str) -> tuple[str, str | None]:
        """Create and initialize LoRA adapter if not already present."""
        try:
            self.engine.collective_rpc("_create_lora_if_not_present", args=(lora_path,))
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

    def generate(
        self, shard: list, sampling_params: SamplingParams, lora_request=None
    ) -> list:
        if self.is_sleeping:
            raise RuntimeError(f"Model {self.model_name} is sleeping. Cannot generate.")
        if not shard:
            return []

        indices, inputs = zip(*shard, strict=False)

        # Pool has already handled DEFAULT_LORA conversion, so we just use what we received
        outputs = self.engine.generate(
            list(inputs),
            sampling_params,
            lora_request=lora_request,
            use_tqdm=should_use_tqdm(),
        )
        return list(zip(indices, outputs, strict=False))

    def chat(
        self, shard: list, sampling_params: SamplingParams, lora_request=None
    ) -> list:
        if self.is_sleeping:
            raise RuntimeError(f"ModelWorker is asleep at level {self.sleep_level}")
        if not shard:
            return []

        indices, inputs = zip(*shard, strict=False)

        # Pool has already handled DEFAULT_LORA conversion, so we just use what we received
        outputs = self.engine.chat(
            list(inputs),
            sampling_params,
            lora_request=lora_request,
            use_tqdm=should_use_tqdm(),
        )
        return list(zip(indices, outputs, strict=False))
