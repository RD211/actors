from __future__ import annotations

import atexit
from collections.abc import Sequence
from typing import Any

import torch
from torch.multiprocessing.reductions import reduce_tensor
from transformers import AutoConfig
from vllm import RequestOutput, SamplingParams
from vllm.platforms import current_platform

from actors.inference.pool import ModelPool
from actors.inference.worker import DEFAULT_LORA
from actors.trainers.base_config import ActorTrainCfg
from actors.utils.logger import Palette, colorize, init_logger

from .base import TrainableLLMActor


class vLLMActor(TrainableLLMActor):
    def __init__(
        self,
        *,
        name: str,
        model_path: str,
        gpu_groups: list[list[int]] | int | None = None,
        use_v1_engine: bool = True,
        engine_kwargs: dict[str, Any] | None = None,
        insomnia: bool = False,  # If true all sleep calls will be ignored
        training_config: ActorTrainCfg | None = None,
    ):
        self.logger = init_logger(name=name)
        model_config = AutoConfig.from_pretrained(model_path)

        if engine_kwargs is None:
            engine_kwargs = {}

        # We extract num_attention_heads if present and check
        # if it is divisible engine_kwargs["tensor_parallel_size"] if tensor parallel size is set
        # TODO: Make this handle all cases when tensor_parallel_size is set etc.
        # if hasattr(
        #     model_config, "num_attention_heads"
        # ):
        #     num_heads = model_config.num_attention_heads

        #     for gpu_g in gpu_groups:
        #         tensor_parallel_size = len(gpu_g)
        #         if num_heads % tensor_parallel_size != 0:
        #             for pipeline_parallel_size in range(1, tensor_parallel_size + 1):
        #                 if (
        #                     tensor_parallel_size % pipeline_parallel_size == 0
        #                     and num_heads % (tensor_parallel_size // pipeline_parallel_size)
        #                     == 0
        #                 ):
        #                     engine_kwargs["tensor_parallel_size"] = (
        #                         tensor_parallel_size // pipeline_parallel_size
        #                     )
        #                     engine_kwargs["pipeline_parallel_size"] = pipeline_parallel_size
        #                     break
        #             self.logger.warning(
        #                 colorize(
        #                     f"num_attention_heads ({num_heads}) is not divisible by tensor_parallel_size ({tensor_parallel_size}). "
        #                     "In order to keep the model working we will set tensor_parallel_size to "
        #                     f"{engine_kwargs['tensor_parallel_size']} and pipeline_parallel_size to {engine_kwargs['pipeline_parallel_size']}.",
        #                     Palette.ERROR,
        #                 )
        #             )

        super().__init__(
            name,
            model_path,
            training_config=training_config,
        )
        self.pool = ModelPool()
        # Prepare engine kwargs with LoRA support if PEFT config is present
        final_engine_kwargs = engine_kwargs.copy() if engine_kwargs else {}

        # Enable LoRA in vLLM if PEFT config is provided
        if self.training_config.peft_config is not None:
            final_engine_kwargs["enable_lora"] = True
            # Set reasonable defaults for LoRA if not already specified
            if "max_lora_rank" not in final_engine_kwargs:
                lora_rank = getattr(
                    self.training_config.peft_config, "r", 16
                )  # Default to rank 16
                final_engine_kwargs["max_lora_rank"] = lora_rank
            if "max_loras" not in final_engine_kwargs:
                final_engine_kwargs["max_loras"] = 1  # Default to 1 LoRA adapter

        self.pool.load_model(
            name=name,
            model_path=model_path,
            gpu_groups=gpu_groups,
            use_v1_engine=use_v1_engine,
            engine_kwargs=final_engine_kwargs,
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
            if hasattr(self, "pool") and self.pool is not None:
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
        with self._with_wake():
            self.pool.finalize_update(self.name)
            self._sleep_level = 0  # This method specifically sets sleep level to 0

    # ═══════════════════════════════════════════════════════════════
    # LoRA/PEFT Support Methods
    # ═══════════════════════════════════════════════════════════════

    def update_lora_weights(self):
        """Update LoRA weights in the vLLM worker."""
        with self._with_wake():
            self.pool.update_lora_weights(self.name)

    def create_lora_if_not_present(self, lora_path: str):
        """Create and initialize LoRA adapter if not already present in the vLLM worker."""
        with self._with_wake():
            self.pool.create_lora_if_not_present(self.name, lora_path)

    def generate(
        self,
        prompts: Sequence[str],
        sampling_params: SamplingParams | None = None,
        lora_request=DEFAULT_LORA,
    ) -> list[RequestOutput]:
        sampling = sampling_params or SamplingParams()
        with self._with_wake():
            return self.pool.generate(self.name, list(prompts), sampling, lora_request)

    def chat(
        self,
        dialogs: Sequence[list],
        sampling_params: SamplingParams | None = None,
        lora_request=DEFAULT_LORA,
    ) -> list[RequestOutput]:
        sampling = sampling_params or SamplingParams()
        with self._with_wake():
            return self.pool.chat(self.name, list(dialogs), sampling, lora_request)

    async def agenerate(
        self,
        prompts: Sequence[str],
        sampling_params: SamplingParams | None = None,
        lora_request=DEFAULT_LORA,
    ) -> list[RequestOutput]:
        sampling = sampling_params or SamplingParams()
        with self._with_wake():
            return await self.pool.agenerate(
                self.name, list(prompts), sampling, lora_request
            )

    async def achat(
        self,
        dialogs: Sequence[list],
        sampling_params: SamplingParams | None = None,
        lora_request=DEFAULT_LORA,
    ) -> list[RequestOutput]:
        sampling = sampling_params or SamplingParams()
        with self._with_wake():
            return await self.pool.achat(
                self.name, list(dialogs), sampling, lora_request
            )

    def start_weight_update(self):
        self.pool.start_update(self.name)

    def update_weights_batch(self, state_dict: dict[str, torch.Tensor]):
        if not state_dict:
            return

        tensors_by_device: dict[torch.device, dict[str, torch.Tensor]] = {}
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
        self.wake()

    def _with_wake(self):
        """Context manager to temporarily wake up and restore previous sleep state."""
        from contextlib import contextmanager

        @contextmanager
        def wake_context():
            previous_sleep_level = self._sleep_level
            self._handle_sleep_state()
            try:
                yield
            finally:
                if previous_sleep_level > 0:
                    self.sleep(level=previous_sleep_level)

        return wake_context()
