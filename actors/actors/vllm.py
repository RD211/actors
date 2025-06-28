from __future__ import annotations
import atexit
from typing import Dict, List, Sequence, Callable
import torch
from torch import nn
from vllm import SamplingParams, RequestOutput
from actors.inference.pool import ModelPool
from actors.inference.worker import DEFAULT_LORA
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
        learning_rate: float = 5e-6,
        optimizer: str | type | callable = "paged_adamw_8bit",
        optimizer_kwargs: Dict | None = None,
        loss: str | type | callable = "grpo",
        loss_kwargs: Dict | None = None,
        scheduler: str | type | callable = "cosine",
        scheduler_kwargs: Dict | None = None,
        model_factory: Callable[[], nn.Module] | None = None,
        reference_model_factory: Callable[[], nn.Module] | None = None,
        # PEFT/LoRA configuration
        peft_config = None,
        # Offloading parameters
        offload_optimizer: bool = False,
        offload_model: bool = False,
        offload_reference_to_cpu: bool = False,
        offload_activations_to_cpu: bool = False,
    ):
        super().__init__(
            name, 
            model_path,
            learning_rate=learning_rate,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            loss=loss,
            loss_kwargs=loss_kwargs,
            scheduler=scheduler,
            scheduler_kwargs=scheduler_kwargs,
            model_factory=model_factory,
            reference_model_factory=reference_model_factory,
            peft_config=peft_config,
            offload_optimizer=offload_optimizer,
            offload_model=offload_model,
            offload_reference_to_cpu=offload_reference_to_cpu,
            offload_activations_to_cpu=offload_activations_to_cpu,
        )
        self.pool = ModelPool()
        if name not in self.pool.list_models():
            # Prepare engine kwargs with LoRA support if PEFT config is present
            final_engine_kwargs = engine_kwargs.copy() if engine_kwargs else {}
            
            # Enable LoRA in vLLM if PEFT config is provided
            if self.training_config.peft_config is not None:
                final_engine_kwargs["enable_lora"] = True
                # Set reasonable defaults for LoRA if not already specified
                if "max_lora_rank" not in final_engine_kwargs:
                    lora_rank = getattr(self.training_config.peft_config, 'r', 16)  # Default to rank 16
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
        self, prompts: Sequence[str], sampling_params: SamplingParams | None = None, lora_request=DEFAULT_LORA
    ) -> List[RequestOutput]:
        sampling = sampling_params or SamplingParams()
        with self._with_wake():
            return self.pool.generate(self.name, list(prompts), sampling, lora_request)

    def chat(
        self, dialogs: Sequence[list], sampling_params: SamplingParams | None = None, lora_request=DEFAULT_LORA
    ) -> List[RequestOutput]:
        sampling = sampling_params or SamplingParams()
        with self._with_wake():
            return self.pool.chat(self.name, list(dialogs), sampling, lora_request)

    async def agenerate(
        self, prompts: Sequence[str], sampling_params: SamplingParams | None = None, lora_request=DEFAULT_LORA
    ) -> List[RequestOutput]:
        sampling = sampling_params or SamplingParams()
        with self._with_wake():
            return await self.pool.agenerate(self.name, list(prompts), sampling, lora_request)

    async def achat(
        self, dialogs: Sequence[list], sampling_params: SamplingParams | None = None, lora_request=DEFAULT_LORA
    ) -> List[RequestOutput]:
        sampling = sampling_params or SamplingParams()
        with self._with_wake():
            return await self.pool.achat(self.name, list(dialogs), sampling, lora_request)

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
