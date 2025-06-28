from __future__ import annotations
import abc
from dataclasses import dataclass, field
import inspect
from shutil import copy
from typing import Callable, Dict, Iterable, Optional, Sequence, Union
import torch
from torch import nn
from torch.optim import Optimizer
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler
from transformers import PreTrainedTokenizer, AutoTokenizer, AutoModelForCausalLM
from actors.losses import GRPOLoss, BaseRLLoss
from liger_kernel.transformers import AutoLigerKernelForCausalLM

# Import PEFT 
from peft import PeftConfig, get_peft_model



class LLMActor(abc.ABC):
    def __init__(self, name: str, model_path: str | None = None):
        self.name = name
        self.model_path = model_path

    @abc.abstractmethod
    def generate(self, prompts: Sequence[str], **kwargs): ...
    @abc.abstractmethod
    def chat(self, dialogs: Sequence[list], **kwargs): ...
    @abc.abstractmethod
    async def agenerate(self, prompts: Sequence[str], **kwargs): ...
    @abc.abstractmethod
    async def achat(self, dialogs: Sequence[list], **kwargs): ...

@dataclass
class TrainingConfig:
    model_path: str
    lr: float = 5e-6
    _tokenizer_factory: Optional[Callable[[], PreTrainedTokenizer]] = field(default=None, repr=False, init=False)
    _model_factory: Optional[Callable[[], nn.Module]] = field(default=None, repr=False, init=False)
    _loss_factory: Callable[[], BaseRLLoss] = field(default_factory=lambda: GRPOLoss(), repr=False, init=False)
    _optim_factory: Optional[Callable[[Iterable[nn.Parameter]], Optimizer]] = field(default=None, repr=False, init=False)
    _scheduler_factory: Union[Callable[[Optimizer], LRScheduler], Callable[[Optimizer, Optional[int]], LRScheduler]] = field(default=lambda opt, steps: opt, repr=False, init=False)
    _reference_model_factory: Optional[Callable[[], nn.Module]] = field(default=None, repr=False, init=False)
    # PEFT/LoRA configuration
    _peft_config: Optional[PeftConfig] = field(default=None, repr=False, init=False)

    def __post_init__(self):
        if self._model_factory is None:
            self._model_factory = lambda: AutoLigerKernelForCausalLM.from_pretrained(self.model_path, trust_remote_code=True, attn_implementation="flash_attention_2")
        if self._optim_factory is None:
            self._optim_factory = lambda p: optim.AdamW(p)
        # When using PEFT, reference model factory should be None since we'll use adapter disabling
        if self._reference_model_factory is None and self._peft_config is None:
            self._reference_model_factory = self._model_factory
        if self._tokenizer_factory is None:
            self._tokenizer_factory = lambda: AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

    def _as_factory(self, obj, **kwargs):
        if isinstance(obj, BaseRLLoss):
            return lambda: obj
        if inspect.isclass(obj):
            if issubclass(obj, Optimizer):
                return lambda p: obj(p, **kwargs)
            if issubclass(obj, BaseRLLoss):
                return lambda: obj(**kwargs)
        if callable(obj):
            return obj
        raise TypeError(
            f"Expected a class or callable, got {type(obj)}. "
        )

    def learning_rate(self, lr: float):
        self.lr = lr
        return self

    def loss(self, loss_obj, **kwargs):
        self._loss_factory = self._as_factory(loss_obj, **kwargs)
        return self

    def optimizer(self, opt_obj=None, **kwargs):
        if opt_obj is None:
            opt_obj = optim.AdamW
        self._optim_factory = self._as_factory(opt_obj, **kwargs)
        return self

    def scheduler(self, factory):
        """
        Set the scheduler factory. Can accept:
        1. A scheduler class (like CosineAnnealingLR) - will auto-pass total_steps as T_max
        2. A lambda with 1 param: lambda opt: SomeScheduler(opt)
        3. A lambda with 2 params: lambda opt, total_steps: SomeScheduler(opt, T_max=total_steps)
        """
        if inspect.isclass(factory):
            # Handle scheduler classes directly
            def class_factory(optimizer, total_steps):
                sig = inspect.signature(factory.__init__)
                params = list(sig.parameters.keys())[2:]  # Skip self, optimizer
                if 'T_max' in params and total_steps is not None:
                    return factory(optimizer, T_max=total_steps)
                elif 'total_iters' in params and total_steps is not None:
                    return factory(optimizer, total_iters=total_steps)
                else:
                    return factory(optimizer)
            self._scheduler_factory = class_factory
        elif callable(factory):
            # Handle lambda functions
            sig = inspect.signature(factory)
            param_count = len(sig.parameters)
            if param_count == 1:
                # Lambda with just optimizer
                self._scheduler_factory = lambda opt, steps: factory(opt)
            elif param_count == 2:
                # Lambda with optimizer and total_steps
                self._scheduler_factory = factory
            else:
                raise ValueError(f"Scheduler factory must accept 1 or 2 parameters, got {param_count}")
        else:
            raise TypeError(f"Expected a class or callable, got {type(factory)}")
        return self
    
    @staticmethod
    def cosine_scheduler(T_max: Optional[int] = None, eta_min: float = 0, **kwargs):
        from torch.optim.lr_scheduler import CosineAnnealingLR
        def factory(optimizer, total_steps):
            effective_T_max = T_max if T_max is not None else (total_steps if total_steps else 1000)
            return CosineAnnealingLR(optimizer, T_max=effective_T_max, eta_min=eta_min, **kwargs)
        return factory
    
    @staticmethod  
    def linear_scheduler(start_factor: float = 1.0, end_factor: float = 0.0, total_iters: Optional[int] = None):
        from torch.optim.lr_scheduler import LinearLR
        def factory(optimizer, total_steps):
            effective_total_iters = total_iters if total_iters is not None else (total_steps if total_steps else 1000)
            return LinearLR(optimizer, start_factor=start_factor, end_factor=end_factor, total_iters=effective_total_iters)
        return factory

    def peft(self, peft_config):
        """
        Set the PEFT configuration for LoRA/QLoRA training.
        
        Args:
            peft_config: PEFT configuration object
        """
        self._peft_config = peft_config
        return self

    @property
    def peft_config(self):
        """Get the PEFT configuration."""
        return self._peft_config

    def reference(self, factory: Callable[[], nn.Module]):
        self._reference_model_factory = factory
        return self

    def model(self, factory: Callable[[], nn.Module]):
        self._model_factory = factory
        if self._reference_model_factory is None:
            self._reference_model_factory = factory
        return self

    def tokenizer(self, tok: PreTrainedTokenizer):
        self._tokenizer_factory = lambda: tok
        return self

    @property
    def model_factory(self):
        return self._model_factory

    @property
    def tokenizer_factory(self):
        return self._tokenizer_factory

    @property
    def loss_factory(self):
        return self._loss_factory

    @property
    def optim_factory(self):
        prev_factory = self._optim_factory
        def _patched_factory(p):
            opt = prev_factory(p)
            for g in opt.param_groups:
                g["lr"] = self.lr
            return opt
        return _patched_factory

    @property
    def scheduler_factory(self):
        return self._scheduler_factory

    @property
    def reference_model_factory(self):
        return self._reference_model_factory

class TrainableLLMActor(LLMActor):
    @abc.abstractmethod
    def sleep(self, level: int = 1) -> None: ...
    @abc.abstractmethod
    def wake(self) -> None: ...
   
    @abc.abstractmethod
    def start_weight_update(self): ...
    @abc.abstractmethod
    def update_weights_batch(self, state_dict: Dict[str, torch.Tensor]): ...
    @abc.abstractmethod
    def finalize_weight_update(self): ...
    

    # ═══════════════════════════════════════════════════════════════
    # LoRA/PEFT Support Methods
    # ═══════════════════════════════════════════════════════════════
    @abc.abstractmethod
    def update_lora_weights(self): ...
    @abc.abstractmethod
    def create_lora_if_not_present(self, lora_path: str): ...

    def __init__(
        self, 
        name: str, 
        model_path: str | None = None,
        *,
        learning_rate: float = 5e-6,
        optimizer: str | type | Callable = "adamw",
        optimizer_kwargs: Dict | None = None,
        loss: str | type | Callable = "grpo",
        loss_kwargs: Dict | None = None,
        scheduler: str | type | Callable = "cosine",
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
        """
        Initialize a trainable LLM actor with configuration options.
        
        Args:
            name: Actor name
            model_path: Path to the model
            learning_rate: Learning rate for training (default: 5e-6)
            optimizer: Optimizer class, string name, or factory. Options: 'adamw', 'adam', 'sgd', 'rmsprop', 'paged_adamw', 'paged_adamw_8bit', 'adamw_8bit', 'adam_8bit'
            optimizer_kwargs: Additional arguments for optimizer
            loss: Loss class, string name, or factory. Options: 'grpo', 'liger_grpo'
            loss_kwargs: Additional arguments for loss
            scheduler: Scheduler class, string name, or factory. Options: 'cosine', 'linear', 'constant', 'exponential', 'step'
            scheduler_kwargs: Additional arguments for scheduler
            model_factory: Factory function to create the main model (default: AutoModelForCausalLM.from_pretrained)
            reference_model_factory: Factory function to create reference model (default: same as main model)
            peft_config: PEFT configuration for LoRA/QLoRA training (default: None)
            offload_optimizer: Whether to offload optimizer states to CPU (default: False)
            offload_model: Whether to offload model parameters to CPU (default: False)
            offload_reference_to_cpu: Whether to use aggressive CPU offloading for reference model (default: False)
            offload_activations_to_cpu: Whether to offload activations to CPU during training (default: False)
        """
        super().__init__(name, model_path)
        self.training_config: TrainingConfig = TrainingConfig(model_path=model_path)
        
        # Configure PEFT if provided
        if peft_config is not None:
            self.training_config.peft(peft_config)
        
        # Store offloading configuration
        self.offload_optimizer = offload_optimizer
        self.offload_model = offload_model
        self.offload_reference_to_cpu = offload_reference_to_cpu
        self.offload_activations_to_cpu = offload_activations_to_cpu
        
        # Configure training with provided parameters
        self.configure_training(
            learning_rate=learning_rate,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            loss=loss,
            loss_kwargs=loss_kwargs,
            scheduler=scheduler,
            scheduler_kwargs=scheduler_kwargs,
        )
        
        # Set model factory if provided
        if model_factory is not None:
            self.training_config.model(model_factory)
        
        # Set reference model factory if provided
        if reference_model_factory is not None:
            self.training_config.reference(reference_model_factory)
    
    # ═══════════════════════════════════════════════════════════════
    # Fluent Configuration API
    # ═══════════════════════════════════════════════════════════════
    
    def configure_training(
        self,
        *,
        learning_rate: float | None = None,
        optimizer: str | type | Callable | None = None,
        optimizer_kwargs: Dict | None = None,
        loss: str | type | Callable | None = None,
        loss_kwargs: Dict | None = None,
        scheduler: str | type | Callable | None = None,
        scheduler_kwargs: Dict | None = None,
    ) -> "TrainableLLMActor":
        """
        Configure training parameters with a single method call.
        
        Args:
            learning_rate: Learning rate for training
            optimizer: Optimizer class, string name, or factory function
            optimizer_kwargs: Additional arguments for optimizer
            loss: Loss class, string name, or factory function  
            loss_kwargs: Additional arguments for loss
            scheduler: Scheduler class, string name, or factory function
            scheduler_kwargs: Additional arguments for scheduler
            
        Returns:
            Self for method chaining
        """
        if learning_rate is not None:
            self.training_config.learning_rate(learning_rate)
            
        if optimizer is not None:
            kwargs = optimizer_kwargs or {}
            if isinstance(optimizer, str):
                optimizer = self._get_optimizer_by_name(optimizer)
            self.training_config.optimizer(optimizer, **kwargs)
            
        if loss is not None:
            kwargs = loss_kwargs or {}
            if isinstance(loss, str):
                loss = self._get_loss_by_name(loss)
            self.training_config.loss(loss, **kwargs)
            
        if scheduler is not None:
            kwargs = scheduler_kwargs or {}
            if isinstance(scheduler, str):
                scheduler = self._get_scheduler_by_name(scheduler, **kwargs)
            self.training_config.scheduler(scheduler)
            
        return self
    
    def set_learning_rate(self, lr: float) -> "TrainableLLMActor":
        """Set the learning rate."""
        self.training_config.learning_rate(lr)
        return self
        
    def set_optimizer(self, optimizer, **kwargs) -> "TrainableLLMActor":
        """Set the optimizer."""
        if isinstance(optimizer, str):
            optimizer = self._get_optimizer_by_name(optimizer)
        self.training_config.optimizer(optimizer, **kwargs)
        return self
        
    def set_loss(self, loss, **kwargs) -> "TrainableLLMActor":
        """Set the loss function."""
        if isinstance(loss, str):
            loss = self._get_loss_by_name(loss)
        self.training_config.loss(loss, **kwargs)
        return self
        
    def set_scheduler(self, scheduler, **kwargs) -> "TrainableLLMActor":
        """Set the learning rate scheduler."""
        if isinstance(scheduler, str):
            scheduler = self._get_scheduler_by_name(scheduler, **kwargs)
        self.training_config.scheduler(scheduler)
        return self
    
    def set_peft_config(self, peft_config) -> "TrainableLLMActor":
        """Set the PEFT configuration for LoRA/QLoRA training."""
        self.training_config.peft(peft_config)
        return self
    
    def set_reference_model(self, factory: Callable[[], nn.Module]) -> "TrainableLLMActor":
        """Set the reference model factory."""
        self.training_config.reference(factory)
        return self
    
    def set_model(self, factory: Callable[[], nn.Module]) -> "TrainableLLMActor":
        """Set the main model factory."""
        self.training_config.model(factory)
        return self
    
    # ═══════════════════════════════════════════════════════════════
    # Convenience Properties and Methods
    # ═══════════════════════════════════════════════════════════════
    
    @property
    def has_peft_config(self) -> bool:
        """Check if PEFT configuration is set."""
        return self.training_config.peft_config is not None
        
    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        """Get the tokenizer instance."""
        return self.training_config.tokenizer_factory()
    
    @property
    def model_factory(self) -> Callable[[], nn.Module]:
        """Get the model factory function."""
        return self.training_config.model_factory
        
    @property
    def current_learning_rate(self) -> float:
        """Get the current learning rate."""
        return self.training_config.lr
        
    def get_training_summary(self) -> Dict:
        """Get a summary of current training configuration."""
        return {
            'learning_rate': self.training_config.lr,
            'model_path': self.training_config.model_path,
            'loss_type': type(self.training_config.loss_factory()).__name__,
            'optimizer_type': type(self.training_config.optim_factory([])).__name__,
        }
    
    # ═══════════════════════════════════════════════════════════════
    # Helper Methods
    # ═══════════════════════════════════════════════════════════════
    
    def _get_optimizer_by_name(self, name: str):
        """Get optimizer class by string name."""
        optimizers = {
            'adamw': optim.AdamW,
            'adam': optim.Adam,
            'sgd': optim.SGD,
            'rmsprop': optim.RMSprop,
        }
        
        # Try to import and add bitsandbytes optimizers if available
        try:
            import bitsandbytes as bnb
            optimizers.update({
                'paged_adamw_32bit': bnb.optim.PagedAdamW32bit,
                'paged_adamw_8bit': bnb.optim.PagedAdamW8bit,
                'adamw_32bit': bnb.optim.AdamW32bit,
                'adamw_8bit': bnb.optim.AdamW8bit,
            })
        except ImportError:
            pass
            
        if name.lower() not in optimizers:
            available = ', '.join(optimizers.keys())
            raise ValueError(f"Unknown optimizer '{name}'. Available: {available}")
            
        return optimizers[name.lower()]
    
    def _get_loss_by_name(self, name: str):
        """Get loss class by string name."""
        from actors.losses import GRPOLoss
        
        losses = {
            'grpo': GRPOLoss,
        }
        
        # Try to import LigerLoss if available
        try:
            from actors.losses.liger_grpo_loss import LigerLoss
            losses['liger_grpo'] = LigerLoss
        except ImportError:
            pass
            
        if name.lower() not in losses:
            available = ', '.join(losses.keys())
            raise ValueError(f"Unknown loss '{name}'. Available: {available}")
            
        return losses[name.lower()]
    
    def _get_scheduler_by_name(self, name: str, **kwargs):
        """Get scheduler factory by string name."""
        if name.lower() == 'cosine':
            return self.training_config.cosine_scheduler(**kwargs)
        elif name.lower() == 'linear':
            return self.training_config.linear_scheduler(**kwargs)
        elif name.lower() == 'constant':
            from torch.optim.lr_scheduler import ConstantLR
            return lambda opt, steps: ConstantLR(opt, **kwargs)
        elif name.lower() == 'exponential':
            from torch.optim.lr_scheduler import ExponentialLR
            gamma = kwargs.get('gamma', 0.95)
            return lambda opt, steps: ExponentialLR(opt, gamma=gamma)
        elif name.lower() == 'step':
            from torch.optim.lr_scheduler import StepLR
            step_size = kwargs.get('step_size', 30)
            gamma = kwargs.get('gamma', 0.1)
            return lambda opt, steps: StepLR(opt, step_size=step_size, gamma=gamma)
        else:
            available = 'cosine, linear, constant, exponential, step'
            raise ValueError(f"Unknown scheduler '{name}'. Available: {available}")