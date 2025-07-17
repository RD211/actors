from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union, Iterable, TYPE_CHECKING
from enum import Enum, auto
import inspect
from transformers import PretrainedConfig, PreTrainedTokenizerBase, PreTrainedModel, PreTrainedTokenizer, AutoTokenizer, AutoModelForCausalLM
import accelerate
import torch
from torch import nn
from torch.optim import Optimizer
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler, CosineAnnealingLR

from actors.losses.base_loss import BaseRLLoss

if TYPE_CHECKING:
    from actors.losses import GRPOLoss, LigerGRPOLoss
    
from liger_kernel.transformers import AutoLigerKernelForCausalLM

from peft import PeftConfig

# ═══════════════════════════════════════════════════════════════════════
# Trainer configuration
# ═══════════════════════════════════════════════════════════════════════

class SaveStrategy(Enum):
    NONE = auto()  # never save
    STEPS = auto()  # checkpoint_every_n only
    FINAL = auto()  # one model save at the very end
    ALL = auto()  # both periodic + final


class EvalStrategy(Enum):
    NONE = auto()  # never evaluate
    STEPS = auto()  # evaluate every eval_every_n steps
    FINAL = auto()  # evaluate only at the end
    ALL = auto()  # evaluate both periodically and at the end

@dataclass
class TrainerCfg:

    # Training
    epochs: int = 1
    batch_size: int = 8
    max_steps: Optional[int] = None
    grad_accumulation_steps: int = 1
    num_iterations: int = 1
    group_size: int = 8
    

    # Logging
    log_every_n: int = 1
    use_wandb: bool = True

    # Eval
    eval_every_n: int = 1000
    eval_strategy: EvalStrategy = EvalStrategy.ALL

    # Checkpointing
    save_strategy: SaveStrategy = SaveStrategy.ALL
    checkpoint_every_n: int = 1000
    max_checkpoints_to_keep: int = 3
    checkpoint_path: str = "checkpoints"


# ═══════════════════════════════════════════════════════════════════════
# Actor configuration
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ActorTrainCfg:
    # Basic training parameters
    learning_rate: float = 1e-6
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True
    reference_batch_size: int = 4
    
    # Advantage calculation and normalization
    advantage_calculator: Optional[Callable[..., List[float]]] = None
    std_normalization: bool = True
    beta: float = 0.1
    loss_temp: float = 1.0
    
    # Model configuration
    use_liger_model: bool = True
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    quantization_config: Optional[Any] = None
    
    # Factory functions (private)
    _tokenizer_factory: Optional[Callable[[], PreTrainedTokenizer]] = field(default=None, repr=False, init=False)
    _model_factory: Optional[Callable[[], nn.Module]] = field(default=None, repr=False, init=False)
    _loss_factory: Callable[[], BaseRLLoss] = field(default=None, repr=False, init=False)
    _optim_factory: Optional[Callable[[Iterable[nn.Parameter]], Optimizer]] = field(default=None, repr=False, init=False)
    _scheduler_factory: Union[Callable[[Optimizer], LRScheduler], Callable[[Optimizer, Optional[int]], LRScheduler]] = field(default=lambda opt, steps: CosineAnnealingLR(opt, T_max=steps if steps else 1000), repr=False, init=False)
    _reference_model_factory: Optional[Callable[[], nn.Module]] = field(default=None, repr=False, init=False)
    
    # PEFT/LoRA configuration
    peft_config: Optional[PeftConfig] = None
    
    # Offloading parameters
    offload_optimizer: bool = False
    offload_model: bool = False
    
    def __init__(
        self,
        *,
        # Basic training parameters
        learning_rate: float = 1e-6,
        max_grad_norm: float = 1.0,
        gradient_checkpointing: bool = True,
        reference_batch_size: int = 4,
        
        # Advantage calculation and normalization
        advantage_calculator: Optional[Callable[..., List[float]]] = None,
        std_normalization: bool = True,
        beta: float = 0.1,
        loss_temp: float = 1.0,
        
        # Model configuration
        use_liger_model: bool = True,
        model_kwargs: Optional[Dict[str, Any]] = None,
        quantization_config: Optional[Any] = None,
        
        # Training components
        optimizer: Optional[Union[str, type, Callable]] = None,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        loss: Optional[Union[str, type, Callable]] = None,
        loss_kwargs: Optional[Dict[str, Any]] = None,
        scheduler: Optional[Union[str, type, Callable]] = None,
        scheduler_kwargs: Optional[Dict[str, Any]] = None,
        
        # Factory functions
        model_factory: Optional[Callable[[], nn.Module]] = None,
        tokenizer_factory: Optional[Callable[[], PreTrainedTokenizer]] = None,
        reference_model_factory: Optional[Callable[[], nn.Module]] = None,
        
        # PEFT/LoRA configuration
        peft_config: Optional[PeftConfig] = None,
        
        # Offloading parameters
        offload_optimizer: bool = False,
        offload_model: bool = False,
    ):
        """
        Initialize ActorTrainCfg with all configuration options.
        
        Args:
            learning_rate: Learning rate for training
            max_grad_norm: Maximum gradient norm for clipping
            gradient_checkpointing: Whether to use gradient checkpointing
            reference_batch_size: Batch size for reference model inference
            advantage_calculator: Optional function to calculate advantages
            std_normalization: Whether to apply standard normalization
            beta: Beta parameter for regularization/weighting
            loss_temp: Temperature parameter for loss function
            use_liger_model: Whether to use Liger kernel models
            model_kwargs: Additional kwargs for model initialization
            quantization_config: Quantization configuration for model loading
            optimizer: Optimizer class, string name, or factory function
            optimizer_kwargs: Additional arguments for optimizer
            loss: Loss class, string name, or factory function
            loss_kwargs: Additional arguments for loss
            scheduler: Scheduler class, string name, or factory function
            scheduler_kwargs: Additional arguments for scheduler
            model_factory: Factory function to create the model
            tokenizer_factory: Factory function to create the tokenizer
            reference_model_factory: Factory function to create reference model
            peft_config: PEFT configuration for LoRA/QLoRA training
            offload_optimizer: Whether to offload optimizer to CPU
            offload_model: Whether to offload model to CPU
        """
        # Set basic parameters
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.gradient_checkpointing = gradient_checkpointing
        self.reference_batch_size = reference_batch_size
        self.advantage_calculator = advantage_calculator
        self.std_normalization = std_normalization
        self.beta = beta
        self.loss_temp = loss_temp
        self.use_liger_model = use_liger_model
        self.model_kwargs = model_kwargs or {}
        self.quantization_config = quantization_config
        self.peft_config = peft_config
        self.offload_optimizer = offload_optimizer
        self.offload_model = offload_model
        
        # Set factories if provided, otherwise keep the dataclass defaults
        if model_factory is not None:
            self._model_factory = model_factory
        if tokenizer_factory is not None:
            self._tokenizer_factory = tokenizer_factory
        if reference_model_factory is not None:
            self._reference_model_factory = reference_model_factory
            
        # Configure optimizer
        if optimizer is not None:
            kwargs = optimizer_kwargs or {}
            if isinstance(optimizer, str):
                optimizer = self._get_optimizer_by_name(optimizer)
            self._optim_factory = self._as_factory(optimizer, **kwargs)
        
        # Configure loss
        if loss is not None:
            kwargs = loss_kwargs or {}
            if isinstance(loss, str):
                loss = self._get_loss_by_name(loss)
            self._loss_factory = self._as_factory(loss, **kwargs)
            
        # Configure scheduler
        if scheduler is not None:
            kwargs = scheduler_kwargs or {}
            if isinstance(scheduler, str):
                scheduler = self._get_scheduler_by_name(scheduler, **kwargs)
            self.set_scheduler(scheduler)
        
        # Call post_init for default setup
        if self._optim_factory is None:
            self._optim_factory = lambda p: optim.AdamW(p)
        if self._loss_factory is None:
            # Set default loss factory with loss_kwargs if provided
            kwargs = loss_kwargs or {}
            from actors.losses import LigerGRPOLoss
            self._loss_factory = lambda: LigerGRPOLoss(config=self, **kwargs)    


    def _as_factory(self, obj, **kwargs):
        if isinstance(obj, BaseRLLoss):
            return lambda: obj
        if inspect.isclass(obj):
            if issubclass(obj, Optimizer):
                return lambda p: obj(p, **kwargs)
            if issubclass(obj, BaseRLLoss):
                # Always pass config to loss functions
                return lambda: obj(config=self, **kwargs)
        if callable(obj):
            return obj
        raise TypeError(
            f"Expected a class or callable, got {type(obj)}. "
        )

    def create_default_factories(self, model_path: str):
        """
        Create default model and tokenizer factories based on model path.
        This should be called by the actor when it has a model_path.
        
        Args:
            model_path: Path to the model for creating factories
        """
        if self._model_factory is None:
            # Merge default kwargs with user-provided kwargs
            default_kwargs = {
                "trust_remote_code": True,
            }
            merged_kwargs = {**default_kwargs, **self.model_kwargs}
            
            # Add quantization config if provided
            if self.quantization_config is not None:
                merged_kwargs["quantization_config"] = self.quantization_config
            
            if self.use_liger_model:
                self._model_factory = lambda: AutoLigerKernelForCausalLM.from_pretrained(model_path, **merged_kwargs)
            else:
                self._model_factory = lambda: AutoModelForCausalLM.from_pretrained(model_path, **merged_kwargs)
        
        # When using PEFT, reference model factory should be None since we'll use adapter disabling
        if self._reference_model_factory is None and self.peft_config is None:
            self._reference_model_factory = self._model_factory
            
        if self._tokenizer_factory is None:
            # Extract tokenizer-specific kwargs (if any)
            tokenizer_kwargs = {k: v for k, v in self.model_kwargs.items() if k in ["trust_remote_code", "use_fast"]}
            if "trust_remote_code" not in tokenizer_kwargs:
                tokenizer_kwargs["trust_remote_code"] = True
            self._tokenizer_factory = lambda: AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)

    def set_learning_rate(self, lr: float):
        """Set the learning rate."""
        self.learning_rate = lr
        return self

    def set_loss(self, loss_obj, **kwargs):
        """Set the loss function."""
        self._loss_factory = self._as_factory(loss_obj, **kwargs)
        return self

    def set_optimizer(self, opt_obj=None, **kwargs):
        """Set the optimizer."""
        if opt_obj is None:
            opt_obj = optim.AdamW
        self._optim_factory = self._as_factory(opt_obj, **kwargs)
        return self

    def set_scheduler(self, factory):
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
    
    def set_peft_config(self, peft_config):
        """
        Set the PEFT configuration for LoRA/QLoRA training.
        
        Args:
            peft_config: PEFT configuration object
        """
        self.peft_config = peft_config
        return self

    @property
    def has_peft_config(self) -> bool:
        """Check if PEFT configuration is set."""
        return self.peft_config is not None

    def set_reference_model_factory(self, factory: Callable[[], nn.Module]):
        """Set the reference model factory."""
        self._reference_model_factory = factory
        return self

    def set_model_factory(self, factory: Callable[[], nn.Module]):
        """Set the main model factory."""
        self._model_factory = factory
        if self._reference_model_factory is None:
            self._reference_model_factory = factory
        return self

    def set_tokenizer_factory(self, factory: Callable[[], PreTrainedTokenizer]):
        """Set the tokenizer factory."""
        self._tokenizer_factory = factory
        return self

    def set_tokenizer(self, tok: PreTrainedTokenizer):
        """Set the tokenizer instance."""
        self._tokenizer_factory = lambda: tok
        return self

    def set_model_kwargs(self, **kwargs):
        """Set additional keyword arguments for model initialization."""
        self.model_kwargs.update(kwargs)
        return self

    def set_use_liger_model(self, use_liger: bool):
        """Set whether to use Liger kernel models."""
        self.use_liger_model = use_liger
        return self
    
    def set_quantization_config(self, quantization_config):
        """Set the quantization configuration for model loading."""
        self.quantization_config = quantization_config
        return self

    @property
    def model_factory(self):
        """Get the model factory."""
        return self._model_factory

    @property
    def tokenizer_factory(self):
        """Get the tokenizer factory."""
        return self._tokenizer_factory

    @property
    def loss_factory(self):
        """Get the loss factory."""
        return self._loss_factory

    @property
    def optim_factory(self):
        """Get the optimizer factory with learning rate applied."""
        prev_factory = self._optim_factory
        def _patched_factory(p):
            opt = prev_factory(p)
            for g in opt.param_groups:
                g["lr"] = self.learning_rate
            return opt
        return _patched_factory

    @property
    def scheduler_factory(self):
        """Get the scheduler factory."""
        return self._scheduler_factory

    @property
    def reference_model_factory(self):
        """Get the reference model factory."""
        return self._reference_model_factory

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
        # Import loss classes at runtime to avoid circular imports
        from actors.losses import GRPOLoss, LigerGRPOLoss
        
        losses = {
            'grpo': GRPOLoss,
            'liger_grpo': LigerGRPOLoss,
        }
            
        if name.lower() not in losses:
            available = ', '.join(losses.keys())
            raise ValueError(f"Unknown loss '{name}'. Available: {available}")
            
        return losses[name.lower()]
    
    def _get_scheduler_by_name(self, name: str, **kwargs):
        """Get scheduler factory by string name."""
        if name.lower() == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingLR
            T_max = kwargs.get('T_max', 1000)
            eta_min = kwargs.get('eta_min', 0)
            return lambda opt, steps: CosineAnnealingLR(opt, T_max=T_max if T_max else (steps if steps else 1000), eta_min=eta_min)
        elif name.lower() == 'linear':
            from torch.optim.lr_scheduler import LinearLR
            start_factor = kwargs.get('start_factor', 1.0)
            end_factor = kwargs.get('end_factor', 0.0)
            total_iters = kwargs.get('total_iters', 1000)
            return lambda opt, steps: LinearLR(opt, start_factor=start_factor, end_factor=end_factor, total_iters=total_iters if total_iters else (steps if steps else 1000))
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

# ═══════════════════════════════════════════════════════════════════════
# Initialized actor state
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ActorTrainState:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    loss_fn: BaseRLLoss
    optim: torch.optim.Optimizer
    accel: accelerate.Accelerator
    model_config: PretrainedConfig
    ref_model: Optional[PreTrainedModel] = None
    sched: Optional[torch.optim.lr_scheduler.LRScheduler] = None
