import inspect
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, fields
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

import accelerate
import torch
from torch import nn, optim
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerBase,
)

from actors.losses.base_loss import BaseRLLoss

if TYPE_CHECKING:
    pass

from liger_kernel.transformers import AutoLigerKernelForCausalLM
from peft.tuners.lora.config import LoraConfig

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
    max_steps: int | None = None
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

    def to_dict(self) -> dict[str, Any]:
        """Converts the config to a dictionary for logging."""
        result = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, Enum):
                result[f.name] = value.name
            else:
                result[f.name] = value
        return result


# ═══════════════════════════════════════════════════════════════════════
# Actor configuration
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class ActorTrainCfg:
    # Basic training parameters
    learning_rate: float = 1e-6
    max_grad_norm: float = 0.1
    gradient_checkpointing: bool = True
    reference_batch_size: int = -1  # -1 is full batch size

    # Advantage calculation and normalization
    advantage_calculator: Callable[..., list[float]] | None = None
    std_normalization: bool = True
    beta: float = 0.0
    loss: str = "liger_gspo"
    loss_temp: float = 1.0

    # Model configuration
    use_liger_model: bool = True
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    quantization_config: Any | None = None

    # Factory functions (private)
    _tokenizer_factory: Callable[[], PreTrainedTokenizer] | None = field(
        default=None, repr=False, init=False
    )
    _model_factory: Callable[[], nn.Module] | None = field(
        default=None, repr=False, init=False
    )
    _loss_factory: Callable[[], BaseRLLoss] = field(
        default=None, repr=False, init=False
    )
    _optim_factory: Callable[[Iterable[nn.Parameter]], Optimizer] | None = field(
        default=None, repr=False, init=False
    )
    _scheduler_factory: (
        Callable[[Optimizer], LRScheduler]
        | Callable[[Optimizer, int | None], LRScheduler]
        | None
    ) = field(
        default=None,
        repr=False,
        init=False,
    )
    _reference_model_factory: Callable[[], nn.Module] | None = field(
        default=None, repr=False, init=False
    )

    # PEFT/LoRA configuration
    peft_config: LoraConfig | None = None

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
        reference_batch_size: int = -1,
        # Advantage calculation and normalization
        advantage_calculator: Callable[..., list[float]] | None = None,
        std_normalization: bool = True,
        beta: float = 0.1,
        loss_temp: float = 1.0,
        # Model configuration
        use_liger_model: bool = True,
        model_kwargs: dict[str, Any] | None = None,
        quantization_config: Any | None = None,
        # Training components
        optimizer: str | type | Callable | None = None,
        optimizer_kwargs: dict[str, Any] | None = None,
        loss: str | type | Callable = "liger_gspo",
        loss_kwargs: dict[str, Any] | None = None,
        scheduler: str | type | Callable | None = "cosine",
        scheduler_kwargs: dict[str, Any] | None = None,
        # Factory functions
        model_factory: Callable[[], nn.Module] | None = None,
        tokenizer_factory: Callable[[], PreTrainedTokenizer] | None = None,
        reference_model_factory: Callable[[], nn.Module] | None = None,
        # PEFT/LoRA configuration
        peft_config: LoraConfig | None = None,
        # Offloading parameters
        offload_optimizer: bool = True,
        offload_model: bool = True,
        # Updating vllm weights
        update_weights_batch_size: int = 300,
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
            peft_config: PEFT configuration for LoRA/QLoRA training (only LoraConfig supported)
            offload_optimizer: Whether to offload optimizer to CPU
            offload_model: Whether to offload model to CPU
            update_weights_batch_size: Batch size for updating weights in vLLM (number of tensors)
        """
        # Validate PEFT config
        if peft_config is not None and not isinstance(peft_config, LoraConfig):
            raise ValueError(
                f"Only LoraConfig is supported for peft_config, got {type(peft_config)}. "
                f"Expected type: peft.tuners.lora.config.LoraConfig"
            )

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
        self.update_weights_batch_size = update_weights_batch_size

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
                self.loss = loss
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
        raise TypeError(f"Expected a class or callable, got {type(obj)}. ")

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
                "torch_dtype": torch.bfloat16,
            }
            merged_kwargs = {**default_kwargs, **self.model_kwargs}

            # Add quantization config if provided
            if self.quantization_config is not None:
                merged_kwargs["quantization_config"] = self.quantization_config

            if self.use_liger_model:
                self._model_factory = (
                    lambda: AutoLigerKernelForCausalLM.from_pretrained(
                        model_path, **merged_kwargs
                    )
                )
            else:
                self._model_factory = lambda: AutoModelForCausalLM.from_pretrained(
                    model_path, **merged_kwargs
                )

        # When using PEFT, reference model factory should be None since we'll use adapter disabling
        if self._reference_model_factory is None and self.peft_config is None:
            self._reference_model_factory = self._model_factory

        if self._tokenizer_factory is None:
            tokenizer_kwargs = {
                k: v
                for k, v in self.model_kwargs.items()
                if k in ["trust_remote_code", "use_fast"]
            }
            if "trust_remote_code" not in tokenizer_kwargs:
                tokenizer_kwargs["trust_remote_code"] = True

            def get_tokenizer():
                tok = AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)
                if tok.pad_token is None:
                    tok.pad_token = tok.eos_token
                return tok

            self._tokenizer_factory = get_tokenizer

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
                if "T_max" in params and total_steps is not None:
                    return factory(optimizer, T_max=total_steps)
                elif "total_iters" in params and total_steps is not None:
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
                raise ValueError(
                    f"Scheduler factory must accept 1 or 2 parameters, got {param_count}"
                )
        else:
            raise TypeError(f"Expected a class or callable, got {type(factory)}")
        return self

    def set_peft_config(self, peft_config):
        """
        Set the PEFT configuration for LoRA/QLoRA training.

        Args:
            peft_config: PEFT configuration object (only LoraConfig supported)
        """
        if peft_config is not None and not isinstance(peft_config, LoraConfig):
            raise ValueError(
                f"Only LoraConfig is supported for peft_config, got {type(peft_config)}. "
                f"Expected type: peft.tuners.lora.config.LoraConfig"
            )
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
            "adamw": optim.AdamW,
            "adam": optim.Adam,
            "sgd": optim.SGD,
            "rmsprop": optim.RMSprop,
        }

        try:
            import bitsandbytes as bnb

            optimizers.update(
                {
                    "paged_adamw_32bit": bnb.optim.PagedAdamW32bit,
                    "paged_adamw_8bit": bnb.optim.PagedAdamW8bit,
                    "adamw_32bit": bnb.optim.AdamW32bit,
                    "adamw_8bit": bnb.optim.AdamW8bit,
                }
            )
        except ImportError:
            pass

        if name.lower() not in optimizers:
            available = ", ".join(optimizers.keys())
            raise ValueError(f"Unknown optimizer '{name}'. Available: {available}")

        return optimizers[name.lower()]

    def _get_loss_by_name(self, name: str):
        """Get loss class by string name."""
        # Import loss classes at runtime to avoid circular imports
        from actors.losses import GRPOLoss, GSPOLoss, LigerGRPOLoss, LigerGSPOLoss

        losses = {
            "grpo": GRPOLoss,
            "liger_grpo": LigerGRPOLoss,
            "gspo": GSPOLoss,
            "liger_gspo": LigerGSPOLoss,
        }

        if name.lower() not in losses:
            available = ", ".join(losses.keys())
            raise ValueError(f"Unknown loss '{name}'. Available: {available}")

        return losses[name.lower()]

    def _get_scheduler_by_name(self, name: str, **kwargs):
        """Get scheduler factory by string name."""
        if name.lower() == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR

            T_max = kwargs.get("T_max", 1000)
            eta_min = kwargs.get("eta_min", 0)
            return lambda opt, steps: CosineAnnealingLR(
                opt,
                T_max=steps if steps is not None else T_max,
                eta_min=eta_min,
            )
        elif name.lower() == "linear":
            from torch.optim.lr_scheduler import LinearLR

            start_factor = kwargs.get("start_factor", 1.0)
            end_factor = kwargs.get("end_factor", 0.0)
            total_iters = kwargs.get("total_iters", 1000)
            return lambda opt, steps: LinearLR(
                opt,
                start_factor=start_factor,
                end_factor=end_factor,
                total_iters=total_iters if total_iters else (steps if steps else 1000),
            )
        elif name.lower() == "constant":
            from torch.optim.lr_scheduler import ConstantLR

            return lambda opt, steps: ConstantLR(opt, **kwargs)
        elif name.lower() == "exponential":
            from torch.optim.lr_scheduler import ExponentialLR

            gamma = kwargs.get("gamma", 0.95)
            return lambda opt, steps: ExponentialLR(opt, gamma=gamma)
        elif name.lower() == "step":
            from torch.optim.lr_scheduler import StepLR

            step_size = kwargs.get("step_size", 30)
            gamma = kwargs.get("gamma", 0.1)
            return lambda opt, steps: StepLR(opt, step_size=step_size, gamma=gamma)
        else:
            available = "cosine, linear, constant, exponential, step"
            raise ValueError(f"Unknown scheduler '{name}'. Available: {available}")

    def to_dict(self, model_path: str | None = None) -> dict[str, Any]:
        """
        Converts the config to a dictionary for logging.
        Handles functions, classes, and complex objects intelligently.

        Args:
            model_path: Model path from the actor (optional)
        """
        result = {}

        if model_path is not None:
            result["model_path"] = model_path

        for f in fields(self):
            value = getattr(self, f.name)

            # Skip None values for private fields to reduce noise
            if f.name.startswith("_") and value is None:
                continue

            # For private fields, use a cleaner name (remove leading underscore)
            field_name = f.name
            serialized_value = self._serialize_value(value)

            if not f.name.startswith("_"):
                result[field_name] = serialized_value

        return result

    def _serialize_value(self, value: Any) -> Any:
        """
        Serialize a value for logging, handling different types.
        """
        if value is None:
            return None
        elif isinstance(value, str | int | float | bool):
            return value
        elif isinstance(value, Enum):
            return value.name
        elif isinstance(value, list | tuple):
            return [self._serialize_value(item) for item in value]
        elif isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        elif isinstance(value, LoraConfig):
            return {
                "r": value.r,
                "lora_alpha": value.lora_alpha,
                "target_modules": value.target_modules,
                "lora_dropout": value.lora_dropout,
                "bias": value.bias,
                "task_type": (
                    value.task_type.value
                    if hasattr(value.task_type, "value")
                    else str(value.task_type)
                ),
            }
        elif inspect.isclass(value):
            return value.__name__
        elif callable(value):
            if hasattr(value, "__name__"):
                name = value.__name__
                if name == "<lambda>":
                    try:
                        source = inspect.getsource(value).strip()
                        return f"{source}"
                    except (OSError, TypeError):
                        return "lambda: <source_unavailable>"
                else:
                    return name
            else:
                return str(type(value).__name__)
        elif hasattr(value, "__dict__"):
            try:
                obj_dict = {}
                for attr_name in dir(value):
                    if not attr_name.startswith("_"):
                        try:
                            attr_value = getattr(value, attr_name)
                            if not callable(attr_value):
                                obj_dict[attr_name] = self._serialize_value(attr_value)
                        except:
                            continue
                return {"type": type(value).__name__, "attributes": obj_dict}
            except:
                return f"<{type(value).__name__}>"
        else:
            return str(value)


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
    ref_model: PreTrainedModel | None = None
    sched: torch.optim.lr_scheduler.LRScheduler | None = None
