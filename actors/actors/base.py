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




class LLMActor(abc.ABC):
    def __init__(self, name: str, model_path: str | None = None):
        self.name = name
        self.model_path = model_path

    @abc.abstractmethod
    def generate(self, prompts: Sequence[str], **kwargs): ...
    @abc.abstractmethod
    def chat(self, dialogs: Sequence[list], **kwargs): ...

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

    def __post_init__(self):
        if self._model_factory is None:
            self._model_factory = lambda: AutoModelForCausalLM.from_pretrained(self.model_path, trust_remote_code=True)
        if self._optim_factory is None:
            self._optim_factory = lambda p: optim.AdamW(p)
        if self._reference_model_factory is None:
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
    def start_weight_update(self): ...
    @abc.abstractmethod
    def update_weights_batch(self, state_dict: Dict[str, torch.Tensor]): ...
    @abc.abstractmethod
    def finalize_weight_update(self): ...
    @abc.abstractmethod
    def sleep(self, level: int = 1) -> None: ...
    @abc.abstractmethod
    def wake(self) -> None: ...

    def __init__(self, name: str, model_path: str | None = None):
        super().__init__(name, model_path)
        self.training_config: TrainingConfig = TrainingConfig(model_path=model_path)