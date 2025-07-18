import gc
from typing import Any

import torch
from torch import nn


def disable_dropout_in_model(model: torch.nn.Module) -> None:
    # taken from: https://github.com/huggingface/trl/blob/main/trl/trainer/utils.py#L847
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0


def free_memory() -> None:
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()


# from: https://github.com/huggingface/trl/blob/main/trl/models/utils.py#L376
class _ForwardRedirection:
    """Implements the `forward-redirection`.

    Taken from Pytorch-lightning:
    https://github.com/Lightning-AI/pytorch-lightning/blob/02311d03fb982560246eead7c08104481fac9579/src/lightning/pytorch/strategies/strategy.py#L602

    A method call to a wrapped module gets rerouted through the wrapper's `forward` method instead.

    """

    def __call__(
        self,
        wrapper_module: nn.Module,
        original_module: nn.Module,
        method: callable,
        *args: Any,
        **kwargs: Any,
    ):
        """Reroutes a method call through the `wrapper_module`'s `forward` method.

        Args:
            wrapper_module: The module that has `original_module` wrapped.
            original_module: The module that was wrapped inside `wrapper_module`.
            method_name: The name of the method that should be called on the `original_module` after inputs get
                redirected through the `wrapper_module`'s `forward` method.
            *args: The positional arguments to the method `method_name`. They will get passed to a patched
                `forward` method instead.
            **kwargs: The keyword arguments to the method `method_name`. They will get passed to a patched
                `forward` method instead.

        """
        original_forward = original_module.forward

        def wrapped_forward(*_args: Any, **_kwargs: Any) -> Any:
            # Unpatch ourselves immediately before calling the method `method_name`
            # because itself may want to call the real `forward`
            original_module.forward = original_forward  # type: ignore[method-assign]
            # Call the actual method e.g. `.training_step(...)`
            out = method(*_args, **_kwargs)
            self.on_after_inner_forward(wrapper_module, original_module)
            return out

        # Patch the original_module's forward so we can redirect the arguments back to the real method
        original_module.forward = wrapped_forward  # type: ignore[method-assign]

        wrapper_output = wrapper_module(*args, **kwargs)
        self.on_after_outer_forward(wrapper_module, original_module)
        return wrapper_output

    def on_after_inner_forward(
        self, wrapper_module: nn.Module, original_module: nn.Module
    ) -> None:
        pass

    def on_after_outer_forward(
        self, wrapper_module: nn.Module, original_module: nn.Module
    ) -> None:
        pass
