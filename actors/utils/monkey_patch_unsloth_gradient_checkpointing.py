# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Taken from unsloth.
import inspect

import torch
import transformers


class Unsloth_Offloaded_Gradient_Checkpointer(torch.autograd.Function):
    """
    Saves VRAM by smartly offloading to RAM.
    Tiny hit to performance, since we mask the movement via non blocking calls.
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, forward_function, hidden_states, *args):
        saved_hidden_states = hidden_states.to("cpu", non_blocking=True)
        with torch.no_grad():
            outputs = forward_function(
                hidden_states, *args
            )  # may be Tensor or tuple/list

        ctx.save_for_backward(saved_hidden_states)
        ctx.forward_function = forward_function
        ctx.args = args

        # Important: return exactly what the forward returned
        return outputs

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, *grad_outputs):
        # grad_outputs matches the number of outputs returned by forward()
        # We only need the gradient for the first output tensor.
        dY = grad_outputs[0] if len(grad_outputs) > 0 else None

        (hidden_states_cpu,) = ctx.saved_tensors
        hidden_states = hidden_states_cpu.to("cuda", non_blocking=True).detach()
        hidden_states.requires_grad = True

        with torch.enable_grad():
            outputs = ctx.forward_function(hidden_states, *ctx.args)
            if isinstance(outputs, (tuple | list)):
                primary = outputs[0]
            else:
                primary = outputs

        torch.autograd.backward(primary, dY)

        # Return grads for each input to forward(): (forward_function, hidden_states, *args)
        return (None, hidden_states.grad) + (None,) * len(ctx.args)


def new_gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
    if not self.supports_gradient_checkpointing:
        raise ValueError(
            f"{self.__class__.__name__} does not support gradient checkpointing."
        )

    gradient_checkpointing_func = Unsloth_Offloaded_Gradient_Checkpointer.apply

    _is_using_old_format = (
        "value" in inspect.signature(self._set_gradient_checkpointing).parameters
    )
    if not _is_using_old_format:
        self._set_gradient_checkpointing(
            enable=True, gradient_checkpointing_func=gradient_checkpointing_func
        )
    else:
        raise NotImplementedError()

    if getattr(self, "_hf_peft_config_loaded", False):
        self.enable_input_require_grads()


previous_method = (
    transformers.modeling_utils.PreTrainedModel.gradient_checkpointing_enable
)


def apply_unsloth_offloaded_gradient_checkpoint_monkey_patch():
    transformers.modeling_utils.PreTrainedModel.gradient_checkpointing_enable = (
        new_gradient_checkpointing_enable
    )


def revert_unsloth_offloaded_gradient_checkpoint_monkey_patch():
    transformers.modeling_utils.PreTrainedModel.gradient_checkpointing_enable = (
        previous_method
    )
