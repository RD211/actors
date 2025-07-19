import logging
from concurrent.futures import ThreadPoolExecutor

import deepspeed
import torch


def gather_and_stream_state_dict(
    accelerator,
    logger: logging.Logger,
    model,
    callback,
    batch_size: int = 300,
    tie_word_embeddings: bool = False,
    lora_only: bool = False,
):
    """
    Gathers the state dictionary from a model distributed with DeepSpeed ZeRO in batches,
    and calls a callback with each batch on the local main process.
    Tensors in the batch are on the GPU of the local main process.

    Parameters:
    -----------
    lora_only : bool
        If True, only gather LoRA adapter parameters (containing 'lora_A' or 'lora_B').
        If False, gather all model parameters.
    """

    # This will keep the tensor on its current device (GPU)
    def _copy_tensor(name_param):
        name, param = name_param
        return name, param.detach()

    # Get all parameters first
    all_params = list(model.named_parameters())

    if lora_only:
        # Filter to only LoRA parameters
        params = [
            (name, param)
            for name, param in all_params
            if "lora_A" in name or "lora_B" in name
        ]

        if not params:
            logger.warning(
                "No LoRA parameters found in model. Make sure this is a PEFT model with LoRA adapters."
            )
            return

        logger.info(
            f"Found {len(params)} LoRA parameters to sync (out of {len(all_params)} total parameters)"
        )
    else:
        # Use all parameters
        params = all_params

        # TODO: Test this for non-qwen models too.
        if tie_word_embeddings:
            params.append(
                (
                    "lm_head.weight",
                    [p[1] for p in model.named_parameters() if "embed" in p[0]][0],
                )
            )

    total = len(params)
    for start in range(0, total, batch_size):
        end = min(total, start + batch_size)
        batch = params[start:end]

        batch_params = [p for _, p in batch]

        # Gathers the full parameters on all processes, but we only process it on the local main process.
        with deepspeed.zero.GatheredParameters(
            batch_params, modifier_rank=None
        ):  # TODO: Check if we gather on all devices by accident.
            if accelerator.is_local_main_process:
                batch_state_dict = {}
                with ThreadPoolExecutor() as executor:
                    futures = executor.map(_copy_tensor, batch)
                    for name, tensor in futures:
                        batch_state_dict[name] = tensor

                # The model on accelerator might have a "module." prefix.
                if any(name.startswith("module.") for name in batch_state_dict):
                    batch_state_dict = {
                        name.replace("module.", "", 1): tensor
                        for name, tensor in batch_state_dict.items()
                    }

                callback(batch_state_dict)

        torch.cuda.empty_cache()

    accelerator.wait_for_everyone()
