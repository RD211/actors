from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import set_seed
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from transformers import AutoModelForCausalLM
import bitsandbytes as bnb
from actors.utils.deepspeed import (
    offload_model_and_optimizer,
    prepare_deepspeed,
    reload_model_and_optimizer,
)
from accelerate.utils import DeepSpeedPlugin, DistributedType
import torch
from actors.utils.logger import Palette, colorize, init_logger
def main():
    logger = init_logger("t")
    ds_plugin = DeepSpeedPlugin(
        
    )
    cfg = ds_plugin.deepspeed_config
    cfg["max_grad_norm"] = 1
    cfg["train_batch_size"] = 2
    cfg["gradient_accumulation_steps"] = 1
    cfg["train_micro_batch_size_per_gpu"] = (
        1
    )
    accel = Accelerator(
        mixed_precision="bf16",
        deepspeed_plugin=ds_plugin,
    )

    logger.info(
        "Before loading model."
    )
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-3B-Instruct", trust_remote_code=True, torch_dtype=torch.bfloat16).train()
    for _ in range(5):
        torch.cuda.empty_cache()
    logger.info(
        "After loading model."
    )
    optimizer = bnb.optim.AdamW8bit(
        model.parameters(),
        lr=2e-6)
    logger.info(
        "After creating optimizer."
    )
    model, optimizer = accel.prepare(model, optimizer)

    logger.info(
        "After preparing model and optimizer."
    )

    offload_model_and_optimizer(
        model,
        optimizer,
        offload_optimizer=True,
        offload_model=False,
    )
    for _ in range(5):
        torch.cuda.empty_cache()
    logger.info(
        "After offloading optimizer."
    )

    offload_model_and_optimizer(
        model,
        optimizer,
        offload_optimizer=True,
        offload_model=True,
    )
    for _ in range(5):
        torch.cuda.empty_cache()
    logger.info(
        "After offloading model."
    )

if __name__ == "__main__":
    main()