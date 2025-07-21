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
from peft import get_peft_model, prepare_model_for_kbit_training
from peft import LoraConfig, TaskType
from transformers import BitsAndBytesConfig
from accelerate.utils import DeepSpeedPlugin, DistributedType
import torch
from actors.utils.logger import Palette, colorize, init_logger
import gc

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

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_quant_storage=torch.bfloat16,
    )

    # Create LoRA configuration
    lora_config = LoraConfig(
        r=256,  # LoRA rank
        lora_alpha=512,  # LoRA scaling parameter
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],  # Target all linear layers
        lora_dropout=0.0,  # LoRA dropout
        bias="none",  # Don't adapt bias parameters
        task_type=TaskType.CAUSAL_LM,  # Task type for causal language modeling
    )
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct", 
        trust_remote_code=True, torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        ).train()
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model, peft_config=lora_config
    ).train()
    for _ in range(5):
        torch.cuda.empty_cache()
    logger.info(
        "After loading model."
    )
    optimizer = bnb.optim.AdamW32bit(
        model.parameters(),
        lr=2e-6)
    logger.info(
        "After creating optimizer."
    )
    model, optimizer = accel.prepare(model, optimizer)

    logger.info(
        "After preparing model and optimizer."
    )
    accel.wait_for_everyone()

    info = offload_model_and_optimizer(
        model,
        optimizer,
        offload_optimizer=False,
        offload_model=True,
    )
    accel.wait_for_everyone()
    print(info)
    for _ in range(5):
        torch.cuda.empty_cache()
    logger.info(
        "After offloading optimizer."
    )

    info = offload_model_and_optimizer(
        model,
        optimizer,
        offload_optimizer=True,
        offload_model=False,
    )
    accel.wait_for_everyone()
    print(info)
    torch.cuda.synchronize()
    for _ in range(5):
        torch.cuda.empty_cache()
    logger.info(
        "After offloading model."
    )
    accel.wait_for_everyone()

    info = offload_model_and_optimizer(
        model,
        optimizer,
        offload_optimizer=True,
        offload_model=False,
    )
    print(torch.cuda.memory_summary())

    gc.collect()
    torch.cuda.empty_cache()          # returns inactive blocks to the driver
    model.optimizer.empty_partition_cache()   # DeepSpeed’s warm‑start stash
    logger.info(
        "After emptying partition cache."
    )

if __name__ == "__main__":
    main()