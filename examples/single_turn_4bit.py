# ═══════════════════════════════════════════════════════════════
# 4-bit QLoRA training on GSM8K
# - Rewards and setup inspired by Unsloth
# ═══════════════════════════════════════════════════════════════

import os
import re

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import BitsAndBytesConfig
from vllm import SamplingParams

from actors import (
    ActorTrainCfg,
    EvalStrategy,
    GRPOTrainer,
    GRPOTrainerCfg,
    SaveStrategy,
    SingleTurnEnvironment,
    vLLMActor,
)
from actors.rewards.base_completion_reward import reward_function

# ═══════════════════════════════════════════════════════════════
# Rewards
# ═══════════════════════════════════════════════════════════════


@reward_function(name="correctness_reward", weight=1.0)
def reward(completion, answer):
    try:
        found_answer = completion.split("<answer>")[1].split("</answer>")[0].strip()
        answer = int(answer.split("### ")[1].strip().replace(",", ""))
        if str(answer).strip() == found_answer:
            return 1.0
        return 0.0
    except:
        return 0.0


@reward_function(name="xml_reward", weight=1.0)
def count_xml(completion) -> float:
    count = 0.0
    if completion.count("<reasoning>\n") == 1:
        count += 0.125
    if completion.count("\n</reasoning>\n") == 1:
        count += 0.125
    if completion.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(completion.split("\n</answer>\n")[-1]) * 0.001
    if completion.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(completion.split("\n</answer>")[-1]) - 1) * 0.001
    return count


@reward_function(name="format_reward", weight=1.0)
def strict_format_reward_func(completion) -> list[float]:
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    return 0.5 if re.match(pattern, completion) else 0.0


# ═══════════════════════════════════════════════════════════════
# Training config
# ═══════════════════════════════════════════════════════════════

# Create quantization configuration
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_quant_storage=torch.bfloat16,
)

# Create LoRA configuration
lora_config = LoraConfig(
    r=64,
    lora_alpha=64,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.0,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# Create training configuration
training_config = ActorTrainCfg(
    learning_rate=5e-6,
    optimizer="adamw_8bit",
    loss="liger_gspo",
    peft_config=lora_config,
    quantization_config=quantization_config,
    offload_model=True,
    offload_optimizer=True,
    beta=0.04,
    max_grad_norm=0.1,
)

# ═══════════════════════════════════════════════════════════════
# Actor
# ═══════════════════════════════════════════════════════════════
actor = vLLMActor(
    name="main",
    model_path="Qwen/Qwen2.5-3B-Instruct",
    engine_kwargs={
        "gpu_memory_utilization": 0.7,
        "max_model_len": 2048,
        "quantization": "bitsandbytes",
    },
    training_config=training_config,
)

tokenizer = actor.tokenizer

# ═══════════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════════

system_prompt = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

# Prepare training data
ds = load_dataset("openai/gsm8k", "main")
ds = ds.map(
    lambda x: {
        "conversation": tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": x["question"]},
            ],
            tokenize=False,
            add_generation_prompt=True,
        ),
    },
)

# ═══════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════


def main():
    # Create environment
    env = SingleTurnEnvironment(
        actor=actor,
        train_data=ds,
        reward_functions=[reward, strict_format_reward_func, count_xml],
        sampling_params=SamplingParams(
            temperature=1.0,
            max_tokens=200,
        ),
        prompt_column="conversation",
        mask_prompt_for_loss=True,
    )

    # Create trainer configuration
    cfg = GRPOTrainerCfg(
        group_size=8,
        batch_size=16,  # Unintuitive but this is the global batch size.
        grad_accumulation_steps=1,
        num_iterations=2,
        log_every_n=1,
        eval_every_n=None,  # No periodic evaluation
        eval_strategy=EvalStrategy.NONE,  # No evaluation
        checkpoint_every_n=30,
        save_strategy=SaveStrategy.ALL,
    )

    # Create trainer with environment and actors
    trainer = GRPOTrainer(cfg=cfg, env=env, actors=[actor])

    import wandb

    if os.getenv("RANK") == "0":
        wandb.init(project="actors", name="3b-qlora")
    trainer.train()


if __name__ == "__main__":
    main()
