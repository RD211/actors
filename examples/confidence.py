import os
import re

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import BitsAndBytesConfig
from vllm import SamplingParams

from actors import (
    ActorTrainCfg,
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


@reward_function(name="conf_reward", weight=1.0)
def reward(completion, answer):
    try:
        found_answer = completion.split("<answer>")[1].split("</answer>")[0].strip()
        found_confidence = (
            completion.split("<confidence>")[1].split("</confidence>")[0].strip()
        )
        if found_confidence not in [str(i) for i in range(1, 11)]:
            return -1.0
        found_confidence = int(found_confidence) / 10.0
        answer = int(answer.split("### ")[1].strip().replace(",", ""))
        if str(answer).strip() == found_answer:
            return torch.log(
                torch.max(torch.tensor(found_confidence), torch.tensor(0.001))
            ).item()
        return torch.log(
            torch.max(1 - torch.tensor(found_confidence), torch.tensor(0.001))
        ).item()
    except:
        return -1.0


@reward_function(name="accuracy_reward", weight=1.0)
def accuracy_reward(completion, answer):
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
    if completion.count("\n<confidence>\n") == 1:
        count += 0.125
        count -= len(completion.split("\n</confidence>\n")[-1]) * 0.001
    if completion.count("\n</confidence>") == 1:
        count += 0.125
        count -= (len(completion.split("\n</confidence>")[-1]) - 1) * 0.001
    return count


@reward_function(name="format_reward", weight=1.0)
def strict_format_reward_func(completion) -> list[float]:
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n<confidence>\n.*?\n</confidence>\n$"
    return 0.5 if re.match(pattern, completion) else 0.0


@reward_function(name="ece_metric", weight=0.0, batched=True)
def ece_metric(completion, answer):
    completions = completion
    answers = answer
    buckets = {i: {"correct": 0, "total": 0} for i in range(1, 11)}
    for completion, answer in zip(completions, answers, strict=False):
        try:
            found_answer = completion.split("<answer>")[1].split("</answer>")[0].strip()
            found_confidence = (
                completion.split("<confidence>")[1].split("</confidence>")[0].strip()
            )
            if found_confidence not in [str(i) for i in range(1, 11)]:
                continue
            found_confidence = int(found_confidence)
            answer = int(answer.split("### ")[1].strip().replace(",", ""))
            buckets[found_confidence]["total"] += 1
            if str(answer).strip() == found_answer:
                buckets[found_confidence]["correct"] += 1
        except:
            continue
    ece = 0.0
    total = sum(buckets[i]["total"] for i in range(1, 11))
    if total == 0:
        return [0.0] * len(completions)
    for i in range(1, 11):
        if buckets[i]["total"] > 0:
            acc = buckets[i]["correct"] / buckets[i]["total"]
            conf = i / 10.0
            ece += (buckets[i]["total"] / total) * abs(acc - conf)
    return [ece] * len(completions)


# ═══════════════════════════════════════════════════════════════
# Training config
# ═══════════════════════════════════════════════════════════════

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_quant_storage=torch.bfloat16,
)

lora_config = LoraConfig(
    r=256,
    lora_alpha=512,
    target_modules="all-linear",
    lora_dropout=0.0,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

training_config = ActorTrainCfg(
    learning_rate=5e-6,
    optimizer="adamw_32bit",
    loss="liger_gspo",
    peft_config=lora_config,
    quantization_config=quantization_config,
    offload_model=True,
    offload_optimizer=True,
    beta=0.001,
    max_grad_norm=0.1,
    reference_batch_size=64,
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
You are tasked with solving a math problem. You must give your answer + your confidence in the answer.
The confidence must be an integer between 1 and 10, where 1 is very uncertain and 10 is absolutely certain.
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
<confidence>
...
</confidence>
"""

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
    env = SingleTurnEnvironment(
        actor=actor,
        train_data=ds,
        reward_functions=[
            reward,
            strict_format_reward_func,
            count_xml,
            accuracy_reward,
            ece_metric,
        ],
        sampling_params=SamplingParams(
            temperature=1.0,
            max_tokens=512,
        ),
        prompt_column="conversation",
        mask_prompt_for_loss=True,
    )

    cfg = GRPOTrainerCfg(
        group_size=8,
        batch_size=128,
        grad_accumulation_steps=8,
        num_iterations=2,
        log_every_n=1,
        checkpoint_every_n=30,
        save_strategy=SaveStrategy.ALL,
    )

    trainer = GRPOTrainer(cfg=cfg, env=env, actors=[actor])

    import wandb

    if os.getenv("RANK") == "0":
        wandb.init(project="actors", name="3b-qlora-256")
    trainer.train()


if __name__ == "__main__":
    main()
