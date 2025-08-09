"""
ParallelEnvironment quick-start:
4 sampler actors + 1 combiner on GSM8K with 4-bit QLoRA (Qwen-2.5-8B-Instruct)
"""

from __future__ import annotations

import os

import torch
from datasets import load_dataset
from math_verify import parse, verify
from peft import LoraConfig, TaskType
from transformers import BitsAndBytesConfig
from vllm import SamplingParams

from actors import (
    ActorTrainCfg,
    CombinerActorConfig,
    EvalStrategy,
    GRPOTrainer,
    GRPOTrainerCfg,
    ParallelActorConfig,
    ParallelEnvironment,
    SaveStrategy,
    vLLMActor,
)
from actors.rewards.base_completion_reward import reward_function

# ═══════════════════════════════════════════════════════════════
# 1. Reward functions
# ═══════════════════════════════════════════════════════════════


def extract_boxed(completion: str) -> str:
    if "boxed{" not in completion:
        return ""
    stack_of_brackets = 1
    boxed = ""
    new_completion = completion.split("boxed{")[-1]
    for char in new_completion:
        if char == "{":
            stack_of_brackets += 1
            boxed += char
        elif char == "}":
            stack_of_brackets -= 1
            if stack_of_brackets == 0:
                break
            boxed += char
        else:
            boxed += char
    return boxed


@reward_function(name="correctness_reward", weight=1.0, batched=False)
def correctness_reward(prompt, completion, actor_name, answer):
    try:
        found_answer = extract_boxed(completion)
        parsed_answer = parse(answer)
        parsed_found_answer = parse(found_answer)
        return verify(parsed_answer, parsed_found_answer)
    except Exception:
        return 0.0


# ═══════════════════════════════════════════════════════════════
# 2. Building blocks
# ═══════════════════════════════════════════════════════════════
def QUANT_CFG():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_quant_storage=torch.bfloat16,
    )


def LORA_CFG():
    return LoraConfig(
        r=256,
        lora_alpha=512,
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


def TRAIN_CFG_TEMPLATE_SAMPLER():
    return ActorTrainCfg(
        learning_rate=2e-6,
        optimizer="adamw_32bit",
        loss="liger_gspo",
        peft_config=LORA_CFG(),
        quantization_config=QUANT_CFG(),
        offload_model=True,
        offload_optimizer=True,
        beta=0.0,
        max_grad_norm=0.1,
        batch_size=64,
        grad_accumulation_steps=8,
        reference_batch_size=16,
    )


def TRAIN_CFG_TEMPLATE_COMBINER():
    return ActorTrainCfg(
        learning_rate=2e-6,
        optimizer="adamw_32bit",
        loss="liger_gspo",
        peft_config=LORA_CFG(),
        quantization_config=QUANT_CFG(),
        offload_model=True,
        offload_optimizer=True,
        beta=0.0,
        max_grad_norm=0.1,
        reference_batch_size=16,
        grad_accumulation_steps=8,
    )


ENGINE_KWARGS_SAMPLER = {
    "gpu_memory_utilization": 0.5,
    "max_model_len": 12_000,  # 16k
    "quantization": "fp8",
}
ENGINE_KWARGS_COMBINER = {
    "gpu_memory_utilization": 0.5,
    "max_model_len": 2**15,  # 32k
    "quantization": "fp8",
}

MODEL_PATH = "Qwen/Qwen3-8B"

# ═══════════════════════════════════════════════════════════════
# 3. Instantiate 5 actors  (4 samplers + 1 combiner)
# ═══════════════════════════════════════════════════════════════


def build_actor(name: str) -> vLLMActor:
    return vLLMActor(
        name=name,
        model_path=MODEL_PATH,
        engine_kwargs=ENGINE_KWARGS_SAMPLER
        if "Sampler" in name
        else ENGINE_KWARGS_COMBINER,
        training_config=TRAIN_CFG_TEMPLATE_SAMPLER()
        if "Sampler" in name
        else TRAIN_CFG_TEMPLATE_COMBINER(),
    )


sampler_names = ["Sampler-A", "Sampler-B", "Sampler-C", "Sampler-D"]
samplers = [build_actor(n) for n in sampler_names]
combiner = build_actor("Combiner")

# ═══════════════════════════════════════════════════════════════
# 4. Data
# ═══════════════════════════════════════════════════════════════

math_rl_data = load_dataset(
    "open-r1/Big-Math-RL-Verified-Processed", "level_5", split="train"
)


math_rl_data = math_rl_data.rename_column("prompt", "problem")
math_rl_data = math_rl_data.rename_column("solution", "answer")

# Rename the prompt column to "problem"
aime25_data = load_dataset("math-ai/aime25", split="test")

# ═══════════════════════════════════════════════════════════════
# 5. ParallelEnvironment configuration
# ═══════════════════════════════════════════════════════════════

sampler_cfgs = [
    ParallelActorConfig(
        actor=a,
        sampling_params=SamplingParams(temperature=1.0, max_tokens=8192),
        num_samples=4,  # one draft per sampler
    )
    for a in samplers
]

combiner_cfg = CombinerActorConfig(
    actor=combiner,
    sampling_params=SamplingParams(temperature=1.0, max_tokens=8192),
)

env = ParallelEnvironment(
    sampler_cfgs=sampler_cfgs,
    final_combiner=combiner_cfg,
    generate_reward_functions=[],
    combiner_reward_functions=[correctness_reward],
    prompt_column="problem",
    train_data=math_rl_data,
    eval_data={
        "aime25": aime25_data,
    },
)

# ═══════════════════════════════════════════════════════════════
# 6. Trainer
# ═══════════════════════════════════════════════════════════════
TRAINER_CFG = GRPOTrainerCfg(
    group_size=8,
    batch_size=64,
    grad_accumulation_steps=32,
    num_iterations=2,
    log_every_n=1,
    eval_every_n=5,
    eval_strategy=EvalStrategy.STEPS,
    checkpoint_every_n=30,
    save_strategy=SaveStrategy.ALL,
)

trainer = GRPOTrainer(
    env=env,
    cfg=TRAINER_CFG,
    actors=samplers + [combiner],
)


# ═══════════════════════════════════════════════════════════════
# 7. Entry-point
# ═══════════════════════════════════════════════════════════════
def main():
    import wandb

    if os.getenv("RANK", "0") == "0":
        wandb.init(project="parallel-qwen8b", name="qwen8b-4sampler-1combiner")

    trainer.train()


if __name__ == "__main__":
    main()
