#!/usr/bin/env python3
"""
ParallelEnvironment quick-start:
4 sampler actors + 1 combiner on GSM8K with 4-bit QLoRA (Qwen-2.5-8B-Instruct)
"""

from __future__ import annotations

import os

import torch
from datasets import load_dataset
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
@reward_function(name="correctness_reward", weight=1.0)
def correctness_reward(completion, answer):
    try:
        found_answer = completion.split("boxed{").split("}")[0].strip()
        answer = int(answer.split("### ")[1].strip().replace(",", ""))
        return 1.0 if str(answer) == found_answer else 0.0
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


def TRAIN_CFG_TEMPLATE():
    return ActorTrainCfg(
        learning_rate=5e-6,
        optimizer="adamw_8bit",
        loss="liger_gspo",
        peft_config=LORA_CFG(),
        quantization_config=QUANT_CFG(),
        offload_model=True,
        offload_optimizer=True,
        beta=0.001,
        max_grad_norm=0.1,
        # Make these explicit so they override trainer defaults if omitted there
        # batch_size=64,
        # grad_accumulation_steps=4,
    )


ENGINE_KWARGS = {
    "gpu_memory_utilization": 0.7,
    "max_model_len": 2**15,  # 32k
    "quantization": "bitsandbytes",
}

MODEL_PATH = "Qwen/Qwen3-0.6B"

# ═══════════════════════════════════════════════════════════════
# 3. Instantiate 5 actors  (4 samplers + 1 combiner)
# ═══════════════════════════════════════════════════════════════


def build_actor(name: str) -> vLLMActor:
    return vLLMActor(
        name=name,
        model_path=MODEL_PATH,
        engine_kwargs=ENGINE_KWARGS,
        training_config=TRAIN_CFG_TEMPLATE(),
    )


sampler_names = ["Sampler-A", "Sampler-B", "Sampler-C", "Sampler-D"]
samplers = [build_actor(n) for n in sampler_names]
combiner = build_actor("Combiner")

# ═══════════════════════════════════════════════════════════════
# 4. Data
# ═══════════════════════════════════════════════════════════════

gsm8k = load_dataset("openai/gsm8k", "main")

# ═══════════════════════════════════════════════════════════════
# 5. ParallelEnvironment configuration
# ═══════════════════════════════════════════════════════════════

sampler_cfgs = [
    ParallelActorConfig(
        actor=a,
        sampling_params=SamplingParams(temperature=1.0, max_tokens=16_000),
        num_samples=1,  # one draft per sampler
    )
    for a in samplers
]

combiner_cfg = CombinerActorConfig(
    actor=combiner,
    sampling_params=SamplingParams(temperature=1.0, max_tokens=16_000),
)

env = ParallelEnvironment(
    sampler_cfgs=sampler_cfgs,
    final_combiner=combiner_cfg,
    generate_reward_functions=[],
    combiner_reward_functions=[correctness_reward],
    prompt_column="question",
    train_data=gsm8k["train"],
    eval_data={"test": gsm8k["test"]},
)

# ═══════════════════════════════════════════════════════════════
# 6. Trainer
# ═══════════════════════════════════════════════════════════════
TRAINER_CFG = GRPOTrainerCfg(
    group_size=8,
    batch_size=64,
    grad_accumulation_steps=4,
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
        wandb.init(project="parallel", name="qwen0.6b-4samplers-combiner")

    trainer.train()


if __name__ == "__main__":
    main()
