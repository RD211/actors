"""
Example demonstrating shared base models with LoRA actors.

This example shows how multiple actors can share the same base model when they have
compatible configurations.

- The first actor (Alice) sets an expected_max_lora_rank which becomes the base capacity
  for the shared model (e.g., r=64).
- Other actors can merge into the same base only if their LoRA rank <= base capacity.

Capacity control keys:
- expected_max_lora_rank: desired base capacity (we pass this to the first actor)
- allow_sharing: opt-out of merging if needed
"""

from __future__ import annotations

import os
import re

from datasets import load_dataset
from math_verify import parse, verify
from peft import LoraConfig, TaskType
from vllm import SamplingParams

from actors import (
    ActorTrainCfg,
    EvalStrategy,
    GRPOTrainer,
    GRPOTrainerCfg,
    SaveStrategy,
    vLLMActor,
)
from actors.environments import (
    CollaborativeActorConfig,
    CollaborativeEnvironment,
)
from actors.rewards.base_conversation_reward import conversation_reward_function

# ----------------------------------------------------------
# Shared base model configuration
# ----------------------------------------------------------

# All actors will use the SAME model path and engine kwargs (except capacity flags).
# This enables automatic base model sharing.
SHARED_MODEL_PATH = "Qwen/Qwen2.5-3B-Instruct"
SHARED_ENGINE_KWARGS = {
    "gpu_memory_utilization": 0.7,  # Keep high but safe enough for sharing
    "max_model_len": 4096,
}
# Explicit capacity we want for the shared base (used by Alice via expected_max_lora_rank)
BASE_CAPACITY_R = 128

print("🚀 Creating actors with shared base model configuration (rank-aware)...")
print(f"📦 Shared model: {SHARED_MODEL_PATH}")
print(f"⚙️  Shared engine base config: {SHARED_ENGINE_KWARGS}")
print(f"📐 Configured base capacity (expected_max_lora_rank): r={BASE_CAPACITY_R}")

# ----------------------------------------------------------
# Training configs for each actor with different LoRA settings
# ----------------------------------------------------------

# Alice: reasoning specialist (r=32), but we set capacity to 64 to allow others to merge
training_config_alice = ActorTrainCfg(
    learning_rate=1e-6,
    optimizer="adamw_32bit",
    offload_model=True,
    offload_optimizer=True,
    reference_batch_size=4,
    beta=0.0,
    peft_config=LoraConfig(
        r=32,  # Actual LoRA rank for Alice
        lora_alpha=64,
        target_modules="all-linear",
        lora_dropout=0.0,
        task_type=TaskType.CAUSAL_LM,
    ),
)

# Bob: creativity specialist (r=64)
training_config_bob = ActorTrainCfg(
    learning_rate=1e-6,
    optimizer="adamw_32bit",
    offload_model=True,
    offload_optimizer=True,
    reference_batch_size=4,
    beta=0.0,
    peft_config=LoraConfig(
        r=64,  # Actual LoRA rank for Bob
        lora_alpha=128,
        target_modules="all-linear",
        lora_dropout=0.0,
        task_type=TaskType.CAUSAL_LM,
    ),
)

# Charlie: accuracy specialist (r=16)
training_config_charlie = ActorTrainCfg(
    learning_rate=1e-6,
    optimizer="adamw_32bit",
    offload_model=True,
    offload_optimizer=True,
    reference_batch_size=4,
    beta=0.0,
    peft_config=LoraConfig(
        r=64,  # Actual LoRA rank for Charlie
        lora_alpha=128,
        target_modules="all-linear",
        lora_dropout=0.0,
        task_type=TaskType.CAUSAL_LM,
    ),
)

# ----------------------------------------------------------
# Creating the actors - they will automatically share the base model!
# ----------------------------------------------------------

print("\n🤖 Creating Alice (reasoning specialist, r=32, base capacity r=64)...")  #
alice_actor = vLLMActor(
    name="Alice",
    model_path=SHARED_MODEL_PATH,
    engine_kwargs=SHARED_ENGINE_KWARGS,
    expected_max_lora_rank=BASE_CAPACITY_R,
    training_config=training_config_alice,
    allow_sharing=True,
)

print("🤖 Creating Bob (creativity specialist, r=64)...")
bob_actor = vLLMActor(
    name="Bob",
    model_path=SHARED_MODEL_PATH,
    engine_kwargs=SHARED_ENGINE_KWARGS,  # Bob requires r=64 and merges into Alice's base
    training_config=training_config_bob,
    allow_sharing=True,
)

print("🤖 Creating Charlie (accuracy specialist, r=16)...")
charlie_actor = vLLMActor(
    name="Charlie",
    model_path=SHARED_MODEL_PATH,
    engine_kwargs=SHARED_ENGINE_KWARGS,  # Charlie (r=16) also merges into the r=64 base
    training_config=training_config_charlie,
    allow_sharing=True,
)

# ----------------------------------------------------------
# Demonstrate shared model information
# ----------------------------------------------------------

print("\n📊 Shared Model Information:")
print("=" * 50)

from actors.inference.pool import ModelPool

pool = ModelPool()

print(f"Total models loaded: {len(pool.models)}")
print(f"Shared base models: {len(pool.shared_models)}")

# Show details for each actor
for actor_name in ["Alice", "Bob", "Charlie"]:
    if actor_name in pool.models:
        record = pool.models[actor_name]
        if record.is_shared and record.shared_config:
            base_id = record.shared_config.base_model_id
            adapter_info = record.lora_adapters[actor_name]
            capacity = (
                record.shared_config.max_lora_rank if record.shared_config else -1
            )
            required_r = (
                training_config_alice.peft_config.r
                if actor_name == "Alice"
                else (
                    training_config_bob.peft_config.r
                    if actor_name == "Bob"
                    else training_config_charlie.peft_config.r
                )
            )
            print(
                f"  🔗 {actor_name}: shared model {base_id[:8]}... "
                f"(adapter_id={adapter_info.adapter_id}, base capacity r={capacity}, required r={required_r})"
            )
        else:
            print(f"  ❌ {actor_name}: not using shared model")

# Show shared model statistics
if pool.shared_models:
    for base_id, shared_record in pool.shared_models.items():
        adapters = list(shared_record.lora_adapters.keys())
        capacity = shared_record.shared_config.max_lora_rank
        print(
            f"  📦 Base model {base_id[:8]}...: {len(adapters)} actors {adapters}, capacity r={capacity}"
        )
        print("     💾 Potential savings: merges avoid extra model loads")

# ----------------------------------------------------------
# Reward: Extract boxed answer from last assistant message and compare to golden
# ----------------------------------------------------------


@conversation_reward_function(name="math_correctness", weight=1.0, batched=True)
def reward_math(conversation, question, answer, **kwargs):
    """Return 1.0 if the last assistant message boxed answer verifies equal to golden; else 0.0."""
    conversations = conversation
    gold_answers = answer

    def extract_boxed(completion: str) -> str:
        if "boxed{" not in completion:
            return ""
        stack = 1
        boxed = ""
        tail = completion.split("boxed{")[-1]
        for ch in tail:
            if ch == "{":
                stack += 1
                boxed += ch
            elif ch == "}":
                stack -= 1
                if stack == 0:
                    break
                boxed += ch
            else:
                boxed += ch
        return boxed

    def extract_gsm8k_gold(ans: str) -> str:
        # Gold format contains a final line like: '#### 42'
        m = re.findall(r"####\s*(.+)", ans)
        return m[-1].strip() if m else ans.strip()

    rewards = []
    for conv, gold in zip(conversations, gold_answers, strict=False):
        last_assistant = conv[-1]["content"]
        pred = extract_boxed(last_assistant)
        gold_clean = extract_gsm8k_gold(str(gold))
        try:
            ok = verify(parse(gold_clean), parse(pred))
            rewards.append(1.0 if ok else 0.0)
        except Exception:
            rewards.append(0.0)
    return rewards


@conversation_reward_function(name="len_penalty", weight=1.0, batched=True)
def reward_len_penalty(conversation, question, answer, **kwargs):
    conversations = conversation
    rewards = []
    for conv in conversations:
        all_lenghts = sum(
            [len(msg["content"]) for msg in conv if msg["role"] == "assistant"]
        )
        rewards.append(-all_lenghts / 8)
    return rewards


# ----------------------------------------------------------
# Create collaborative environment
# ----------------------------------------------------------

print("\n🏗️  Setting up collaborative environment...")

# Load datasets consistent with other math examples
train_data = load_dataset("openai/gsm8k", "main", split="train")
eval_data = load_dataset("openai/gsm8k", "main", split="test")

# System prompts: require boxed answers in every assistant message
system_prompt_alice = (
    "You are Alice, a reasoning specialist. Collaborate with Bob and Charlie to solve the math problem. "
    "Always end your message with the final answer in the format \\boxed{your_answer}."
)
system_prompt_bob = (
    "You are Bob, a creativity specialist. Explore solution strategies. "
    "Always end your message with the final answer in the format \\boxed{your_answer}."
)
system_prompt_charlie = (
    "You are Charlie, an accuracy specialist. Verify computations and ensure correctness. Make your messages very short. "
    "Always end your message with the final answer in the format \\boxed{your_answer}."
)

# Create collaborative environment
env = CollaborativeEnvironment(
    actor_cfgs=[
        CollaborativeActorConfig(
            actor=alice_actor,
            system_prompt=system_prompt_alice,
            sampling_params=SamplingParams(
                temperature=1.0,
                max_tokens=1024,
            ),
        ),
        CollaborativeActorConfig(
            actor=bob_actor,
            system_prompt=system_prompt_bob,
            sampling_params=SamplingParams(
                temperature=1.0,
                max_tokens=1024,
            ),
        ),
        CollaborativeActorConfig(
            actor=charlie_actor,
            system_prompt=system_prompt_charlie,
            sampling_params=SamplingParams(
                temperature=1.0,
                max_tokens=1024,
            ),
        ),
    ],
    round_spec="Alice -> Bob -> Charlie",
    reward_functions=[reward_math, reward_len_penalty],
    run_concurrently=False,
    prompt_column="question",
    mask_other_agents_for_loss=True,
    train_data=train_data,
    eval_data=eval_data,
)

# ----------------------------------------------------------
# Training configuration
# ----------------------------------------------------------

cfg = GRPOTrainerCfg(
    group_size=8,
    batch_size=64,
    grad_accumulation_steps=2,
    num_iterations=1,
    log_every_n=1,
    eval_every_n=10,
    eval_strategy=EvalStrategy.ALL,
    checkpoint_every_n=10,
    save_strategy=SaveStrategy.FINAL,
    checkpoint_path="checkpoints/shared_models_demo",
)

# ----------------------------------------------------------
# Run training
# ----------------------------------------------------------


def main():
    """Main training function."""
    print(
        "\n🚀 Starting collaborative training with shared base models (rank-aware)..."
    )
    print("=" * 60)

    trainer = GRPOTrainer(
        cfg=cfg, env=env, actors=[alice_actor, bob_actor, charlie_actor]
    )

    if os.getenv("RANK", "0") == "0":
        try:
            import wandb

            wandb.init(
                project="actors-shared-models",
                name="shared-base-gsm8k",
                config={
                    "model_path": SHARED_MODEL_PATH,
                    "shared_actors": ["Alice", "Bob", "Charlie"],
                    "base_capacity_r": BASE_CAPACITY_R,
                },
            )
        except ImportError:
            print("💡 Wandb not available, skipping logging")

    trainer.train()

    print("\n✅ Training completed!")


if __name__ == "__main__":
    main()
