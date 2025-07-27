import os
import re

import wandb
from datasets import load_dataset
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
# Training configs for the two actors.
# ----------------------------------------------------------

# Create training configuration for first actor
training_config_alice = ActorTrainCfg(
    learning_rate=1e-6,
    optimizer="adamw_8bit",
    loss="liger_grpo",
    offload_model=True,
    offload_optimizer=True,
    beta=0.0,
)

# Create training configuration for second actor
training_config_bob = ActorTrainCfg(
    learning_rate=1e-6,
    optimizer="adamw_8bit",
    loss="liger_grpo",
    offload_model=True,
    offload_optimizer=True,
    beta=0.0,
)

# ----------------------------------------------------------
# Creating the two actors.
# ----------------------------------------------------------

# Create first actor (Alice)
alice_actor = vLLMActor(
    name="Alice",
    model_path="Qwen/Qwen3-4B",
    engine_kwargs={
        "gpu_memory_utilization": 0.65,
        "max_model_len": 8192,
    },
    training_config=training_config_alice,
    gpu_groups=[[0, 1]],
)

# Create second actor (Bob)
bob_actor = vLLMActor(
    name="Bob",
    model_path="Qwen/Qwen3-4B",
    engine_kwargs={
        "gpu_memory_utilization": 0.65,
        "max_model_len": 8192,
    },
    training_config=training_config_bob,
    gpu_groups=[[0, 1]],
)

# ----------------------------------------------------------
# Judge actor for correctness evaluation.
# ----------------------------------------------------------

# This judge allows us to evaluate even hard to equate answers.
judge_actor = vLLMActor(
    name="Judge",
    model_path="Qwen/Qwen3-4B",
    engine_kwargs={
        "gpu_memory_utilization": 0.65,
        "max_model_len": 4096,
    },
    non_trainable=True,
    gpu_groups=[[0, 1]],
)

# ----------------------------------------------------------

# ------------------------------------------------------------
# Reward function for the collaborative task.
# ------------------------------------------------------------


@conversation_reward_function(name="correctness_reward", weight=1.0, batched=True)
def reward(conversation, problem, answer):
    conversations = conversation
    problems = problem
    answers = answer

    # We average over this number of judge attempts.
    out_of = 4
    texts = [
        "".join([msg["content"] for msg in conv if msg.get("role", "") != "system"])
        for conv in conversations
    ]
    extracted_answers = [
        re.findall(r"boxed{(.*?)}", text)[-1]
        if re.search(r"boxed{(.*?)}", text)
        else ""
        for text in texts
    ]
    bonus = [0.25 if ans else 0.0 for ans in extracted_answers]
    # Create the prompts for the judge
    prompts_judge = [
        [
            {
                "role": "user",
                "content": (
                    f"Problem: {problem}\n"
                    f"Golden Answer: {answer}\n"
                    f"Extracted Answer: {extracted_answer}\n"
                    "Is the answer correct? Reason and then respond with '\\boxed{yes}' or '\\boxed{no}'."
                ),
            }
        ]
        for problem, answer, extracted_answer in zip(
            problems, answers, extracted_answers, strict=True
        )
    ]
    # Use the judge actor to evaluate the correctness
    judge_responses = judge_actor.chat(
        prompts_judge * out_of,
        sampling_params=SamplingParams(temperature=0.2, max_tokens=2048),
    )
    # Extract the responses
    judge_responses = [
        out.text for response in judge_responses for out in response.outputs
    ]
    # Convert responses to rewards
    judge_responses = [
        1.0 if re.search(r"\\boxed{yes}", response) else 0.0
        for response in judge_responses
    ]
    # We average the rewards cause we do avg@4.
    rewards = [0.0] * len(conversations)
    for i, reward in enumerate(judge_responses):
        rewards[i % len(rewards)] += reward / out_of

    # Add bonus for giving any answer
    for i in range(len(rewards)):
        rewards[i] += bonus[i]

    return rewards


# ----------------------------------------------------------


# -----------------------------------------------------------
# Main function to run the training.
# -----------------------------------------------------------
def main():
    # Dataset loading
    train_dataset = load_dataset("rl-actors/Big-Math-RL-Verified-Subset", split="train")
    eval_dataset = load_dataset("rl-actors/Big-Math-RL-Verified-Subset", split="test")

    # We give specific prompts to both actors.
    system_prompt_alice = (
        "You are an Alice, you must collaborate with Bob to solve the problem step by step and provide the final answer in a boxed format."
        "You will alternate turns with Bob, and your responses should be clear and concise."
        "Make sure to provide a final answer in the format '\\boxed{your_answer}' at the end of each message with your best current guess."
    )

    system_prompt_bob = (
        "You are Bob, a creative and curious AI assistant. You will collaborate with Alice to solve the problem step by step."
        "Your responses should be engaging and concise, and you should always provide a final answer in the format '\\boxed{your_answer}'."
        "You will alternate turns with Alice, and your responses should encourage Alice to think critically and creatively."
        "Make sure to provide a final answer in the format '\\boxed{your_answer}' at the end of each message with your best current guess."
    )

    # Create collaborative environment
    env = CollaborativeEnvironment(
        actor_cfgs=[
            CollaborativeActorConfig(
                actor=alice_actor,
                system_prompt=system_prompt_alice,
                sampling_params=SamplingParams(
                    temperature=0.8,
                    max_tokens=1500,
                ),
            ),
            CollaborativeActorConfig(
                actor=bob_actor,
                system_prompt=system_prompt_bob,
                sampling_params=SamplingParams(
                    temperature=0.8,
                    max_tokens=1500,
                ),
            ),
        ],
        round_spec="Alice -> Bob -> Alice -> Bob",  # This is the definition of the order of turns.
        reward_functions=[reward],
        run_concurrently=False,
        prompt_column="problem",
        mask_other_agents_for_loss=True,
        train_data=train_dataset,
        eval_data=eval_dataset,
    )

    # Create trainer configuration
    cfg = GRPOTrainerCfg(
        group_size=8,
        batch_size=128,
        grad_accumulation_steps=16,
        num_iterations=2,
        log_every_n=1,
        eval_every_n=25,
        eval_strategy=EvalStrategy.ALL,
        checkpoint_every_n=20,
        save_strategy=SaveStrategy.ALL,
    )

    # Create trainer with environment and both actors
    trainer = GRPOTrainer(cfg=cfg, env=env, actors=[alice_actor, bob_actor])

    # Initialize wandb for logging
    if os.getenv("RANK") == "0":
        wandb.init(
            project="actors",
            name="Alice-Bob-Collaborative",
        )

    # Train the collaborative system
    trainer.train()


if __name__ == "__main__":
    main()
