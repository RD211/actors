import torch
from datasets import Dataset
from vllm import SamplingParams

from actors import (
    ActorTrainCfg,
    EvalStrategy,
    GRPOTrainer,
    GRPOTrainerCfg,
    SaveStrategy,
    SimpleSingleTurnEnvironment,
    vLLMActor,
)


def length_reward(completion: str) -> float:
    """Rewards shorter responses."""
    return -min(
        len(completion) / 500, 5.0
    )  # Negative reward for length, capped at -5.0


def get_lr_scheduler(optimizer, max_step):
    warmup_steps = 2
    # part 1 – warm-up: linearly increase from 0.1× to 1.0× base_lr
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,  # 0.1 × base_lr -> 100 µ after 1st step
        end_factor=1.0,
        total_iters=warmup_steps,
    )

    # part 2 – linear decay all the way to 0
    decay = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.0, total_iters=max_step - warmup_steps
    )

    # stitch them together
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, decay], milestones=[warmup_steps]
    )
    return scheduler


def main():
    # Create training configuration
    training_config = ActorTrainCfg(
        learning_rate=2e-6,
        optimizer="adamw_8bit",
        loss="liger_grpo",
        scheduler=get_lr_scheduler,
        offload_model=True,
        offload_optimizer=True,
        beta=0.04,
    )

    # Create actor with improved configuration API
    actor = vLLMActor(
        name="main",
        model_path="Qwen/Qwen2.5-0.5B-Instruct",
        engine_kwargs={
            "gpu_memory_utilization": 0.5,
            "max_model_len": 2048,
            "quantization": "fp8",
        },
        training_config=training_config,
        gpu_groups=[
            [0, 1],        
        ],
    )
    tokenizer = actor.tokenizer

    # Prepare training data
    data = [
        {"text": "What is the capital of France?"},
        {"text": "Explain the theory of relativity."},
        {"text": "How does quantum computing work?"},
        {"text": "What is the weather like today?"},
        {"text": "Tell me a joke."},
        {"text": "What is the largest mammal?"},
        {"text": "Who wrote 'To Kill a Mockingbird'?"},
        {"text": "What is the speed of light?"},
        {"text": "How do you make a cake?"},
    ] * 50

    train_data = [
        {
            "conversation": tokenizer.apply_chat_template(
                [{"role": "user", "content": item["text"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
        }
        for item in data
    ]
    train_dataset = Dataset.from_list(train_data)

    # Create multiple eval datasets with custom names
    general_qa = [
        {"text": "What is the capital of Italy?"},
        {"text": "Explain photosynthesis briefly."},
        {"text": "What is the largest ocean?"},
    ] * 5

    science_qa = [
        {"text": "How does machine learning work?"},
        {"text": "What is quantum entanglement?"},
        {"text": "Explain the theory of evolution."},
    ] * 5

    creative_qa = [
        {"text": "Tell me about space exploration."},
        {"text": "Write a haiku about rain."},
        {"text": "Describe a perfect day."},
    ] * 5

    # Convert to proper format and create named eval datasets
    eval_datasets = {
        "general": Dataset.from_list(
            [
                {
                    "conversation": tokenizer.apply_chat_template(
                        [{"role": "user", "content": item["text"]}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                }
                for item in general_qa
            ]
        ),
        "science": Dataset.from_list(
            [
                {
                    "conversation": tokenizer.apply_chat_template(
                        [{"role": "user", "content": item["text"]}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                }
                for item in science_qa
            ]
        ),
        "creative": Dataset.from_list(
            [
                {
                    "conversation": tokenizer.apply_chat_template(
                        [{"role": "user", "content": item["text"]}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                }
                for item in creative_qa
            ]
        ),
    }

    # Create environment with data and actor
    env = SimpleSingleTurnEnvironment(
        actor=actor,
        train_data=train_dataset,
        eval_data=eval_datasets,
        reward_functions=[length_reward],
        sampling_params=SamplingParams(
            temperature=1.0,
            max_tokens=256,
        ),
        prompt_column="conversation",
        mask_prompt_for_loss=True,
    )

    # Create trainer configuration
    cfg = GRPOTrainerCfg(
        group_size=16,
        batch_size=64,
        grad_accumulation_steps=8,
        num_iterations=2,
        log_every_n=1,
        eval_every_n=50,  # Run evaluation every 50 steps
        eval_strategy=EvalStrategy.ALL,  # Run evaluation both periodically and at the end
        checkpoint_every_n=30,
        save_strategy=SaveStrategy.ALL,
    )

    # Create trainer with environment and actors
    trainer = GRPOTrainer(cfg=cfg, env=env, actors=[actor])

    import wandb

    wandb.init(project="test_actors-2", entity="rd211", name="0.5B")
    trainer.train()
    trainer.push_to_hub(
        "rd211/test_actors_main",
        private=True,
    )


if __name__ == "__main__":
    main()
