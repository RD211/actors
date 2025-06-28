import torch
from actors.environments.env_base import Environment
from actors.environments.types import EnvironmentOutput, ActorOutput
from actors.actors import vLLMActor
from transformers import AutoTokenizer, AutoModelForCausalLM
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from actors.losses.grpo_loss import GRPOLoss
from actors.losses.liger_grpo_loss import LigerLoss
from torch.optim.lr_scheduler import LinearLR, ConstantLR
from vllm import SamplingParams
from actors import Trainer
from actors.trainers.trainer import EvalStrategy
import bitsandbytes as bnb
from actors.environments import SimpleSingleTurnEnvironment
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import PEFT for LoRA configuration
from peft import LoraConfig, TaskType

def length_reward(completion: str) -> float:
    """Rewards shorter responses."""
    return -min(len(completion) / 500, 5.0)  # Negative reward for length, capped at -5.0

def main():
    # Create LoRA configuration
    lora_config = LoraConfig(
        r=64,  # LoRA rank
        lora_alpha=64,  # LoRA scaling parameter
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # Target all linear layers
        lora_dropout=0.1,  # LoRA dropout
        bias="none",  # Don't adapt bias parameters
        task_type=TaskType.CAUSAL_LM,  # Task type for causal language modeling
    )
    
    # Create actor with PEFT configuration
    actor = vLLMActor(
        name="main",
        model_path="Qwen/Qwen2.5-3B-Instruct",
        engine_kwargs={
            "gpu_memory_utilization": 0.6,
            "max_model_len": 2048,
            "quantization": "fp8"
        },
        # Training configuration now directly in constructor
        learning_rate=2e-4,  # Higher learning rate for LoRA training
        optimizer="adamw_32bit",  # Using string for convenience
        loss="liger_grpo",  # Using string for liger loss
        loss_kwargs={"beta": 0.0, "temperature": 1.0},
        scheduler="cosine",  # Using string for cosine scheduler
        # PEFT/LoRA configuration
        peft_config=lora_config,
        # Offloading configuration now in actor
        offload_model=True,
        offload_optimizer=True,
        offload_reference_to_cpu=True,  # Enable aggressive CPU offloading for reference model
        offload_activations_to_cpu=True,  # Enable CPU activation offloading for training model
    )
    tokenizer = actor.tokenizer

    env = SimpleSingleTurnEnvironment(
        actor=actor,
        reward_functions=[length_reward],
        sampling_params=SamplingParams(
            temperature=1.0,
            max_tokens=256,
        ),
        prompt_column='conversation',
        mask_prompt_for_loss=True,
    )

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
    ] * 120

    data = [{'conversation':tokenizer.apply_chat_template([{'role': 'user', 'content': item['text']}], tokenize=False, add_generation_prompt=True)} for item in data]
    
    # Create multiple eval datasets with custom names
    general_qa = [
        {"text": "What is the capital of Italy?"},
        {"text": "Explain photosynthesis briefly."},
        {"text": "What is the largest ocean?"},
    ]
    
    science_qa = [
        {"text": "How does machine learning work?"},
        {"text": "What is quantum entanglement?"},
        {"text": "Explain the theory of evolution."},
    ]
    
    creative_qa = [
        {"text": "Tell me about space exploration."},
        {"text": "Write a haiku about rain."},
        {"text": "Describe a perfect day."},
    ]
    
    # Convert to proper format and create named eval datasets
    eval_data = {
        "general": [{'conversation': tokenizer.apply_chat_template([{'role': 'user', 'content': item['text']}], tokenize=False, add_generation_prompt=True)} for item in general_qa],
        "science": [{'conversation': tokenizer.apply_chat_template([{'role': 'user', 'content': item['text']}], tokenize=False, add_generation_prompt=True)} for item in science_qa],
        "creative": [{'conversation': tokenizer.apply_chat_template([{'role': 'user', 'content': item['text']}], tokenize=False, add_generation_prompt=True)} for item in creative_qa],
    }
    
    trainer = Trainer(
        env,
        group_size=16,
        batch_size=64,
        grad_accumulation_steps=4,
        num_iterations=2,
        reference_batch_size=4,
        log_every_n=1,
        data=data,
        std_normalization=True,
        eval_data=eval_data,
        eval_every_n=None,  # Run evaluation every 2 steps
        eval_strategy=EvalStrategy.NONE,  # Run evaluation both periodically and at the end
    )

    import wandb

    wandb.init(project="test_actors", entity="rd211", name="3b-lora")
    trainer.train(checkpoint_every_n=30)
    trainer.push_to_hub(
        "rd211/test_actors_lora_main",
        private=True,
    )


if __name__ == "__main__":
    main()
