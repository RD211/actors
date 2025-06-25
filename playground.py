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

def length_reward(completion: str) -> float:
    """Rewards shorter responses."""
    return -min(len(completion) / 500, 5.0)  # Negative reward for length, capped at -5.0

def custom_model_factory():
    """Example custom model factory that applies modifications before training."""
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct", 
        trust_remote_code=True,
        # You can add custom loading parameters here
        # torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",
    )
    
    # Apply custom modifications
    # model.config.use_cache = False  # Example: disable cache
    # model.lm_head.requires_grad_(False)  # Example: freeze output layer
    
    print("✅ Custom model factory applied modifications!")
    return model

def main():
    # Create actor with improved configuration API
    actor = vLLMActor(
        name="main",
        model_path="Qwen/Qwen2.5-0.5B-Instruct",
        engine_kwargs={
            "gpu_memory_utilization": 0.5,
            "max_model_len": 2048,
            "quantization": "fp8"
        },
        # Training configuration now directly in constructor
        learning_rate=2e-6,
        optimizer="paged_adamw_8bit",  # Using string for convenience
        loss="liger_grpo",  # Using string for liger loss
        loss_kwargs={"beta": 0.0, "temperature": 1.0},
        scheduler="cosine",  # Using string for cosine scheduler
        model_factory=custom_model_factory,  # Use custom model factory
    )
    
    # Get tokenizer directly from actor property
    tokenizer = actor.tokenizer

    # Alternative: You can still use fluent configuration if needed
    # actor.set_learning_rate(1e-5).set_optimizer("adamw").set_scheduler("linear")
    
    # Or configure multiple things at once
    # actor.configure_training(
    #     learning_rate=1e-5,
    #     optimizer="adamw_8bit", 
    #     scheduler="linear",
    #     scheduler_kwargs={"start_factor": 1.0, "end_factor": 0.1}
    # )
    
    # You can also set a custom reference model factory if needed
    # from transformers import AutoModelForCausalLM
    # actor.set_reference_model(lambda: AutoModelForCausalLM.from_pretrained("some/other/model"))
    
    # Or set a custom model factory for the main model after initialization
    # def another_model_factory():
    #     from transformers import AutoModelForCausalLM
    #     model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
    #     # Apply different modifications here
    #     # model.config.some_parameter = some_value
    #     # model.some_layer.requires_grad_(False)  # Freeze specific layers
    #     return model
    # actor.set_model(another_model_factory)

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
    ] * 10

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
        reference_batch_size=2,
        log_every_n=1,
        data=data,
        std_normalization=False,
        eval_data=eval_data,
        eval_every_n=2,  # Run evaluation every 2 steps
        eval_strategy=EvalStrategy.ALL,  # Run evaluation both periodically and at the end
    )

    import wandb

    wandb.init(project="test_actors", entity="rd211", name="test")
    trainer.train(checkpoint_every_n=30)
    trainer.push_to_hub(
        "rd211/test_actors_main",
        private=True,
    )


if __name__ == "__main__":
    main()
