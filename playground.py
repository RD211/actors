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
import bitsandbytes as bnb
from actors.environments import SimpleSingleTurnEnvironment
from torch.optim.lr_scheduler import CosineAnnealingLR

def length_reward(completion: str) -> float:
    """Rewards shorter responses."""
    return -min(len(completion) / 500, 1.0)  # Negative reward for length, capped at -1.0
def main():
    actor = vLLMActor(
        name="main",
        model_path="Qwen/Qwen2.5-0.5B-Instruct",
        engine_kwargs={
            "gpu_memory_utilization": 0.5,
            "max_model_len": 2048,
        },
    )
    tokenizer = actor.training_config.tokenizer_factory()
    actor.training_config.learning_rate(2e-6).loss(
        LigerLoss(beta=0.04, temperature=1.0)
    ).optimizer(
       bnb.optim.PagedAdam32bit
    ).scheduler(
        lambda o, t_max: CosineAnnealingLR(o, T_max=t_max)
    )

    actor.sleep(1)
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
    ] * 5

    data = [{'conversation':tokenizer.apply_chat_template([{'role': 'user', 'content': item['text']}], tokenize=False, add_generation_prompt=True)} for item in data]
    trainer = Trainer(
        env,
        group_size=16,
        batch_size=64,
        grad_accumulation_steps=4,
        num_iterations=4,
        reference_batch_size=2,
        log_every_n=1,
        data=data,
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
