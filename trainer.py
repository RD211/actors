import torch
from actors.environments.env_base import Environment, ActorSpec
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

class MyEnv(Environment):
    def __init__(self):
        super().__init__()
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

        actor = vLLMActor(name="main", model_path="Qwen/Qwen2.5-0.5B-Instruct",
                          engine_kwargs=
            {
                "gpu_memory_utilization": 0.5,
                "max_model_len": 2048,
            }
        )
        actor.sleep(1)
        self.tokenizer = tok
        spec  = ActorSpec(
            actor_name   ="main",
            model_factory=lambda: AutoLigerKernelForCausalLM.from_pretrained(
                "Qwen/Qwen2.5-0.5B-Instruct", torch_dtype=torch.bfloat16, use_cache=False, trust_remote_code=True
            ),
            tokenizer    = tok,
            loss_factory = lambda: GRPOLoss(beta=0.0, temperature=1.0),
            optim_factory=lambda p: bnb.optim.PagedAdam8bit(p, lr=2e-6),
            scheduler_factory=lambda o: ConstantLR(o, factor=1.0, total_iters=1),
            reference_model_factory=lambda: AutoLigerKernelForCausalLM.from_pretrained(
                "Qwen/Qwen2.5-0.5B-Instruct", torch_dtype=torch.bfloat16, use_cache=False, trust_remote_code=True
            )
        )

        actor2 = vLLMActor(name="main2", model_path="Qwen/Qwen2.5-0.5B-Instruct",
                          engine_kwargs=
            {
                "gpu_memory_utilization": 0.5,
                "max_model_len": 2048,
            }
        )
        spec2 = ActorSpec(
            actor_name   ="main2",
            model_factory=lambda: AutoLigerKernelForCausalLM.from_pretrained(
                "Qwen/Qwen2.5-0.5B-Instruct", torch_dtype=torch.bfloat16, use_cache=False, trust_remote_code=True
            ),
            tokenizer    = tok,
            loss_factory = lambda: LigerLoss(beta=0.0, temperature=1.0, loss_type="bnpo"),
            optim_factory=lambda p: bnb.optim.PagedAdam8bit(p, lr=2e-6),
            scheduler_factory=lambda o: LinearLR(o, start_factor=1.0, end_factor=0.0, total_iters=1),
            reference_model_factory=lambda: AutoLigerKernelForCausalLM.from_pretrained(
                "Qwen/Qwen2.5-0.5B-Instruct", torch_dtype=torch.bfloat16, use_cache=False, trust_remote_code=True
            )
        )
        actor2.sleep(1)
        self.register(actor, spec)
        self.actor = actor
        self.register(actor2, spec2)
        self.actor2 = actor2

    def __call__(self, batch):
        
        self.actor2.sleep(1)
        self.actor.wake()
        texts = batch["text"]
        generations = self.actor.chat(
            [[{ "role": "user", "content": text }] for text in texts],
            sampling_params=SamplingParams(
                temperature=1.0,
                max_tokens=256,
            )
        )
        generated_texts = [self.tokenizer.apply_chat_template([{"role": "user", "content": text}, 
                                                               {"role": "assistant", "content": gen.outputs[0].text}], tokenize=False) for text, gen in zip(texts, generations)]
        
        # Calculate different types of rewards
        length_penalties = [-len(gen)/1000 for text, gen in zip(texts, generated_texts)]
        quality_scores = [0.5 * (1 + hash(gen) % 100 / 100) for gen in generated_texts]  # Mock quality score
        total_rewards = [lp + qs for lp, qs in zip(length_penalties, quality_scores)]
        
        tokenized = self.tokenizer(generated_texts)
        
        self.actor.sleep(1)
        return EnvironmentOutput(
            actors={
                "main": ActorOutput(
                    input_ids=tokenized.input_ids,
                    attention_mask=tokenized.attention_mask,
                    rewards=total_rewards,
                    reward_components={
                        "length_penalty": length_penalties,
                        "quality_score": quality_scores,
                    }
                ),
                "main2": ActorOutput(
                    input_ids=tokenized.input_ids,
                    attention_mask=tokenized.attention_mask,
                    rewards=total_rewards,
                    reward_components={
                        "length_penalty": length_penalties,
                        "quality_score": quality_scores,
                    }
                )
            }
        )
    

def main():
    env = MyEnv()
    data = [
            {"text": "What is the capital of France?"},
            {"text":"Explain the theory of relativity."},
            {"text": "How does quantum computing work?"},
            {"text": "What is the weather like today?"},
            {"text": "Tell me a joke."},
            {"text": "What is the largest mammal?"},
            {"text": "Who wrote 'To Kill a Mockingbird'?"},
            {"text": "What is the speed of light?"},
            {"text": "How do you make a cake?"},
    ] * 120
    trainer = Trainer(env, 
                      group_size=4, 
                      batch_size=16,
                      grad_accumulation_steps=1, 
                      num_iterations=1,
                      reference_batch_size=2,
                      log_every_n=1,
                      data=data,
                      )

    # Initialize wandb
    import wandb
    wandb.init(project="test_actors", entity="rd211", name="test")
    trainer.train(checkpoint_every_n=30)
    trainer.push_to_hub({
        'main': 'rd211/test_actors_main',
        'main2': 'rd211/test_actors_main2'
    }, private=True)

if __name__ == "__main__":
    main()