import torch
from actors.environments.env_base import Environment, ActorSpec
from actors.actors import vLLMActor
from transformers import AutoTokenizer, AutoModelForCausalLM
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
        self.tokenizer = tok
        spec  = ActorSpec(
            actor_name   ="main",
            model_factory=lambda: AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2.5-0.5B-Instruct", torch_dtype=torch.bfloat16, use_cache=False, trust_remote_code=True
            ),
            tokenizer    = tok,
            loss_factory = lambda: GRPOLoss(beta=0.0, temperature=1.0),
            optim_factory=lambda p: bnb.optim.PagedAdam8bit(p, lr=2e-6),
            scheduler_factory=lambda o: ConstantLR(o, factor=1.0, total_iters=1),
            # reference_model_factory=lambda: AutoModelForCausalLM.from_pretrained(
            #     "Qwen/Qwen2.5-3B-Instruct", torch_dtype=torch.bfloat16, use_cache=False, trust_remote_code=True
            # )
        )
        actor.sleep(1)
        self.register(actor, spec)
        self.actor = actor

    def __call__(self, batch):

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
        rewards = [-len(gen)/1000 for text, gen in zip(texts, generated_texts)]
        self.actor.sleep(1)
        return {
            "main": {
                "input_ids": self.tokenizer(generated_texts).input_ids,
                "rewards": rewards,
                "attention_mask": self.tokenizer(generated_texts).attention_mask,
            }
        }
    

def main():
    env = MyEnv()
    trainer = Trainer(env, 
                      group_size=4, 
                      batch_size=16,
                      grad_accumulation_steps=1, 
                      num_iterations=2,
                      reference_batch_size=2,
                      log_every_n=1,
                      )
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

    # Initialize wandb
    import wandb
    wandb.init(project="test_actors", entity="rd211", name="test")
    trainer.train(data, checkpoint_every_n=30)

if __name__ == "__main__":
    main()