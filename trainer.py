import torch
from src.trainer.environments.env_base import Environment, ActorSpec
from src.trainer.actors.actors import vLLMActor
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.trainer.losses.grpo_loss import GRPOLoss
from src.trainer.losses.liger_grpo_loss import LigerLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, ConstantLR
from vllm import SamplingParams
from src.trainer.trainers.trainer import Trainer
import bitsandbytes as bnb

class MyEnv(Environment):
    def __init__(self):
        super().__init__()
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

        actor = vLLMActor(name="main", model_path="Qwen/Qwen2.5-0.5B-Instruct",
                          engine_kwargs=
            {
                "gpu_memory_utilization": 0.3,
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
            loss_factory = lambda: GRPOLoss(beta=0.04, temperature=1.0),
            optim_factory=lambda p: bnb.optim.Adam8bit(p, lr=2e-6, is_paged=True),
            scheduler_factory=lambda o: ConstantLR(o, factor=1.0, total_iters=1),
            reference_model_factory=lambda: AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2.5-0.5B-Instruct", torch_dtype=torch.bfloat16, use_cache=False, trust_remote_code=True
            )
        )
        actor.sleep(1)
        self.register(actor, spec)
        self.actor = actor

    def __call__(self, batch):

        self.actor.wake()
        texts = batch["text"]
        print("Input texts:", texts)
        generations = self.actor.chat(
            [[{ "role": "user", "content": text }] for text in texts],
            sampling_params=SamplingParams(
                temperature=1.0,
                max_tokens=256,
            )
        )
        generated_texts = [self.tokenizer.apply_chat_template([{"role": "user", "content": text}, 
                                                               {"role": "assistant", "content": gen.outputs[0].text}], tokenize=False) for text, gen in zip(texts, generations)]
        print("Generated texts:", generated_texts[0])
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
                      grad_accumulation_steps=16, 
                      reference_batch_size=2)
    data = {
        "text": [
            "What is the capital of France?",
            "Explain the theory of relativity.",
            "What is the meaning of life?",
            "Describe the process of photosynthesis."
        ]
    }

    for _ in range(10000):  # Run for 10 training steps
        metrics = trainer.train_step(data)
        # Pretty print the metrics
        print("-" * 40)
        for actor_name, actor_metrics in metrics.items():
            print(f"Actor: {actor_name}")
            for metric_name, metric_value in actor_metrics.items():
                print(f"  {metric_name}: {metric_value:.4f}")
        print("-" * 40)
    print("Training step completed successfully.")

if __name__ == "__main__":
    main()