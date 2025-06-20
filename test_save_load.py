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
from deepspeed.ops.adam import DeepSpeedCPUAdam

import os, json, torch, random

from src.trainer.environments.env_base import Environment
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
    

# ---------------- helpers ----------------------------------------------------
def print_step(tr, msg):
    if tr.accel.is_main_process:
        print(f"{msg}  current step = {tr._step}")

def run_n_steps(trainer, n, batch):
    for _ in range(n):
        trainer.train_step(batch)

# ---------------- main -------------------------------------------------------
def main():
    # reproducibility for the test only
    torch.manual_seed(0)
    random.seed(0)

    batch = {
        "text": [
            "What is the capital of France?",
            "Explain the theory of relativity.",
            "What is the meaning of life?",
            "Describe the process of photosynthesis.",
        ]
    }

    # --- 1) create env + trainer ------------------------------------------------
    env1     = MyEnv()
    trainer1 = Trainer(
        env1,
        group_size=4,
        batch_size=16,
        grad_accumulation_steps=1,
        num_iterations=1,
        reference_batch_size=2,
        gradient_checkpointing=False,
    )

    # --- 2) run a few steps -----------------------------------------------------
    run_n_steps(trainer1, n=3, batch=batch)
    print_step(trainer1, "After first 3 steps:")

    # --- 3) save checkpoint -----------------------------------------------------
    ckpt_dir = "./ckpt_demo"
    trainer1.save_checkpoint(ckpt_dir)
    if trainer1.accel.is_main_process:
        print(f"Checkpoint written to {ckpt_dir}")
        print("Files in checkpoint dir:", os.listdir(ckpt_dir))
        # Show that trainer_state.json captured the step
        print("trainer_state.json:", json.load(open(os.path.join(ckpt_dir, 'trainer_state.json'))))

    # ---------------------------------------------------------------------------
    # --- 4) start a *new* trainer and load the checkpoint -----------------------
    env2     = MyEnv()                      # need fresh env/actors
    trainer2 = Trainer(
        env2,
        group_size=4,
        batch_size=16,
        grad_accumulation_steps=1,
        num_iterations=1,
        reference_batch_size=2,
        gradient_checkpointing=False,
    )

    trainer2.load_checkpoint(ckpt_dir)
    print_step(trainer2, "After load_checkpoint:")

    # --- 5) keep training; step counter should continue ------------------------
    run_n_steps(trainer2, n=2, batch=batch)
    print_step(trainer2, "After 2 more steps:")

if __name__ == "__main__":
    main()
