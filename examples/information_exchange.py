import difflib
import os
import re

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import BitsAndBytesConfig
from vllm import SamplingParams

from actors import (
    ActorTrainCfg,
    CollaborativeActorConfig,
    CollaborativeEnvironment,
    GRPOTrainer,
    GRPOTrainerCfg,
    conversation_reward_function,
    vLLMActor,
)

# Create quantization configuration
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_quant_storage=torch.bfloat16,
)

# Initialize two actors
ben = vLLMActor(
    name="Ben",
    model_path="Qwen/Qwen2.5-3B-Instruct",
    engine_kwargs={
        "gpu_memory_utilization": 0.5,
        "max_model_len": 2048,
        "quantization": "bitsandbytes",
    },
    training_config=ActorTrainCfg(
        learning_rate=5e-6,
        optimizer="adamw_32bit",
        peft_config=LoraConfig(
            r=64,
            lora_alpha=128,
            target_modules="all-linear",
            lora_dropout=0.0,
            task_type=TaskType.CAUSAL_LM,
        ),
        quantization_config=quantization_config,
        beta=0.0,
    ),
)

alice = vLLMActor(
    name="Alice",
    model_path="meta-llama/Llama-3.2-3B-Instruct",
    engine_kwargs={
        "gpu_memory_utilization": 0.5,
        "max_model_len": 2048,
        "quantization": "bitsandbytes",
    },
    training_config=ActorTrainCfg(
        learning_rate=5e-6,
        optimizer="adamw_32bit",
        peft_config=LoraConfig(
            r=64,
            lora_alpha=128,
            target_modules="all-linear",
            lora_dropout=0.0,
            task_type=TaskType.CAUSAL_LM,
        ),
        quantization_config=quantization_config,
        beta=0.0,
    ),
)


ds = load_dataset("ugursa/Yahoo-Finance-News-Sentences", split="train").remove_columns(
    ["label"]
)

# We split it into two parts.
ben_ds = ds.select(range(0, len(ds), 2)).shuffle()
alice_ds = ds.select(range(1, len(ds), 2)).shuffle()

# We combine in one ds with text_ben and text_alice columns.
ben_ds = ben_ds.rename_column("text", "text_ben")
alice_ds = alice_ds.rename_column("text", "text_alice")
ds = ben_ds.add_column("text_alice", alice_ds["text_alice"])


ben_prompt = r"""
You are **Ben** and you are chatting with **Alice**.
Alice has a secret sentence for you, and you have one for her.
Your goal is to swap the two sentences as quickly as possible.

**What to do**

1. Read the sentence written below (it is only for you).
2. After Alice successfully sends you *her* secret sentence, reply with **only** the following format:

<Answer>
(the sentence from Alice)
</Answer>

Do **not** put your own sentence inside these tags.
Keep the conversation as short as possible—if it drags on, the police might catch you before the swap is done.

Here is your sentence:
{{ text_ben }}
"""

alice_prompt = r"""
You are **Alice** and you are chatting with **Ben**.
Ben has a secret sentence for you, and you have one for him.
Your goal is to swap the two sentences as quickly as possible.

**What to do**

1. Read the sentence written below (it is only for you).
2. After Ben successfully sends you *his* secret sentence, reply with **only** the following format:

<Answer>
(the sentence from Ben)
</Answer>

Do **not** put your own sentence inside these tags.
Keep the conversation as short as possible—if it drags on, the police might catch you before the swap is done.

Here is your sentence:
{{ text_alice }}
"""

sampling_params = SamplingParams(temperature=1.0, max_tokens=256)

ben_config = CollaborativeActorConfig(
    actor=ben, system_prompt=ben_prompt, sampling_params=sampling_params
)

alice_config = CollaborativeActorConfig(
    actor=alice, system_prompt=alice_prompt, sampling_params=sampling_params
)

ben_tokenizer = ben.tokenizer
alice_tokenizer = alice.tokenizer


@conversation_reward_function(name="len_penalty")
def len_penalty(conversation, actor_name):
    messages_assistant = "".join(
        [turn["content"] for turn in conversation if turn["role"] == "assistant"]
    )
    messages_user = "".join(
        [turn["content"] for turn in conversation if turn["role"] == "user"]
    )
    if actor_name == "Ben":
        len_tokens_ben = len(ben_tokenizer.encode(messages_assistant))
        len_tokens_alice = len(alice_tokenizer.encode(messages_user))
    else:
        len_tokens_ben = len(ben_tokenizer.encode(messages_user))
        len_tokens_alice = len(alice_tokenizer.encode(messages_assistant))

    return -(len_tokens_ben + len_tokens_alice) / 256.0


@conversation_reward_function(name="correct_answer_reward", batched=True)
def correct_answer_reward(conversation, actor_name, text_ben, text_alice):
    conversations = conversation
    actor_names = actor_name
    rewards = []
    for conv, actor_name, txt_ben, txt_alice in zip(
        conversations, actor_names, text_ben, text_alice, strict=False
    ):
        messages_assistant = "".join(
            [turn["content"] for turn in conv if turn["role"] == "assistant"]
        )
        messages_user = "".join(
            [turn["content"] for turn in conv if turn["role"] == "user"]
        )
        if actor_name == "Ben":
            answer_ben = re.findall(
                r"<Answer>\s*(.*?)\s*</Answer>", messages_assistant, re.DOTALL
            )
            answer_alice = re.findall(
                r"<Answer>\s*(.*?)\s*</Answer>", messages_user, re.DOTALL
            )
        else:
            answer_ben = re.findall(
                r"<Answer>\s*(.*?)\s*</Answer>", messages_user, re.DOTALL
            )
            answer_alice = re.findall(
                r"<Answer>\s*(.*?)\s*</Answer>", messages_assistant, re.DOTALL
            )

        if not answer_ben or not answer_alice:
            rewards.append(0.0)
            continue
        answer_ben = answer_ben[-1].strip().lower()
        answer_alice = answer_alice[-1].strip().lower()
        correct_answer_ben = txt_alice.strip().lower()
        correct_answer_alice = txt_ben.strip().lower()
        similarity = difflib.SequenceMatcher(
            None, answer_ben, correct_answer_ben
        ).ratio()
        similarity += difflib.SequenceMatcher(
            None, answer_alice, correct_answer_alice
        ).ratio()
        similarity /= 2.0  # Average similarity for both answers
        rewards.append(similarity)

    return rewards


def main():
    env = CollaborativeEnvironment(
        actor_cfgs=[ben_config, alice_config],
        train_data=ds,
        mask_other_agents_for_loss=True,
        reward_functions=[len_penalty, correct_answer_reward],
        round_spec="Alice -> Ben -> Alice -> Ben",
        prompt_column="text_ben",  # Not actually used here.
    )

    trainer = GRPOTrainer(
        env=env,
        cfg=GRPOTrainerCfg(
            batch_size=32,
            group_size=8,
            grad_accumulation_steps=2,
            num_iterations=2,
            max_steps=500,
        ),
        actors=[alice, ben],
    )

    import wandb

    if os.getenv("RANK") == "0":
        wandb.init(project="actors", name="information_exchange")

    trainer.train()


if __name__ == "__main__":
    main()
