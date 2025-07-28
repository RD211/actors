# Actors: Multi‑(Agent, Turn, Env) RL

<p align="center">
  <img src="https://i.imgur.com/Mk0fSSa.png" alt="Long Banner" width="400">
</p>

<p align="center">
 A hackable library for doing <strong>Multi‑Turn Multi‑Agent RL</strong> with LLMs for the <strong>GPU poor</strong> and <strong>middle class</strong>. Supports some fun environments and makes it very easy to add new ones.
</p>

<p align="center">
  <a href="https://huggingface.co/rl-actors">
    <img alt="Hugging Face Hub" src="https://img.shields.io/badge/🤗%20Hub-RL--Actors-yellow">
  </a>
  <a href="https://pypi.org/project/rl-actors/">
    <img alt="PyPI" src="https://img.shields.io/pypi/v/rl-actors">
  </a>
</p>

---

## Multi‑Trainable‑Agents

This library supports training **multiple different** models together using [Accelerate](https://huggingface.co/docs/accelerate/en/usage_guides/deepspeed_multiple_model).

This allows you to do some very fun stuff, such as adversarial training, collaborative problem solving, multi‑agent collaboration, etc.

Here is a quick simplified example for collaborative problem solving:

```python
# 2 completely different models, both trainable.
bob_actor = vLLMActor(
  name="Bob",
  model_path="Qwen/Qwen2.5-7B-Instruct",
)
alice_actor = vLLMActor(
  name="Alice",
  model_path="meta-llama/Llama-3.1-8B-Instruct",
)

# Loading a math dataset
ds = load_dataset('rl-actors/GSM8K-Easy-Math')

# In this environment they will take turns improving their solution.
env = CollaborativeEnvironment(
  actor_cfgs=[
    CollaborativeActorConfig(
      actor=alice_actor,
      system_prompt="You are Alice",
    ),
    CollaborativeActorConfig(
      actor=bob_actor,
      system_prompt="You are Bob",
    ),
  ],
  reward_functions=[
    # Omitted for brevity.
  ],
  # The order of the rounds is specified with a tiny DSL.
  # Bob starts and then Alice followed by random 5 turns.
  round_spec='Bob -> Alice -> (Bob/Alice)*5',
  train_dataset=ds
)
```

---

## Installation

```bash
git clone https://github.com/RD211/actors.git
pip install .
```

You should always run the code with **accelerate** using a **ZeRO‑3** configuration to be able to use all the features of the library.

```bash
accelerate launch --config_file zero3.yaml your_script.py
```

The library uses **Accelerate**, **DeepSpeed**, **bitsandbytes**, **vLLM**, and **PEFT**, and supports **LoRA** and **QLoRA** training.

Some quickstart examples can be found at `examples/`.

---

## Environments

We plan to have the following environments; suggestions for new environments are welcome:

| Category               | Environment                       | Status | Description                                                                                                                |
| ---------------------- | --------------------------------- | :----: | -------------------------------------------------------------------------------------------------------------------------- |
| Single Trainable Agent | **SingleTurnEnvironment**         |    ✅   | Standard environment with only one actor and one turn.                                                                     |
| Multi Trainable Agent  | **CollaborativeEnvironment**      |    ✅   | Iterates on a task together in alternating turns.                                                                          |
| Multi Trainable Agent  | **ParallelEnvironment**           |    ⏳   | Samples multiple solutions in parallel and combines them at the end. This is probably what Grok 4 Heavy does.              |
| Fun Environments       | **JailbreakEnvironment**          |    ⏳   | One trainable actor tries to convince a frozen actor to do unsafe things from this [dataset](rl-actors/Jailbreak-dataset). |
| Fun Environments       | **CodeforcesParallelEnvironment** |    ⏳   | Same as the parallel environment but with code execution feedback.                                                         |

### Creating a new environment

It is pretty easy to add a new environment, and we recommend making a new environment rather than trying to adapt the current environments for specific tasks.

```python
class CustomEnv(Environment):
  async def generate(self, batch: Map[str, Any]) -> EnvironmentOutput:
    # 1. Sample using your actor.
    problems = batch['problem']
    generations = await alice_actor.agenerate(problems)
    txt_gen = [gen.outputs[0].text for gen in generations]

    # 2. Give rewards (simplified).
    answers = batch['answer']
    rewards = [int(answer in txt) for answer, txt in zip(answers, txt_gen)]

    # 3. We now return the environment results.
    tok = alice_actor.tokenizer

    alice_output = ActorOutput(
      input_ids=tok(txt_gen)['input_ids'],
      rewards=rewards,
    )

    return EnvironmentOutput(
      actors={'Alice': alice_output},
    )
```

### Combining environments

Combining environments is pretty cool. There are two major use cases we see:

* Training on multiple different tasks with different rewards and completely different goals. Coding + Math, Coding + Creative Writing, etc.
* Easily adding evaluation environments to your training.

Here are some examples:

```python
# Training env for Codeforces.
codeforces_env = CodeforcesParallelEnvironment(
  actors=[bob_actor],
  reward_functions=[codeforces_reward]
)

# Training env for math.
math_env = SingleTurnEnvironment(
  actors=[bob_actor],
  reward_functions=[math_correctness],
  prompt_column='problem',
  train_data=load_dataset('rl-actors/GSM8K-Easy-Math', split='train'),
  eval_data={
    'gsm8k': load_dataset('rl-actors/GSM8K-Easy-Math', split='test')
  }
)

# Evaluation environment for AIME.
aime_eval = SingleTurnEnvironment(
  actors=[bob_actor],
  reward_functions=[math_correctness],
  prompt_column='problem',
  eval_data={
    'aime25': load_dataset('math-ai/aime25')
  }
)

# Final combined environment.
env = codeforces_env + math_env + aime_eval
```

---

## Rewards

We do not provide many predefined reward functions as of now, but they can be easily created.
The rewards are made to super easily support judges and very complex workflows.
If you create your own environment you do not even need to explicitly create a reward function, as they can just be part of your environment directly.

However, for our predefined environments you can make rewards as follows:

```python
# Single turn reward
@reward_function(name='math_reward', weight=1.0)
def length_reward(prompt: str, completion: str) -> float:
  return -len(prompt) / 1024

# We support batched rewards and weights too.
@conversation_reward_function(name='math_reward', weight=1.0, batched=True)
def math_reward(conversations: list,
                problem: list,  # Dataset field
                answer: list,   # Also dataset field
                actor_names: list  # allows actor-specific rewards.
              ) -> list[float]:
  # Batched reward functions are designed for Judges.
  # You can use Actors freely in the reward function.
  # ...
  return rewards

# The parameters for the reward functions are automatically filled in as follows:
# For Single turn you will always get the prompt and completion.
# For Conversation you will always get conversations and actor_names.
# For both of them you will get all dataset attributes too, such as `answer` for math data.
```

---

## Memory efficiency

Training multiple models at the same time requires a lot of careful VRAM management. We have thus implemented the following features:

* Full offloading of optimizer states and parameters. This is done during inference but also when switching between different models during the training part. [More details here.](docs/offloading.md)
* Triton kernel for computing log‑probabilities. Helps with long context a bit. [More details here.](docs/logps_kernel.md)
* [Liger kernels](https://github.com/linkedin/Liger-Kernel) for computing the GRPO loss.
* Efficient streamed implementation for updating vLLM weights along with LoRA in‑memory updates. [More details here.](docs/updating_weights.md)

#### Debugging VRAM

In order to debug memory issues try running with `ACTORS_LOGGING_LEVEL='verbose'`.

Sometimes memory becomes very fragmented and can cause OOM errors when switching to the inference part. You can try running with: `PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.3,max_split_size_mb:64` and it might fix the problem.

Sometimes, after a failed run, memory might remain allocated for a while. Make sure to terminate all previous processes before starting a new run.

---

## RL algorithms

Currently there is a **GRPO** and **[GSPO](https://www.arxiv.org/abs/2507.18071)** implementation. Both implementations have both a torch version and a Liger-Kernel chunked version.
> [!NOTE]  
> You can also get a lot of the other implementations such as DAPO, Dr. GRPO just by configuring the existing losses and advantage function.

---

## Actors

We support both hosted API actors and local/trainable actors.

```python
# OpenAI‑style API actor (frozen or for judgment / orchestration)
openai_actor = OpenAIActor(
  name="Judge",
  api_key=os.environ["OPENAI_API_KEY"],
  # base_url can be customized to point at compatible endpoints
)

# Trainable vLLM actors
train_cfg = ActorTrainCfg(
  learning_rate=1e-6,
  beta=0.01,                      # Controls KL
  peft_config=LoraConfig(r=16),   # pass a PEFT/LoRA config if desired
  offload_optimizer=True,
  offload_model=True,
)

bob = vLLMActor(
  name="Bob",
  model_path="Qwen/Qwen2.5-7B-Instruct",
  gpu_groups=[[0, 1]],            # on what GPUs we put the model; allows data‑parallel
  training_config=train_cfg,
)

alice = vLLMActor(
  name="Alice",
  model_path="meta-llama/Llama-3.1-8B-Instruct",
  gpu_groups=1,
  training_config=train_cfg,
)
```

* The **`gpu_groups`** for the `vLLMActor` are on what GPUs we put the model on, and it allows for data‑parallel.

---

## Inspiration

Inspired by [Verifiers](https://github.com/willccbb/verifiers), [TRL](https://github.com/huggingface/trl), and [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF).
