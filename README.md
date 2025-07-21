# Actors: Multi-(Agent, Turn, Env) RL

<p align="center">
  <img src="https://i.imgur.com/Mk0fSSa.png" alt="Long Banner" width="400">
</p>

<p align="center">
 A hackable library for doing <strong>Multi-Turn Multi-Agent RL</strong> with LLMs for the <strong>GPU poor</strong> and <strong>middle class</strong>.  Supports some fun environments and makes it very easy to add new ones.
</p>

<p align="center">
  <a href="https://huggingface.co/rl-actors">
    <img alt="Hugging Face Hub" src="https://img.shields.io/badge/🤗%20Hub-RL--Actors-yellow">
  </a>
</p>


## Multi-Trainable-Agents
This library supports training **multiple different** models together using the [experimental API from Accelerate](https://huggingface.co/docs/accelerate/en/usage_guides/deepspeed_multiple_model).

This allows you to do some very fun stuff. such as adversarial training, collaborative problem solving, multi-agent collaboration etc.

Here is a quick example for collaborative problem solving:
```python
# 2 Completly different models, both trainable.
main_actor = vLLMActor(
    name="main_actor",
    model_path="Qwen/Qwen2.5-7B-Instruct",
)
second_actor = vLLMActor(
    name="second_actor",
    model_path="meta-llama/Llama-3.1-8B-Instruct",
)

# In this environment they will take turns improving their solution.
env = CollaborativeMathProblemSolving(
    actors=[main_actor, second_actor],
    max_turns=5
)
```
## Memory Efficiency

Training multiple models at the same time requires a lot of careful VRAM management. We have thus implemented the following features:

- Full offloading of optimizer states and parameters. This is done during inference but also when switching from different models during the training part. [More details here.](docs/offloading.md)
- Custom triton kernel for computing log-probabilities. Helps with long context. [More details here.](docs/logps_kernel.md)
- [Liger kernels](https://github.com/linkedin/Liger-Kernel) for computing the GRPO loss.
- Efficient streamed implementation for updating vLLM weights and LoRA weight updates fully in memory. No need to write to disk. [More details here](docs/updating_weights.md)

## RL Algorithms
Currently there is only a GRPO implementation, however by configuring the advantage calculator and some loss settings you can get some of the variants like Dr. GRPO.

We plan on releasing and making special trainers for more RL methods after all basic functionality and performance is good enough.


## Environments

We plan to have the following:

### Single Trainable Agent
- SimpleSingleTurnEnvironment - Standard environment
- CollaborativeProblemSolvingEnvironment - Iterates on a problem in a round robin fashion with the provided models. 
- HeavyAgentProblemSolvingEnvironment - Given a problem does thinking with multiple agents in parallel and then merges solutions. Probably what Grok 4 heavy does I suppose or the pro versions of OpenAI models.
- JailbreakEnvironment - two agents, one is frozen and one is training, the model that is training tries to trick the frozen model into performing tasks it was instructed not to perform.

### Multi Trainable Agent
- HeavyMultiAgentProblemSolvingEnvironment - Same as the single agent one but now the multiple samples come from different models and can improve diversity. The model that combines the solutions can be selected as one of the models or a non-trainable one.
- CollaborativeProblemSolvingEnvironment - Iterates on a problem together for specific number of rounds.
- 

### Fun
- AdversarialChessEnvironment - chess game basically.
- SelfPlayChessEnvironment - plays chess on its own.