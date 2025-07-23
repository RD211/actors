# Updating Weights

We want to avoid making a copy of the full model in memory in order to send it to vLLM. We also cannot support colocating the vLLM instance with the trainable one like in [Unsloth](https://github.com/unslothai/unsloth), as we can have multiple vLLM instances of the same model, and the distribution of parameters will also be different. Obviously, this would not work for multi-node setups either.

Thus, we do something fun: we **stream weights in batches**. When a vLLM worker receives them, it moves them to a CPU cache, and once all are received, we send a signal to perform the update.

---

Updating weights is generally pretty fast. You can adjust how many tensors are shared at once by setting `update_weights_batch_size` in the actor config:

```python
training_config = ActorTrainCfg(
  update_weights_batch_size=1e9,  # Just one big batch.
)
```

---

## Details

For full parameter finetuning, we make sure each instance of the model in vLLM gets every parameter. That means that if `tensor_parallel_size` is set to 4, then each of the GPUs in this group will get a fourth of the tensors.

If there are multiple instances of the model (data parallel), then we send all required parameters to all instances.

For LoRA, however, we send all parameters to everybody — they’re small, and they need to be split by TP rank and are thus annoying to split up before sending. [Check out](../actors/inference/rlhf_utils.py) the implementation of LoRA weight updating for the most hacky code.

---

### 🕒 WIP (Coming Soon)

If there is only one big batch, there’s no need to move it to CPU and then back to GPU — we should probably skip this step.

FP8 weight loading with inflight quantization is coming soon (hopefully). It currently works on a single GPU that supports FP8, such as a 4090.
