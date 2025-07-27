# Offloading

#### ⚠️ Note
| Offloading  is **only supported with DeepSpeed ZeRO-3**.

You can enable **model and optimizer offloading** with just a couple of flags:

```python
# Create training config
training_config = ActorTrainCfg(
    offload_model=True,
    offload_optimizer=True,
)

# Create actor with offloading
actor = vLLMActor(
    name="main",
    model_path="Qwen/Qwen2.5-7B-Instruct",
    training_config=training_config,
)
```

With this setup, the **model and optimizer are offloaded when not in use**, reducing GPU memory usage. Unlike standard DeepSpeed CPU offloading—which repeatedly moves weights between CPU and GPU—we move the model to GPU once, run all necessary computations, and then offload.

This is particularly useful when training with **multiple actors**: only the active one stays on GPU, while the others remain fully offloaded.

Model load/offload time is generally negligible, so both `offload_model` and `offload_optimizer` are **enabled by default** in the actor config.

---

### 🛠 Implementation Details

See [`deepspeed.py`](../actors/utils/deepspeed.py) for the implementation. We've extended DeepSpeed’s `offload_states` and `reload_states` to:

* Support more optimizer types
* Offload *all* memory, not just partial states
* Work reliably with **LoRA**, which previously had issues in our setup.

---

### 🕒 WIP (Coming Soon)

We are trying to see if we can offload parts of the optimizer states during loss calculations. This would give us a bit more headroom for higher context lengths.
