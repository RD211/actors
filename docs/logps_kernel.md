# Efficient Log-Probabilities Kernel Implementation

When computing reference log probabilities for GRPO, the standard approach requires storing a tensor of shape `BxSxV` (batch size × sequence length × vocabulary size). For large models, this quickly becomes memory-intensive.

For example:

* **Model**: Qwen2.5-7B (vocab size = 152,064)
* **Sequence length**: 16,000 tokens
* **Batch size**: 8

This results in a tensor occupying approximately **40 GB of memory** in BF16 precision, which is substantial.

While typically overshadowed by memory consumption during the subsequent loss calculation stage, optimizing this step to be more memory-efficient allows us to have larger batches.

Our implementation, inspired by the chunked Cross-Entropy loss in [Liger-Kernel](https://github.com/linkedin/Liger-Kernel), computes log probabilities in smaller, manageable chunks. This approach avoids materializing the entire logits tensor at once.


TODO: Add a small plot here showing vram usage vs normal.