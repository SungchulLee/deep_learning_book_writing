# Model Sharding

## Learning Objectives

- Understand model sharding strategies
- Calculate memory requirements for different configurations
- Compare sharding frameworks

## Why Shard?

Large models don't fit on a single GPU:

| Model | FP16 Size | Single GPU? |
|-------|----------|-------------|
| LLaMA-7B | 14 GB | Yes (A100 80GB) |
| LLaMA-13B | 26 GB | Yes (A100 80GB) |
| LLaMA-70B | 140 GB | No → 2+ GPUs |
| LLaMA-405B | 810 GB | No → 10+ GPUs |

## Sharding Strategies

| Strategy | How It Works | Communication | Best For |
|----------|-------------|---------------|---------|
| Tensor Parallelism | Split individual layers | All-reduce per layer | Latency-sensitive |
| Pipeline Parallelism | Assign whole layers to GPUs | Point-to-point | Throughput |
| Data Parallelism | Replicate model, split data | All-reduce gradients | Training |
| Expert Parallelism | Assign MoE experts to GPUs | All-to-all routing | MoE models |

## Memory Calculation

Per-GPU memory for tensor parallelism with $P$ GPUs:

$$M_{\text{per\_GPU}} = \frac{M_{\text{model}}}{P} + M_{\text{KV\_cache}} + M_{\text{activations}}$$

Note: KV cache and activations are **not** divided by $P$ in tensor parallelism.

```python
def estimate_gpu_memory(
    model_params_B: float,
    n_gpus: int,
    precision_bytes: int = 2,  # fp16
    context_length: int = 4096,
    batch_size: int = 1,
    n_layers: int = 80,
    hidden_dim: int = 8192,
    n_kv_heads: int = 8,
):
    # Model weights per GPU (tensor parallel)
    model_mem = model_params_B * 1e9 * precision_bytes / n_gpus

    # KV cache per GPU (not split in TP)
    head_dim = hidden_dim // (n_kv_heads * 8)  # Assuming GQA
    kv_per_layer = 2 * batch_size * context_length * n_kv_heads * head_dim * precision_bytes
    kv_total = kv_per_layer * n_layers

    total = model_mem + kv_total
    return {"model_GB": model_mem / 1e9, "kv_GB": kv_total / 1e9,
            "total_GB": total / 1e9}
```

## Frameworks

| Framework | Tensor Parallel | Pipeline Parallel | Key Feature |
|-----------|----------------|-------------------|-------------|
| DeepSpeed | ZeRO stages | Yes | Memory optimization |
| FSDP (PyTorch) | Sharded params | Limited | Native PyTorch |
| Megatron-LM | Yes | Yes | NVIDIA-optimized |
| vLLM | Yes | No | Inference-focused |

## References

1. Shoeybi, M., et al. (2020). "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism." *arXiv*.
2. Rajbhandari, S., et al. (2020). "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models." *SC*.
