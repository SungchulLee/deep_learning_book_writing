# Inference Optimization Overview

## Learning Objectives

- Understand why LLM inference is memory-bandwidth bound
- Distinguish prefill and decode phases
- Identify key inference metrics and optimization techniques

## The Inference Challenge

LLM inference is fundamentally **memory-bandwidth bound**, not compute-bound. During autoregressive decoding, each token generation requires reading the entire model weights from memory but performs relatively few FLOPs per byte read.

**Arithmetic intensity** during decoding:

$$\text{AI} = \frac{\text{FLOPs}}{\text{Bytes}} \approx \frac{2N}{2N / \text{batch\_size}} = \text{batch\_size}$$

For batch size 1, AI $\approx 1$ FLOP/byte—far below GPU peak compute/bandwidth ratios (~100-300 for modern GPUs).

## Two Phases of Inference

### Prefill Phase

Process the entire input prompt in parallel:

- **Compute-bound**: Full matrix multiplications over all input tokens
- **Latency**: Proportional to prompt length
- **Metric**: Time to First Token (TTFT)

### Decode Phase

Generate tokens one at a time, autoregressively:

- **Memory-bound**: Each step reads full model weights for one token
- **Latency**: Per-token generation time
- **Metric**: Tokens Per Second (TPS)

## Key Metrics

| Metric | Definition | Target |
|--------|-----------|--------|
| TTFT | Time to first output token | <500ms (interactive) |
| TPS | Tokens generated per second | >30 TPS (real-time) |
| Throughput | Total tokens/second across batch | Maximize |
| Latency (P99) | 99th percentile total response time | <5s |

## Optimization Landscape

| Technique | Phase | Benefit | Section |
|-----------|-------|---------|---------|
| KV Cache | Decode | Avoid recomputation | 15.7 |
| Flash Attention | Both | Memory-efficient attention | 15.7 |
| Quantization | Both | Reduce memory, increase throughput | 15.7 |
| Paged Attention | Decode | Efficient memory management | 15.7 |
| Continuous Batching | Decode | Higher throughput | 15.7 |
| Speculative Decoding | Decode | Lower latency | 15.7 |
| Tensor Parallelism | Both | Distribute across GPUs | 15.7 |
| Pipeline Parallelism | Both | Distribute across GPUs | 15.7 |

## Memory Breakdown

For a 70B parameter model (fp16):

```
Model weights:     140 GB
KV cache (2048 ctx, batch 32):  ~40 GB
Activations:       ~10 GB
─────────────────────────
Total:            ~190 GB → 3× A100 80GB
```

## References

1. Pope, R., et al. (2023). "Efficiently Scaling Transformer Inference." *MLSys*.
2. Kwon, W., et al. (2023). "Efficient Memory Management for LLM Serving with PagedAttention." *SOSP*.
