# Paged Attention

## Learning Objectives

- Understand the KV cache memory management problem
- Learn how paged attention applies virtual memory concepts
- Understand vLLM's architecture

## The Problem

Standard KV cache allocation pre-allocates a contiguous block for the maximum sequence length per request. This leads to:

1. **Internal fragmentation**: Short sequences waste pre-allocated memory
2. **External fragmentation**: Variable-length sequences create unusable gaps
3. **Over-allocation**: Must reserve for max possible length

Typical memory waste: **60-80%** of allocated KV cache memory.

## Paged Attention Solution

**PagedAttention** (Kwon et al., 2023) applies operating system virtual memory concepts:

- KV cache is divided into fixed-size **blocks** (pages)
- Blocks are allocated **on demand** as the sequence grows
- A **block table** maps logical positions to physical blocks
- Blocks can be **non-contiguous** in GPU memory

```
Logical KV Cache:  [Block 0][Block 1][Block 2][Block 3]
                      ↓        ↓        ↓        ↓
Physical Memory:   [Block 7][Block 2][Block 5][Block 9]
                   (mapped via block table)
```

## Benefits

| Aspect | Standard | Paged Attention |
|--------|---------|----------------|
| Memory waste | 60-80% | <4% |
| Max batch size | Limited | 2-4x larger |
| Memory sharing | Impossible | Shared prefixes |
| Throughput | Baseline | 2-4x improvement |

## vLLM

vLLM is the reference implementation of PagedAttention:

```python
from vllm import LLM, SamplingParams

# Initialize with PagedAttention
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=4,
    gpu_memory_utilization=0.95,  # Use most of GPU memory
)

# Efficient batch inference
prompts = ["Analyze AAPL earnings...", "Summarize the Fed minutes..."]
sampling_params = SamplingParams(temperature=0.7, max_tokens=512)
outputs = llm.generate(prompts, sampling_params)
```

## References

1. Kwon, W., et al. (2023). "Efficient Memory Management for LLM Serving with PagedAttention." *SOSP*.
