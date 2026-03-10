# Efficiency

Efficiency measures how computational resources (time, memory, energy) scale with problem size. In deep learning, efficiency determines whether a model can be trained in days versus years and whether it can run inference within a latency budget.

## Definition

The efficiency of an algorithm is characterized by its time complexity $T(n)$ and space complexity $S(n)$ as functions of input size $n$. For neural networks, the relevant quantities are:

$$
\text{Training cost} = O(\text{epochs} \times n \times d \times L) \qquad \text{Inference cost} = O(d^2 \times L)
$$

where $n$ is the dataset size, $d$ is the hidden dimension, and $L$ is the number of layers.

## Explanation

Efficiency improvements often matter more than algorithmic novelty. The key complexity classes relevant to deep learning:

| Complexity | Example | Feasibility |
|---|---|---|
| $O(1)$ | Hash lookup, cached embedding | Instant |
| $O(n)$ | Linear scan, single forward pass | Fast |
| $O(n \log n)$ | Sorting, FFT-based convolution | Practical |
| $O(n^2)$ | Self-attention (standard Transformer) | Limits sequence length |
| $O(n^3)$ | Matrix inversion, naive attention | Only small $n$ |

The Transformer's $O(n^2)$ attention complexity explains why standard models cap sequence length at a few thousand tokens. Linear attention variants and sparse attention reduce this to $O(n)$ or $O(n \log n)$, enabling much longer sequences.

Memory efficiency is equally critical: training a model that fits in GPU memory requires techniques like gradient checkpointing, mixed-precision training, and gradient accumulation.

## Examples

```python
import torch
import time

# Compare O(n^2) attention vs O(n) linear scan
def quadratic_attention(Q, K, V):
    """Standard dot-product attention: O(n^2 d)."""
    scores = Q @ K.transpose(-2, -1) / Q.shape[-1] ** 0.5
    weights = torch.softmax(scores, dim=-1)
    return weights @ V

# Measure scaling
for n in [128, 512, 2048]:
    d = 64
    Q = K = V = torch.randn(1, n, d)
    start = time.time()
    for _ in range(10):
        _ = quadratic_attention(Q, K, V)
    elapsed = (time.time() - start) / 10
    print(f"n={n:>5d}: attention time={elapsed*1000:.2f}ms")

# Memory: estimate model size
def model_memory_mb(params):
    return params * 4 / (1024 ** 2)  # float32 = 4 bytes

d, L = 768, 12  # BERT-base dimensions
params = L * (4 * d * d + 4 * d)  # approximate
print(f"\nBERT-base-like: ~{params:,} params = {model_memory_mb(params):.1f} MB")
```
