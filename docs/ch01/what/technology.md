# Algorithms as Technology

Algorithms are a technology that often matters more than hardware. In deep learning, algorithmic improvements (attention mechanisms, residual connections, better optimizers) have driven progress far more than raw compute increases alone.

## Definition

Total system performance is the product of hardware capability and algorithmic efficiency:

$$
\text{Total performance} = \text{Hardware speed} \times \text{Algorithm efficiency}
$$

An algorithmic improvement that reduces complexity from $O(n^2)$ to $O(n \log n)$ provides a speedup that grows with $n$, eventually dwarfing any constant-factor hardware advantage.

## Explanation

The history of deep learning demonstrates the power of algorithmic innovation:

- **Residual connections** (2015): Enabled training of networks with 100+ layers where previous architectures failed beyond 20 layers. No hardware change was required.
- **Attention mechanism** (2017): Replaced sequential RNN computation with parallelizable matrix operations, enabling training on much longer sequences with the same hardware.
- **Mixed-precision training** (2018): Reduced memory and computation by using float16 for most operations while maintaining float32 for critical accumulations. This is a pure algorithmic/software improvement.
- **Flash Attention** (2022): Achieved 2-4x speedup on attention computation by rearranging memory access patterns -- same hardware, same mathematical result, much faster execution.

The implication for practitioners: before purchasing more compute, check whether a better algorithm, architecture, or training recipe exists. Algorithmic improvements are free and often provide larger gains than hardware upgrades.

## Examples

```python
import torch
import time

# Naive attention vs fused attention (algorithmic improvement)
def naive_attention(Q, K, V):
    """Materializes full n x n attention matrix."""
    scores = Q @ K.transpose(-2, -1) / Q.shape[-1] ** 0.5
    weights = torch.softmax(scores, dim=-1)
    return weights @ V

n, d = 1024, 64
Q = K = V = torch.randn(1, n, d)

# Naive: allocates O(n^2) intermediate
start = time.time()
for _ in range(10):
    out_naive = naive_attention(Q, K, V)
naive_time = (time.time() - start) / 10

# PyTorch scaled_dot_product_attention (uses memory-efficient algorithm)
start = time.time()
for _ in range(10):
    out_fused = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
fused_time = (time.time() - start) / 10

print(f"Naive attention:  {naive_time*1000:.2f} ms")
print(f"Fused attention:  {fused_time*1000:.2f} ms")
print(f"Same result: {torch.allclose(out_naive, out_fused, atol=1e-5)}")
```
