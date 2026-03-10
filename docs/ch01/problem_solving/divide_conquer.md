# Divide and Conquer

Divide and conquer breaks a problem into independent subproblems, solves them recursively, and combines the results. This paradigm appears throughout deep learning in multi-head attention, recursive neural networks, and hierarchical feature extraction.

## Definition

A divide-and-conquer algorithm has three steps: **divide** the problem into $a$ subproblems of size $n/b$, **conquer** each subproblem recursively, and **combine** the solutions. The running time satisfies the recurrence:

$$
T(n) = a \, T\!\left(\frac{n}{b}\right) + f(n)
$$

where $f(n)$ is the cost of dividing and combining.

## Explanation

The key insight is that solving smaller subproblems independently and combining is often faster than solving the full problem directly. The Master Theorem gives the solution:

- If $f(n) = O(n^{\log_b a - \epsilon})$, then $T(n) = \Theta(n^{\log_b a})$ (recursion dominates)
- If $f(n) = \Theta(n^{\log_b a})$, then $T(n) = \Theta(n^{\log_b a} \log n)$ (balanced)
- If $f(n) = \Omega(n^{\log_b a + \epsilon})$, then $T(n) = \Theta(f(n))$ (combine dominates)

In deep learning, divide and conquer appears in:

- **Multi-head attention**: Divides the embedding dimension into $h$ heads, computes attention independently, then concatenates (combines).
- **Hierarchical models**: U-Net processes features at multiple resolutions, splitting and merging at each level.
- **Parallel computation**: Splitting a batch across GPUs, computing gradients independently, and averaging (combining) is divide and conquer.

## Examples

```python
import torch

# Divide-and-conquer matrix multiply (Strassen-style concept)
# Standard: O(n^3). Strassen: O(n^2.81) via 7 recursive multiplies
def recursive_sum(x: torch.Tensor) -> torch.Tensor:
    """Divide and conquer sum: split, recurse, combine."""
    if x.numel() == 1:
        return x.squeeze()
    mid = x.numel() // 2
    left = recursive_sum(x[:mid])
    right = recursive_sum(x[mid:])
    return left + right

x = torch.arange(1, 9, dtype=torch.float32)
print(f"Recursive sum of {x.tolist()}: {recursive_sum(x).item()}")
print(f"Direct sum: {x.sum().item()}")

# Multi-head attention as divide and conquer
d_model, n_heads = 64, 4
d_head = d_model // n_heads
Q = torch.randn(1, 8, d_model)  # (batch, seq_len, d_model)

# Divide: split into heads
heads = Q.view(1, 8, n_heads, d_head).transpose(1, 2)  # (1, n_heads, 8, d_head)
print(f"Divided into {n_heads} heads of dim {d_head}")

# Combine: concatenate heads
combined = heads.transpose(1, 2).contiguous().view(1, 8, d_model)
print(f"Combined back to shape {combined.shape}")
print(f"Reconstruction matches: {torch.allclose(Q, combined)}")
```
