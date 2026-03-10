# Mathematical Induction

Mathematical induction is the standard technique for proving statements about all natural numbers. In deep learning, induction arguments appear in proofs about network depth, recursive architectures, and convergence after $n$ iterations.

## Definition

Mathematical induction proves that a property $P(n)$ holds for all integers $n \geq n_0$ via two steps:

1. **Base case**: Prove $P(n_0)$
2. **Inductive step**: Prove that $P(k) \implies P(k+1)$ for all $k \geq n_0$

The principle rests on the well-ordering of natural numbers: if the set of counterexamples were non-empty, it would have a smallest element, which the base case and inductive step together exclude.

## Explanation

The inductive step does not prove $P(k+1)$ in isolation. It proves the implication: *if* $P(k)$ holds, *then* $P(k+1)$ follows. Combined with the base case, this creates a chain of implications covering all $n \geq n_0$.

A standard example: the sum of the first $n$ integers satisfies

$$
\sum_{i=1}^{n} i = \frac{n(n+1)}{2}
$$

**Base case**: $P(1)$: $1 = \frac{1 \cdot 2}{2}$. True.

**Inductive step**: Assume $\sum_{i=1}^{k} i = \frac{k(k+1)}{2}$. Then

$$
\sum_{i=1}^{k+1} i = \frac{k(k+1)}{2} + (k+1) = \frac{(k+1)(k+2)}{2}
$$

In deep learning, induction is used to prove that a network with $L$ layers composes $L$ affine-nonlinear transformations, or that gradient descent reduces loss at each step under certain conditions.

## Examples

```python
import torch

# Verify the sum formula by induction-style checking
def sum_formula(n):
    return n * (n + 1) // 2

# Base case
assert sum_formula(1) == 1, "Base case failed"

# Inductive step: verify P(k) + (k+1) = P(k+1) for many k
for k in range(1, 100):
    lhs = sum_formula(k) + (k + 1)
    rhs = sum_formula(k + 1)
    assert lhs == rhs, f"Inductive step failed at k={k}"
print("Induction verified for n = 1..100")

# Deep learning connection: verify that composing L linear layers
# is equivalent to a single linear layer (induction on depth)
torch.manual_seed(42)
d = 4
x = torch.randn(d)

L = 5
matrices = [torch.randn(d, d) for _ in range(L)]

# Apply layers sequentially
result = x.clone()
for W in matrices:
    result = W @ result

# Compose into single matrix
composed = torch.eye(d)
for W in matrices:
    composed = W @ composed

print(f"Sequential result: {result[:3]}")
print(f"Composed result:   {(composed @ x)[:3]}")
print(f"Match: {torch.allclose(result, composed @ x, atol=1e-4)}")
```
