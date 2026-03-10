# Proof by Contradiction

Proof by contradiction is a fundamental reasoning technique used throughout mathematics and theoretical machine learning. It establishes truth by showing that the negation leads to an impossibility.

## Definition

A proof by contradiction (reductio ad absurdum) proves a statement $P$ by assuming $\neg P$ and deriving a logical contradiction. Since a consistent system cannot contain contradictions, the assumption $\neg P$ must be false, and therefore $P$ is true.

## Explanation

The structure of a contradiction proof is:

1. Assume $\neg P$ (the claim is false)
2. Through valid logical steps, derive a statement that contradicts a known fact or the assumption itself
3. Conclude $P$ must be true

This technique is especially useful when direct proof is difficult. In machine learning theory, contradiction proofs appear in:

- **No Free Lunch theorems**: Proving that no single algorithm dominates all others by assuming one does and deriving a contradiction
- **Lower bounds on sample complexity**: Showing that fewer than $n$ samples cannot suffice by constructing indistinguishable distributions
- **Impossibility results**: Proving certain learning tasks require specific conditions (e.g., that a hypothesis class must be finite for consistent convergence without structural assumptions)

A classic example: proving $\sqrt{2}$ is irrational. Assume $\sqrt{2} = p/q$ with $\gcd(p, q) = 1$. Then $2q^2 = p^2$, so $p$ is even. Write $p = 2k$, giving $q^2 = 2k^2$, so $q$ is also even. This contradicts $\gcd(p, q) = 1$.

## Examples

```python
import torch

# Demonstrate contradiction logic numerically:
# If sqrt(2) were rational p/q, then p^2 = 2*q^2 exactly.
# We show no small integer pair satisfies this.

max_val = 1000
found = False
for q in range(1, max_val):
    p_squared = 2 * q * q
    p = int(p_squared ** 0.5)
    if p * p == p_squared:
        found = True
        break

print(f"Found exact integer p/q with p^2 = 2q^2 for q < {max_val}: {found}")

# Contradiction in optimization: a convex function cannot have
# two distinct global minima. Verify with a quadratic.
x = torch.linspace(-5, 5, 1000)
f = x ** 2 + 1  # strictly convex
min_val = f.min().item()
min_indices = (f - min_val).abs() < 1e-6
num_minima = min_indices.sum().item()
print(f"Number of global minima of x^2 + 1: {num_minima}")
print(f"Minimum value: {min_val:.4f}")
```
