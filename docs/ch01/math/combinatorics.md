# Combinatorics

Combinatorics counts the number of ways to arrange or select objects. In deep learning, combinatorial reasoning arises when analyzing search spaces, counting model architectures in neural architecture search, and understanding the capacity of discrete structures.

## Definition

Combinatorics is the branch of mathematics concerned with counting, arranging, and selecting elements from finite sets. The fundamental quantities are:

$$
\text{Permutations: } P(n, k) = \frac{n!}{(n-k)!} \qquad \text{Combinations: } \binom{n}{k} = \frac{n!}{k!(n-k)!}
$$

Permutations count ordered arrangements of $k$ items from $n$, while combinations count unordered selections.

## Explanation

Three counting principles cover most situations in practice:

- **Multiplication principle**: If task A has $m$ outcomes and task B has $n$ outcomes, the pair has $m \cdot n$ outcomes. This gives $n^k$ arrangements when choosing $k$ items from $n$ with repetition allowed.
- **Combinations**: Selecting $k$ items from $n$ without regard to order yields $\binom{n}{k}$ possibilities. This appears when choosing which neurons to drop in dropout or selecting subsets of features.
- **Binomial theorem**: Connects combinations to algebra:

$$
(x + y)^n = \sum_{k=0}^{n} \binom{n}{k} x^{n-k} y^k
$$

In deep learning, combinatorial explosion explains why brute-force hyperparameter search is infeasible. A grid search over $p$ hyperparameters with $v$ values each requires $v^p$ evaluations, motivating random search and Bayesian optimization.

## Examples

```python
import torch
from math import comb, factorial

# Permutations and combinations
n, k = 10, 3
perms = factorial(n) // factorial(n - k)
combs = comb(n, k)
print(f"P({n},{k}) = {perms}")
print(f"C({n},{k}) = {combs}")

# Hyperparameter search space explosion
params = 5
values_per_param = 4
grid_size = values_per_param ** params
print(f"Grid search: {params} params x {values_per_param} values = {grid_size} configs")

# Verify binomial theorem with PyTorch
x, y = torch.tensor(2.0), torch.tensor(3.0)
n_val = 4
lhs = (x + y) ** n_val
rhs = sum(comb(n_val, k) * x ** (n_val - k) * y ** k for k in range(n_val + 1))
print(f"(x+y)^{n_val} = {lhs.item():.0f}, sum of binomial terms = {rhs.item():.0f}")
```
