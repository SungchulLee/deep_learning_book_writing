# Brute Force Search

Brute force exhaustively enumerates all candidate solutions. In deep learning, brute force provides the conceptual baseline that motivates smarter approaches like gradient descent, random search over hyperparameter grids, and learned heuristics.

## Definition

A brute force algorithm evaluates every element of the solution space and selects the best. Its time complexity is:

$$
T(n) = |\text{Solution Space}| \times \text{Cost per evaluation}
$$

The solution space is often exponential ($2^n$ subsets, $n!$ permutations), making brute force infeasible for large inputs.

## Explanation

Brute force is valuable in three situations:

- **Correctness baseline**: When developing an optimized algorithm, a brute force implementation serves as a reference to verify correctness. In deep learning, a naive forward pass computation verifies that an optimized CUDA kernel produces correct results.
- **Small search spaces**: Hyperparameter grid search with few parameters is brute force that remains practical because the space is small.
- **Understanding the problem**: Writing the brute force solution first clarifies the structure of the problem and reveals patterns that suggest optimization.

The transition from brute force to gradient-based optimization is foundational to deep learning: instead of evaluating all possible weight configurations (intractable), we follow the gradient to iteratively improve a single configuration.

## Examples

```python
import torch

# Brute force: find the weight that minimizes loss
# (intractable for real networks, but illustrative)
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([2.0, 4.0, 6.0])

best_w, best_loss = 0.0, float("inf")
for w in torch.linspace(-5, 5, 1000):
    loss = ((w * x - y) ** 2).mean().item()
    if loss < best_loss:
        best_w, best_loss = w.item(), loss
print(f"Brute force: w={best_w:.4f}, loss={best_loss:.6f}")

# Gradient descent: finds the same answer efficiently
w = torch.tensor(0.0, requires_grad=True)
optimizer = torch.optim.SGD([w], lr=0.01)
for _ in range(200):
    loss = ((w * x - y) ** 2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print(f"Gradient descent: w={w.item():.4f}, loss={loss.item():.6f}")
```
