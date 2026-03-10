# Properties of Algorithms

Every algorithm possesses fundamental properties that determine its reliability and usefulness. Understanding these properties helps distinguish well-designed deep learning systems from fragile ones.

## Definition

The five essential properties of an algorithm are:

$$
\begin{array}{ll}
\text{Correctness} & \text{Produces the right output for every valid input} \\
\text{Efficiency} & \text{Uses time and space wisely as input grows} \\
\text{Finiteness} & \text{Terminates after a finite number of steps} \\
\text{Definiteness} & \text{Each step is precisely defined} \\
\text{Generality} & \text{Solves a class of problems, not just one instance}
\end{array}
$$

## Explanation

In deep learning, these properties take specific forms:

- **Correctness**: A training algorithm is correct if it converges to a (local) minimum of the loss. Verification involves gradient checking, loss curve monitoring, and evaluation on held-out data.
- **Efficiency**: Training and inference must complete within resource budgets. Efficient architectures (e.g., depthwise separable convolutions) trade minimal accuracy for large speedups.
- **Finiteness**: Training must terminate. This requires explicit stopping criteria: maximum epochs, early stopping on validation loss, or learning rate reaching a minimum threshold.
- **Definiteness**: Every operation must be unambiguous. Floating-point non-determinism (different GPU results across runs) violates strict definiteness but is acceptable in practice.
- **Generality**: A good architecture (ResNet, Transformer) generalizes across tasks. A model overfit to one dataset lacks generality.

**Deterministic vs stochastic**: Classical algorithms are deterministic. Neural network training is stochastic (random initialization, mini-batch sampling, dropout). The stochasticity is deliberate -- it provides regularization and enables exploration of the loss landscape.

## Examples

```python
import torch
import torch.nn as nn

# Demonstrate finiteness: early stopping
model = nn.Linear(10, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
x, y = torch.randn(50, 10), torch.randn(50, 1)

best_loss, patience, wait = float("inf"), 5, 0
for epoch in range(1000):  # max epochs ensures finiteness
    loss = nn.functional.mse_loss(model(x), y)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    if loss.item() < best_loss - 1e-4:
        best_loss = loss.item()
        wait = 0
    else:
        wait += 1
    if wait >= patience:
        print(f"Early stopping at epoch {epoch}, loss={loss.item():.4f}")
        break

# Demonstrate determinism vs stochasticity
torch.manual_seed(42)
a = torch.randn(3)
torch.manual_seed(42)
b = torch.randn(3)
print(f"Same seed -> same output: {torch.equal(a, b)}")

torch.manual_seed(0)
c = torch.randn(3)
print(f"Different seed -> different output: {not torch.equal(a, c)}")
```
