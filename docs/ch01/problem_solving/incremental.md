# Incremental Algorithms

Incremental algorithms build solutions one element at a time, maintaining correctness at each step. This paradigm directly maps to the training loop in deep learning: each gradient step incrementally improves the model parameters.

## Definition

An incremental algorithm constructs a solution by processing elements sequentially, updating the current solution with each new element:

$$
\text{Solution}_i = \text{Update}(\text{Solution}_{i-1}, \text{element}_i)
$$

The invariant is that $\text{Solution}_i$ is correct for the first $i$ elements.

## Explanation

The incremental approach appears throughout deep learning:

- **Stochastic gradient descent**: Each mini-batch update incrementally improves the parameters. The invariant is that the loss generally decreases (in expectation) after each step.
- **Online learning**: Models update with each arriving data point, never revisiting old data. This is incremental by nature.
- **Running statistics**: Batch normalization tracks running mean and variance incrementally across training batches.

The key advantage is simplicity: the update rule is often easy to derive and implement. The disadvantage is that greedy incremental choices may not yield globally optimal solutions (similar to how SGD finds local rather than global minima).

## Examples

```python
import torch

# Incremental mean and variance (Welford's algorithm)
# Used in BatchNorm running statistics
def incremental_stats(data: torch.Tensor):
    """Compute mean and variance incrementally."""
    mean = torch.tensor(0.0)
    m2 = torch.tensor(0.0)
    for i, x in enumerate(data, 1):
        delta = x - mean
        mean += delta / i
        delta2 = x - mean
        m2 += delta * delta2
    variance = m2 / len(data)
    return mean, variance

data = torch.randn(1000)
inc_mean, inc_var = incremental_stats(data)
print(f"Incremental mean: {inc_mean.item():.6f}")
print(f"Incremental var:  {inc_var.item():.6f}")
print(f"Direct mean:      {data.mean().item():.6f}")
print(f"Direct var:       {data.var(correction=0).item():.6f}")

# SGD as incremental optimization
torch.manual_seed(42)
w = torch.randn(1, requires_grad=True)
x = torch.randn(100)
y = 3.0 * x + torch.randn(100) * 0.1

for i in range(50):
    idx = i % len(x)  # one element at a time
    pred = w * x[idx]
    loss = (pred - y[idx]) ** 2
    loss.backward()
    with torch.no_grad():
        w -= 0.01 * w.grad
        w.grad.zero_()
print(f"Learned w: {w.item():.4f} (true: 3.0)")
```
