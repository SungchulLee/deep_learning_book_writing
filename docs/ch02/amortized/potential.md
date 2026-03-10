# Potential Method

The potential method defines a potential function over the data structure's state, converting actual costs into amortized costs via the change in potential. This is the most powerful amortized analysis technique and provides a framework for understanding adaptive optimizers in deep learning.

## Definition

Define a potential function $\Phi(D_i)$ mapping data structure state after operation $i$ to a non-negative real number. The amortized cost of operation $i$ is:

$$
\hat{c}_i = c_i + \Phi(D_i) - \Phi(D_{i-1})
$$

Summing over $n$ operations:

$$
\sum_{i=1}^{n} \hat{c}_i = \sum_{i=1}^{n} c_i + \Phi(D_n) - \Phi(D_0)
$$

If $\Phi(D_n) \geq \Phi(D_0)$, the amortized total is an upper bound on the actual total.

## Explanation

The potential function captures "stored energy" in the system. Cheap operations that increase potential are saving up for future expensive operations that will decrease potential.

In deep learning, the potential method provides insight into:

- **Adaptive learning rates**: Adam maintains running estimates of first and second moments ($m_t, v_t$). The potential function is the accumulated state of these estimates. Early updates have high amortized cost (building up the estimates), while later updates have lower cost because the estimates stabilize.
- **Dynamic computation graphs**: PyTorch's autograd graph grows during the forward pass (increasing potential) and releases memory during backward (decreasing potential).
- **Learning rate warmup**: The warmup phase builds "potential" (model moves toward a good region), enabling the subsequent high-learning-rate phase to be effective.

## Examples

```python
import torch

# Potential method for dynamic array: Phi = 2*size - capacity
# When array is half full: Phi = 0 (minimum)
# When array is full: Phi = size (maximum, triggers doubling)

def potential(size, capacity):
    return max(0, 2 * size - capacity)

cap = 2
size = 0
total_actual = 0
total_amortized = 0

for i in range(16):
    if size == cap:
        actual_cost = size + 1  # copy all + insert
        cap *= 2
    else:
        actual_cost = 1  # just insert
    old_pot = potential(size, cap if size < cap else cap // 2)
    size += 1
    new_pot = potential(size, cap)
    amortized = actual_cost + new_pot - old_pot
    total_actual += actual_cost
    total_amortized += amortized

print(f"Total actual cost: {total_actual}")
print(f"Amortized cost per op: {total_actual / 16:.2f}")

# Adam optimizer: moment accumulation as potential buildup
model = torch.nn.Linear(10, 1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
x, y = torch.randn(32, 10), torch.randn(32, 1)

for step in range(50):
    loss = torch.nn.functional.mse_loss(model(x), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Inspect accumulated state (the "potential")
state = optimizer.state[list(model.parameters())[0]]
print(f"\nAdam moment estimates after 50 steps:")
print(f"  exp_avg norm:  {state['exp_avg'].norm().item():.6f}")
print(f"  exp_avg_sq norm: {state['exp_avg_sq'].norm().item():.6f}")
print(f"  step count: {state['step']}")
```
