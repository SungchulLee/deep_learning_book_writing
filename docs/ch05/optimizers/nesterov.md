# Nesterov Accelerated Gradient

## Overview

Nesterov momentum (NAG) improves upon classical momentum by computing the gradient at the "look-ahead" position $\theta_t - \eta \mu v_{t-1}$ rather than the current position. This provides a form of gradient correction that leads to better convergence, especially near optima.

## Update Rule

$$v_t = \mu \, v_{t-1} + \nabla_\theta \mathcal{L}(\theta_t - \eta \mu \, v_{t-1})$$
$$\theta_{t+1} = \theta_t - \eta \, v_t$$

The key difference from classical momentum: the gradient is evaluated at the anticipated next position, not the current one.

## Intuition

Classical momentum blindly follows the accumulated velocity. Nesterov momentum "looks ahead" to where the momentum would carry the parameters and corrects the update based on the gradient at that lookahead point. This correction is particularly valuable when approaching a minimum, where the gradient changes rapidly.

## PyTorch Implementation

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01,
                            momentum=0.9, nesterov=True)
```

## Convergence Properties

For convex functions, Nesterov momentum achieves an optimal convergence rate of $O(1/t^2)$ compared to $O(1/t)$ for classical momentum. In practice, for non-convex deep learning loss surfaces, the improvement is often modest but consistent.

## When to Use

Nesterov momentum is generally preferred over classical momentum when using SGD. It adds no computational overhead and provides modest but reliable improvements. It is the default recommendation when SGD with momentum is the chosen optimizer family.

## Key Takeaways

- Nesterov momentum evaluates the gradient at the look-ahead position.
- This provides a correction term that improves convergence near optima.
- Use `nesterov=True` with `torch.optim.SGD` for a free improvement over classical momentum.
