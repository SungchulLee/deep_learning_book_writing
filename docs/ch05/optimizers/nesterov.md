# Nesterov Accelerated Gradient

[Classical momentum](momentum.md) accumulates a velocity vector from past gradients and uses it to accelerate parameter updates.  While this speeds convergence in low-curvature directions, the accumulated velocity can overshoot when the loss surface curves sharply — for instance near a minimum, where the gradient changes direction rapidly.  **Nesterov Accelerated Gradient** (NAG; Nesterov, 1983) mitigates this by evaluating the gradient at a **look-ahead position** rather than the current position.  This "peek ahead" provides an early warning of curvature changes, allowing the optimizer to slow down before it overshoots.

## Update Rule

At each time step $t$, let $\theta_t$ denote the current parameters, $\mu \in [0, 1)$ the momentum coefficient, and $\eta > 0$ the learning rate.  The velocity vector $v_t$ is initialized to zero.

**Step 1.** Compute the look-ahead position and evaluate the gradient there:

$$
\tilde{\theta}_t = \theta_t + \mu \, v_{t-1}
$$

**Step 2.** Update the velocity using the gradient at the look-ahead position:

$$
v_t = \mu \, v_{t-1} - \eta \, \nabla_\theta \mathcal{L}(\tilde{\theta}_t)
$$

**Step 3.** Update the parameters:

$$
\theta_{t+1} = \theta_t + v_t
$$

The critical difference from classical momentum is in Step 1: the gradient is evaluated at $\tilde{\theta}_t = \theta_t + \mu \, v_{t-1}$ (where momentum would carry us) rather than at $\theta_t$ itself.

!!! note "Equivalent reformulation"
    Many frameworks, including PyTorch, implement an algebraically equivalent form that avoids explicitly computing the look-ahead position.  Defining $\hat{\theta}_t = \theta_t + \mu \, v_t$ and substituting, one can show that the update reduces to a modified velocity step applied directly at $\theta_t$.  The mathematical effect is identical.

## Intuition: Corrective Look-Ahead

Consider a ball rolling downhill with momentum.  Classical momentum evaluates the slope at the ball's current position and adds it to the accumulated velocity.  If the ball is approaching a valley floor, the current gradient still points downhill, so the ball accelerates further and overshoots.

Nesterov momentum first moves the ball to where momentum *would* carry it, then evaluates the slope at that future position.  If the future position is already past the valley floor, the gradient there points *uphill*, which reduces the velocity and prevents overshoot.

??? example "Comparison on a 1D quadratic"
    Consider minimizing $f(\theta) = \frac{1}{2}\theta^2$ starting from $\theta_0 = 10$ with $\eta = 0.1$ and $\mu = 0.9$.

    **Classical momentum** evaluates $\nabla f(\theta_t) = \theta_t$ at the current position, so the velocity accumulates a strong downhill signal even as the parameters approach zero, causing oscillation.

    **Nesterov momentum** evaluates $\nabla f(\theta_t + \mu v_{t-1})$ at the look-ahead position.  When the velocity is about to carry $\theta$ past zero, the look-ahead gradient points in the opposite direction, damping the oscillation earlier.

## Convergence Properties

For **smooth convex** functions with Lipschitz-continuous gradients, the convergence rates are:

| Method | Rate |
|---|---|
| Gradient descent (no momentum) | $O(1/t)$ |
| Nesterov accelerated gradient | $O(1/t^2)$ |

The $O(1/t^2)$ rate achieved by NAG is **optimal** among first-order methods for this function class (Nesterov, 1983).  This quadratic speedup in the convergence rate is the theoretical motivation for Nesterov acceleration.

For **non-convex** loss surfaces typical in deep learning, the theoretical guarantees do not directly apply, but empirical results consistently show that Nesterov momentum reduces oscillation and converges slightly faster than classical momentum.

## PyTorch Example

```python
"""Nesterov Accelerated Gradient demonstration using PyTorch's SGD."""

import torch
import torch.nn as nn


# === Model and Optimizer Setup ===

if __name__ == "__main__":
    model = nn.Linear(10, 1)

    # Enable Nesterov momentum by setting nesterov=True
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9,
        nesterov=True,
    )

    # Training loop (illustrative)
    for step in range(100):
        x = torch.randn(32, 10)
        y = torch.randn(32, 1)
        loss = nn.functional.mse_loss(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Final loss: {loss.item():.4f}")
```

## When to Use

Nesterov momentum is generally preferred over classical momentum whenever SGD is the chosen optimizer family:

- **No extra cost**: the look-ahead gradient evaluation adds negligible overhead (PyTorch's reformulation avoids a second forward pass).
- **Reduced oscillation**: the corrective look-ahead reliably dampens oscillation near minima.
- **Standard recommendation**: set `nesterov=True` in `torch.optim.SGD` as a default when using SGD with momentum.

For adaptive-rate optimizers like [Adam](adam.md), the Nesterov idea has been incorporated into variants such as NAdam, but standard Adam is more commonly used in practice.

## Reference

- Nesterov, Y. (1983). A Method of Solving a Convex Programming Problem with Convergence Rate $O(1/k^2)$. *Soviet Mathematics Doklady*, 27(2), 372--376.
- Sutskever, I., Martens, J., Dahl, G., & Hinton, G. (2013). On the Importance of Initialization and Momentum in Deep Learning. *ICML 2013*.
