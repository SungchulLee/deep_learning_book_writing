# Adadelta

[Adagrad](adagrad.md) adapts the learning rate per parameter by dividing by the square root of all past squared gradients.  While effective for sparse problems, this accumulation grows without bound, causing the effective learning rate to shrink to zero over time.  Adadelta (Zeiler, 2012) addresses this problem by replacing the unbounded sum with an **exponentially decaying average** of squared gradients.  It also introduces a novel "unit correction" mechanism that eliminates the need for an initial learning rate hyperparameter altogether.

## Motivation: The Unit Mismatch Problem

In standard SGD, the parameter update $\Delta\theta = -\eta \, g$ has units proportional to the gradient, not to the parameter itself.  If $\theta$ represents a weight in kilograms and $g = \partial L / \partial \theta$ has units of loss per kilogram, then $\eta \, g$ has units that depend on the arbitrary choice of $\eta$.  Adadelta resolves this by scaling the update so that $\Delta\theta$ has the same "units" as $\theta$.

## Update Rule

At each time step $t$, let $g_t = \nabla_\theta L(\theta_t)$ denote the gradient of the loss with respect to the parameters.  All squaring, square roots, and divisions below are **element-wise** operations.

**Step 1.** Update the exponentially decaying average of squared gradients, denoted $s_t$:

$$
s_t = \rho \, s_{t-1} + (1 - \rho) \, g_t^2
$$

**Step 2.** Compute the parameter update $\Delta\theta_t$ using the ratio of the RMS of past parameter updates to the RMS of the current gradient accumulator:

$$
\Delta\theta_t = -\frac{\sqrt{\delta_{t-1} + \epsilon}}{\sqrt{s_t + \epsilon}} \, g_t
$$

**Step 3.** Update the exponentially decaying average of squared parameter updates, denoted $\delta_t$:

$$
\delta_t = \rho \, \delta_{t-1} + (1 - \rho) \, (\Delta\theta_t)^2
$$

**Step 4.** Apply the update:

$$
\theta_{t+1} = \theta_t + \Delta\theta_t
$$

Here $\rho \in [0, 1)$ is the **decay rate** (typically $0.9$), and $\epsilon$ is a small constant (e.g., $10^{-6}$) that prevents division by zero.

## Key Insight: Unit Correction

The denominator $\sqrt{s_t + \epsilon}$ is the RMS of recent gradients, and the numerator $\sqrt{\delta_{t-1} + \epsilon}$ is the RMS of recent parameter updates.  Their ratio cancels the "gradient units," so the update $\Delta\theta_t$ ends up with the same units as $\theta$ itself.  This is the reason Adadelta can function without an explicit learning rate.

!!! note "Comparison with RMSProp"
    RMSProp (developed independently and concurrently) shares the exponentially decaying average of squared gradients (the denominator), but uses a fixed learning rate $\eta$ in the numerator instead of the RMS of past updates.  Adadelta's unit correction makes it the more principled formulation, though in practice both have been superseded by [Adam](adam.md).

## PyTorch Example

```python
"""Adadelta optimizer demonstration on a simple regression task."""

import torch
import torch.nn as nn


# === Model and Optimizer Setup ===

if __name__ == "__main__":
    # Simple linear regression
    model = nn.Linear(10, 1)
    optimizer = torch.optim.Adadelta(model.parameters(), rho=0.9, eps=1e-6)

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

!!! warning "Learning rate in PyTorch"
    Although Adadelta was designed to be learning-rate-free, PyTorch's implementation includes a `lr` parameter (default `1.0`) that multiplies the update.  Setting `lr=1.0` recovers the original algorithm.  Tuning `lr` away from 1.0 can sometimes improve convergence but defeats the purpose of the unit correction.

## Properties

- **No learning rate required** (in the original formulation): the RMS ratio provides automatic scaling.
- **Bounded accumulation**: the exponential decay in $s_t$ prevents the runaway accumulation that plagues Adagrad.
- **Unit-correct updates**: the parameter update has the same dimensional units as the parameters.
- **Historical significance**: Adadelta and RMSProp were developed at nearly the same time and share the same key insight (exponential moving average of squared gradients).
- **Modern usage**: rarely used in current practice; [Adam](adam.md) and [AdamW](adamw.md) are preferred for most applications.

## Reference

- Zeiler, M. D. (2012). ADADELTA: An Adaptive Learning Rate Method. *arXiv:1212.5701*.
