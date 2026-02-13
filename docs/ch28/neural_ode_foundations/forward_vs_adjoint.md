# Forward vs Adjoint Sensitivity

## Overview

Computing gradients through an ODE solve requires differentiating the integration process. Two approaches exist: forward sensitivity and the adjoint method, with different memory and computational tradeoffs.

## Forward Sensitivity

Differentiate the ODE directly:

$$\frac{d}{dt}\frac{\partial z}{\partial \theta} = \frac{\partial f}{\partial z} \frac{\partial z}{\partial \theta} + \frac{\partial f}{\partial \theta}$$

This requires solving an augmented ODE of dimension $d + d \times p$ (where $d$ is the state dimension and $p$ is the number of parameters), which is prohibitive for large models.

## Adjoint Method

The adjoint method (Pontryagin, 1962; Chen et al., 2018) computes gradients by solving a backward ODE:

$$\frac{da(t)}{dt} = -a(t)^T \frac{\partial f}{\partial z}$$

where $a(t) = \partial \mathcal{L} / \partial z(t)$ is the adjoint state.

**Memory**: $O(1)$ in the number of ODE solver steps (vs. $O(T)$ for backprop through the solver). The adjoint method does not store intermediate states.

**Cost**: requires solving the ODE backward in time, plus one vector-Jacobian product per step.

## Comparison

| Aspect | Forward Sensitivity | Adjoint Method |
|--------|-------------------|----------------|
| Memory | $O(T \cdot (d + dp))$ | $O(d + p)$ |
| Compute | One augmented forward ODE | Forward + backward ODE |
| Accuracy | Exact | Numerical errors from backward solve |
| Implementation | Straightforward | Requires careful numerical handling |
| Use case | Small $p$ | Large $p$ (neural networks) |

## Practical Issues with Adjoint

The adjoint method solves the ODE backward, which can be numerically unstable for certain dynamics. The reconstructed trajectory $z(t)$ during the backward pass may diverge from the original forward trajectory, leading to inaccurate gradients.

**Checkpointing**: a practical compromise is to store intermediate states at a few checkpoints and use the adjoint method between checkpoints.

```python
from torchdiffeq import odeint_adjoint

# Memory-efficient training
z_T = odeint_adjoint(dynamics_fn, z_0, t_span, method='dopri5')
```
