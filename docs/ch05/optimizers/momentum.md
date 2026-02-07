# SGD with Momentum

## Overview

Momentum accumulates an exponentially decaying moving average of past gradients, accelerating convergence along consistent gradient directions and dampening oscillations in directions with high curvature.

## Update Rule

$$v_t = \mu \, v_{t-1} + g_t$$
$$\theta_{t+1} = \theta_t - \eta \, v_t$$

where $\mu \in [0, 1)$ is the momentum coefficient (typically 0.9) and $v_t$ is the velocity (accumulated gradient).

## Intuition

Consider a loss landscape shaped like an elongated valley. Vanilla SGD oscillates across the narrow dimension while making slow progress along the long dimension. Momentum accumulates velocity along the consistent (long) direction and cancels out oscillations across the narrow direction—analogous to a ball rolling downhill with inertia.

## PyTorch Implementation

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

## Effect of Momentum Coefficient

- $\mu = 0$: Reduces to vanilla SGD.
- $\mu = 0.9$: Standard choice. The effective step size is amplified by up to $1/(1-\mu) = 10\times$ in consistent gradient directions.
- $\mu = 0.99$: More aggressive smoothing, useful when gradients are very noisy.

## Dampening

PyTorch supports dampening, which reduces the contribution of the current gradient:

$$v_t = \mu \, v_{t-1} + (1 - d) \, g_t$$

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01,
                            momentum=0.9, dampening=0.1)
```

## Key Takeaways

- Momentum accelerates convergence by accumulating gradient history.
- The standard momentum coefficient is 0.9.
- SGD with momentum is the default optimizer for many computer vision tasks and often matches or outperforms adaptive methods in final generalization when properly tuned.
