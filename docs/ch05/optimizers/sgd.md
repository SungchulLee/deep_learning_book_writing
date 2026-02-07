# SGD (Stochastic Gradient Descent)

## Overview

Vanilla SGD is the simplest optimization algorithm: update parameters by moving in the direction of the negative gradient, scaled by a learning rate.

## Update Rule

$$\theta_{t+1} = \theta_t - \eta \, g_t$$

where $g_t = \nabla_\theta \mathcal{L}(\theta_t; \mathcal{B}_t)$ is the mini-batch gradient and $\eta$ is the learning rate.

## PyTorch Implementation

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

## Properties

**Convergence**: For convex problems with appropriate learning rate decay, SGD converges to the global minimum. For non-convex problems (deep learning), SGD converges to a local minimum or saddle point.

**Noise**: The stochasticity of mini-batch gradients acts as implicit regularization, helping escape sharp minima and favoring flatter regions of the loss landscape that generalize better.

**Learning rate sensitivity**: SGD is highly sensitive to the learning rate. Too high and training diverges; too low and convergence is prohibitively slow.

## With Weight Decay

SGD with L2 regularization (weight decay):

$$\theta_{t+1} = \theta_t - \eta \, (g_t + \lambda \theta_t) = (1 - \eta\lambda)\theta_t - \eta \, g_t$$

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
```

For SGD, L2 regularization and weight decay are mathematically equivalent.

## When to Use

SGD is the simplest baseline. In practice, it is rarely used without momentum (see next section) but serves as a reference point for understanding all other optimizers.

## Key Takeaways

- SGD updates parameters proportional to the negative gradient.
- Gradient noise provides implicit regularization.
- SGD requires careful learning rate tuning and is typically enhanced with momentum.
