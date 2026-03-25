# Adagrad

Standard SGD applies the same learning rate $\eta$ to every parameter, but different parameters often need different step sizes.  A weight connected to a frequently activated feature receives many gradient updates and may need a smaller step to avoid oscillation, while a weight connected to a rare feature receives few updates and benefits from a larger step.  **Adagrad** (Adaptive Gradient, Duchi et al., 2011) addresses this by maintaining a per-parameter accumulation of past squared gradients and using it to scale the learning rate individually for each parameter.

## Update Rule

At each time step $t$, let $g_t = \nabla_\theta L(\theta_t)$ denote the gradient of the loss with respect to the parameters.  All squaring, square roots, and divisions below are **element-wise** operations.

**Step 1.** Accumulate the sum of squared gradients in the state variable $s_t$:

$$
s_t = s_{t-1} + g_t^2
$$

**Step 2.** Update the parameters using the per-parameter effective learning rate $\eta / \sqrt{s_t + \epsilon}$:

$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{s_t + \epsilon}} \, g_t
$$

Here $\eta > 0$ is the global learning rate, and $\epsilon$ is a small constant (typically $10^{-8}$) that prevents division by zero.  The state $s_0$ is initialized to zero.

## Sparse Feature Handling

Adagrad's per-parameter scaling is particularly effective for sparse data.  Parameters associated with infrequent features (e.g., rare words in NLP, uncommon user--item interactions in recommendation systems) accumulate small $s_t$ values, so they receive a relatively large effective learning rate $\eta / \sqrt{s_t + \epsilon}$.  Conversely, parameters tied to frequent features accumulate large $s_t$ and receive smaller updates.  This automatic balancing means Adagrad can make meaningful progress on rare features without overshooting on common ones.

??? example "Intuition with two parameters"
    Suppose parameter $\theta_1$ receives gradients of magnitude $\approx 10$ at every step, while parameter $\theta_2$ receives gradients of magnitude $\approx 0.1$ (it corresponds to a rare feature).  After $T$ steps:

    - $s_{T,1} \approx 100T$, so the effective LR for $\theta_1$ is $\eta / \sqrt{100T} = \eta / (10\sqrt{T})$.
    - $s_{T,2} \approx 0.01T$, so the effective LR for $\theta_2$ is $\eta / \sqrt{0.01T} = 10\eta / \sqrt{T}$.

    Parameter $\theta_2$ gets an effective learning rate that is $100\times$ larger than $\theta_1$'s, compensating for the rarity of its gradient signal.

## The Decay Problem

Because $s_t$ is a sum of non-negative terms, it grows monotonically.  The effective learning rate $\eta / \sqrt{s_t + \epsilon}$ therefore **decreases monotonically** over time.  For convex problems with short training horizons, this automatic annealing is a feature.  For deep learning, however, training may run for millions of steps, and the effective learning rate can shrink so aggressively that the model effectively stops learning long before convergence.

This fundamental limitation motivated the development of [RMSProp](rmsprop.md), [Adadelta](adadelta.md), and eventually [Adam](adam.md), all of which replace the unbounded sum with an exponentially decaying average.

## PyTorch Example

```python
"""Adagrad optimizer demonstration on a simple regression task."""

import torch
import torch.nn as nn


# === Model and Optimizer Setup ===

if __name__ == "__main__":
    model = nn.Linear(10, 1)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01, eps=1e-10)

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

## When to Use Adagrad

Adagrad is well-suited for:

- **Sparse features**: NLP tasks with large vocabularies, recommendation systems with many items, and any setting where most features are inactive in a given sample.
- **Short training runs**: the decay is less harmful when training terminates early.
- **Convex objectives**: the monotonic decay provably aids convergence for convex problems.

For long training on non-convex deep learning objectives, prefer [Adam](adam.md) or [AdamW](adamw.md).

## Reference

- Duchi, J., Hazan, E., & Singer, Y. (2011). Adaptive Subgradient Methods for Online Learning and Stochastic Optimization. *Journal of Machine Learning Research*, 12, 2121--2159.
