# Optimizer Overview

## Overview

Optimizers implement the parameter update rule that converts gradients computed by backpropagation into parameter changes. The choice of optimizer and its hyperparameters profoundly affects convergence speed, training stability, and generalization.

## The Optimization Problem

Training minimizes a loss function $\mathcal{L}(\theta)$ over parameters $\theta$:

$$\theta^* = \arg\min_\theta \mathcal{L}(\theta) = \arg\min_\theta \frac{1}{N} \sum_{i=1}^N \ell(f_\theta(x_i), y_i)$$

Since computing the full gradient is expensive, we use **stochastic gradient descent**: estimate $\nabla_\theta \mathcal{L}$ from a mini-batch $\mathcal{B} \subset \{1, \ldots, N\}$:

$$g_t = \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \nabla_\theta \ell(f_\theta(x_i), y_i)$$

## PyTorch Optimizer Interface

All optimizers share a common interface:

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training step
optimizer.zero_grad()       # Clear gradients
loss.backward()             # Compute gradients
optimizer.step()            # Apply update rule
```

## Parameter Groups

Optimizers accept parameter groups with different hyperparameters:

```python
optimizer = torch.optim.SGD([
    {'params': model.backbone.parameters(), 'lr': 1e-4},
    {'params': model.head.parameters(), 'lr': 1e-3}
], momentum=0.9)
```

This is commonly used for differential learning rates in transfer learning (lower LR for pretrained layers, higher LR for new layers).

## Optimizer Taxonomy

Optimizers fall into three families:

**First-order, non-adaptive**: SGD and its momentum variants. Use the same learning rate for all parameters.

**First-order, adaptive**: Adagrad, RMSprop, Adam, and their variants. Maintain per-parameter learning rates based on historical gradient information.

**Second-order**: L-BFGS, natural gradient methods. Use curvature information for faster convergence but at higher computational cost.

| Optimizer | Adaptive | Momentum | Key Property |
|-----------|----------|----------|---|
| SGD | No | No | Simplest baseline |
| SGD + Momentum | No | Yes | Accelerated convergence |
| Adagrad | Yes | No | Good for sparse features |
| RMSprop | Yes | No | Adagrad with decay |
| Adam | Yes | Yes | Adaptive + momentum |
| AdamW | Yes | Yes | Decoupled weight decay |
| L-BFGS | N/A | N/A | Quasi-Newton, second-order |

## Key Takeaways

- Optimizers implement the update rule: $\theta_{t+1} = \theta_t - \eta \cdot \text{update}(g_t, \text{state})$.
- Parameter groups enable per-layer learning rate and regularization tuning.
- Adaptive optimizers adjust learning rates per parameter; non-adaptive optimizers use a global rate.
- The remaining sections in 5.4 derive and implement each optimizer in detail.
