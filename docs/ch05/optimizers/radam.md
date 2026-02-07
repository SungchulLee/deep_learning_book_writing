# RAdam

## Overview

RAdam (Rectified Adam) addresses Adam's variance issue in early training. During the first few hundred steps, Adam's second moment estimate has high variance, leading to unreliable adaptive learning rates. RAdam dynamically switches between SGD (when variance is high) and Adam (when the estimate stabilizes).

## The Variance Problem

Adam's second moment $v_t$ is initialized to zero and computed from a small number of gradient samples in early training. The resulting $1/\sqrt{\hat{v}_t}$ term has high variance, causing erratic early updates that can prevent convergence.

## Update Rule

RAdam computes the variance of the inverse second moment estimate and only applies adaptive scaling when this variance falls below a threshold:

$$\rho_t = \rho_\infty - \frac{2 t \beta_2^t}{1 - \beta_2^t}$$

where $\rho_\infty = 2/(1-\beta_2) - 1$. When $\rho_t > 5$ (variance is tractable), RAdam uses the full Adam update with a rectification term. Otherwise, it falls back to SGD with momentum.

## PyTorch Implementation

```python
optimizer = torch.optim.RAdam(model.parameters(), lr=0.001,
                              betas=(0.9, 0.999), eps=1e-8)
```

## RAdam vs. Warmup

RAdam provides an automatic warmup effect: it uses SGD during early training (when Adam would be unreliable) and transitions to Adam as estimates stabilize. This often eliminates the need for explicit learning rate warmup schedules.

## Key Takeaways

- RAdam automatically handles the unreliable early phase of Adam training.
- Dynamically switches between SGD and Adam based on variance estimates.
- Can replace Adam + warmup with a single optimizer, reducing hyperparameter tuning.
