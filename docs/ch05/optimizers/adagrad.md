# Adagrad

## Overview

Adagrad (Adaptive Gradient) adapts the learning rate for each parameter individually based on the historical sum of squared gradients. Parameters with large accumulated gradients receive smaller updates, and parameters with small accumulated gradients receive larger updates.

## Update Rule

$$s_t = s_{t-1} + g_t^2$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{s_t} + \epsilon} \, g_t$$

where $s_t$ is the accumulated sum of squared gradients and $\epsilon \approx 10^{-8}$ prevents division by zero.

## PyTorch Implementation

```python
optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01, eps=1e-10)
```

## Properties

**Sparse feature handling**: Adagrad excels with sparse data. Infrequent features (rare words, uncommon financial indicators) accumulate small $s_t$ values, receiving effectively larger learning rates that allow meaningful updates when these features appear.

**Learning rate decay**: The monotonically increasing $s_t$ causes the effective learning rate to decrease over time. This is both a strength (automatic annealing) and a weakness (premature convergence).

## The Decay Problem

The denominator $\sqrt{s_t}$ only grows, so the effective learning rate $\eta / \sqrt{s_t}$ monotonically decreases. In deep learning, where training runs for many iterations, this decay can become too aggressive, effectively stopping learning before convergence. This limitation motivated RMSprop and Adadelta.

## When to Use

Adagrad is well-suited for problems with sparse features (NLP, recommendation systems, sparse financial features) and short training runs. For longer training, prefer RMSprop or Adam, which address the learning rate decay issue.

## Key Takeaways

- Adagrad adapts learning rates per parameter based on accumulated squared gradients.
- Excellent for sparse features; problematic for long training due to aggressive learning rate decay.
- Motivated the development of RMSprop and Adam.
