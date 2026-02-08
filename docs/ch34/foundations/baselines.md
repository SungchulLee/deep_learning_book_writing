# 34.1.4 Baseline Methods

## Introduction

Baselines are a fundamental variance reduction technique in policy gradient methods. By subtracting a baseline $b(s)$ from the return, we can dramatically reduce the variance of gradient estimates without introducing bias (provided the baseline does not depend on the action). This section covers the theory, common baseline choices, and the optimal baseline.

## Theory of Baselines

### Unbiasedness Proof

The key theoretical result: subtracting any function $b(s_t)$ that depends only on the state (not the action) from the return preserves the expected gradient:

$$\mathbb{E}_{a \sim \pi_\theta}\left[\nabla_\theta \log \pi_\theta(a|s) \cdot b(s)\right] = b(s) \cdot \nabla_\theta \sum_a \pi_\theta(a|s) = b(s) \cdot \nabla_\theta 1 = 0$$

Therefore, the modified gradient estimate remains unbiased:

$$\mathbb{E}\left[\nabla_\theta \log \pi_\theta(a|s)(G_t - b(s_t))\right] = \nabla_\theta J(\theta)$$

### Variance Reduction

The variance of the gradient estimate with baseline $b$ is:

$$\text{Var}[\hat{g}] = \mathbb{E}\left[\left(\nabla_\theta \log \pi_\theta(a|s)(G_t - b(s_t))\right)^2\right] - \left(\nabla_\theta J(\theta)\right)^2$$

Choosing $b(s_t) \approx \mathbb{E}[G_t | s_t] = V^{\pi}(s_t)$ minimizes this variance by centering the gradient weights around zero—some actions have positive advantage (better than average) and some have negative advantage (worse than average).

## Common Baseline Choices

### Constant Baseline

The simplest baseline is the average return across episodes:

$$b = \frac{1}{N} \sum_{i=1}^{N} G_0^{(i)}$$

Advantages: trivial to compute, no additional parameters. Limitation: does not account for state-dependent value differences.

### Moving Average Baseline

A running exponential average of returns:

$$b_t = (1 - \beta) \cdot b_{t-1} + \beta \cdot G_0^{(t)}$$

Adapts to changing policy performance over training but still state-independent.

### State-Dependent Value Baseline

The theoretically optimal form uses the value function:

$$b(s_t) = V^{\pi_\theta}(s_t) \approx V_\phi(s_t)$$

This transforms returns into advantages: $A_t = G_t - V_\phi(s_t)$. This is the basis of actor-critic methods (Section 34.2).

### Optimal Constant Baseline

The variance-minimizing constant baseline has the analytical form:

$$b^* = \frac{\mathbb{E}\left[\left(\nabla_\theta \log \pi_\theta\right)^2 G_t\right]}{\mathbb{E}\left[\left(\nabla_\theta \log \pi_\theta\right)^2\right]}$$

This is a weighted average of returns, where weights are the squared score function norms. In practice, the value function baseline provides better variance reduction.

## From Baselines to Advantages

Subtracting the value function baseline from returns yields the advantage:

$$A^{\pi}(s_t, a_t) = Q^{\pi}(s_t, a_t) - V^{\pi}(s_t) = G_t - V^{\pi}(s_t)$$

The advantage has a natural interpretation:
- $A > 0$: Action $a_t$ was better than the average action in state $s_t$
- $A < 0$: Action $a_t$ was worse than average
- $A = 0$: Action $a_t$ was exactly average

This centering is what drives effective variance reduction.

## Value Function Learning

When using a learned value function baseline, we jointly optimize:

1. **Policy objective**: $\max_\theta \mathbb{E}[\log \pi_\theta(a|s) \cdot A_t]$
2. **Value objective**: $\min_\phi \mathbb{E}[(G_t - V_\phi(s_t))^2]$

The value function is trained via regression on observed returns. Common choices:
- Mean squared error (MSE) loss
- Huber loss (more robust to outliers)
- Clipped value loss (used in PPO)

## Advantage Estimation Methods

Several methods approximate the advantage with different bias-variance profiles:

| Method | Estimate | Bias | Variance |
|--------|----------|------|----------|
| Monte Carlo | $G_t - V(s_t)$ | None | High |
| 1-step TD | $r_t + \gamma V(s_{t+1}) - V(s_t)$ | High (if $V$ imperfect) | Low |
| n-step TD | $\sum_{k=0}^{n-1}\gamma^k r_{t+k} + \gamma^n V(s_{t+n}) - V(s_t)$ | Medium | Medium |
| GAE($\lambda$) | $\sum_{k=0}^{\infty} (\gamma\lambda)^k \delta_{t+k}$ | Tunable | Tunable |

GAE (Generalized Advantage Estimation) provides a smooth interpolation and is covered in detail in Section 34.2.4.

## Practical Considerations

### Advantage Normalization

After computing advantages, normalize them per batch:

$$\hat{A}_t = \frac{A_t - \mu_A}{\sigma_A + \epsilon}$$

This keeps the gradient scale stable across training, independent of reward magnitude.

### Separate Value Training

The value function should be trained with sufficient capacity to provide an accurate baseline. Common practices:
- Use more gradient steps for value update than policy
- Larger network capacity for value function
- Separate learning rate for value function

### Bootstrapping Trade-off

Using $V(s_{t+1})$ in the advantage estimate (bootstrapping) reduces variance but introduces bias proportional to value function error. Early in training when $V_\phi$ is inaccurate, Monte Carlo returns may be preferable.

## Summary

Baselines are the primary variance reduction tool in policy gradients. The value function baseline $V^{\pi}(s)$ converts returns to advantages, providing the theoretical foundation for actor-critic methods. Proper advantage estimation—balancing bias and variance—is critical for efficient policy gradient learning.
