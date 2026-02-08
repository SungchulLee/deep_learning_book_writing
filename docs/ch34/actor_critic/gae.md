# 34.2.4 Generalized Advantage Estimation (GAE)

## Introduction

Generalized Advantage Estimation (Schulman et al., 2016) provides a principled framework for trading off bias and variance in advantage estimation. GAE computes an exponentially-weighted average of multi-step advantage estimates, controlled by a single hyperparameter $\lambda \in [0, 1]$. It has become the standard advantage estimator in modern policy gradient algorithms including PPO.

## Motivation

Different advantage estimators offer different bias-variance profiles:

- **1-step TD**: $\hat{A}_t^{(1)} = \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ — low variance, high bias
- **2-step**: $\hat{A}_t^{(2)} = \delta_t + \gamma \delta_{t+1}$ — medium variance, medium bias
- **$n$-step**: $\hat{A}_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k \delta_{t+k}$ — higher variance, lower bias
- **Monte Carlo**: $\hat{A}_t^{(\infty)} = G_t - V(s_t)$ — highest variance, no bias

GAE smoothly interpolates between these extremes.

## GAE Formula

$$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{k=0}^{T-t-1} (\gamma \lambda)^k \delta_{t+k}$$

where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ is the TD error.

### Special Cases

| $\lambda$ | Estimator | Bias | Variance |
|-----------|-----------|------|----------|
| 0 | 1-step TD ($\delta_t$) | High | Low |
| 0.5 | Balanced mixture | Medium | Medium |
| 0.95 | Standard choice | Low | Moderate |
| 1 | Monte Carlo ($G_t - V(s_t)$) | None | High |

## Efficient Computation

GAE is computed recursively in a single backward pass:

$$\hat{A}_{T-1} = \delta_{T-1}$$
$$\hat{A}_t = \delta_t + \gamma \lambda (1 - d_{t+1}) \hat{A}_{t+1}$$

where $d_{t+1}$ indicates terminal states. This is $O(T)$ in time and memory.

## Bias-Variance Analysis

The bias of GAE stems from value function approximation error. With perfect value function, GAE is unbiased for all $\lambda$. In practice:

- Smaller $\lambda$: More reliance on value function, more bias if $V$ is inaccurate, less variance
- Larger $\lambda$: Less reliance on value function, less bias, more variance from Monte Carlo-like estimation

The effective discount for GAE is $\gamma_\text{eff} = \gamma \lambda$, meaning GAE with $\gamma=0.99, \lambda=0.95$ has an effective horizon of $\frac{1}{1-\gamma\lambda} \approx 19$ steps.

## GAE with PPO

In PPO, GAE advantages are used for both the clipped policy objective and the return targets:

1. Compute GAE advantages: $\hat{A}_t^{\text{GAE}}$
2. Normalize: $\hat{A}_t \leftarrow \frac{\hat{A}_t - \mu}{\sigma + \epsilon}$
3. Return targets: $\hat{R}_t = \hat{A}_t^{\text{GAE}} + V_\phi(s_t)$

The value function is then trained on $\hat{R}_t$, not on Monte Carlo returns directly.

## Practical Considerations

### Choosing $\lambda$

- $\lambda = 0.95$: Default choice for most problems
- $\lambda = 0.97$: Better for long-horizon tasks
- $\lambda = 0.9$: Better for tasks with accurate value functions
- Lower $\lambda$ when value function is well-trained; higher when exploration is critical

### Interaction with $\gamma$

The effective bias-variance trade-off depends on both $\gamma$ and $\lambda$:
- $\gamma$ controls the actual horizon of the MDP
- $\lambda$ controls how much of that horizon is estimated via bootstrapping vs. actual rewards
- $\gamma \lambda$ determines the effective weighting decay

### Mini-batch Computation

When using mini-batch updates (as in PPO), GAE is computed once for the full rollout, then advantages are used across multiple epochs of mini-batch optimization.

## Summary

GAE provides a unified, tunable framework for advantage estimation that subsumes both TD and Monte Carlo methods. The single hyperparameter $\lambda$ offers a clean bias-variance knob. Combined with PPO's clipped objective, GAE enables stable, sample-efficient policy optimization across a wide range of tasks.
