# 34.1.3 REINFORCE

## Introduction

REINFORCE (Williams, 1992) is the foundational policy gradient algorithm. It uses Monte Carlo sampling of complete episodes to estimate the policy gradient, making it conceptually simple but practically challenged by high variance. Understanding REINFORCE is essential as it forms the basis for all subsequent policy gradient methods.

## Algorithm

The REINFORCE algorithm follows three steps:

1. **Sample**: Collect a trajectory $\tau = (s_0, a_0, r_0, \ldots, s_T, a_T, r_T)$ by following $\pi_\theta$
2. **Estimate**: Compute returns $G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k$ for each timestep
3. **Update**: Perform gradient ascent on $\theta$:

$$\theta \leftarrow \theta + \alpha \sum_{t=0}^{T} \gamma^t \nabla_\theta \log \pi_\theta(a_t | s_t) \, G_t$$

### Pseudocode

```
Initialize policy parameters θ
For each episode:
    Generate trajectory τ = (s₀, a₀, r₀, ..., sₜ, aₜ, rₜ) following π_θ
    For t = T, T-1, ..., 0:
        Gₜ = rₜ + γ · G_{t+1}   (with G_{T+1} = 0)
    Compute loss: L(θ) = -Σₜ log π_θ(aₜ|sₜ) · Gₜ
    Update: θ ← θ - α ∇_θ L(θ)
```

## Properties

### Unbiasedness

REINFORCE produces unbiased gradient estimates because Monte Carlo returns are unbiased estimates of $Q^{\pi_\theta}(s_t, a_t)$:

$$\mathbb{E}_{\tau}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t) G_t\right] = \nabla_\theta J(\theta)$$

### High Variance

The variance of REINFORCE gradients comes from multiple sources:

1. **Trajectory randomness**: Different episodes produce very different returns
2. **Compounding stochasticity**: Both policy randomness and environment stochasticity accumulate over long horizons
3. **Credit assignment**: The return $G_t$ is a noisy signal for individual action quality

Variance scales with episode length and reward magnitude, often requiring thousands of episodes for meaningful learning.

### Convergence

Under standard conditions (diminishing step size $\sum_t \alpha_t = \infty$, $\sum_t \alpha_t^2 < \infty$), REINFORCE converges to a local optimum. In practice, Adam optimizer with fixed learning rate is commonly used.

## Variance Reduction Techniques

### Reward-to-Go (Causality)

Using $G_t$ instead of the total trajectory reward $R(\tau)$ reduces variance by removing rewards that precede action $a_t$. This does not introduce bias because:

$$\mathbb{E}_{a_t \sim \pi_\theta}\left[\nabla_\theta \log \pi_\theta(a_t|s_t) \sum_{k=0}^{t-1} \gamma^k r_k\right] = 0$$

### Discounting

Multiplying the gradient weight by $\gamma^t$ reduces the contribution of distant timesteps. This introduces a mild bias but significantly reduces variance in long-horizon tasks.

### Return Normalization

Normalizing returns to have zero mean and unit variance within a batch:

$$\hat{G}_t = \frac{G_t - \bar{G}}{\text{std}(G) + \epsilon}$$

This simple technique can dramatically improve stability without any theoretical justification other than keeping the gradient scale manageable.

## Limitations

REINFORCE has several practical limitations:

- **Sample inefficiency**: Requires complete episodes and many of them
- **On-policy only**: Cannot reuse data from previous policies
- **High variance**: Slow convergence even with variance reduction
- **No bootstrapping**: Must wait until episode termination
- **Sensitive to hyperparameters**: Learning rate must be carefully tuned

These limitations motivate the actor-critic methods discussed in Section 34.2.

## Summary

REINFORCE is the simplest policy gradient algorithm, directly implementing the policy gradient theorem with Monte Carlo return estimates. While its high variance limits practical utility, it remains foundational. All modern policy gradient methods can be viewed as variance-reduced, stabilized extensions of REINFORCE.
