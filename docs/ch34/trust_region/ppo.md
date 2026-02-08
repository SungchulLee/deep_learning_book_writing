# 34.3.3 Proximal Policy Optimization (PPO)

## Introduction

PPO (Schulman et al., 2017) achieves the stability benefits of TRPO while being dramatically simpler to implement. Instead of solving a constrained optimization problem, PPO uses a clipped surrogate objective that implicitly constrains the policy update. PPO has become the most widely used policy gradient algorithm due to its strong performance, robustness, and ease of implementation.

## Core Idea

PPO's insight is that clipping the probability ratio $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)}$ provides a first-order approximation to the trust region constraint:

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t\right)\right]$$

where $\epsilon$ is the clipping parameter (typically $0.2$).

## The Clipped Objective

### How Clipping Works

The clipped objective creates a pessimistic bound on policy improvement:

**When $A_t > 0$ (good action)**:
- If $r_t > 1 + \epsilon$: Gradient is clipped (prevents overexploitation)
- If $r_t \leq 1 + \epsilon$: Normal gradient (encourages the action)

**When $A_t < 0$ (bad action)**:
- If $r_t < 1 - \epsilon$: Gradient is clipped (prevents overcorrection)
- If $r_t \geq 1 - \epsilon$: Normal gradient (discourages the action)

The `min` operator ensures we take the more conservative update.

### Comparison with Unclipped

Without clipping, the surrogate objective $r_t(\theta) A_t$ can lead to:
- Very large policy changes when $r_t$ deviates far from 1
- Catastrophic performance collapse
- Sensitivity to learning rate

Clipping bounds the effective policy change per update.

## Full PPO Objective

The complete PPO loss combines three terms:

$$L(\theta) = -L^{\text{CLIP}}(\theta) + c_v L^{\text{VF}}(\theta) - c_e S[\pi_\theta]$$

where:
- $L^{\text{CLIP}}$: Clipped policy surrogate
- $L^{\text{VF}} = (V_\theta(s_t) - V_t^{\text{target}})^2$: Value function loss
- $S[\pi_\theta]$: Entropy bonus
- $c_v = 0.5$, $c_e = 0.01$ (typical coefficients)

## PPO Algorithm

```
For each iteration:
    Collect T timesteps of data using π_θ_old
    Compute advantages using GAE(γ, λ)
    
    For K epochs:
        For each minibatch:
            Compute r_t(θ) = π_θ(a|s) / π_θ_old(a|s)
            L_clip = min(r_t · A_t, clip(r_t, 1-ε, 1+ε) · A_t)
            L_value = (V_θ(s) - V_target)²
            L = -L_clip + c_v · L_value - c_e · entropy
            Update θ using gradient of L
```

### Key Design Choices

1. **Multiple epochs**: PPO reuses collected data for $K$ gradient steps (typically $K = 3\text{--}10$)
2. **Minibatch updates**: Splits the rollout into random minibatches for each epoch
3. **Advantage normalization**: Normalize advantages per minibatch for stable gradients
4. **Gradient clipping**: Additional safeguard via max gradient norm

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| $\epsilon$ (clip) | 0.2 | Clipping range |
| Learning rate | $2.5 \times 10^{-4}$ | Adam step size |
| $\gamma$ | 0.99 | Discount factor |
| $\lambda$ (GAE) | 0.95 | GAE parameter |
| $K$ (epochs) | 4 | Update epochs per rollout |
| Minibatch size | 64 | Gradient update batch size |
| $c_v$ | 0.5 | Value loss coefficient |
| $c_e$ | 0.01 | Entropy coefficient |
| Max grad norm | 0.5 | Gradient clipping |
| $N_\text{envs}$ | 8 | Parallel environments |
| $T$ (rollout) | 128 | Steps per rollout |

## PPO Variants

### PPO-Clip (Standard)
Uses the clipped surrogate objective described above.

### PPO-Penalty
Uses an adaptive KL penalty instead of clipping:

$$L(\theta) = \mathbb{E}_t\left[r_t(\theta) A_t - \beta D_{\text{KL}}(\pi_{\theta_\text{old}} \| \pi_\theta)\right]$$

$\beta$ is adjusted: increased if KL > target, decreased if KL < target. Less commonly used than PPO-Clip.

## Why PPO Works

PPO's effectiveness comes from several factors:
1. **Bounded updates**: Clipping prevents catastrophic policy changes
2. **Data efficiency**: Multiple epochs extract more learning per rollout
3. **Robustness**: Works across diverse environments without extensive tuning
4. **Simplicity**: No second-order optimization or line search required
5. **Scalability**: Easily parallelized with vectorized environments

## Summary

PPO achieves TRPO-level stability through a simple clipped objective, enabling multiple gradient steps per rollout while preventing destructive policy updates. Its combination of performance, simplicity, and robustness has made it the de facto standard for on-policy policy optimization.
