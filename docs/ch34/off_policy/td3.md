# 34.4.2 Twin Delayed DDPG (TD3)

## Introduction

TD3 (Fujimoto et al., 2018) addresses three critical issues in DDPG through three targeted modifications: twin critics to combat overestimation, delayed policy updates, and target policy smoothing. These simple changes dramatically improve stability and performance.

## Three Key Modifications

### 1. Clipped Double-Q Learning (Twin Critics)

DDPG suffers from Q-value overestimation due to the max operator in the Bellman backup. TD3 maintains two independent critic networks and takes the minimum:

$$y = r + \gamma (1-d) \min_{i=1,2} Q_{\phi'_i}(s', \tilde{a}')$$

This pessimistic estimate counteracts the overestimation bias.

### 2. Delayed Policy Updates

The actor is updated less frequently than the critic (typically every 2 critic updates). This allows the critic to converge to more accurate Q-values before the actor exploits them, reducing the accumulation of errors.

### 3. Target Policy Smoothing

Noise is added to target actions to smooth the Q-function:

$$\tilde{a}' = \text{clip}(\mu_{\theta'}(s') + \text{clip}(\epsilon, -c, c), a_\text{low}, a_\text{high})$$
$$\epsilon \sim \mathcal{N}(0, \sigma)$$

This regularizes the critic by preventing it from developing sharp peaks that the actor could exploit.

## TD3 Algorithm

```
Initialize actors μ_θ, critics Q_{φ1}, Q_{φ2}, targets
Initialize replay buffer D

For each timestep:
    a = μ_θ(s) + ε, ε ~ N(0, σ)
    Store (s, a, r, s', done) in D
    Sample minibatch from D
    
    # Target with smoothing
    ã' = clip(μ_{θ'}(s') + clip(ε, -c, c), a_low, a_high)
    y = r + γ(1-d) min(Q_{φ1'}(s', ã'), Q_{φ2'}(s', ã'))
    
    # Update both critics
    Update φ1: minimize (Q_{φ1}(s,a) - y)²
    Update φ2: minimize (Q_{φ2}(s,a) - y)²
    
    # Delayed actor update (every d steps)
    If t mod d == 0:
        Update θ: maximize Q_{φ1}(s, μ_θ(s))
        Soft update all targets
```

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Policy delay $d$ | 2 | Actor update frequency |
| Target noise $\sigma$ | 0.2 | Smoothing noise std |
| Noise clip $c$ | 0.5 | Smoothing noise bound |
| Exploration noise | 0.1 | Action noise std |
| $\tau$ | 0.005 | Soft update coefficient |

## TD3 vs DDPG

| Issue | DDPG | TD3 |
|-------|------|-----|
| Q-overestimation | Single critic | Twin critics (min) |
| Actor-critic coupling | Simultaneous | Delayed actor |
| Target smoothness | None | Regularized |
| Stability | Brittle | Robust |

## Summary

TD3's three modifications—twin critics, delayed updates, and target smoothing—provide a principled fix for DDPG's instabilities. It remains a strong baseline for continuous control tasks, offering competitive performance with straightforward implementation.
