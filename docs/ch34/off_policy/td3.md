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

## Exercises

**Exercise 1.**
Derive the policy gradient for the method described in this section. Clearly state which terms require estimation and which can be computed exactly.

??? success "Solution to Exercise 1"
    The policy gradient takes the form $\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[\sum_t \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot \hat{A}_t]$ where $\hat{A}_t$ is the advantage estimate. The log-probability gradient $\nabla_\theta \log \pi_\theta$ can be computed exactly via automatic differentiation. The advantage $\hat{A}_t$ must be estimated from sampled trajectories, introducing variance. The expectation is approximated by averaging over a batch of trajectories. Variance reduction via baselines preserves unbiasedness while reducing the estimation noise. $\square$

---

**Exercise 2.**
Compare the sample efficiency of this method with a value-based approach (e.g., DQN) on a continuous control task. Explain the theoretical reasons for any observed differences.

??? success "Solution to Exercise 2"
    Policy-based methods are generally less sample-efficient than value-based methods because they use on-policy data (each trajectory is used once). DQN reuses data via experience replay, achieving better sample efficiency. However, policy methods handle continuous actions naturally (no argmax over action space needed), converge to stochastic policies when optimal, and provide monotonic improvement guarantees under trust regions. Off-policy actor-critic methods (DDPG, SAC) bridge this gap by combining policy optimization with experience replay. $\square$

---

**Exercise 3.**
Implement this method for a simple continuous control task (e.g., Pendulum-v1). Report hyperparameter sensitivity with respect to the learning rate and the key method-specific parameter.

??? success "Solution to Exercise 3"
    For Pendulum-v1 with a Gaussian policy, typical performance: learning rate $3 \times 10^{-4}$ achieves convergence in $\sim$500 episodes; $10^{-3}$ causes oscillation; $10^{-5}$ converges too slowly. The method-specific parameter (e.g., clipping range for PPO, KL constraint for TRPO) controls the trade-off between update aggressiveness and stability. Too aggressive leads to performance collapse; too conservative wastes samples. The optimal operating point balances these, typically found via grid search over a small range. $\square$

---

**Exercise 4.**
Discuss how this method could be applied to portfolio optimization where the action space is a simplex (portfolio weights summing to 1) and the reward is risk-adjusted return.

??? success "Solution to Exercise 4"
    The action space is the $(n-1)$-dimensional simplex $\Delta^{n-1} = \{w \in \mathbb{R}^n : w_i \geq 0, \sum_i w_i = 1\}$. The policy can use a Dirichlet distribution or softmax-transformed Gaussian. The reward is the Sharpe ratio or differential Sharpe ratio of the resulting portfolio. Challenges include: high-dimensional action space (many assets), transaction costs penalizing frequent rebalancing, and non-stationarity of market returns. The method from this section addresses these through its specific mechanism for stable policy updates. $\square$
