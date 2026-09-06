# 34.4.1 Deep Deterministic Policy Gradient (DDPG)
## Introduction

DDPG (Lillicrap et al., 2016) extends DQN to continuous action spaces by combining a deterministic policy gradient with an off-policy actor-critic framework. It maintains a deterministic actor $\mu_\theta(s)$ and a Q-function critic $Q_\phi(s,a)$, using experience replay and target networks for stable off-policy learning.

## Key Idea

Unlike stochastic policies that output distributions, DDPG's actor outputs a single deterministic action:

$$a = \mu_\theta(s)$$

The deterministic policy gradient (Silver et al., 2014):

$$\nabla_\theta J(\theta) = \mathbb{E}_{s \sim \mathcal{D}}\left[\nabla_a Q_\phi(s, a)\big|_{a=\mu_\theta(s)} \cdot \nabla_\theta \mu_\theta(s)\right]$$

The gradient flows through the Q-function into the actor, requiring no sampling over actions.

## Algorithm

### Components

1. **Actor** $\mu_\theta(s)$: Deterministic policy network
2. **Critic** $Q_\phi(s, a)$: Action-value function
3. **Target actor** $\mu_{\theta'}$: Slowly-updated copy of actor
4. **Target critic** $Q_{\phi'}$: Slowly-updated copy of critic
5. **Replay buffer** $\mathcal{D}$: Stores transition tuples $(s, a, r, s', d)$

### Update Rules

**Critic update** (minimize TD error):

$$L(\phi) = \mathbb{E}_{(s,a,r,s',d) \sim \mathcal{D}}\left[\left(Q_\phi(s,a) - y\right)^2\right]$$

$$y = r + \gamma (1-d) Q_{\phi'}(s', \mu_{\theta'}(s'))$$

**Actor update** (maximize Q):

$$\nabla_\theta J = \mathbb{E}_{s \sim \mathcal{D}}\left[\nabla_a Q_\phi(s,a)\big|_{a=\mu_\theta(s)} \nabla_\theta \mu_\theta(s)\right]$$

**Target updates** (Polyak averaging):

$$\theta' \leftarrow \tau \theta + (1-\tau) \theta'$$

$$\phi' \leftarrow \tau \phi + (1-\tau) \phi'$$

where $\tau = 0.005$ (soft update coefficient).

### Exploration

Since the policy is deterministic, exploration requires adding noise:

$$a = \mu_\theta(s) + \mathcal{N}(0, \sigma)$$

Common noise processes:

- **Gaussian**: $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$ — simple and effective
- **Ornstein-Uhlenbeck**: Temporally correlated noise for smoother exploration

## DDPG Pseudocode

```
Initialize actor μ_θ, critic Q_φ, target networks θ' ← θ, φ' ← φ
Initialize replay buffer D

For each timestep:
    Select action: a = μ_θ(s) + ε, ε ~ N(0, σ)
    Execute a, observe r, s'
    Store (s, a, r, s', done) in D
    
    Sample minibatch from D
    Compute target: y = r + γ(1-d)Q_{φ'}(s', μ_{θ'}(s'))
    Update critic: minimize (Q_φ(s,a) - y)²
    Update actor: maximize Q_φ(s, μ_θ(s))
    Soft update targets: θ' ← τθ + (1-τ)θ', φ' ← τφ + (1-τ)φ'
```

## Known Issues

DDPG suffers from several problems that motivated TD3 and SAC:

1. **Q-value overestimation**: The critic tends to overestimate Q-values, causing the actor to exploit errors
2. **Brittleness**: Sensitive to hyperparameters, especially learning rates
3. **Exploration**: Gaussian noise may be insufficient for complex environments
4. **Value divergence**: The interplay of function approximation, bootstrapping, and off-policy learning can cause instability

## Hyperparameters

| Parameter | Typical Value | Description |
|-----------|--------------|-------------|
| Actor LR | $1 \times 10^{-4}$ | Actor learning rate |
| Critic LR | $1 \times 10^{-3}$ | Critic learning rate |
| $\tau$ | 0.005 | Soft update coefficient |
| $\gamma$ | 0.99 | Discount factor |
| Buffer size | $10^6$ | Replay buffer capacity |
| Batch size | 256 | Minibatch size |
| Noise $\sigma$ | 0.1 | Exploration noise std |
| Warmup steps | 25000 | Random actions before training |

## Summary

DDPG pioneered continuous-action deep RL by combining deterministic policy gradients with DQN-style stabilization techniques. While effective, its brittleness and overestimation issues led to improved variants: TD3 addresses overestimation, and SAC replaces the deterministic policy with a maximum entropy framework.

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
