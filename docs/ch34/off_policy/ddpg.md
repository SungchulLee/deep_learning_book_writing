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
