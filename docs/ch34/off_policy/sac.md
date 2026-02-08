# 34.4.3 Soft Actor-Critic (SAC)

## Introduction

SAC (Haarnoja et al., 2018) combines off-policy actor-critic learning with maximum entropy reinforcement learning. By augmenting the reward with an entropy bonus, SAC encourages exploration while learning near-optimal policies. The stochastic policy, automatic temperature tuning, and twin critics make SAC one of the most robust and sample-efficient continuous control algorithms.

## Maximum Entropy Objective

SAC maximizes the entropy-augmented return:

$$J(\pi) = \sum_{t=0}^{T} \mathbb{E}_{(s_t, a_t) \sim \rho_\pi}\left[r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t))\right]$$

where $\alpha > 0$ is the temperature parameter controlling the exploration-exploitation trade-off.

The optimal policy under this objective is:

$$\pi^*(a|s) \propto \exp\left(\frac{1}{\alpha} Q^*(s, a)\right)$$

## Algorithm Components

### 1. Stochastic Policy (Squashed Gaussian)

Unlike DDPG/TD3's deterministic policies, SAC uses a stochastic policy parameterized as a squashed Gaussian:

$$a = \tanh(\mu_\theta(s) + \sigma_\theta(s) \odot \epsilon), \quad \epsilon \sim \mathcal{N}(0, I)$$

The reparameterization trick enables gradient flow through the sampling.

### 2. Twin Q-Functions

SAC uses two Q-networks (like TD3) to mitigate overestimation:

$$Q_\text{target} = r + \gamma(1-d)\left(\min_{i=1,2} Q_{\phi'_i}(s', a') - \alpha \log \pi_\theta(a'|s')\right)$$

### 3. Automatic Temperature Tuning

SAC automatically adjusts $\alpha$ to maintain a target entropy $\bar{\mathcal{H}}$:

$$\alpha^* = \arg\min_\alpha \mathbb{E}_{a \sim \pi}\left[-\alpha \log \pi(a|s) - \alpha \bar{\mathcal{H}}\right]$$

Target entropy is typically set to $\bar{\mathcal{H}} = -\dim(\mathcal{A})$ (negative action dimension).

## SAC Update Rules

**Q-function update**:
$$L(\phi_i) = \mathbb{E}\left[\left(Q_{\phi_i}(s,a) - y\right)^2\right]$$
$$y = r + \gamma(1-d)\left(\min_j Q_{\phi'_j}(s', \tilde{a}') - \alpha \log \pi_\theta(\tilde{a}'|s')\right)$$

**Policy update**:
$$L(\theta) = \mathbb{E}_{s \sim \mathcal{D}}\left[\alpha \log \pi_\theta(\tilde{a}|s) - \min_i Q_{\phi_i}(s, \tilde{a})\right]$$

where $\tilde{a}$ is sampled via reparameterization.

**Temperature update**:
$$L(\alpha) = \mathbb{E}_{a \sim \pi}\left[-\alpha(\log \pi_\theta(a|s) + \bar{\mathcal{H}})\right]$$

## SAC Advantages

- **Robust exploration**: Entropy maximization prevents premature convergence
- **Sample efficient**: Off-policy with replay buffer
- **Automatic tuning**: Temperature adapts to the task
- **Stable**: Twin critics + soft updates + stochastic policy
- **No noise tuning**: Exploration emerges from the maximum entropy objective

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Actor/Critic LR | $3 \times 10^{-4}$ | Learning rates |
| $\alpha$ LR | $3 \times 10^{-4}$ | Temperature learning rate |
| $\gamma$ | 0.99 | Discount factor |
| $\tau$ | 0.005 | Soft update coefficient |
| Target entropy | $-\dim(\mathcal{A})$ | Entropy target |
| Buffer size | $10^6$ | Replay capacity |
| Batch size | 256 | Minibatch size |

## Summary

SAC achieves state-of-the-art sample efficiency for continuous control by unifying maximum entropy RL with off-policy actor-critic learning. The combination of stochastic policy, automatic temperature tuning, and twin critics creates a robust algorithm that requires minimal hyperparameter tuning.
