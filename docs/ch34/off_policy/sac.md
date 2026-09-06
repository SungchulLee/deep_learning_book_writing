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
