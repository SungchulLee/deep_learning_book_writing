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

### Choosing lambda

- $\lambda = 0.95$: Default choice for most problems
- $\lambda = 0.97$: Better for long-horizon tasks
- $\lambda = 0.9$: Better for tasks with accurate value functions
- Lower $\lambda$ when value function is well-trained; higher when exploration is critical

### Interaction with gamma

The effective bias-variance trade-off depends on both $\gamma$ and $\lambda$:

- $\gamma$ controls the actual horizon of the MDP
- $\lambda$ controls how much of that horizon is estimated via bootstrapping vs. actual rewards
- $\gamma \lambda$ determines the effective weighting decay

### Mini-batch Computation

When using mini-batch updates (as in PPO), GAE is computed once for the full rollout, then advantages are used across multiple epochs of mini-batch optimization.

## Summary

GAE provides a unified, tunable framework for advantage estimation that subsumes both TD and Monte Carlo methods. The single hyperparameter $\lambda$ offers a clean bias-variance knob. Combined with PPO's clipped objective, GAE enables stable, sample-efficient policy optimization across a wide range of tasks.

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
