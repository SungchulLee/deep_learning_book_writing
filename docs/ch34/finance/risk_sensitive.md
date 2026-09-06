# 34.7.3 Risk-Sensitive Reinforcement Learning
## Introduction

Standard RL maximizes expected cumulative reward, ignoring the distribution of outcomes. In finance, risk management is paramount—a strategy with high expected return but catastrophic tail risk is unacceptable. Risk-sensitive RL incorporates risk measures directly into the optimization objective.

## Risk Measures

### Variance-Penalized Return

$$J(\pi) = \mathbb{E}[R] - \lambda \text{Var}(R)$$

Simple but penalizes upside variance equally with downside variance.

### Conditional Value at Risk (CVaR)

$$\text{CVaR}_\alpha = \mathbb{E}[R | R \leq \text{VaR}_\alpha]$$

Expected loss in the worst $\alpha$% of scenarios. More appropriate for tail risk management.

### Sharpe Ratio Objective

$$J(\pi) = \frac{\mathbb{E}[R] - R_f}{\sqrt{\text{Var}(R)}}$$

Risk-adjusted return that balances mean and variance.

### Maximum Drawdown Constraint

$$\text{MDD} = \max_t \left(\max_{s \leq t} V_s - V_t\right) / \max_{s \leq t} V_s$$

Constraining MDD limits the worst peak-to-trough decline.

## Approaches to Risk-Sensitive RL

### 1. Reward Modification
Incorporate risk into the reward function:

$$r_t^{\text{risk}} = r_t - \lambda \cdot \text{risk\_measure}(s_t, a_t)$$

Simple but changes the MDP semantics.

### 2. Distributional RL
Learn the full return distribution $Z^\pi(s, a)$ instead of just $Q^\pi(s, a)$:

- Quantile regression DQN (QR-DQN)
- Implicit quantile networks (IQN)
- Use the learned distribution to compute risk measures

### 3. Constrained MDPs
Optimize return subject to risk constraints:

$$\max_\pi \mathbb{E}[R] \quad \text{s.t.} \quad \text{CVaR}_\alpha(R) \geq \tau$$

Solved via Lagrangian relaxation with adaptive multipliers.

### 4. Mean-Variance Policy Gradient
Direct optimization of Sharpe-like objectives using modified policy gradients:

$$\nabla_\theta J_\text{MV} = \nabla_\theta \mathbb{E}[R] - \lambda \nabla_\theta \text{Var}(R)$$

Requires careful estimation of the variance gradient.

## Practical Implementation

### CVaR Optimization via Sorting
1. Collect batch of episode returns
2. Sort returns in ascending order
3. Take the bottom $\alpha$% as the CVaR sample
4. Optimize policy to improve these worst-case returns

### Lagrangian Approach for Constraints
```
For each iteration:
    Update policy to maximize: L(π, λ) = E[R] - λ(CVaR_constraint - CVaR(π))
    Update λ: λ ← max(0, λ + α_λ(CVaR_constraint - CVaR(π)))
```

## Summary

Risk-sensitive RL is essential for financial applications where tail risk management is as important as return maximization. CVaR-based objectives, distributional RL, and constrained MDPs provide principled frameworks for incorporating risk awareness into policy optimization.

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
