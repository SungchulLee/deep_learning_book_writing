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
