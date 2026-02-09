# 35.4.1 Risk-Adjusted Rewards

## Learning Objectives

- Design reward functions that incorporate risk into RL objectives
- Implement Sharpe ratio, Sortino ratio, and Calmar ratio as rewards
- Understand the differential Sharpe ratio for single-step optimization
- Compare risk-neutral vs. risk-adjusted policy behavior

## Introduction

In financial RL, maximizing raw returns without risk consideration leads to catastrophic outcomes—high leverage, concentrated positions, and extreme drawdowns. Risk-adjusted rewards ensure the agent learns to balance return generation with risk management, producing policies suitable for real-world deployment.

## Risk-Adjusted Reward Functions

### 1. Risk-Penalized Return

$$r_t^{\text{adj}} = r_t^{\text{return}} - \lambda \cdot \text{Risk}_t$$

where $\lambda$ controls the risk aversion level.

### 2. Differential Sharpe Ratio

Moody & Saffell (2001) derived a single-step reward that approximates the gradient of the rolling Sharpe ratio:

$$D_t = \frac{B_{t-1} \Delta A_t - \frac{1}{2} A_{t-1} \Delta B_t}{(B_{t-1} - A_{t-1}^2)^{3/2}}$$

where:
- $A_t = A_{t-1} + \eta(r_t - A_{t-1})$ (exponential moving average of returns)
- $B_t = B_{t-1} + \eta(r_t^2 - B_{t-1})$ (EMA of squared returns)

### 3. Sortino-Based Reward

Penalizes only downside volatility:

$$r_t^{\text{sortino}} = r_t - \lambda \cdot \max(0, -r_t)^2$$

### 4. Return-to-Drawdown Reward

$$r_t^{\text{calmar}} = r_t - \lambda \cdot \frac{V_{\text{peak}} - V_t}{V_{\text{peak}}}$$

### 5. Risk Parity Reward

Encourages equal risk contribution across assets:

$$r_t = r_t^{\text{return}} - \lambda \cdot \sum_i \left(\text{RC}_i - \frac{1}{N}\right)^2$$

## Comparison of Reward Functions

| Reward | Optimizes | Behavior |
|--------|-----------|----------|
| Raw return | $\mathbb{E}[R]$ | Aggressive, high variance |
| Sharpe-based | $\mathbb{E}[R]/\sigma$ | Balanced risk-return |
| Sortino-based | $\mathbb{E}[R]/\sigma_{\text{down}}$ | Tolerates upside volatility |
| Calmar-based | $\mathbb{E}[R]/\text{MDD}$ | Drawdown-averse |
| CVaR-adjusted | $\mathbb{E}[R] + \lambda\text{CVaR}$ | Tail-risk aware |

## Risk Aversion Parameter $\lambda$

The choice of $\lambda$ significantly impacts policy behavior:

- $\lambda = 0$: Pure return maximization (risk-neutral)
- $\lambda \in (0, 1)$: Moderate risk aversion
- $\lambda > 1$: Strong risk aversion, conservative positions

In practice, $\lambda$ can be treated as a hyperparameter tuned via validation, or the agent can be conditioned on $\lambda$ for a family of policies.

## Summary

Risk-adjusted rewards are essential for financial RL. The differential Sharpe ratio provides an elegant single-step reward, while Sortino and Calmar variants target specific risk dimensions. The risk aversion parameter $\lambda$ controls the aggressiveness of the learned policy.

## References

- Moody, J. & Saffell, M. (2001). Learning to Trade via Direct Reinforcement. IEEE Transactions on Neural Networks.
- Sharpe, W. (1966). Mutual Fund Performance. Journal of Business.
- Sortino, F. & van der Meer, R. (1991). Downside Risk. Journal of Portfolio Management.
