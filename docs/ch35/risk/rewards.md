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

## Risk Aversion Parameter lambda

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

## Exercises

**Exercise 1.**
Design a Gymnasium-compatible environment for the financial problem described in this section. Specify the observation space, action space, reward function, and episode termination conditions.

??? success "Solution to Exercise 1"
    Observation space: a vector containing recent returns, current position, portfolio value, and relevant market features (e.g., volatility, volume). Action space: depends on the problem (discrete for buy/hold/sell, continuous for position sizing). Reward: risk-adjusted return per step (e.g., log return minus penalty for risk). Episode terminates after a fixed horizon (e.g., one trading year) or if portfolio value drops below a threshold (margin call). The environment must handle transaction costs, slippage, and market impact realistically. $\square$

---

**Exercise 2.**
Analyze the reward shaping trade-offs for this financial RL problem. Compare at least three candidate reward functions and discuss which properties of the optimal policy each preserves.

??? success "Solution to Exercise 2"
    Candidate rewards: (1) raw PnL -- simple but high variance and delayed; (2) Sharpe-based differential reward $D_t = \frac{\Delta A_t B_{t-1} - \frac{1}{2}\Delta B_t A_{t-1}}{(B_{t-1} - A_{t-1}^2)^{3/2}}$ -- directly optimizes the Sharpe ratio but complex; (3) log return with drawdown penalty $r_t = \log(V_t/V_{t-1}) - \lambda \max(0, DD_t - \tau)$ -- balances return with risk control. The potential-based shaping theorem guarantees that adding $\gamma\Phi(s') - \Phi(s)$ preserves the optimal policy. The Sharpe-based reward changes the optimization objective (may alter optimal policy), while pure PnL preserves it but learns slowly. $\square$

---

**Exercise 3.**
Discuss the non-stationarity challenge specific to this financial application. Propose a concrete strategy for adapting the RL agent to regime changes.

??? success "Solution to Exercise 3"
    Financial markets exhibit regime changes (bull/bear, high/low volatility) that violate the MDP stationarity assumption. A concrete adaptation strategy: (1) include a regime indicator as part of the state (e.g., HMM-estimated regime probabilities); (2) use a meta-learning approach where the agent maintains multiple policies and selects based on detected regime; (3) implement continual learning with an expanding replay buffer weighted toward recent experience (exponential decay weights). The agent should also monitor its own performance and reduce position sizes when out-of-distribution inputs are detected. $\square$

---

**Exercise 4.**
Compare the backtesting results one would expect from this RL approach versus a simple heuristic baseline. What statistical tests should be used to determine if the RL agent genuinely outperforms?

??? success "Solution to Exercise 4"
    Baselines: buy-and-hold, equal-weight, or momentum strategy. The RL agent should be evaluated on walk-forward out-of-sample periods (never on training data). Statistical tests: (1) paired t-test on daily returns for mean difference; (2) bootstrap confidence interval on the Sharpe ratio difference; (3) multiple hypothesis testing correction (e.g., Bonferroni or Holm) if comparing multiple strategies. A common pitfall is p-hacking through hyperparameter tuning on the test set. The evaluation must use a hold-out period that was never used for any model selection. Report both statistical significance and economic significance (transaction costs, capacity). $\square$
