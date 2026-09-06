# 35.4.3 VaR Constraints
## Learning Objectives

- Implement Value-at-Risk constraints in RL portfolio optimization
- Understand parametric, historical, and Monte Carlo VaR estimation
- Design constrained RL formulations with VaR limits
- Handle VaR constraints via Lagrangian relaxation in policy optimization

## Introduction

Value-at-Risk (VaR) quantifies the maximum expected loss over a given time horizon at a specified confidence level. For example, a 1-day 95% VaR of \$1M means there is a 5% chance of losing more than \$1M in a single day. Regulatory requirements (Basel III) mandate VaR-based capital reserves, making VaR constraints essential in production systems.

## VaR Definition

$$\text{VaR}_\alpha = -\inf\{x : P(R \leq x) > \alpha\}$$

At confidence level $(1-\alpha)$: the loss that is exceeded with probability $\alpha$.

## VaR Estimation Methods

### 1. Parametric (Variance-Covariance)

Assumes Gaussian returns:

$$\text{VaR}_\alpha = -(\mu_p - z_\alpha \cdot \sigma_p) \cdot V$$

where $z_\alpha$ is the standard normal quantile, $\mu_p$ and $\sigma_p$ are portfolio return mean and standard deviation.

### 2. Historical Simulation

Use the empirical distribution of past returns:

$$\text{VaR}_\alpha = -\text{Quantile}_\alpha(\{R_1, R_2, \ldots, R_T\}) \cdot V$$

### 3. Monte Carlo

Simulate future returns from a fitted model and compute the quantile.

## Constrained RL with VaR

### Lagrangian Relaxation

$$\mathcal{L}(\theta, \lambda) = \mathbb{E}_\pi\left[\sum r_t\right] + \lambda \cdot \left(\text{VaR}_\alpha^{\text{limit}} - \text{VaR}_\alpha(\pi)\right)$$

Dual update: $\lambda \leftarrow \max(0, \lambda + \eta (\text{VaR}_\alpha(\pi) - \text{VaR}_\alpha^{\text{limit}}))$

### Reward Penalty

$$r_t^{\text{constrained}} = r_t - \lambda_{\text{var}} \cdot \max(0, \hat{\text{VaR}}_t - \text{VaR}_{\text{limit}})$$

### Action Masking

Before executing an action, check if the resulting portfolio VaR exceeds the limit. If so, scale down the action.

## Summary

VaR constraints ensure RL policies operate within regulatory and risk management bounds. The combination of Lagrangian relaxation for soft constraints and action masking for hard constraints provides robust VaR control.

## References

- Jorion, P. (2006). Value at Risk: The New Benchmark for Managing Financial Risk. McGraw-Hill.
- Chow, Y., et al. (2017). Risk-Constrained Reinforcement Learning with Percentile Risk Criteria. JMLR.
- Tamar, A., et al. (2015). Optimizing the CVaR via Sampling. AAAI.

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
