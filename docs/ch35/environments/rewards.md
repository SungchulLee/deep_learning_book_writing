# 35.1.4 Reward Engineering
## Learning Objectives

- Design reward functions that align agent behavior with financial objectives
- Implement risk-adjusted reward formulations
- Understand reward shaping and its impact on learned policies
- Handle the exploration-exploitation trade-off in financial rewards

## Introduction

Reward engineering is perhaps the most consequential design decision in financial RL. The reward function defines what the agent optimizes for, and misaligned rewards lead to catastrophic real-world outcomes. Unlike games where the objective is clear (score, win/loss), financial objectives involve subtle trade-offs between returns, risk, and costs.

## Reward Function Taxonomy

### 1. Simple Return Rewards

**Raw portfolio return**:

$$r_t = \frac{V_t - V_{t-1}}{V_{t-1}}$$

where $V_t$ is portfolio value at time $t$.

**Log return** (preferred for compounding):

$$r_t = \ln\left(\frac{V_t}{V_{t-1}}\right)$$

**Problem**: Maximizing raw returns ignores risk entirely. The agent may learn to take extreme leveraged positions.

### 2. Risk-Adjusted Returns

**Differential Sharpe Ratio** (Moody & Saffell, 2001):

An incremental approximation to the Sharpe ratio suitable for online learning:

$$r_t^{\text{DSR}} = \frac{B_{t-1} \Delta A_t - \frac{1}{2} A_{t-1} \Delta B_t}{(B_{t-1} - A_{t-1}^2)^{3/2}}$$

where $A_t$ and $B_t$ are exponential moving averages of returns and squared returns:

$$A_t = A_{t-1} + \eta (R_t - A_{t-1})$$

$$B_t = B_{t-1} + \eta (R_t^2 - B_{t-1})$$

**Sortino-Based Reward**:

$$r_t = R_t - \lambda \cdot \max(0, -R_t)^2$$

This penalizes downside returns more heavily.

**Risk-Penalized Return**:

$$r_t = R_t - \lambda \cdot \text{Risk}_t$$

where $\text{Risk}_t$ can be variance, drawdown, VaR, or CVaR.

### 3. Benchmark-Relative Rewards

**Tracking error**:

$$r_t = -(R_t^{\text{portfolio}} - R_t^{\text{benchmark}})^2$$

**Information ratio increment**:

$$r_t = (R_t^p - R_t^b) - \lambda \cdot (R_t^p - R_t^b - \overline{\alpha})^2$$

### 4. Profit and Loss (P&L) Based

**Realized P&L**:

$$r_t = \sum_i (\text{sell\_price}_i - \text{buy\_price}_i) \times \text{quantity}_i$$

**Mark-to-Market P&L**:

$$r_t = \sum_i w_{t,i} \cdot R_{t,i} - c \cdot ||\Delta w_t||_1$$

where $c \cdot ||\Delta w_t||_1$ accounts for transaction costs.

### 5. Multi-Objective Rewards

Combine multiple objectives with weighted sum:

$$r_t = \alpha_1 R_t - \alpha_2 \text{DD}_t - \alpha_3 \text{TC}_t + \alpha_4 \mathbb{1}[\text{constraints satisfied}]$$

| Component | Meaning |
|-----------|---------|
| $R_t$ | Portfolio return |
| $\text{DD}_t$ | Drawdown penalty |
| $\text{TC}_t$ | Transaction cost penalty |
| $\mathbb{1}[\cdot]$ | Constraint satisfaction bonus |

## Reward Shaping

### Potential-Based Shaping

Add a shaping term that doesn't change the optimal policy:

$$r'_t = r_t + \gamma \Phi(s_{t+1}) - \Phi(s_t)$$

where $\Phi(s)$ is a potential function. Example: use portfolio Sharpe ratio as potential.

### Temporal Considerations

**Dense vs. sparse rewards**: Dense per-step rewards (daily returns) provide faster learning but may cause myopic behavior. Sparse episode-end rewards (total Sharpe) are cleaner but slower to learn.

**Practical compromise**: Use dense per-step returns with an episode-end bonus:

$$r_t = R_t + \mathbb{1}[t = T] \cdot \text{bonus}(\text{episode\_sharpe})$$

### Reward Scaling

RL algorithms are sensitive to reward scale. Normalize rewards to have roughly unit variance:

$$\hat{r}_t = \frac{r_t - \mu_r}{\sigma_r + \epsilon}$$

Use running statistics for online normalization.

## Transaction Cost Integration

Transaction costs must be included in the reward to prevent excessive trading:

$$r_t = \underbrace{w_t^\top R_t}_{\text{gross return}} - \underbrace{c \cdot \|\Delta w_t\|_1}_{\text{proportional cost}} - \underbrace{c_{\text{fixed}} \cdot \|\Delta w_t\|_0}_{\text{fixed cost}}$$

where:

- $c$ is the proportional cost rate (e.g., 10 bps)
- $c_{\text{fixed}}$ is a fixed per-trade cost
- $\|\Delta w_t\|_0$ counts the number of trades

## Common Pitfalls

1. **Reward hacking**: The agent finds degenerate strategies that maximize the reward but are financially meaningless (e.g., churning to exploit a reward bug)
2. **Delayed consequences**: Today's portfolio construction affects tomorrow's rebalancing cost. Myopic rewards miss this.
3. **Reward non-stationarity**: Market regimes change, causing the same actions to yield different rewards over time.
4. **Scale sensitivity**: Mixing percentage returns with dollar amounts creates imbalanced gradients.

## Summary

Reward engineering in financial RL requires careful alignment of the reward signal with true financial objectives. Risk-adjusted rewards (differential Sharpe ratio, Sortino-based) are preferred over raw returns. Transaction costs must be embedded in the reward. Multi-objective formulations with appropriate weighting capture the complexity of real trading objectives.

## References

- Moody, J., & Saffell, M. (2001). Learning to Trade via Direct Reinforcement. IEEE Transactions on Neural Networks
- Zhang, Z., Zohren, S., & Roberts, S. (2020). Deep Reinforcement Learning for Trading

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
