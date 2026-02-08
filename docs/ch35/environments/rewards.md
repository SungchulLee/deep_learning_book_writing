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
