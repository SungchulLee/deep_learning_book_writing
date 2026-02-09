# 35.2.1 Problem Formulation

## Learning Objectives

- Formulate portfolio management as a Markov Decision Process
- Define appropriate state, action, and reward spaces for portfolio optimization
- Understand the relationship between classical portfolio theory and RL approaches
- Map Markowitz mean-variance optimization to the RL framework

## Introduction

Portfolio management—the continuous allocation of capital across a universe of assets—is one of the most natural applications of reinforcement learning in finance. At each time step, the agent observes market conditions and its current portfolio, then decides how to rebalance its holdings to maximize risk-adjusted returns.

The RL formulation extends classical portfolio theory (Markowitz, 1952) by allowing the agent to learn non-linear, time-varying allocation policies directly from data, without assuming fixed return distributions or static covariance structures.

## MDP Formulation

### State Space

The portfolio management state combines market information with portfolio-specific variables:

$$s_t = \left( \mathbf{X}_t^{\text{market}}, \mathbf{X}_t^{\text{portfolio}}, \mathbf{X}_t^{\text{context}} \right)$$

where:

| Component | Description | Examples |
|-----------|-------------|----------|
| $\mathbf{X}_t^{\text{market}}$ | Market features | Returns, volatility, correlations, volume |
| $\mathbf{X}_t^{\text{portfolio}}$ | Portfolio state | Current weights, cash position, unrealized P&L |
| $\mathbf{X}_t^{\text{context}}$ | Contextual info | Time features, regime indicators, macro variables |

### Action Space

The action $a_t$ represents the **target portfolio weights** $\mathbf{w}_t \in \mathbb{R}^N$ for $N$ assets:

$$a_t = \mathbf{w}_t = (w_{t,1}, w_{t,2}, \ldots, w_{t,N})$$

Subject to constraints:

- **Long-only**: $w_{t,i} \geq 0 \; \forall i$
- **Fully invested**: $\sum_{i=1}^{N} w_{t,i} = 1$ (or $\leq 1$ with cash)
- **Position limits**: $w_{t,i} \leq w_{\max}$
- **Leverage limits**: $\sum_{i=1}^{N} |w_{t,i}| \leq L_{\max}$

### Transition Dynamics

The portfolio evolves through market returns and rebalancing:

$$\tilde{w}_{t+1,i} = \frac{w_{t,i} (1 + r_{t+1,i})}{\sum_{j=1}^{N} w_{t,j} (1 + r_{t+1,j})}$$

where $\tilde{w}_{t+1}$ are the **drift weights** before rebalancing, and $r_{t+1,i}$ is the return of asset $i$.

### Reward Function

The single-step reward captures portfolio performance net of costs:

$$r_t = \ln\left(\frac{V_t}{V_{t-1}}\right) - c \cdot \|\mathbf{w}_t - \tilde{\mathbf{w}}_t\|_1$$

where $V_t$ is portfolio value and $c$ is the transaction cost rate. More sophisticated reward functions incorporate risk penalties:

$$r_t^{\text{risk-adj}} = r_t^{\text{return}} - \lambda \cdot \text{Risk}(r_{t-W:t})$$

## Connection to Classical Portfolio Theory

### Markowitz Mean-Variance

The classical Markowitz optimization solves:

$$\max_{\mathbf{w}} \; \mathbf{w}^T \boldsymbol{\mu} - \frac{\lambda}{2} \mathbf{w}^T \boldsymbol{\Sigma} \mathbf{w}$$

This is a **single-period, static** optimization. The RL formulation generalizes this in several ways:

| Aspect | Markowitz | RL Approach |
|--------|-----------|-------------|
| Horizon | Single period | Multi-period with discounting |
| Distributions | Assumes Gaussian | Distribution-free, learned |
| Parameters | Estimates $\mu$, $\Sigma$ | Learned end-to-end |
| Transaction costs | Typically ignored | Explicitly modeled |
| Non-linearity | Linear weights | Non-linear policy |
| Adaptivity | Static rebalance | Dynamic, state-dependent |

### Dynamic Programming Connection

The RL value function for portfolio management is:

$$V^\pi(s_t) = \mathbb{E}_\pi \left[ \sum_{k=0}^{T-t} \gamma^k r_{t+k} \mid s_t \right]$$

When the reward is the log-return, maximizing the cumulative reward is equivalent to maximizing the growth rate of the portfolio (Kelly criterion):

$$\max_\pi \; \mathbb{E}\left[\sum_{t=0}^{T} \ln\left(\frac{V_{t+1}}{V_t}\right)\right] = \max_\pi \; \mathbb{E}\left[\ln\left(\frac{V_T}{V_0}\right)\right]$$

## Policy Architecture

The policy network maps states to portfolio weights with built-in constraint satisfaction:

```
State → Encoder → Hidden Layers → Softmax → Portfolio Weights
         |                |
    (Market features)  (Dropout/LayerNorm)
```

The **softmax output** naturally satisfies the simplex constraint $\sum w_i = 1, \; w_i \geq 0$. For portfolios allowing short positions, alternative output layers are needed (e.g., tanh with normalization).

## Objective Function Variants

Different RL objectives lead to different portfolio behaviors:

| Objective | Formula | Behavior |
|-----------|---------|----------|
| Maximum Return | $\max \mathbb{E}[\sum r_t]$ | Aggressive, high variance |
| Sharpe Ratio | $\max \frac{\mathbb{E}[R]}{\text{std}(R)}$ | Risk-adjusted performance |
| Sortino Ratio | $\max \frac{\mathbb{E}[R]}{\text{downside\_std}(R)}$ | Penalizes downside only |
| Max Drawdown | $\max \mathbb{E}[R] - \lambda \cdot \text{MDD}$ | Drawdown-aware |
| CVaR | $\max \mathbb{E}[R] + \lambda \cdot \text{CVaR}_\alpha$ | Tail-risk aware |

## Summary

Formulating portfolio management as an MDP enables learning dynamic, non-linear allocation policies that adapt to market conditions. The RL framework generalizes classical portfolio theory by handling multi-period optimization, transaction costs, and non-Gaussian return distributions naturally. Key design decisions include the choice of state features, action representation, and reward function, each significantly impacting the learned policy.

## References

- Markowitz, H. (1952). Portfolio Selection. The Journal of Finance.
- Jiang, Z., Xu, D., & Liang, J. (2017). A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem. arXiv:1706.10059.
- Moody, J. & Saffell, M. (2001). Learning to Trade via Direct Reinforcement. IEEE Transactions on Neural Networks.
