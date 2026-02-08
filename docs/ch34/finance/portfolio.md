# 34.7.1 Portfolio Optimization with Policy-Based RL

## Introduction

Portfolio optimization using policy-based deep RL frames asset allocation as a sequential decision problem. The agent learns to dynamically rebalance a portfolio across multiple assets, maximizing risk-adjusted returns while accounting for transaction costs and market constraints.

## MDP Formulation

### State Space
The observation at time $t$ includes:
- **Price features**: Returns, moving averages, volatility for each asset
- **Portfolio state**: Current asset weights $w_t$
- **Market features**: VIX, interest rates, sector indicators
- **Account features**: Cash balance, unrealized P&L

$$s_t = (\text{price\_features}_t, w_t, \text{market\_features}_t, \text{account}_t)$$

### Action Space
The action represents target portfolio weights:
$$a_t = (w_1^{\text{target}}, \ldots, w_N^{\text{target}})$$

With constraints: $\sum_i w_i = 1$ (fully invested), $w_i \geq 0$ (long-only) or relaxed for long-short.

### Reward Function
Risk-adjusted return minus transaction costs:
$$r_t = \underbrace{w_t^\top R_t}_{\text{portfolio return}} - \underbrace{\lambda_\text{tc} \sum_i |w_{i,t} - w_{i,t-1}| c_i}_{\text{transaction costs}} - \underbrace{\lambda_\text{risk} \cdot \text{risk}(w_t, R_t)}_{\text{risk penalty}}$$

Common risk measures: variance, maximum drawdown, CVaR.

## Policy Architecture

### Portfolio Policy Network
```
Observation → Feature Extraction → Hidden Layers → Softmax → Portfolio Weights
```

The softmax output naturally satisfies the simplex constraint ($\sum w_i = 1, w_i \geq 0$).

### Temperature-Scaled Softmax
$$w_i = \frac{\exp(h_i / \tau)}{\sum_j \exp(h_j / \tau)}$$

Lower temperature → more concentrated portfolios; higher temperature → more diversified.

## Training Considerations

### Transaction Cost Modeling
Realistic costs include:
- Proportional costs (bid-ask spread, commissions)
- Market impact (price movement from large trades)
- Slippage (execution price differs from decision price)

### Turnover Regularization
Penalize excessive trading to encourage stable portfolios:
$$L_\text{turnover} = \lambda \sum_t \|w_t - w_{t-1}\|_1$$

### Multi-Period Optimization
The RL agent naturally optimizes over multiple periods, considering how today's actions affect future opportunities—a key advantage over single-period mean-variance optimization.

## Comparison with Classical Methods

| Method | Multi-period | Transaction costs | Non-linear constraints | Adaptivity |
|--------|-------------|-------------------|----------------------|------------|
| Mean-Variance | No | Difficult | Limited | No |
| Black-Litterman | No | Difficult | Limited | Partial |
| Risk Parity | No | No | Limited | No |
| Policy-Based RL | Yes | Natural | Any | Yes |

## Summary

Policy-based RL enables dynamic portfolio optimization that naturally incorporates transaction costs, adapts to changing market conditions, and handles complex constraints. The key challenges are realistic simulation, avoiding overfitting to historical data, and ensuring robustness to regime changes.
