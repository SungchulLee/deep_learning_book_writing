# 35.3.4 Momentum Trading

## Learning Objectives

- Formulate momentum trading as an RL problem
- Implement trend-following and cross-sectional momentum strategies
- Design momentum signals as state features for RL agents
- Handle regime-dependent momentum with adaptive policies

## Introduction

Momentum is one of the most well-documented anomalies in financial markets: assets that have performed well recently tend to continue performing well (and vice versa). RL enhances momentum strategies by learning optimal signal combination, position sizing, and timing of entry/exit based on market conditions.

## Types of Momentum

### Time-Series Momentum (Trend Following)

Each asset is evaluated independently based on its own past returns:

$$\text{signal}_t^i = \text{sign}\left(\sum_{k=1}^{L} r_{t-k}^i\right)$$

Position: go long if positive momentum, short if negative.

### Cross-Sectional Momentum

Assets are ranked relative to each other:

$$\text{rank}_t^i = \text{percentile}\left(R_t^i, \{R_t^1, \ldots, R_t^N\}\right)$$

Long the top decile, short the bottom decile (market-neutral).

### Momentum Signals

| Signal | Formula | Lookback |
|--------|---------|----------|
| Simple return | $R_{t,L} = P_t / P_{t-L} - 1$ | 1-12 months |
| Risk-adjusted | $R_{t,L} / \sigma_{t,L}$ | Sharpe of trailing period |
| Residual | $R_{t,L} - \beta R_{t,L}^{\text{market}}$ | Factor-adjusted |
| Dual momentum | $\max(R^{\text{abs}}, R^{\text{rel}})$ | Absolute + relative |

## MDP Formulation

### State

$$s_t = \left(\mathbf{R}_t^{1m}, \mathbf{R}_t^{3m}, \mathbf{R}_t^{6m}, \mathbf{R}_t^{12m}, \sigma_t, V_t, \text{regime}_t, \text{position}_t\right)$$

Multiple lookback returns capture different momentum frequencies.

### Action

For $N$ assets: $a_t = \mathbf{w}_t \in [-1, 1]^N$ (position per asset).

### Reward

$$r_t = \sum_i w_{t,i} \cdot r_{t+1,i} - c \cdot \|\Delta \mathbf{w}_t\|_1$$

## Momentum Crashes

Momentum strategies are vulnerable to sudden reversals ("momentum crashes"):
- Typically occur after market downturns followed by sharp recovery
- Short positions in beaten-down stocks rally violently
- RL can learn to reduce exposure during high-crash-risk periods

**Crash indicators**:
- Market volatility spike ($\text{VIX} > 30$)
- Extreme winner-loser spread
- High market drawdown followed by rapid recovery

## Dynamic Momentum with RL

RL improves over static momentum by:
1. **Adaptive lookback**: Selecting the most predictive lookback period
2. **Regime-dependent sizing**: Reducing exposure in unfavorable regimes
3. **Signal combination**: Weighting multiple momentum signals non-linearly
4. **Turnover optimization**: Smoothing position changes to reduce costs

## Summary

RL-based momentum trading adapts to changing market conditions, combining multiple momentum signals and managing crash risk dynamically. The key advantage over rule-based momentum is the ability to learn non-linear signal combinations and regime-dependent position sizing.

## References

- Jegadeesh, N. & Titman, S. (1993). Returns to Buying Winners and Selling Losers. Journal of Finance.
- Moskowitz, T., Ooi, Y., & Pedersen, L. (2012). Time Series Momentum. Journal of Financial Economics.
- Daniel, K. & Moskowitz, T. (2016). Momentum Crashes. Journal of Financial Economics.
