# 33.7.3 Discrete Trading

## Problem Formulation

**Discrete trading** applies DQN to the classic buy/sell/hold decision problem. The agent observes market features and selects from a discrete set of position changes.

### MDP Components

- **State**: $s_t = (\text{position}_t, \text{features}_t)$
  - Current position: flat (0), long (+1), short (-1)
  - Market features: returns, moving averages, RSI, volume, volatility, etc.

- **Action**: $a_t \in \{\text{Buy}, \text{Hold}, \text{Sell}\}$ or equivalently $\{+1, 0, -1\}$

- **Reward**: PnL from position changes minus transaction costs
  $$r_t = \text{position}_t \cdot (p_{t+1} - p_t) - c \cdot |\Delta \text{position}_t|$$

## Feature Engineering

### Technical Features
- Price returns (1, 5, 20 day)
- Moving average crossovers (SMA, EMA)
- RSI, MACD, Bollinger Band position
- Realized volatility (multiple windows)
- Volume ratios

### Normalization
All features should be normalized to approximately zero mean, unit variance:
- Returns: Already centered, scale by rolling std
- Oscillators (RSI): Map to [-1, 1]
- Volume: Log-transform, then z-score

## Reward Design

Several reward formulations exist:

1. **Raw PnL**: $r_t = \text{pos}_t \cdot \text{return}_t - \text{cost}_t$
2. **Risk-adjusted (Sharpe)**: Maximize rolling Sharpe ratio
3. **Differential Sharpe**: $r_t \approx \frac{\partial \text{Sharpe}}{\partial \theta}$ at each step
4. **Log return**: $r_t = \log(1 + \text{pos}_t \cdot \text{return}_t)$ — accounts for compounding

## Transaction Cost Modeling

| Cost component | Typical magnitude |
|---------------|------------------|
| Commission | 0.001% – 0.01% |
| Spread | 0.01% – 0.1% |
| Market impact | 0.01% – 0.5% |
| Slippage | 0.01% – 0.05% |

Transaction costs are critical: many RL agents overfit to noise and trade excessively. Include realistic costs to ensure practical strategies.

## Common Pitfalls

1. **Overfitting**: DQN can memorize training data patterns → use out-of-sample validation
2. **Look-ahead bias**: Features must use only past data
3. **Non-stationarity**: Market regimes change; periodically retrain
4. **Excessive trading**: Without transaction costs, agents may trade every step
5. **Survivorship bias**: Use delisted stocks/instruments in backtest
6. **Data snooping**: Hyperparameter tuning on test set inflates results

## Evaluation Metrics

| Metric | Description | Good Value |
|--------|-------------|------------|
| Sharpe ratio | Risk-adjusted return | > 1.0 |
| Maximum drawdown | Worst peak-to-trough | < 20% |
| Win rate | Fraction of profitable trades | > 50% |
| Profit factor | Gross profit / gross loss | > 1.5 |
| Calmar ratio | Annual return / max drawdown | > 1.0 |
| Trading frequency | Average trades per day | Depends on strategy |
