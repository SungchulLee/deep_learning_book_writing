# 35.7.2 Paper Trading

## Learning Objectives

- Implement paper trading for RL strategy validation
- Design realistic paper trading that matches production conditions
- Measure and compare paper vs. backtest performance
- Identify discrepancies between simulation and reality

## Introduction

Paper trading (simulated trading with real-time data but no real money) bridges the gap between backtesting and live trading. It validates that the full system—data pipeline, feature computation, inference, and order management—works correctly in real-time.

## Paper Trading vs. Backtesting

| Aspect | Backtesting | Paper Trading |
|--------|------------|--------------|
| Data | Historical | Real-time |
| Speed | Fast (replay) | Wall-clock |
| Fills | Simulated | Simulated (realistic) |
| Bugs | May be hidden | Exposed by real data |
| Duration | Minutes | Weeks to months |

## Key Validation Checks

### 1. Feature Consistency
- Compare paper-trading features with backtest features for overlapping periods
- Alert on significant divergences

### 2. Signal Stability
- Monitor signal distribution shift between backtest and paper trading
- Check for data quality issues (missing data, stale prices)

### 3. Execution Realism
- Model realistic fills (spread, partial fills, rejects)
- Track simulated slippage vs. expected slippage

### 4. Performance Tracking
- Compare paper PnL to backtest expectations
- Statistical test: is paper Sharpe within confidence interval of backtest Sharpe?

## Minimum Paper Trading Duration

Based on the strategy's expected Sharpe ratio:

$$T_{\min} \approx \left(\frac{2}{\text{SR}_{\text{daily}}}\right)^2 \text{ days}$$

For daily SR of 0.1: minimum ~400 trading days (unrealistic). In practice, 3-6 months is standard with additional qualitative checks.

## Summary

Paper trading is an essential validation step that catches system bugs, data issues, and performance discrepancies before real capital is at risk.

## References

- de Prado, M.L. (2018). Advances in Financial Machine Learning. Wiley.
