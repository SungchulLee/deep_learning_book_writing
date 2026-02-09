# 35.6.3 Performance Metrics

## Learning Objectives

- Compute comprehensive performance metrics for RL trading strategies
- Understand the relationship between different risk-return metrics
- Implement rolling and conditional performance analysis
- Compare RL agents against benchmark strategies

## Introduction

Performance metrics quantify the quality of a trading strategy. No single metric captures all dimensions of performance; a comprehensive evaluation requires examining returns, risk, tail behavior, and efficiency.

## Key Metrics

### Return Metrics

| Metric | Formula | Interpretation |
|--------|---------|---------------|
| Total Return | $(V_T/V_0 - 1) \times 100$ | Cumulative percentage gain |
| CAGR | $(V_T/V_0)^{252/T} - 1$ | Annualized compound growth rate |

### Risk Metrics

| Metric | Description |
|--------|-------------|
| Volatility | Annualized standard deviation |
| Max Drawdown | Worst peak-to-trough decline |
| VaR (95%) | 5th percentile loss |
| CVaR (95%) | Expected tail loss |

### Risk-Adjusted Metrics

| Metric | Formula |
|--------|---------|
| Sharpe Ratio | Mean excess return / volatility |
| Sortino Ratio | Mean excess return / downside vol |
| Calmar Ratio | CAGR / Max Drawdown |
| Information Ratio | Active return / Tracking error |

### Trading Metrics

| Metric | Description |
|--------|-------------|
| Win Rate | Fraction of profitable days |
| Profit Factor | Gross profit / Gross loss |
| Turnover | Average daily portfolio turnover |

## Summary

A comprehensive performance evaluation requires multiple metrics spanning returns, risk, risk-adjusted performance, and trading efficiency.

## References

- Bacon, C. (2008). Practical Portfolio Performance Measurement and Attribution. Wiley.
