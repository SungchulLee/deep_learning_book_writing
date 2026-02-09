# 35.6.1 Backtesting Framework

## Learning Objectives

- Design a comprehensive backtesting framework for RL trading strategies
- Implement event-driven and vectorized backtesting engines
- Handle realistic market simulation including costs, slippage, and fills
- Avoid common backtesting pitfalls (look-ahead bias, survivorship bias)

## Introduction

Backtesting is the process of evaluating a trading strategy on historical data. For RL strategies, backtesting must simulate the full interaction loop: observation, action, execution, and portfolio update. A robust backtesting framework is essential for distinguishing genuine alpha from overfitting artifacts.

## Architecture

```
┌─────────────────────────────────────────────┐
│              Backtesting Engine              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │   Data    │  │  Strategy │  │ Portfolio │  │
│  │  Handler  │  │   Agent   │  │  Tracker  │  │
│  └─────┬────┘  └─────┬────┘  └─────┬────┘  │
│        │              │              │        │
│        ▼              ▼              ▼        │
│  ┌──────────────────────────────────────┐   │
│  │         Execution Simulator          │   │
│  └──────────────┬───────────────────────┘   │
│                 │                            │
│  ┌──────────────▼───────────────────────┐   │
│  │         Performance Analyzer         │   │
│  └──────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
```

## Common Pitfalls

| Pitfall | Description | Prevention |
|---------|-------------|-----------|
| Look-ahead bias | Using future data | Strict temporal indexing |
| Survivorship bias | Only testing on surviving assets | Include delisted assets |
| Fill assumptions | Assuming instant fills at mid-price | Model spread + slippage |
| Ignoring costs | No transaction costs | Realistic cost model |
| Ignoring capacity | Strategy can't scale | Model market impact |

## Summary

A well-designed backtesting framework separates data handling, strategy logic, execution simulation, and performance analysis into modular components. Realistic cost modeling and strict bias prevention are essential for reliable results.

## References

- de Prado, M.L. (2018). Advances in Financial Machine Learning. Wiley.
- Chan, E. (2013). Algorithmic Trading: Winning Strategies and Their Rationale. Wiley.
