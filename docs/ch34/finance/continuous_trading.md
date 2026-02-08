# 34.7.2 Continuous Trading with Policy-Based RL

## Introduction

Continuous trading applies policy-based RL to real-time execution decisions: position sizing, order timing, and order type selection. Unlike portfolio optimization which operates at daily frequency, continuous trading operates at intraday or tick-level timescales.

## MDP Formulation

### State Space
- **Market microstructure**: Bid-ask spread, order book depth, recent trades
- **Technical indicators**: Price momentum, volume profile, volatility
- **Position state**: Current position, unrealized P&L, time in position
- **Execution state**: Remaining quantity to trade, time until deadline

### Action Space
Continuous actions for position management:
- **Position size**: $a \in [-1, 1]$ (short to long, as fraction of max position)
- **Order type**: Market/limit (discrete), limit price offset (continuous)
- **Execution speed**: Aggressive/passive trade scheduling

### Reward
$$r_t = \text{realized P\&L}_t + \Delta\text{unrealized P\&L}_t - \text{costs}_t$$

## Trading-Specific Challenges

1. **Non-stationarity**: Market dynamics change across regimes
2. **Partial observability**: Order flow and other participants' intentions are hidden
3. **High-frequency noise**: Tick data is extremely noisy
4. **Execution costs**: Slippage and market impact are state-dependent
5. **Risk constraints**: Hard limits on position size and drawdown

## Policy Architecture for Trading

### Feature Processing
Time-series features processed via:
- LSTM/GRU for sequential dependencies
- Temporal convolutions for multi-scale patterns
- Attention mechanisms for relevant historical events

### Action Output
For continuous position management:
```
Features → LSTM → Hidden → Tanh → Position target ∈ [-1, 1]
```

## Training with Market Simulators

Realistic training requires:
- Order book simulation with realistic queue dynamics
- Market impact models (temporary and permanent)
- Latency modeling (decision-to-execution delay)
- Fee structures (maker/taker, exchange fees)

## Summary

Continuous trading with policy-based RL enables adaptive execution strategies that respond to real-time market conditions. The main challenges are realistic simulation, handling non-stationarity, and managing execution costs.
