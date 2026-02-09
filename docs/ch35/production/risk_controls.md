# 35.7.4 Risk Controls

## Learning Objectives

- Implement production risk controls for RL trading systems
- Design pre-trade, real-time, and post-trade risk checks
- Build kill switches and automated position flattening
- Ensure regulatory compliance with risk management requirements

## Introduction

Production risk controls are the safety net that prevents catastrophic losses. They operate independently of the RL agent and enforce hard limits on positions, losses, and trading activity. Even a well-trained agent can produce dangerous actions due to model errors, data issues, or unprecedented market conditions.

## Risk Control Layers

### Pre-Trade Controls
- Position limit checks
- Sector/factor exposure limits
- Order size limits
- Leverage limits
- Fat-finger checks (unusually large orders)

### Real-Time Controls
- Drawdown circuit breaker
- PnL loss limit (daily, weekly)
- Volatility-based position scaling
- Correlation-based exposure reduction

### Post-Trade Controls
- End-of-day reconciliation
- Performance attribution
- Risk report generation
- Compliance checks

## Kill Switch Implementation

A kill switch immediately flattens all positions when triggered:

**Triggers**:
- Daily loss exceeds threshold
- Drawdown exceeds limit
- System malfunction detected
- Manual override by risk manager

**Actions**:
1. Cancel all open orders
2. Send market orders to close all positions
3. Disable new order generation
4. Alert risk management team
5. Require manual re-enable

## Position Sizing Framework

$$\text{Size}_t = \min\left(\text{Model Size}, \frac{\text{Risk Budget}}{\text{Vol}_t}, \text{Liquidity Limit}\right)$$

Each trade must pass through all three constraints.

## Summary

Production risk controls provide defense-in-depth against model failures, data issues, and extreme market events. They operate independently of the RL agent and enforce hard limits that cannot be overridden by the model.

## References

- Kissell, R. (2013). The Science of Algorithmic Trading and Portfolio Management. Academic Press.
- Narang, R. (2013). Inside the Black Box. Wiley.
