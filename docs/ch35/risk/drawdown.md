# 35.4.2 Drawdown Control

## Learning Objectives

- Implement drawdown-aware RL policies for portfolio management
- Design reward functions and constraints that limit maximum drawdown
- Understand the relationship between drawdown and risk of ruin
- Build drawdown circuit breakers and position scaling mechanisms

## Introduction

Maximum drawdown (MDD)—the largest peak-to-trough decline in portfolio value—is often the most scrutinized risk metric in practice. A strategy with excellent risk-adjusted returns but a 50% drawdown will lose investor capital as clients withdraw during the drawdown. RL agents must be explicitly trained to control drawdowns.

## Drawdown Metrics

$$\text{Drawdown}_t = \frac{V_{\text{peak}} - V_t}{V_{\text{peak}}}$$

$$\text{MDD} = \max_t \text{Drawdown}_t$$

**Maximum Drawdown Duration**: Time from peak to recovery.

**Underwater curve**: Time series of current drawdown at each point.

## Drawdown Control Methods

### 1. Reward Engineering

Penalize drawdown in the reward function:

$$r_t = R_t - \lambda_{\text{dd}} \cdot \text{Drawdown}_t$$

Or use a drawdown threshold:

$$r_t = R_t - \lambda_{\text{dd}} \cdot \max(0, \text{Drawdown}_t - d_{\text{threshold}})^2$$

### 2. Position Scaling

Reduce position sizes proportionally as drawdown increases:

$$\text{scale}_t = \max\left(0, 1 - \frac{\text{Drawdown}_t}{d_{\max}}\right)$$

$$\mathbf{w}_t^{\text{scaled}} = \text{scale}_t \cdot \mathbf{w}_t^{\text{target}}$$

### 3. Circuit Breaker

Hard stop when drawdown exceeds a limit:

$$\text{if Drawdown}_t > d_{\text{limit}}: \quad \mathbf{w}_t = \mathbf{0} \text{ (go to cash)}$$

### 4. Constrained Policy Optimization

Add drawdown as a constraint in the RL objective:

$$\max_\theta \mathbb{E}[\sum r_t] \quad \text{s.t.} \quad \mathbb{P}(\text{MDD} > d_{\max}) \leq \epsilon$$

This can be solved via Lagrangian relaxation (CPO, RCPO).

## State Augmentation

Include drawdown information in the state:

$$s_t^{\text{aug}} = (s_t, \text{Drawdown}_t, \text{DD\_duration}_t, V_t / V_{\text{peak}})$$

This gives the agent direct visibility into its drawdown status.

## Summary

Drawdown control is critical for practical RL trading systems. A combination of reward engineering, position scaling, and circuit breakers provides layered protection. The agent should have direct visibility into its drawdown status through state augmentation.

## References

- Chekhlov, A., Uryasev, S., & Zabarankin, M. (2005). Drawdown Measure in Portfolio Optimization. International Journal of Theoretical and Applied Finance.
- Grossman, S. & Zhou, Z. (1993). Optimal Investment Strategies for Controlling Drawdowns. Mathematical Finance.
