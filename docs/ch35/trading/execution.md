# 35.3.1 Execution Algorithms

## Learning Objectives

- Formulate optimal trade execution as a reinforcement learning problem
- Implement RL-based execution strategies that minimize market impact
- Understand the TWAP, VWAP, and Almgren-Chriss benchmarks
- Design state representations and rewards for execution agents

## Introduction

Trade execution is the process of converting a trading decision (e.g., "buy 10,000 shares of AAPL") into actual market orders. The challenge is executing the desired quantity while minimizing the total cost—balancing **urgency** (completing the order quickly to capture the alpha signal) against **market impact** (large orders moving the price against the trader).

RL is particularly well-suited to execution because the problem is naturally sequential: at each time step, the agent decides how much of the remaining quantity to execute, adapting to real-time market conditions.

## MDP Formulation for Execution

### State

$$s_t = \left(q_t^{\text{remaining}}, t, T, \text{price}_t, \text{spread}_t, \text{volume}_t, \text{volatility}_t, \text{imbalance}_t\right)$$

where $q_t^{\text{remaining}}$ is the remaining quantity to execute.

### Action

$$a_t = \frac{v_t}{q_t^{\text{remaining}}} \in [0, 1]$$

The fraction of remaining quantity to execute in this time step.

### Reward

$$r_t = -\left(p_t^{\text{exec}} - p_0^{\text{arrival}}\right) \cdot v_t$$

The negative implementation shortfall: the difference between execution price and the arrival price (decision price), weighted by the executed volume.

### Terminal Reward

If any quantity remains at deadline $T$:

$$r_T = -\lambda_{\text{penalty}} \cdot q_T^{\text{remaining}} \cdot p_T$$

## Benchmark Strategies

### TWAP (Time-Weighted Average Price)

Execute equal quantities at each time step:

$$v_t = \frac{Q}{T}$$

Simple but ignores market conditions entirely.

### VWAP (Volume-Weighted Average Price)

Execute proportional to expected volume:

$$v_t = Q \cdot \frac{\hat{V}_t}{\sum_{k=0}^{T} \hat{V}_k}$$

where $\hat{V}_t$ is the predicted volume at time $t$.

### Almgren-Chriss Optimal Trajectory

The analytical solution minimizes a combination of expected cost and risk:

$$q_t^* = Q \cdot \frac{\sinh\left(\kappa(T-t)\right)}{\sinh(\kappa T)}$$

where $\kappa = \sqrt{\frac{\lambda \sigma^2}{\eta}}$, $\lambda$ is risk aversion, $\sigma$ is volatility, and $\eta$ is the temporary impact coefficient.

## RL Advantages Over Benchmarks

| Aspect | Benchmarks | RL Execution |
|--------|-----------|-------------|
| Adaptivity | Static schedule | Adapts to real-time conditions |
| Volume | VWAP uses predicted volume | Reacts to actual volume |
| Volatility | Fixed in A-C model | Adjusts execution speed |
| Spread | Ignored | Can wait for favorable spread |
| Order book | Not considered | Can use depth information |

## Summary

RL-based execution algorithms learn to adapt their trading speed to real-time market conditions, outperforming static benchmarks like TWAP and VWAP. The key design decisions are the granularity of the time steps, the market features included in the state, and whether to use implementation shortfall or VWAP tracking as the reward signal.

## References

- Almgren, R. & Chriss, N. (2001). Optimal Execution of Portfolio Transactions. Journal of Risk.
- Nevmyvaka, Y., Feng, Y., & Kearns, M. (2006). Reinforcement Learning for Optimized Trade Execution. ICML.
- Ning, B., Lin, F.H.T., & Jaimungal, S. (2021). Double Deep Q-Learning for Optimal Execution. Applied Mathematical Finance.
