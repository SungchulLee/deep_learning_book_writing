# 35.2.3 Rebalancing Strategies

## Learning Objectives

- Implement RL-based dynamic rebalancing strategies
- Compare time-based, threshold-based, and learned rebalancing triggers
- Model the trade-off between tracking error and transaction costs
- Design rebalancing policies that adapt to market conditions

## Introduction

Rebalancing is the process of adjusting portfolio weights back toward target allocations as market movements cause drift. The fundamental tension is between two costs: **tracking error** (deviation from the optimal portfolio) and **transaction costs** (trading to reduce that deviation). An RL agent can learn to optimally balance these competing objectives.

## Rebalancing Approaches

### 1. Calendar Rebalancing

Fixed schedule (daily, weekly, monthly). Simple but ignores market conditions:

$$\text{Rebalance if } t \mod T_{\text{rebal}} = 0$$

### 2. Threshold Rebalancing

Trigger when drift exceeds a threshold:

$$\text{Rebalance if } \|\tilde{\mathbf{w}}_t - \mathbf{w}^*\|_1 > \delta$$

where $\tilde{\mathbf{w}}_t$ are the drift weights and $\mathbf{w}^*$ is the target.

### 3. Optimal Band Rebalancing (No-Trade Zone)

Maintain a band $[w_i^* - \delta_i, w_i^* + \delta_i]$ around each target weight. Trade only positions that breach their band:

$$a_{t,i} = \begin{cases} w_i^* - \tilde{w}_{t,i} & \text{if } \tilde{w}_{t,i} > w_i^* + \delta_i \\ w_i^* - \tilde{w}_{t,i} & \text{if } \tilde{w}_{t,i} < w_i^* - \delta_i \\ 0 & \text{otherwise} \end{cases}$$

### 4. RL-Based Rebalancing

The agent learns **when** and **how much** to rebalance. The action includes both target weights and a rebalancing intensity:

$$a_t = (\mathbf{w}_t^{\text{target}}, \alpha_t)$$

where $\alpha_t \in [0, 1]$ controls how aggressively to rebalance:

$$\mathbf{w}_t^{\text{new}} = (1 - \alpha_t) \tilde{\mathbf{w}}_t + \alpha_t \mathbf{w}_t^{\text{target}}$$

## Cost-Tracking Error Trade-off

The optimal rebalancing frequency minimizes the total cost:

$$\text{Total Cost} = \underbrace{\text{Tracking Error}(\delta)}_{\text{decreases with frequency}} + \underbrace{\text{Transaction Costs}(f)}_{\text{increases with frequency}}$$

The RL agent implicitly learns this trade-off through the reward function:

$$r_t = R_t^{\text{portfolio}} - c \cdot \text{turnover}_t - \lambda \cdot \|\mathbf{w}_t - \mathbf{w}_t^*\|^2$$

## Partial Rebalancing

Rather than fully rebalancing to the target, partial rebalancing trades a fraction:

$$\mathbf{w}_t^{\text{new}} = \tilde{\mathbf{w}}_t + \alpha \cdot (\mathbf{w}_t^* - \tilde{\mathbf{w}}_t)$$

The agent can learn the optimal $\alpha$ based on current conditions (volatility, spread, momentum).

## Drift Dynamics

Between rebalancing events, portfolio weights drift according to relative asset returns:

$$\tilde{w}_{t+1,i} = \frac{w_{t,i}(1 + r_{t+1,i})}{\sum_j w_{t,j}(1 + r_{t+1,j})}$$

High-volatility assets drift faster, requiring more frequent rebalancing. The RL agent learns to anticipate drift based on current volatility estimates.

## Summary

Rebalancing strategy selection significantly impacts portfolio performance. RL agents can learn adaptive policies that consider market conditions, transaction costs, and portfolio state when deciding when and how much to rebalance. The no-trade zone and partial rebalancing approaches provide practical middle ground between naive calendar rebalancing and continuous optimization.

## References

- Almgren, R. & Chriss, N. (2001). Optimal execution of portfolio transactions. Journal of Risk.
- Sun, S., et al. (2019). Optimal Rebalancing Strategy Using Dynamic Programming. Quantitative Finance.
- DeMiguel, V., et al. (2009). A Generalized Approach to Portfolio Optimization. Review of Financial Studies.
