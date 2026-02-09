# 35.2.4 Transaction Costs

## Learning Objectives

- Model realistic transaction costs in RL portfolio environments
- Understand the components of trading costs: commissions, spread, slippage, market impact
- Implement proportional, fixed, and non-linear cost models
- Design cost-aware reward functions that lead to practical trading policies

## Introduction

Transaction costs are the single most important difference between simulated and live trading performance. An RL agent trained without realistic cost modeling will learn to trade too frequently, generating policies that are profitable in simulation but unprofitable in practice. Accurate cost modeling is essential for policies that transfer to real markets.

## Components of Transaction Costs

### 1. Explicit Costs

| Component | Typical Range | Description |
|-----------|--------------|-------------|
| Commissions | 0-5 bps | Broker fees per trade |
| Exchange fees | 0.1-1 bps | Exchange transaction fees |
| Taxes | 0-50 bps | Stamp duty, financial transaction tax |
| Clearing fees | 0.01-0.1 bps | Post-trade settlement costs |

### 2. Implicit Costs

| Component | Typical Range | Description |
|-----------|--------------|-------------|
| Bid-ask spread | 1-50 bps | Half-spread cost for immediate execution |
| Market impact | Variable | Price movement caused by the trade itself |
| Slippage | 1-10 bps | Difference between intended and actual execution price |
| Opportunity cost | Variable | Cost of not executing when signal is present |

### 3. Total Cost Model

$$\text{TC}(v) = \underbrace{c_{\text{fixed}}}_{\text{fixed}} + \underbrace{c_{\text{prop}} \cdot |v|}_{\text{proportional}} + \underbrace{c_{\text{impact}} \cdot |v|^{3/2}}_{\text{market impact}}$$

where $v$ is the trade volume (in dollars or shares).

## Cost Models

### Proportional Cost (Simplest)

$$\text{TC} = c \cdot \sum_{i=1}^N |w_i^{\text{new}} - w_i^{\text{old}}| \cdot V_t$$

Linear in turnover, parameterized by a single rate $c$ (typically 5-20 bps).

### Spread-Based Cost

$$\text{TC}_i = \frac{s_i}{2} \cdot |\Delta w_i| \cdot V_t$$

where $s_i$ is the bid-ask spread of asset $i$. The half-spread represents the cost of crossing the spread for immediate execution.

### Market Impact (Square Root Model)

The Almgren-Chriss model for temporary market impact:

$$\Delta p_i = \eta_i \cdot \sigma_i \cdot \text{sign}(v_i) \cdot \sqrt{\frac{|v_i|}{V_i^{\text{ADV}}}}$$

where $\sigma_i$ is daily volatility, $V_i^{\text{ADV}}$ is average daily volume, and $\eta_i$ is the impact coefficient.

### Slippage Model

Execution price differs from the decision price:

$$p_i^{\text{exec}} = p_i^{\text{mid}} + \frac{s_i}{2} \cdot \text{sign}(v_i) + \epsilon_i$$

where $\epsilon_i \sim \mathcal{N}(0, \sigma_{\text{slip}}^2)$ is random slippage.

## Impact on Portfolio Performance

Transaction costs compound over time. For a strategy with annual turnover $\tau$ and average cost $c$:

$$\text{Annual cost drag} = \tau \cdot c$$

For example, a strategy with 500% annual turnover and 10 bps average cost loses 50 bps per year to trading costs alone.

## Cost-Aware RL

### Augmented Reward

$$r_t = R_t^{\text{portfolio}} - \text{TC}_t(\Delta \mathbf{w}_t)$$

### Turnover Penalty

Add explicit turnover regularization:

$$r_t = R_t^{\text{portfolio}} - \text{TC}_t - \lambda_{\text{turn}} \cdot \|\Delta \mathbf{w}_t\|_1$$

### Action Smoothing

Penalize rapid changes in portfolio:

$$r_t = R_t^{\text{portfolio}} - \text{TC}_t - \lambda_{\text{smooth}} \cdot \|\mathbf{w}_t - \mathbf{w}_{t-1}\|_2^2$$

## Summary

Transaction cost modeling is critical for developing RL policies that work in practice. The choice of cost model should match the asset class and trading frequency. Proportional costs suffice for daily strategies with liquid assets, while market impact models are essential for large orders or illiquid markets. Cost-aware rewards naturally teach the agent to balance signal exploitation against trading friction.

## References

- Almgren, R. & Chriss, N. (2001). Optimal Execution of Portfolio Transactions. Journal of Risk.
- Kissell, R. (2013). The Science of Algorithmic Trading and Portfolio Management. Academic Press.
- Frazzini, A., Israel, R., & Moskowitz, T. (2018). Trading Costs. SSRN.
