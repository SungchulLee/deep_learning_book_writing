# 35.3.2 Market Making

## Learning Objectives

- Formulate market making as a reinforcement learning problem
- Understand the market maker's inventory risk and adverse selection challenges
- Implement RL-based quoting strategies with inventory management
- Compare Avellaneda-Stoikov baseline with learned policies

## Introduction

Market makers provide liquidity by continuously quoting bid and ask prices. They profit from the bid-ask spread but face two key risks: **inventory risk** (holding unwanted positions) and **adverse selection** (being picked off by informed traders). RL can learn dynamic quoting strategies that adapt spread and inventory management to market conditions.

## MDP Formulation

### State

$$s_t = \left(q_t, p_t^{\text{mid}}, \sigma_t, V_t, \text{OFI}_t, t/T\right)$$

| Variable | Description |
|----------|-------------|
| $q_t$ | Current inventory position |
| $p_t^{\text{mid}}$ | Mid-price |
| $\sigma_t$ | Realized volatility |
| $V_t$ | Trading volume |
| $\text{OFI}_t$ | Order flow imbalance |
| $t/T$ | Time fraction |

### Action

The market maker sets bid and ask quotes:

$$a_t = (\delta_t^{\text{bid}}, \delta_t^{\text{ask}})$$

where $\delta$ is the offset from mid-price:

$$p_t^{\text{bid}} = p_t^{\text{mid}} - \delta_t^{\text{bid}}, \quad p_t^{\text{ask}} = p_t^{\text{mid}} + \delta_t^{\text{ask}}$$

### Reward

$$r_t = \text{PnL}_t - \lambda \cdot q_t^2$$

The quadratic inventory penalty encourages the market maker to keep inventory near zero.

## Avellaneda-Stoikov Model

The classical analytical solution (Avellaneda & Stoikov, 2008):

**Reservation price** (adjusted mid accounting for inventory):

$$r_t = p_t^{\text{mid}} - q_t \cdot \gamma \sigma^2 (T - t)$$

**Optimal spread**:

$$\delta^* = \gamma \sigma^2 (T-t) + \frac{2}{\gamma} \ln\left(1 + \frac{\gamma}{k}\right)$$

where $\gamma$ is risk aversion, $\sigma$ is volatility, and $k$ is the order arrival intensity parameter.

## Inventory Management

The core challenge: maintaining approximately zero inventory while earning the spread. Strategies include:

- **Symmetric quoting**: Equal bid/ask offsets (no inventory skew)
- **Asymmetric quoting**: Skew quotes to attract trades that reduce inventory
- **Hedging**: Aggressively cross the spread to flatten inventory
- **Position limits**: Hard constraints on maximum inventory

## Fill Probability

The probability of a limit order being filled depends on its distance from mid:

$$P(\text{fill}|\delta) = A \cdot e^{-k\delta}$$

where $A$ is the baseline fill rate and $k$ controls the sensitivity to distance.

## Summary

RL-based market making learns to dynamically adjust quotes based on inventory, volatility, and order flow. The learned policies can outperform the Avellaneda-Stoikov model by adapting to non-stationary market conditions and complex order flow patterns. Key challenges include modeling realistic fill probabilities and handling the high-frequency nature of market making.

## References

- Avellaneda, M. & Stoikov, S. (2008). High-frequency trading in a limit order book. Quantitative Finance.
- Guéant, O., Lehalle, C.A., & Fernandez-Tapia, J. (2013). Dealing with the inventory risk. Mathematics and Financial Economics.
- Spooner, T., et al. (2018). Market Making via Reinforcement Learning. AAMAS.
