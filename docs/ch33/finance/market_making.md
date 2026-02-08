# 33.7.2 Market Making

## Problem Formulation

A **market maker** continuously provides buy and sell quotes (bids and asks) to the market, profiting from the bid-ask spread while managing inventory risk. The RL agent learns optimal quoting strategies.

### MDP Components

- **State**: $s_t = (q_t, p_t^{\text{mid}}, \sigma_t, S_t, \text{imbalance}_t, t)$
  - $q_t$: Current inventory position
  - $p_t^{\text{mid}}$: Mid-price
  - $\sigma_t$: Estimated volatility
  - $S_t$: Current spread
  - Order book imbalance, time of day features

- **Action**: Discrete choices for bid/ask offsets from mid-price
  - Example: $a = (a_{\text{bid}}, a_{\text{ask}})$ where each is an offset level (e.g., 1, 2, 3, 5, 10 ticks)

- **Reward**: PnL minus inventory penalty
  $$r_t = \text{PnL}_t - \lambda \cdot q_t^2$$

### Inventory Risk

Market makers face **adverse selection**: informed traders trade against the market maker, causing losses. Large inventory positions expose the market maker to directional risk. The inventory penalty $\lambda q_t^2$ encourages mean-reversion of the position.

## Avellaneda-Stoikov Framework

The classical market-making model provides an analytical benchmark:

$$\delta^* = \frac{1}{\gamma}\ln\left(1 + \frac{\gamma}{\kappa}\right) + \frac{q \cdot \gamma \sigma^2 (T - t)}{2}$$

where $\gamma$ is risk aversion, $\kappa$ is order arrival rate intensity, and $q$ is inventory. The RL agent should match or exceed this on realistic market dynamics.

## DQN for Market Making

The agent selects spread levels for buy and sell quotes. The action space is discretized:

| Action | Bid offset | Ask offset |
|--------|-----------|------------|
| 0 | 1 tick | 1 tick (tight spread) |
| 1 | 1 tick | 2 ticks |
| 2 | 2 ticks | 1 tick |
| ... | ... | ... |

### Key Design Choices

1. **Symmetric vs asymmetric spreads**: Allow different bid/ask offsets for inventory management
2. **Order size**: Fixed lot size or variable (adds action dimensions)
3. **Position limits**: Constrain $|q_t| \leq Q_{\max}$
4. **Reward shaping**: Balance spread capture vs inventory risk

## Practical Considerations

- **Latency**: Real market making requires microsecond decisions; RL computes policies offline
- **Adverse selection modeling**: Fill probability depends on toxicity of order flow
- **Multiple venues**: Market makers often operate across exchanges
- **Regulatory constraints**: Obligations to quote continuously, maximum spread limits
