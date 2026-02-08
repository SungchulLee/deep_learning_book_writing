# 33.7.1 Order Execution

## Problem Formulation

**Optimal order execution** is the problem of liquidating a large position $Q$ shares over a time horizon $T$ while minimizing market impact costs. This is naturally formulated as an MDP.

### MDP Components

- **State**: $s_t = (q_t, t, p_t, v_t, \sigma_t, \text{features}_t)$
  - $q_t$: Remaining inventory to execute
  - $t$: Time remaining
  - $p_t$: Current price
  - $v_t$: Recent volume
  - $\sigma_t$: Recent volatility
  - Additional market microstructure features (spread, order book imbalance)

- **Action**: $a_t \in \{0, 1, 2, ..., K\}$ — number of shares to execute in this period (discretized)

- **Reward**: $r_t = -\text{execution\_cost}_t = -(p_t^{\text{exec}} - p_0) \cdot n_t$ where $n_t$ is the number of shares traded and $p_t^{\text{exec}}$ includes market impact

- **Transition**: Market dynamics (prices, volumes) evolve stochastically

### Market Impact Model

Execution costs arise from:
1. **Temporary impact**: Price moves against you proportional to trade size, $\eta \cdot n_t / v_t$
2. **Permanent impact**: Each trade permanently shifts the price, $\gamma \cdot n_t$
3. **Spread costs**: Bid-ask spread paid on each execution

## DQN for Order Execution

The agent learns $Q_\theta(s_t, a_t)$ where $a_t$ represents the fraction of remaining inventory to execute:

$$a_t \in \left\{0, \frac{1}{K}, \frac{2}{K}, ..., 1\right\}$$

### Reward Design

Several reward formulations:
- **Implementation shortfall**: $r_t = -(p_t^{\text{exec}} - p_{\text{arrival}}) \cdot n_t$
- **VWAP benchmark**: $r_t = (\text{VWAP}_t - p_t^{\text{exec}}) \cdot n_t$
- **Risk-adjusted**: $r_t = -\text{cost}_t - \lambda \cdot \text{risk}_t$

### Constraints
- Inventory must be fully liquidated by $T$: $\sum_t n_t = Q$
- Non-negative trades: $n_t \geq 0$ (no buying during sell execution)
- Maximum participation rate: $n_t \leq \alpha \cdot v_t$

## Baseline: TWAP and VWAP

- **TWAP** (Time-Weighted Average Price): Execute $Q/T$ shares per period — simplest baseline
- **VWAP**: Execute proportional to expected volume — standard industry benchmark
- **Almgren-Chriss**: Analytical optimal solution under linear impact model

The RL agent should outperform these baselines by adapting to real-time market conditions.

## Practical Considerations

1. **Action masking**: Ensure the agent can't trade more than remaining inventory
2. **Terminal constraint**: Force execution of remaining shares at $t = T$
3. **Normalization**: Normalize state features (price returns, not levels)
4. **Simulation fidelity**: Market impact model must be realistic for transfer to live trading
5. **Transaction costs**: Include commissions, exchange fees
