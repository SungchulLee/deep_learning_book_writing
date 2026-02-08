# 32.2.2 States and Actions

## State Space

The **state space** $\mathcal{S}$ is the set of all possible situations the agent may encounter. The state must capture sufficient information to make the Markov property hold.

### Discrete State Spaces

In discrete MDPs, states are enumerable: $\mathcal{S} = \{s_1, s_2, \ldots, s_n\}$. Examples include grid positions, game board configurations, or inventory levels.

### Continuous State Spaces

Continuous state spaces are subsets of $\mathbb{R}^n$. Examples include physical positions/velocities, portfolio weights, or sensor readings. These require function approximation for practical algorithms.

### State Representation Design

Good state representations should be:

1. **Markov**: Contain enough information to predict the future
2. **Compact**: Avoid unnecessary dimensions that slow learning
3. **Informative**: Distinguish states that require different actions
4. **Normalizable**: Features should be on comparable scales

### Feature Engineering for States

| Domain | Raw Observation | Engineered State Features |
|--------|----------------|--------------------------|
| Trading | Price series | Returns, volatility, RSI, MACD, volume ratios |
| Robotics | Joint angles | Positions, velocities, contact forces |
| Games | Raw pixels | CNN embeddings, object positions |
| Inventory | Stock levels | Days of supply, demand forecast, seasonality |

## Action Space

The **action space** $\mathcal{A}$ (or $\mathcal{A}(s)$ if state-dependent) is the set of choices available to the agent.

### Discrete Actions

$\mathcal{A} = \{a_1, a_2, \ldots, a_m\}$. Common in games, routing, and classification-like decisions.

For trading: $\mathcal{A} = \{\text{strong\_sell}, \text{sell}, \text{hold}, \text{buy}, \text{strong\_buy}\}$

### Continuous Actions

$\mathcal{A} \subseteq \mathbb{R}^m$. Required when decisions are naturally continuous.

For portfolio allocation: $\mathcal{A} = \{\mathbf{w} \in \mathbb{R}^n : w_i \geq 0, \sum_i w_i = 1\}$ (simplex constraint)

### Action Space Design Considerations

| Consideration | Discrete | Continuous |
|--------------|----------|-----------|
| **Algorithm choice** | Q-learning, DQN | Policy gradient, DDPG, SAC |
| **Exploration** | ε-greedy, UCB | Gaussian noise, entropy bonus |
| **Curse of dimensionality** | Exponential in factors | Managed by policy networks |
| **Constraints** | Embedded in action set | Projection or penalty methods |

### Discretization of Continuous Actions

When using discrete-action algorithms for continuous problems:

- **Uniform grid**: Divide each dimension into bins
- **Adaptive**: Finer resolution in high-value regions
- **Action branching**: Factored discrete actions for each dimension

## State-Action Pairs

The joint space $\mathcal{S} \times \mathcal{A}$ is fundamental to many RL algorithms. The **action-value function** $Q(s,a)$ assigns a value to each state-action pair.

For finite MDPs: $|\mathcal{S} \times \mathcal{A}| = |\mathcal{S}| \cdot |\mathcal{A}|$

This product space can become very large, motivating function approximation.

## Financial State-Action Design

### Market State Representation

A comprehensive trading state might include:

**Price features**: Returns at multiple horizons, realized volatility, bid-ask spread

**Technical indicators**: Moving averages, RSI, Bollinger bands, MACD

**Portfolio state**: Current positions, unrealized P&L, margin utilization

**Market microstructure**: Order book imbalance, trade flow, volume profile

**Macro features**: Interest rates, VIX, sector indices, economic indicators

### Trading Action Design

| Approach | Action Space | Pros | Cons |
|----------|-------------|------|------|
| Discrete signals | {sell, hold, buy} | Simple, compatible with Q-learning | Coarse control |
| Position targets | Target weight ∈ [−1, 1] | Precise, natural | Requires continuous methods |
| Order parameters | (price, size, type) | Realistic | High-dimensional |
| Delta positions | Change in position | Incremental, bounded | May oscillate |

## Summary

Proper design of state and action spaces is critical for RL success. States must satisfy the Markov property and be efficiently representable, while actions must capture the full range of meaningful decisions. In financial applications, both spaces require careful feature engineering to balance informativeness with computational tractability.
