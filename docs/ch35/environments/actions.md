# 35.1.3 Action Spaces

## Learning Objectives

- Design appropriate action spaces for financial RL agents
- Compare discrete vs. continuous action formulations
- Implement portfolio weight actions with constraints
- Handle action masking and feasibility constraints

## Introduction

The action space defines what decisions the RL agent can make at each time step. In finance, actions typically represent trading decisions: how much of each asset to buy or sell. The choice between discrete and continuous action spaces has significant implications for algorithm selection, training stability, and the expressiveness of the learned policy.

## Action Space Formulations

### 1. Discrete Actions

Map trading decisions to a finite set of choices:

**Simple Buy/Sell/Hold (per asset)**:
$$a_t \in \{\text{buy}, \text{hold}, \text{sell}\}^N$$

For $N$ assets, the combinatorial action space has $3^N$ elements, which grows exponentially.

**Discretized Allocation**:
$$a_t \in \{0, 0.1, 0.2, \ldots, 1.0\}^N \quad \text{subject to} \quad \sum_i a_{t,i} = 1$$

**Advantages**: Works with DQN-family algorithms; clear exploration via ε-greedy.
**Disadvantages**: Exponential scaling with assets; cannot express fine-grained allocations.

### 2. Continuous Actions

Represent target portfolio weights directly:

$$a_t \in [0, 1]^N \quad \text{(long-only)}, \quad a_t \in [-1, 1]^N \quad \text{(long-short)}$$

The raw network output is transformed to satisfy constraints:

**Softmax normalization** (long-only, fully invested):
$$w_i = \frac{e^{a_i}}{\sum_j e^{a_j}}$$

**Simplex projection** (long-only with cash):
$$w = \text{proj}_{\Delta}(a), \quad \Delta = \{w : w_i \geq 0, \sum_i w_i \leq 1\}$$

**Tanh with rescaling** (long-short):
$$w_i = \frac{\tanh(a_i)}{\sum_j |\tanh(a_j)|} \cdot L_{\max}$$

where $L_{\max}$ is maximum gross leverage.

**Advantages**: Fine-grained control; works with PPO, SAC, TD3.
**Disadvantages**: Exploration is harder; constraint satisfaction requires careful design.

### 3. Hybrid Actions

Combine discrete and continuous components:

- **Discrete**: Which assets to trade (attention/selection)
- **Continuous**: How much to allocate to selected assets

This is useful when the universe is large but only a few positions change at each step.

## Constraint Handling

### Portfolio Constraints

Financial portfolios typically have constraints that must be enforced:

| Constraint | Mathematical Form | Enforcement |
|-----------|-------------------|-------------|
| Fully invested | $\sum_i w_i = 1$ | Softmax normalization |
| Long-only | $w_i \geq 0$ | Clamp + renormalize |
| Max position | $w_i \leq w_{\max}$ | Clamp + redistribute |
| Max leverage | $\sum_i |w_i| \leq L$ | Scale if exceeded |
| Sector limits | $\sum_{i \in S_k} w_i \leq c_k$ | Iterative projection |
| Min trade size | $|\Delta w_i| \geq \delta$ or $0$ | Threshold + snap to zero |

### Action Transformation Pipeline

```python
def transform_action(self, raw_action):
    """Transform raw network output to valid portfolio weights."""
    
    # Step 1: Apply activation
    if self.long_only:
        weights = torch.softmax(raw_action, dim=-1)
    else:
        weights = torch.tanh(raw_action)
    
    # Step 2: Enforce position limits
    weights = torch.clamp(weights, -self.max_position, self.max_position)
    
    # Step 3: Enforce leverage constraint
    gross_leverage = weights.abs().sum()
    if gross_leverage > self.max_leverage:
        weights = weights * (self.max_leverage / gross_leverage)
    
    # Step 4: Compute trades (difference from current)
    trades = weights - self.current_weights
    
    # Step 5: Enforce minimum trade size
    small_trades = trades.abs() < self.min_trade
    trades[small_trades] = 0.0
    weights = self.current_weights + trades
    
    return weights
```

## Action Representations Compared

### Target Weights vs. Trade Deltas

**Target weights**: $a_t = w_t^{\text{target}}$
- Agent outputs desired portfolio composition
- Environment computes required trades
- Simpler for the agent; stateless action interpretation

**Trade deltas**: $a_t = \Delta w_t = w_t^{\text{target}} - w_{t-1}$
- Agent outputs changes to current portfolio
- More natural for transaction cost awareness
- Requires knowing current position (state-dependent interpretation)

### Order-Based Actions

For execution-level RL:

$$a_t = (\text{side}, \text{quantity}, \text{price}, \text{order\_type})$$

- **Side**: Buy or sell
- **Quantity**: Number of shares/contracts
- **Price**: Limit price or market order flag
- **Order type**: Market, limit, stop-loss, etc.

## Implementation

```python
# Continuous action space for N assets (long-only with cash)
action_space = spaces.Box(
    low=0.0,
    high=1.0,
    shape=(num_assets,),
    dtype=np.float32
)

# Discrete action space (3 actions per asset, factored)
action_space = spaces.MultiDiscrete([3] * num_assets)

# Hybrid: select K assets, then allocate
action_space = spaces.Dict({
    'selection': spaces.MultiBinary(num_assets),
    'allocation': spaces.Box(0, 1, (num_assets,))
})
```

## Action Masking

Some actions may be infeasible at certain states:

- Cannot sell an asset not currently held (if short selling is prohibited)
- Cannot buy if no cash is available
- Cannot trade a halted stock

```python
def get_action_mask(self):
    mask = np.ones(self.action_space.n, dtype=bool)
    
    # Cannot buy without cash
    if self.portfolio.cash <= 0:
        mask[self.BUY_ACTIONS] = False
    
    # Cannot sell without position
    for i, pos in enumerate(self.portfolio.positions):
        if pos <= 0:
            mask[self.SELL_ACTIONS[i]] = False
    
    return mask
```

## Summary

Action space design involves trade-offs between expressiveness, scalability, and training complexity. Continuous actions with softmax or simplex projection are most common for portfolio allocation, while discrete actions suit execution-level decisions. Constraint handling through action transformation pipelines ensures feasibility without modifying the RL algorithm.

## References

- Lillicrap, T. P., et al. (2016). Continuous control with deep reinforcement learning (DDPG)
- Ye, Y., et al. (2020). Reinforcement-Learning based Portfolio Management with Augmented Asset Movement Prediction States
