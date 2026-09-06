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

## Exercises

**Exercise 1.**
Design a Gymnasium-compatible environment for the financial problem described in this section. Specify the observation space, action space, reward function, and episode termination conditions.

??? success "Solution to Exercise 1"
    Observation space: a vector containing recent returns, current position, portfolio value, and relevant market features (e.g., volatility, volume). Action space: depends on the problem (discrete for buy/hold/sell, continuous for position sizing). Reward: risk-adjusted return per step (e.g., log return minus penalty for risk). Episode terminates after a fixed horizon (e.g., one trading year) or if portfolio value drops below a threshold (margin call). The environment must handle transaction costs, slippage, and market impact realistically. $\square$

---

**Exercise 2.**
Analyze the reward shaping trade-offs for this financial RL problem. Compare at least three candidate reward functions and discuss which properties of the optimal policy each preserves.

??? success "Solution to Exercise 2"
    Candidate rewards: (1) raw PnL -- simple but high variance and delayed; (2) Sharpe-based differential reward $D_t = \frac{\Delta A_t B_{t-1} - \frac{1}{2}\Delta B_t A_{t-1}}{(B_{t-1} - A_{t-1}^2)^{3/2}}$ -- directly optimizes the Sharpe ratio but complex; (3) log return with drawdown penalty $r_t = \log(V_t/V_{t-1}) - \lambda \max(0, DD_t - \tau)$ -- balances return with risk control. The potential-based shaping theorem guarantees that adding $\gamma\Phi(s') - \Phi(s)$ preserves the optimal policy. The Sharpe-based reward changes the optimization objective (may alter optimal policy), while pure PnL preserves it but learns slowly. $\square$

---

**Exercise 3.**
Discuss the non-stationarity challenge specific to this financial application. Propose a concrete strategy for adapting the RL agent to regime changes.

??? success "Solution to Exercise 3"
    Financial markets exhibit regime changes (bull/bear, high/low volatility) that violate the MDP stationarity assumption. A concrete adaptation strategy: (1) include a regime indicator as part of the state (e.g., HMM-estimated regime probabilities); (2) use a meta-learning approach where the agent maintains multiple policies and selects based on detected regime; (3) implement continual learning with an expanding replay buffer weighted toward recent experience (exponential decay weights). The agent should also monitor its own performance and reduce position sizes when out-of-distribution inputs are detected. $\square$

---

**Exercise 4.**
Compare the backtesting results one would expect from this RL approach versus a simple heuristic baseline. What statistical tests should be used to determine if the RL agent genuinely outperforms?

??? success "Solution to Exercise 4"
    Baselines: buy-and-hold, equal-weight, or momentum strategy. The RL agent should be evaluated on walk-forward out-of-sample periods (never on training data). Statistical tests: (1) paired t-test on daily returns for mean difference; (2) bootstrap confidence interval on the Sharpe ratio difference; (3) multiple hypothesis testing correction (e.g., Bonferroni or Holm) if comparing multiple strategies. A common pitfall is p-hacking through hyperparameter tuning on the test set. The evaluation must use a hold-out period that was never used for any model selection. Report both statistical significance and economic significance (transaction costs, capacity). $\square$
