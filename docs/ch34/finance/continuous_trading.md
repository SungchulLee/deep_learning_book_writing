# 34.7.2 Continuous Trading with Policy-Based RL
## Introduction

Continuous trading applies policy-based RL to real-time execution decisions: position sizing, order timing, and order type selection. Unlike portfolio optimization which operates at daily frequency, continuous trading operates at intraday or tick-level timescales.

## MDP Formulation

### State Space
- **Market microstructure**: Bid-ask spread, order book depth, recent trades
- **Technical indicators**: Price momentum, volume profile, volatility
- **Position state**: Current position, unrealized P&L, time in position
- **Execution state**: Remaining quantity to trade, time until deadline

### Action Space
Continuous actions for position management:

- **Position size**: $a \in [-1, 1]$ (short to long, as fraction of max position)
- **Order type**: Market/limit (discrete), limit price offset (continuous)
- **Execution speed**: Aggressive/passive trade scheduling

### Reward

$$r_t = \text{realized P\&L}_t + \Delta\text{unrealized P\&L}_t - \text{costs}_t$$

## Trading-Specific Challenges

1. **Non-stationarity**: Market dynamics change across regimes
2. **Partial observability**: Order flow and other participants' intentions are hidden
3. **High-frequency noise**: Tick data is extremely noisy
4. **Execution costs**: Slippage and market impact are state-dependent
5. **Risk constraints**: Hard limits on position size and drawdown

## Policy Architecture for Trading

### Feature Processing
Time-series features processed via:

- LSTM/GRU for sequential dependencies
- Temporal convolutions for multi-scale patterns
- Attention mechanisms for relevant historical events

### Action Output
For continuous position management:
```
Features → LSTM → Hidden → Tanh → Position target ∈ [-1, 1]
```

## Training with Market Simulators

Realistic training requires:

- Order book simulation with realistic queue dynamics
- Market impact models (temporary and permanent)
- Latency modeling (decision-to-execution delay)
- Fee structures (maker/taker, exchange fees)

## Summary

Continuous trading with policy-based RL enables adaptive execution strategies that respond to real-time market conditions. The main challenges are realistic simulation, handling non-stationarity, and managing execution costs.

## Exercises

**Exercise 1.**
Derive the policy gradient for the method described in this section. Clearly state which terms require estimation and which can be computed exactly.

??? success "Solution to Exercise 1"
    The policy gradient takes the form $\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[\sum_t \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot \hat{A}_t]$ where $\hat{A}_t$ is the advantage estimate. The log-probability gradient $\nabla_\theta \log \pi_\theta$ can be computed exactly via automatic differentiation. The advantage $\hat{A}_t$ must be estimated from sampled trajectories, introducing variance. The expectation is approximated by averaging over a batch of trajectories. Variance reduction via baselines preserves unbiasedness while reducing the estimation noise. $\square$

---

**Exercise 2.**
Compare the sample efficiency of this method with a value-based approach (e.g., DQN) on a continuous control task. Explain the theoretical reasons for any observed differences.

??? success "Solution to Exercise 2"
    Policy-based methods are generally less sample-efficient than value-based methods because they use on-policy data (each trajectory is used once). DQN reuses data via experience replay, achieving better sample efficiency. However, policy methods handle continuous actions naturally (no argmax over action space needed), converge to stochastic policies when optimal, and provide monotonic improvement guarantees under trust regions. Off-policy actor-critic methods (DDPG, SAC) bridge this gap by combining policy optimization with experience replay. $\square$

---

**Exercise 3.**
Implement this method for a simple continuous control task (e.g., Pendulum-v1). Report hyperparameter sensitivity with respect to the learning rate and the key method-specific parameter.

??? success "Solution to Exercise 3"
    For Pendulum-v1 with a Gaussian policy, typical performance: learning rate $3 \times 10^{-4}$ achieves convergence in $\sim$500 episodes; $10^{-3}$ causes oscillation; $10^{-5}$ converges too slowly. The method-specific parameter (e.g., clipping range for PPO, KL constraint for TRPO) controls the trade-off between update aggressiveness and stability. Too aggressive leads to performance collapse; too conservative wastes samples. The optimal operating point balances these, typically found via grid search over a small range. $\square$

---

**Exercise 4.**
Discuss how this method could be applied to portfolio optimization where the action space is a simplex (portfolio weights summing to 1) and the reward is risk-adjusted return.

??? success "Solution to Exercise 4"
    The action space is the $(n-1)$-dimensional simplex $\Delta^{n-1} = \{w \in \mathbb{R}^n : w_i \geq 0, \sum_i w_i = 1\}$. The policy can use a Dirichlet distribution or softmax-transformed Gaussian. The reward is the Sharpe ratio or differential Sharpe ratio of the resulting portfolio. Challenges include: high-dimensional action space (many assets), transaction costs penalizing frequent rebalancing, and non-stationarity of market returns. The method from this section addresses these through its specific mechanism for stable policy updates. $\square$
