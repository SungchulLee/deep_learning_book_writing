# 34.7.1 Portfolio Optimization with Policy-Based RL
## Introduction

Portfolio optimization using policy-based deep RL frames asset allocation as a sequential decision problem. The agent learns to dynamically rebalance a portfolio across multiple assets, maximizing risk-adjusted returns while accounting for transaction costs and market constraints.

## MDP Formulation

### State Space
The observation at time $t$ includes:

- **Price features**: Returns, moving averages, volatility for each asset
- **Portfolio state**: Current asset weights $w_t$
- **Market features**: VIX, interest rates, sector indicators
- **Account features**: Cash balance, unrealized P&L

$$s_t = (\text{price\_features}_t, w_t, \text{market\_features}_t, \text{account}_t)$$

### Action Space
The action represents target portfolio weights:

$$a_t = (w_1^{\text{target}}, \ldots, w_N^{\text{target}})$$

With constraints: $\sum_i w_i = 1$ (fully invested), $w_i \geq 0$ (long-only) or relaxed for long-short.

### Reward Function
Risk-adjusted return minus transaction costs:

$$r_t = \underbrace{w_t^\top R_t}_{\text{portfolio return}} - \underbrace{\lambda_\text{tc} \sum_i |w_{i,t} - w_{i,t-1}| c_i}_{\text{transaction costs}} - \underbrace{\lambda_\text{risk} \cdot \text{risk}(w_t, R_t)}_{\text{risk penalty}}$$

Common risk measures: variance, maximum drawdown, CVaR.

## Policy Architecture

### Portfolio Policy Network
```
Observation → Feature Extraction → Hidden Layers → Softmax → Portfolio Weights
```

The softmax output naturally satisfies the simplex constraint ($\sum w_i = 1, w_i \geq 0$).

### Temperature-Scaled Softmax

$$w_i = \frac{\exp(h_i / \tau)}{\sum_j \exp(h_j / \tau)}$$

Lower temperature → more concentrated portfolios; higher temperature → more diversified.

## Training Considerations

### Transaction Cost Modeling
Realistic costs include:

- Proportional costs (bid-ask spread, commissions)
- Market impact (price movement from large trades)
- Slippage (execution price differs from decision price)

### Turnover Regularization
Penalize excessive trading to encourage stable portfolios:

$$L_\text{turnover} = \lambda \sum_t \|w_t - w_{t-1}\|_1$$

### Multi-Period Optimization
The RL agent naturally optimizes over multiple periods, considering how today's actions affect future opportunities—a key advantage over single-period mean-variance optimization.

## Comparison with Classical Methods

| Method | Multi-period | Transaction costs | Non-linear constraints | Adaptivity |
|--------|-------------|-------------------|----------------------|------------|
| Mean-Variance | No | Difficult | Limited | No |
| Black-Litterman | No | Difficult | Limited | Partial |
| Risk Parity | No | No | Limited | No |
| Policy-Based RL | Yes | Natural | Any | Yes |

## Summary

Policy-based RL enables dynamic portfolio optimization that naturally incorporates transaction costs, adapts to changing market conditions, and handles complex constraints. The key challenges are realistic simulation, avoiding overfitting to historical data, and ensuring robustness to regime changes.

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
