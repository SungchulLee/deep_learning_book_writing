# 32.3.4 Bellman Optimality Equations

## Optimal Value Functions

The **optimal state value function** and **optimal action value function** represent the best possible performance:

$$V^*(s) = \max_\pi V_\pi(s) \quad \text{for all } s$$

$$Q^*(s,a) = \max_\pi Q_\pi(s,a) \quad \text{for all } s, a$$

## Bellman Optimality Equation for V*

Instead of averaging over actions (as in the Bellman equation for $V_\pi$), the optimality equation takes the **maximum**:

$$V^*(s) = \max_a \left[R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^*(s')\right]$$

The optimal agent always selects the best action — no averaging needed.

## Bellman Optimality Equation for Q*

$$Q^*(s,a) = R(s,a) + \gamma \sum_{s'} P(s'|s,a) \max_{a'} Q^*(s',a')$$

This is the foundation of Q-learning: the optimal Q-value for $(s,a)$ equals the immediate reward plus the discounted maximum Q-value at the next state.

## Relationship Between V* and Q*

$$V^*(s) = \max_a Q^*(s,a)$$

$$Q^*(s,a) = R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^*(s')$$

## The Optimal Policy

Given $Q^*$, the optimal policy is:

$$\pi^*(a|s) = \begin{cases} 1 & \text{if } a = \arg\max_{a'} Q^*(s,a') \\ 0 & \text{otherwise} \end{cases}$$

Given $V^*$ (requires model):

$$\pi^*(s) = \arg\max_a \left[R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^*(s')\right]$$

Key insight: $Q^*$ enables **model-free** policy extraction (no need for $P$ or $R$), while $V^*$ requires model knowledge.

## Bellman Optimality Operator

The **Bellman optimality operator** $\mathcal{T}^*$:

$$(\mathcal{T}^* V)(s) = \max_a \left[R(s,a) + \gamma \sum_{s'} P(s'|s,a) V(s')\right]$$

$V^*$ is its unique fixed point: $\mathcal{T}^* V^* = V^*$.

### Contraction Property

$\mathcal{T}^*$ is also a $\gamma$-contraction in the max norm:

$$\|\mathcal{T}^* V_1 - \mathcal{T}^* V_2\|_\infty \leq \gamma \|V_1 - V_2\|_\infty$$

This guarantees that **value iteration** ($V_{k+1} = \mathcal{T}^* V_k$) converges to $V^*$.

## Comparison: Bellman Equation vs. Bellman Optimality Equation

| Property | Bellman Equation ($V_\pi$) | Bellman Optimality ($V^*$) |
|----------|--------------------------|---------------------------|
| Operator | $\sum_a \pi(a|s)[\cdots]$ | $\max_a[\cdots]$ |
| Linearity | Linear in $V$ | **Nonlinear** (due to max) |
| Solution | Matrix inversion | Iterative methods required |
| Fixed point | $V_\pi$ (given $\pi$) | $V^*$ (optimal) |
| Complexity | $O(\|S\|^3)$ direct | $O(\|S\|^2 \|A\|)$ per iteration |

The nonlinearity of the Bellman optimality equation means it generally cannot be solved in closed form — iterative methods (value iteration, policy iteration) are needed.

## Existence and Uniqueness

For finite MDPs with $\gamma < 1$:

1. **$V^*$ exists and is unique**: Follows from the contraction mapping theorem
2. **At least one deterministic optimal policy exists**: Can always select $\arg\max_a Q^*(s,a)$
3. **All optimal policies share the same value functions**: $V_{\pi^*} = V^*$ and $Q_{\pi^*} = Q^*$

## Limitations of Bellman Optimality

Solving the Bellman optimality equations directly requires:

1. **Known dynamics**: $P(s'|s,a)$ and $R(s,a)$ must be available
2. **Enumerable states**: Must iterate over all states
3. **Sufficient computation**: Each iteration costs $O(|\mathcal{S}|^2 |\mathcal{A}|)$

For large or continuous state spaces, **approximate** methods are needed (function approximation, deep RL).

## Financial Interpretation

In portfolio optimization:

- $V^*(s)$: The maximum achievable expected return from market state $s$ under any trading strategy
- $Q^*(s, \text{buy})$: The best expected outcome if we buy in state $s$ and act optimally thereafter
- $\pi^*(s) = \arg\max_a Q^*(s,a)$: The optimal trading action in each market state
- The gap $V^*(s) - V_\pi(s)$ measures the **suboptimality** of strategy $\pi$ in state $s$

## Summary

The Bellman optimality equations characterize the optimal value functions through a recursive max operation. Unlike the Bellman equations for a fixed policy, the optimality equations are nonlinear and require iterative solution. The optimal policy can be extracted directly from $Q^*$ without a model, making Q-function methods central to model-free RL. These equations provide the theoretical foundation for dynamic programming (Section 32.4) and temporal difference methods (Section 32.6).
