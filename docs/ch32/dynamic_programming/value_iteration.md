# 32.4.4 Value Iteration

## Overview

**Value iteration** combines policy evaluation and improvement into a single update by applying the Bellman optimality operator. It directly computes $V^*$ without explicitly maintaining a policy.

## Algorithm

$$V_{k+1}(s) = \max_a \left[R(s,a) + \gamma \sum_{s'} P(s'|s,a) V_k(s')\right] \quad \text{for all } s$$

This is equivalent to one sweep of policy evaluation followed by policy improvement — the "truncated" version of policy iteration with $k=1$ evaluation sweep.

### Pseudocode

```
Initialize V(s) = 0 for all s ∈ S

Repeat until convergence (Δ < θ):
    Δ = 0
    For each s ∈ S:
        v = V(s)
        V(s) = max_a [R(s,a) + γ Σ_s' P(s'|s,a) V(s')]
        Δ = max(Δ, |v - V(s)|)

# Extract policy
For each s ∈ S:
    π(s) = argmax_a [R(s,a) + γ Σ_s' P(s'|s,a) V(s')]

Return V ≈ V*, π ≈ π*
```

## Convergence

Value iteration converges because the Bellman optimality operator is a $\gamma$-contraction:

$$\|V_{k+1} - V^*\|_\infty \leq \gamma \|V_k - V^*\|_\infty$$

After $k$ iterations: $\|V_k - V^*\|_\infty \leq \gamma^k \|V_0 - V^*\|_\infty$

To achieve $\epsilon$-accuracy: $k \geq \frac{\log(\|V_0 - V^*\|_\infty / \epsilon)}{\log(1/\gamma)}$

## Computational Cost

- **Per iteration**: $O(|\mathcal{S}|^2 |\mathcal{A}|)$ — for each state, evaluate all actions and their transitions
- **Iterations to $\epsilon$-convergence**: $O\left(\frac{1}{1-\gamma} \log \frac{R_{\max}}{(1-\gamma)\epsilon}\right)$

## Relationship to Q-Value Iteration

Value iteration can also be expressed in terms of Q-values:

$$Q_{k+1}(s, a) = R(s,a) + \gamma \sum_{s'} P(s'|s,a) \max_{a'} Q_k(s', a')$$

$$V_{k+1}(s) = \max_a Q_{k+1}(s, a)$$

Q-value iteration is the basis for **Q-learning** (the sample-based, model-free version).

## Practical Considerations

### Stopping Criteria

1. **Value convergence**: Stop when $\max_s |V_{k+1}(s) - V_k(s)| < \theta$
2. **Policy convergence**: Stop when the greedy policy doesn't change
3. **Bounded error**: If $\|V_k - V_{k-1}\|_\infty < \epsilon(1-\gamma)/(2\gamma)$, then the greedy policy w.r.t. $V_k$ is $\epsilon$-optimal

### Asynchronous Value Iteration

Instead of sweeping all states, update states in any order (even one at a time). This is useful when:
- Some states are more important than others
- The state space is very large
- Real-time updates are needed (like in a trading system)

**Prioritized sweeping**: Update states with the largest Bellman error first, focusing computation where it matters most.

## Value Iteration in Finance

For a discretized portfolio management MDP:

1. Initialize expected returns $V(s) = 0$ for all market states
2. For each market state, find the portfolio allocation that maximizes: immediate expected return + discounted expected future returns
3. Iterate until the value function converges
4. Extract the optimal trading strategy

This is essentially **backward induction** applied to multi-period portfolio optimization.

## Summary

Value iteration is a simple, efficient algorithm that computes $V^*$ by repeatedly applying the Bellman optimality operator. Each iteration performs a one-step lookahead over all actions, making it straightforward to implement. Combined with the contraction mapping guarantee, value iteration provides a reliable path to the optimal value function and policy.
