# 32.4.2 Policy Improvement

## Overview

**Policy improvement** takes a value function $V_\pi$ and produces a better (or equal) policy $\pi'$. Combined with policy evaluation, this forms the policy iteration algorithm.

## The Policy Improvement Theorem

If for all $s \in \mathcal{S}$:

$$Q_\pi(s, \pi'(s)) \geq V_\pi(s)$$

then policy $\pi'$ is at least as good as $\pi$:

$$V_{\pi'}(s) \geq V_\pi(s) \quad \text{for all } s$$

If strict inequality holds for at least one state, then $\pi'$ is strictly better.

## Greedy Policy Improvement

Given $V_\pi$, construct the **greedy policy**:

$$\pi'(s) = \arg\max_a Q_\pi(s, a) = \arg\max_a \left[R(s,a) + \gamma \sum_{s'} P(s'|s,a) V_\pi(s')\right]$$

This is guaranteed to improve the policy (or maintain it if already optimal) because:

$$Q_\pi(s, \pi'(s)) = \max_a Q_\pi(s, a) \geq Q_\pi(s, \pi(s)) = V_\pi(s)$$

## When Improvement Stops

If $\pi' = \pi$ (the greedy policy equals the current policy), then for all $s$:

$$V_\pi(s) = \max_a \left[R(s,a) + \gamma \sum_{s'} P(s'|s,a) V_\pi(s')\right]$$

This is exactly the **Bellman optimality equation**, meaning $V_\pi = V^*$ and $\pi$ is optimal.

## Stochastic Policy Improvement

For stochastic policies, improvement can be done by shifting probability toward better actions:

$$\pi'(a|s) \propto \pi(a|s) \cdot \exp\left(\frac{A_\pi(s,a)}{\tau}\right)$$

where $A_\pi(s,a) = Q_\pi(s,a) - V_\pi(s)$ is the advantage and $\tau$ is a temperature parameter.

## Key Properties

1. **Monotonic improvement**: $V_{\pi'} \geq V_\pi$ element-wise
2. **Finite convergence**: Since there are finitely many deterministic policies, improvement must terminate at $\pi^*$
3. **Model-dependent**: Requires $P(s'|s,a)$ and $R(s,a)$ to compute greedy actions (unless using Q-values)
4. **Model-free with Q**: If we have $Q_\pi$, greedy improvement is $\pi'(s) = \arg\max_a Q_\pi(s,a)$ — no model needed

## Financial Interpretation

Policy improvement in trading is analogous to strategy refinement:

1. **Evaluate** current strategy performance across market states ($V_\pi$)
2. **Identify** states where alternative actions would yield higher expected returns ($Q_\pi(s,a) > V_\pi(s)$)
3. **Update** strategy to take better actions in those states ($\pi'$)

This is essentially what a portfolio manager does during strategy review — evaluating performance, identifying improvements, and updating the strategy.

## Summary

Policy improvement constructs a strictly better policy by acting greedily with respect to the current value function. The policy improvement theorem guarantees monotonic improvement, and termination occurs when the Bellman optimality equation is satisfied.
