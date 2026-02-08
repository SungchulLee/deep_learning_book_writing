# 32.5.2 Monte Carlo Control

## Overview

**MC control** uses Monte Carlo methods to find the optimal policy. Following the Generalized Policy Iteration (GPI) framework, it alternates between MC policy evaluation and greedy policy improvement.

## Why Q-Values?

MC control uses $Q_\pi(s,a)$ instead of $V_\pi(s)$ because:

- $V$-based improvement requires a model: $\pi'(s) = \arg\max_a [R(s,a) + \gamma \sum_{s'} P(s'|s,a) V(s')]$
- $Q$-based improvement is **model-free**: $\pi'(s) = \arg\max_a Q(s,a)$

## MC Control with Exploring Starts

### Assumption

**Exploring starts**: Every state-action pair has a non-zero probability of being selected as the starting pair. This ensures all $(s,a)$ pairs are visited.

### Algorithm (MC-ES)

```
Initialize: Q(s,a) arbitrary, π(s) = argmax_a Q(s,a), Returns(s,a) = []

For each episode:
    Choose S_0, A_0 randomly (exploring starts)
    Generate episode following π
    G = 0
    For t = T-1, T-2, ..., 0:
        G = γG + R_{t+1}
        If (S_t, A_t) not in earlier steps:
            Append G to Returns(S_t, A_t)
            Q(S_t, A_t) = mean(Returns(S_t, A_t))
            π(S_t) = argmax_a Q(S_t, a)
```

This converges to $Q^*$ and $\pi^*$ as episodes $\to \infty$.

## MC Control without Exploring Starts

Exploring starts is often impractical (we can't always control the start state). The solution: **on-policy** methods using **soft policies**.

### ε-Greedy Policy

$$\pi(a|s) = \begin{cases} 1 - \epsilon + \frac{\epsilon}{|\mathcal{A}|} & \text{if } a = \arg\max_{a'} Q(s, a') \\ \frac{\epsilon}{|\mathcal{A}|} & \text{otherwise} \end{cases}$$

This ensures all actions have minimum probability $\frac{\epsilon}{|\mathcal{A}|} > 0$, guaranteeing exploration.

### On-Policy MC Control

```
Initialize: Q(s,a) arbitrary, π = ε-greedy w.r.t. Q

For each episode:
    Generate episode following π
    G = 0
    For t = T-1, ..., 0:
        G = γG + R_{t+1}
        If (S_t, A_t) not in earlier steps:
            Append G to Returns(S_t, A_t)
            Q(S_t, A_t) = mean(Returns(S_t, A_t))
            π = ε-greedy w.r.t. Q
```

**Limitation**: Converges to the best ε-soft policy, not necessarily $\pi^*$ (unless $\epsilon \to 0$).

## GLIE (Greedy in the Limit with Infinite Exploration)

A sequence of policies is GLIE if:

1. All state-action pairs are explored infinitely often
2. The policy converges to greedy: $\pi_k \to \pi^*$ as $k \to \infty$

**Example**: $\epsilon$-greedy with $\epsilon_k = 1/k$ satisfies GLIE.

MC control with GLIE exploration converges to $Q^*$ and $\pi^*$.

## Financial Application

MC control for trading:

1. Start with a random trading strategy
2. Execute it over a historical episode, collect returns
3. Update Q-estimates for each (market state, action) pair
4. Improve the strategy by acting greedily w.r.t. Q
5. Repeat with decaying exploration

This is essentially a **systematic backtesting and strategy refinement loop**.

## Summary

MC control extends MC prediction to policy optimization by maintaining Q-values and improving the policy greedily. Exploring starts or ε-greedy exploration ensure sufficient coverage of the state-action space. GLIE conditions guarantee convergence to the optimal policy.
