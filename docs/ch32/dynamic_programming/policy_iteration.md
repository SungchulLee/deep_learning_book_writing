# 32.4.3 Policy Iteration

## Overview

**Policy iteration** alternates between policy evaluation and policy improvement to find the optimal policy. It is one of the two classical dynamic programming algorithms for MDPs.

## Algorithm

$$\pi_0 \xrightarrow{\text{evaluate}} V_{\pi_0} \xrightarrow{\text{improve}} \pi_1 \xrightarrow{\text{evaluate}} V_{\pi_1} \xrightarrow{\text{improve}} \pi_2 \to \cdots \to \pi^*$$

### Pseudocode

```
Initialize π(s) arbitrarily for all s ∈ S

Repeat:
    # Policy Evaluation
    Compute V_π (solve V = r_π + γ P_π V)

    # Policy Improvement
    policy_stable = True
    For each s ∈ S:
        old_action = π(s)
        π(s) = argmax_a [R(s,a) + γ Σ_s' P(s'|s,a) V_π(s')]
        If old_action ≠ π(s):
            policy_stable = False

    If policy_stable:
        Return π, V_π  (optimal)
```

## Convergence

1. Each policy evaluation produces the exact $V_\pi$
2. Each policy improvement produces a strictly better policy (or the same if optimal)
3. There are at most $|\mathcal{A}|^{|\mathcal{S}|}$ deterministic policies
4. Therefore, policy iteration terminates in a finite number of steps

In practice, policy iteration typically converges in very few iterations (often 5-10 for moderate MDPs), because each step makes a big improvement.

## Computational Cost

| Component | Cost |
|-----------|------|
| Policy Evaluation (exact) | $O(\|\mathcal{S}\|^3)$ per iteration |
| Policy Evaluation (iterative) | $O(\|\mathcal{S}\|^2 \|\mathcal{A}\| \cdot k)$ for $k$ sweeps |
| Policy Improvement | $O(\|\mathcal{S}\|^2 \|\mathcal{A}\|)$ per iteration |
| Total iterations | Typically $O(\|\mathcal{S}\| \|\mathcal{A}\|)$ in the worst case |

## Variants

### Modified Policy Iteration

Don't run policy evaluation to full convergence — use only $k$ sweeps before improving:

$$V^{(k)} \approx V_\pi \quad \text{(approximate evaluation)}$$
$$\pi' \leftarrow \text{greedy}(V^{(k)})$$

This interpolates between policy iteration ($k = \infty$) and value iteration ($k = 1$).

### Generalized Policy Iteration (GPI)

The concept of alternating evaluation and improvement, not necessarily to completion, is called **Generalized Policy Iteration**. Most RL algorithms can be viewed as instances of GPI:

- **Policy evaluation**: Any process that makes $V$ more accurate for the current $\pi$
- **Policy improvement**: Any process that makes $\pi$ greedier with respect to the current $V$

GPI drives both $V$ and $\pi$ toward their optimal values, even with partial updates.

## Policy Iteration vs. Value Iteration

| Feature | Policy Iteration | Value Iteration |
|---------|-----------------|-----------------|
| Evaluation | Full (exact $V_\pi$) | None (1 sweep = 1 Bellman optimality backup) |
| Per-iteration cost | Higher (full eval) | Lower (single sweep) |
| Number of iterations | Fewer (big steps) | More (small steps) |
| Overall efficiency | Often faster for small MDPs | Often faster for large MDPs |
| Convergence | Finite steps (exact) | Asymptotic ($\epsilon$-optimal) |

## Summary

Policy iteration is a powerful algorithm that finds the exact optimal policy by alternating between evaluating the current policy and improving it. Its strength is rapid convergence in few iterations, though each iteration can be expensive due to full policy evaluation. The GPI framework generalizes this alternation to encompass most RL algorithms.
