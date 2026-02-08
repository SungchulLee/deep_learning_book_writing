# 32.7.4 Eligibility Traces

## Overview

**Eligibility traces** are a mechanism that bridges TD and MC methods by maintaining a short-term memory of recently visited states. They provide the backward-view implementation of TD(λ) and enable efficient, online computation of λ-returns.

## Types of Traces

### Accumulating Traces

$$e_t(s) = \gamma \lambda \, e_{t-1}(s) + \mathbb{1}[S_t = s]$$

Each visit adds 1 to the trace, and the trace decays by $\gamma\lambda$ at each step. Multiple visits accumulate.

### Replacing Traces

$$e_t(s) = \begin{cases} 1 & \text{if } S_t = s \\ \gamma \lambda \, e_{t-1}(s) & \text{otherwise} \end{cases}$$

Resets the trace to 1 upon each visit instead of accumulating. Often performs better in practice.

### Dutch Traces

$$e_t(s) = \gamma \lambda \, e_{t-1}(s) + \alpha(1 - \gamma \lambda \, e_{t-1}(s)) \mathbb{1}[S_t = s]$$

Designed to exactly reproduce the offline λ-return algorithm in the linear function approximation setting.

## TD(λ) with Eligibility Traces

```
Initialize V(s), e(s) = 0 for all s
Parameters: α, γ, λ

For each episode:
    S = initial state, e(s) = 0 for all s
    For each step:
        A ~ π(S)
        Take A, observe R, S'
        δ = R + γ V(S') - V(S)          # TD error
        e(S) = e(S) + 1                   # Update trace (accumulating)
        For all s:
            V(s) ← V(s) + α δ e(s)       # Update values
            e(s) ← γ λ e(s)              # Decay traces
        S ← S'
```

## SARSA(λ) — Control with Traces

Extend eligibility traces to state-action pairs:

$$e_t(s, a) = \gamma \lambda \, e_{t-1}(s, a) + \mathbb{1}[S_t = s, A_t = a]$$
$$\delta_t = R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)$$
$$Q(s, a) \leftarrow Q(s, a) + \alpha \delta_t e_t(s, a) \quad \text{for all } s, a$$

## Watkins's Q(λ) — Off-Policy with Traces

Cut traces when the greedy action differs from the taken action:

$$e_t(s, a) = \begin{cases} \gamma \lambda \, e_{t-1}(s, a) + \mathbb{1}[S_t=s, A_t=a] & \text{if } A_t = \arg\max_{a'} Q(S_t, a') \\ \mathbb{1}[S_t=s, A_t=a] & \text{otherwise (reset)} \end{cases}$$

This ensures traces are only propagated along greedy trajectories.

## Intuition: Credit Assignment

Eligibility traces solve the **temporal credit assignment** problem:

- When a reward is received, which past states/actions were responsible?
- The trace $e(s)$ measures each state's "eligibility" for credit
- Recently and frequently visited states get more credit
- The trace decays with rate $\gamma\lambda$, fading out older states

## Computational Considerations

| Approach | Memory | Per-step Cost |
|----------|--------|---------------|
| N-step (forward) | Store n steps | $O(n)$ |
| Eligibility traces | One trace per state | $O(\|\mathcal{S}\|)$ per step |
| Sparse traces | Only active states | $O(\text{active states})$ |

For large state spaces, sparse trace implementations only update states with non-negligible traces.

## The Unified View

Eligibility traces provide a unifying framework:

$$\text{TD}(0) \xleftrightarrow{\lambda \in [0,1]} \text{MC}$$
$$\text{SARSA}(0) \xleftrightarrow{\lambda \in [0,1]} \text{MC Control}$$

The parameter $\lambda$ smoothly interpolates between pure bootstrapping and pure sampling.

## Summary

Eligibility traces are a powerful mechanism for credit assignment that enables efficient implementation of multi-step TD methods. They maintain a decaying memory of visited states and distribute TD errors across all eligible states at each step. The choice of trace type (accumulating, replacing, Dutch) and the λ parameter control the balance between TD and MC behavior.
