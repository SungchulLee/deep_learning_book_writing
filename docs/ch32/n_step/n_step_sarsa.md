# 32.7.2 N-Step SARSA

## Overview

**N-step SARSA** extends SARSA to use n-step returns for Q-value updates, providing a control algorithm that interpolates between one-step SARSA and Monte Carlo control.

## N-Step Q-Return

$$G_{t:t+n} = R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^n Q(S_{t+n}, A_{t+n})$$

The update rule:

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[G_{t:t+n} - Q(S_t, A_t)\right]$$

## Algorithm

```
Initialize Q(s,a) for all s, a
Parameters: n, α, γ, ε

For each episode:
    Store S_0, A_0 ~ ε-greedy(Q, S_0); T = ∞
    For t = 0, 1, 2, ...:
        If t < T:
            Take A_t, observe R_{t+1}, S_{t+1}
            If S_{t+1} is terminal: T = t + 1
            Else: A_{t+1} ~ ε-greedy(Q, S_{t+1})
        τ = t - n + 1
        If τ ≥ 0:
            G = Σ_{i=τ+1}^{min(τ+n,T)} γ^{i-τ-1} R_i
            If τ + n < T: G += γ^n Q(S_{τ+n}, A_{τ+n})
            Q(S_τ, A_τ) ← Q(S_τ, A_τ) + α[G - Q(S_τ, A_τ)]
    Until τ = T - 1
```

## N-Step Off-Policy SARSA

For off-policy learning with behavior policy $b$ and target policy $\pi$:

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \rho_{t+1:t+n-1} \left[G_{t:t+n} - Q(S_t, A_t)\right]$$

where $\rho_{t+1:t+n-1} = \prod_{k=t+1}^{\min(t+n-1, T-1)} \frac{\pi(A_k|S_k)}{b(A_k|S_k)}$

Note: the product starts at $t+1$ (not $t$) because $A_t$ has already been taken.

## Choosing n

Optimal $n$ depends on the problem. Common guidelines:
- Start with $n \in \{4, 8, 16\}$ and tune
- Larger $n$ for problems with delayed rewards
- Smaller $n$ for problems with noisy transitions
- The performance curve as a function of $n$ is typically U-shaped (in terms of error)

## Summary

N-step SARSA provides a flexible control algorithm that balances bias and variance through the choice of $n$. It is particularly effective when rewards are delayed or when the optimal trade-off between bootstrapping and direct experience is problem-specific.
