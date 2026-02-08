# 32.7.1 N-Step TD Prediction

## Overview

**N-step TD** methods generalize TD(0) and Monte Carlo by using $n$ steps of actual rewards before bootstrapping. They provide a spectrum between TD(0) ($n=1$) and MC ($n=\infty$).

## N-Step Return

The **n-step return** from time $t$ is:

$$G_{t:t+n} = R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^n V(S_{t+n})$$

- $n = 1$: $G_{t:t+1} = R_{t+1} + \gamma V(S_{t+1})$ — TD(0)
- $n = \infty$: $G_{t:\infty} = R_{t+1} + \gamma R_{t+2} + \cdots$ — Monte Carlo
- Intermediate $n$: Uses $n$ real rewards + bootstrapped estimate

## N-Step TD Prediction Algorithm

$$V(S_t) \leftarrow V(S_t) + \alpha \left[G_{t:t+n} - V(S_t)\right]$$

### Pseudocode

```
Initialize V(s) for all s
Parameters: n (number of steps), α, γ

For each episode:
    Store S_0; T = ∞
    For t = 0, 1, 2, ...:
        If t < T:
            Take A_t ~ π(S_t), observe R_{t+1}, S_{t+1}
            If S_{t+1} is terminal: T = t + 1
        τ = t - n + 1    (time step being updated)
        If τ ≥ 0:
            G = Σ_{i=τ+1}^{min(τ+n,T)} γ^{i-τ-1} R_i
            If τ + n < T: G += γ^n V(S_{τ+n})
            V(S_τ) ← V(S_τ) + α[G - V(S_τ)]
    Until τ = T - 1
```

## Bias-Variance Trade-off

| n | Bias | Variance | Update Delay |
|---|------|----------|-------------|
| 1 (TD) | High (heavy bootstrap) | Low | 1 step |
| Small | Moderate | Moderate | n steps |
| Large | Low | High | n steps |
| ∞ (MC) | None | Highest | End of episode |

The optimal $n$ is problem-dependent and often found empirically (typically $n \in [3, 10]$ works well).

## Financial Application

N-step TD for trading strategy evaluation allows updating value estimates using $n$ trading periods of actual returns before bootstrapping. This balances the immediate feedback of TD(0) with the longer-horizon accuracy of MC — analogous to evaluating a strategy over a rolling window rather than daily or end-of-quarter.
