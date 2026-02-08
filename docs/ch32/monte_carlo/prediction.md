# 32.5.1 Monte Carlo Prediction

## Overview

**Monte Carlo (MC) prediction** estimates the value function $V_\pi$ by averaging returns from sampled episodes. Unlike dynamic programming, MC methods don't require a model of the environment — they learn directly from experience.

## Key Properties

1. **Model-free**: No knowledge of $P(s'|s,a)$ or $R(s,a)$ needed
2. **Episode-based**: Requires complete episodes (episodic tasks only)
3. **Unbiased**: Sample averages converge to true expected values
4. **No bootstrapping**: Uses actual returns, not estimated future values

## First-Visit MC Prediction

Estimate $V_\pi(s)$ using the return following the **first** visit to state $s$ in each episode.

### Algorithm

```
Initialize: V(s) = 0, Returns(s) = [] for all s

For each episode:
    Generate episode: S_0, A_0, R_1, S_1, A_1, R_2, ..., S_T
    G = 0
    For t = T-1, T-2, ..., 0:
        G = γG + R_{t+1}
        If S_t not in {S_0, ..., S_{t-1}}:     # First visit
            Append G to Returns(S_t)
            V(S_t) = mean(Returns(S_t))
```

**Convergence**: As visits $\to \infty$, $V(s) \to V_\pi(s)$ by the law of large numbers.

## Every-Visit MC Prediction

Uses returns from **every** visit to $s$, not just the first. Simpler to implement and also converges to $V_\pi$, though estimates from the same episode are correlated.

### Comparison

| Property | First-Visit | Every-Visit |
|----------|------------|------------|
| Bias | Unbiased | Biased (slightly, vanishes) |
| Variance | Lower (for few visits) | Higher (correlated samples) |
| Convergence | $O(1/\sqrt{n})$ | $O(1/\sqrt{n})$ |
| Implementation | Track visited states | Simpler |

## Incremental Implementation

Instead of storing all returns, update incrementally:

$$V(s) \leftarrow V(s) + \alpha \left[G_t - V(s)\right]$$

With constant step size $\alpha$, this gives more weight to recent episodes (useful for non-stationary environments). With $\alpha = 1/N(s)$, this is equivalent to the sample mean.

## MC vs. DP

| Feature | Dynamic Programming | Monte Carlo |
|---------|-------------------|-------------|
| Model required | Yes | No |
| Bootstrapping | Yes | No |
| Episode required | No | Yes |
| Bias | None (exact) | None (unbiased) |
| Variance | None | From sampling |
| Computation | $O(\|S\|^2\|A\|)$ per sweep | $O(T)$ per episode |

## Financial Application

MC prediction naturally suits finance:

- **Backtesting**: Generate episodes from historical data, compute average returns → MC estimate of strategy value
- **Option pricing**: MC simulation of price paths to estimate option values
- **Risk estimation**: Sample P&L scenarios to estimate VaR/CVaR

## Summary

MC prediction provides a simple, model-free approach to estimating value functions by averaging returns from complete episodes. Its unbiased nature and simplicity make it a fundamental building block, though the requirement for complete episodes limits its applicability to episodic tasks.
