# 32.7.3 TD(λ)

## Overview

**TD(λ)** combines all n-step returns into a single update using an exponentially weighted average. This avoids choosing a specific $n$ and smoothly interpolates between TD(0) and MC.

## The λ-Return

The **λ-return** is a weighted average of all n-step returns:

$$G_t^\lambda = (1 - \lambda) \sum_{n=1}^{T-t-1} \lambda^{n-1} G_{t:t+n} + \lambda^{T-t-1} G_t$$

where $\lambda \in [0, 1]$ controls the weighting:
- $\lambda = 0$: Only $G_{t:t+1}$ contributes → TD(0)
- $\lambda = 1$: Only $G_t$ contributes → Monte Carlo
- Intermediate: Geometrically decaying mixture

## Weight Distribution

The weights $(1-\lambda)\lambda^{n-1}$ form a geometric distribution that sums to 1:

| n-step | Weight |
|--------|--------|
| 1-step | $(1-\lambda)$ |
| 2-step | $(1-\lambda)\lambda$ |
| 3-step | $(1-\lambda)\lambda^2$ |
| ... | ... |
| Terminal | $\lambda^{T-t-1}$ (remainder) |

## Forward View vs. Backward View

### Forward View (Theoretical)

Look forward from state $S_t$ and compute the λ-return using future rewards:

$$V(S_t) \leftarrow V(S_t) + \alpha [G_t^\lambda - V(S_t)]$$

This requires waiting until the end of the episode (like MC).

### Backward View (Practical)

Use **eligibility traces** to update all previously visited states at each step:

$$\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$$
$$e_t(s) = \gamma \lambda e_{t-1}(s) + \mathbb{1}[S_t = s]$$
$$V(s) \leftarrow V(s) + \alpha \delta_t e_t(s) \quad \text{for all } s$$

The forward and backward views produce identical updates over a complete episode.

## Special Cases

| $\lambda$ | Algorithm | Trace | Behavior |
|-----------|----------|-------|----------|
| 0 | TD(0) | No trace needed | One-step bootstrap |
| 0.5 | TD(0.5) | Moderate traces | Balanced |
| 0.9 | TD(0.9) | Long traces | Near-MC |
| 1 | TD(1) = MC | Full episode trace | Monte Carlo |

## Choosing λ

- **$\lambda \approx 0.9$** is often a good default
- The optimal $\lambda$ is problem-dependent
- Lower $\lambda$: faster learning in MDPs with reliable value estimates
- Higher $\lambda$: better in MDPs with sparse or noisy rewards
- Can be tuned via cross-validation or meta-learning

## Financial Application

TD(λ) provides a natural framework for multi-horizon strategy evaluation in finance:
- Low $\lambda$: Focus on short-term prediction (day trading)
- High $\lambda$: Emphasize long-term cumulative returns (portfolio management)
- The exponential weighting naturally models the decreasing relevance of distant future states
