# 32.6.1 TD Prediction

## Overview

**Temporal Difference (TD) prediction** estimates value functions by combining ideas from Monte Carlo (learning from experience) and dynamic programming (bootstrapping from estimates). TD methods update value estimates after every step, without waiting for the episode to end.

## The Key Insight

TD methods use the **TD target** $R_{t+1} + \gamma V(S_{t+1})$ as a proxy for the true return $G_t$:

$$V(S_t) \leftarrow V(S_t) + \alpha \left[R_{t+1} + \gamma V(S_{t+1}) - V(S_t)\right]$$

The term $\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$ is the **TD error** — the difference between the estimated value and the bootstrapped target.

## Comparison: MC vs. TD vs. DP

| Property | MC | TD | DP |
|----------|----|----|-----|
| Model-free | Yes | Yes | No |
| Bootstrapping | No | Yes | Yes |
| Online (per-step) | No | Yes | N/A |
| Complete episodes | Required | Not required | Not required |
| Bias | Unbiased | Biased (bootstrap) | None (exact) |
| Variance | High | Lower | None |

## Why TD Works

TD methods work because:
1. The TD target $R_{t+1} + \gamma V(S_{t+1})$ is a **sample** of the Bellman equation
2. Under appropriate step sizes, the updates converge to the fixed point of the Bellman equation
3. Bootstrapping reduces variance (at the cost of some bias) compared to MC

## TD vs. MC: Bias-Variance Trade-off

- **MC return** $G_t$: Unbiased but high variance (depends on many random variables)
- **TD target** $R_{t+1} + \gamma V(S_{t+1})$: Biased (uses estimated $V$) but lower variance (depends on only one transition)

In practice, the lower variance often makes TD methods learn faster than MC.

## Financial Interpretation

TD prediction enables real-time strategy evaluation: after each trading period, update the estimated value of the current market state using the observed reward and the estimated value of the next state. No need to wait until the end of the evaluation period (as MC would require).
