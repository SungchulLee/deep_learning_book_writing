# 33.3.3 V-Trace

## Motivation

**V-trace** (Espeholt et al., 2018) was developed for the IMPALA architecture, which uses distributed actors to collect experience in parallel while a central learner updates the policy. The key challenge: by the time experience reaches the learner, the policy has changed, creating significant off-policy lag.

## Algorithm

V-trace corrects for the policy lag using truncated importance sampling:

$$v_s = V(s) + \sum_{t=s}^{s+n-1} \gamma^{t-s} \left(\prod_{i=s}^{t-1} c_i\right) \delta_t V$$

where:
- $\delta_t V = \rho_t (r_t + \gamma V(s_{t+1}) - V(s_t))$ is the corrected TD error
- $\rho_t = \min\left(\bar{\rho}, \frac{\pi(a_t|s_t)}{\mu(a_t|s_t)}\right)$ is the truncated IS ratio for value updates
- $c_i = \min\left(\bar{c}, \frac{\pi(a_i|s_i)}{\mu(a_i|s_i)}\right)$ is the truncated IS ratio for trace propagation
- $\bar{\rho}$ and $\bar{c}$ are truncation thresholds

## Key Differences from Retrace

| Aspect | Retrace(λ) | V-Trace |
|--------|-----------|---------|
| Truncation | $\lambda \cdot \min(1, \pi/\mu)$ | $\min(\bar{c}, \pi/\mu)$ |
| Value correction | Via Q-values | Via V-values |
| Clipping params | Single $\lambda$ | Two: $\bar{\rho}$ and $\bar{c}$ |
| Primary use | Single-agent replay | Distributed actors |
| Fixed point | $Q^\pi$ | Between $\pi_{\bar{\rho}}$ and $\pi$ |

## Truncation Thresholds

- **$\bar{\rho}$ (rho-bar)**: Controls the fixed point of V-trace. At $\bar{\rho} = \infty$, converges to $V^\pi$. At $\bar{\rho} = 1$, more conservative
- **$\bar{c}$ (c-bar)**: Controls trace propagation speed. Higher values propagate information faster but with more variance
- **Typical values**: $\bar{\rho} = \bar{c} = 1.0$

## Application to Q-Learning

For value-based methods, V-trace can be adapted:

$$Q^{\text{ret}}(s_t, a_t) = r_t + \gamma \left[v_{t+1} + (1 - \bar{c}) \cdot \pi(a_{t+1}|s_{t+1}) \cdot (Q(s_{t+1}, a_{t+1}) - V(s_{t+1}))\right]$$

where $v_{t+1}$ is the V-trace corrected value.

## Distributed Setting

V-trace is designed for architectures where:
1. Multiple **actors** run copies of the policy and collect experience
2. A single **learner** updates the policy using batched experience
3. There's a **lag** of $k$ updates between actors and learner

V-trace's truncated IS ratios bound the variance introduced by this lag, making it practical for large-scale distributed training.

## Practical Considerations

- V-trace is most beneficial when there's significant policy lag (distributed training, large replay buffers)
- For single-agent DQN with modest buffer sizes, simpler methods (n-step, Retrace) may suffice
- The truncation thresholds add robustness at the cost of some bias
