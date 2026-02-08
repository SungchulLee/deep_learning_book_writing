# 33.3.2 Retrace(λ)

## The Off-Policy Problem in Multi-Step Learning

N-step returns assume on-policy data: the $n$ transitions were collected by the current policy. With experience replay, transitions come from older policies, creating a distribution mismatch. **Retrace(λ)** (Munos et al., 2016) provides safe off-policy correction for multi-step targets.

## Algorithm

Retrace(λ) computes the target as:

$$Q^{\text{ret}}(s_t, a_t) = Q(s_t, a_t) + \sum_{k=t}^{t+n-1} \gamma^{k-t} \left(\prod_{i=t+1}^{k} c_i\right) \delta_k$$

where:
- $\delta_k = r_k + \gamma Q(s_{k+1}, a^*_{k+1}) - Q(s_k, a_k)$ is the TD error at step $k$ (with $a^* = \arg\max Q$)
- $c_i = \lambda \min\left(1, \frac{\pi(a_i | s_i)}{\mu(a_i | s_i)}\right)$ are the trace coefficients
- $\pi$ is the target (current) policy
- $\mu$ is the behavior (data collection) policy

## Key Properties

1. **Safety**: The trace coefficients $c_i$ are bounded by $\lambda$, ensuring the algorithm doesn't diverge regardless of how different $\pi$ and $\mu$ are
2. **Efficiency**: Unlike full importance sampling, Retrace doesn't require multiplying many IS ratios (which can have exponential variance)
3. **Convergence**: Retrace(λ) provably converges to $Q^\pi$ in the tabular setting
4. **No policy ratio estimation needed for deterministic policies**: When the target policy is greedy, $\pi(a|s) = 1$ for the greedy action and 0 otherwise, simplifying $c_i$

## Comparison with Other Methods

| Method | Trace coefficient | Properties |
|--------|------------------|------------|
| N-step | $c_i = 1$ (no correction) | Biased off-policy |
| Full IS | $c_i = \frac{\pi(a_i|s_i)}{\mu(a_i|s_i)}$ | Unbiased but high variance |
| Tree-backup | $c_i = \pi(a_i|s_i)$ | Safe but ignores behavior |
| Retrace(λ) | $c_i = \lambda\min(1, \frac{\pi}{\mu})$ | Safe + efficient |

## Practical Implementation

For DQN with ε-greedy behavior policy:
- $\pi(a|s) = 1$ if $a = \arg\max Q$ else $0$ (greedy target policy)
- $\mu(a|s) = \epsilon/|\mathcal{A}|$ if random, else $1 - \epsilon + \epsilon/|\mathcal{A}|$ (ε-greedy behavior)
- $c_i = \lambda$ if $a_i$ is greedy, else $0$

This means Retrace effectively truncates traces when the behavior policy took a non-greedy action, providing automatic variance reduction.

## Hyperparameters

| Parameter | Typical | Notes |
|-----------|---------|-------|
| $\lambda$ | 0.95–1.0 | Higher = more multi-step, lower = more TD |
| Trace length | 5–20 | Number of steps to look ahead |
