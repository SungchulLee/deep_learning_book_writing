# 33.3.1 N-Step Returns

## From 1-Step to N-Step

Standard DQN uses 1-step TD targets:

$$y^{(1)} = r_t + \gamma \max_{a'} Q(s_{t+1}, a')$$

**N-step returns** extend this by using $n$ actual rewards before bootstrapping:

$$y^{(n)} = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n \max_{a'} Q(s_{t+n}, a')$$

## The Bias-Variance Trade-off

| Method | Bias | Variance | Description |
|--------|------|----------|-------------|
| 1-step TD | High | Low | Bootstraps immediately; biased by Q-estimate errors |
| N-step TD | Medium | Medium | Uses more real rewards; less bootstrap bias |
| Monte Carlo | None | High | Full return; no bootstrapping; high variance |

N-step returns interpolate between TD (n=1) and Monte Carlo (n=∞):
- **Small n**: Low variance, high bias (relies on Q-estimates)
- **Large n**: High variance, low bias (uses more real rewards)
- **Optimal n**: Typically 3–5 for most environments

## Implementation

N-step returns require buffering $n$ consecutive transitions before computing the target:

```
n-step buffer: [(s_t, a_t, r_t), (s_{t+1}, a_{t+1}, r_{t+1}), ..., (s_{t+n-1}, ...)]

R^(n) = r_t + γ r_{t+1} + γ² r_{t+2} + ... + γ^{n-1} r_{t+n-1}
Target: y = R^(n) + γ^n max_a' Q(s_{t+n}, a')
```

### Handling Episode Boundaries

When an episode terminates before $n$ steps:
- Use the available rewards and set the bootstrap term to zero
- Flush the n-step buffer at episode end

### Storage

Store $(s_t, a_t, R^{(n)}_t, s_{t+n}, \text{done})$ in the replay buffer, where $R^{(n)}_t$ is the pre-computed n-step return.

## Hyperparameter: Choosing $n$

| Environment | Recommended $n$ |
|-------------|----------------|
| Atari | 3 (Rainbow default) |
| CartPole / simple | 3–5 |
| Continuous control | 1–3 |
| Financial trading | 5–10 (longer credit assignment) |

Larger $n$ works better when:
- Rewards are sparse (need to propagate signal further)
- Q-function estimates are inaccurate (less reliance on bootstrapping)
- Episodes are long (more intermediate rewards to use)

## Off-Policy Correction

N-step returns create an off-policy issue: the $n$ transitions in the buffer were collected by an older policy, but we're using them to update the current policy. For small $n$ (3–5), this mismatch is often acceptable. For larger $n$, importance sampling correction is needed (see Retrace and V-trace).

## Combining with Other Improvements

N-step returns are fully compatible with:
- **Double DQN**: Use Double DQN for the bootstrap term at step $t+n$
- **Distributional RL**: Apply n-step to the distributional Bellman operator
- **Prioritized Replay**: Priorities based on n-step TD errors
- **Rainbow**: N-step is one of the six Rainbow components
