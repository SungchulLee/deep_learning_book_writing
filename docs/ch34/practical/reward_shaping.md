# 34.6.1 Reward Shaping

## Introduction

Reward shaping modifies the reward signal to accelerate learning without changing the optimal policy. Poorly designed rewards can lead to reward hacking, while well-shaped rewards dramatically reduce training time.

## Potential-Based Reward Shaping

The only form guaranteed to preserve the optimal policy (Ng et al., 1999):

$$r'(s, a, s') = r(s, a, s') + \gamma \Phi(s') - \Phi(s)$$

where $\Phi(s)$ is a potential function. This adds a "progress signal" without changing what behavior is optimal.

## Common Reward Design Patterns

### Dense vs Sparse Rewards
- **Sparse**: Reward only at goal (e.g., +1 for winning). Hard to learn from.
- **Dense**: Continuous feedback (e.g., distance to goal). Faster learning but may introduce bias.

### Reward Scaling
Normalizing rewards to a consistent scale (e.g., dividing by running std) stabilizes training across environments with different reward magnitudes.

### Reward Clipping
Clipping rewards to $[-1, 1]$ (as in DQN) prevents outlier rewards from destabilizing training but loses information about reward magnitude.

## Reward Shaping Pitfalls

1. **Reward hacking**: Agent exploits shaped reward in unintended ways
2. **Local optima**: Intermediate rewards can create attractors that prevent reaching the true goal
3. **Misalignment**: Shaped reward doesn't capture the true objective

## Finance-Specific Reward Design

- **Sharpe ratio**: Risk-adjusted return as reward
- **Drawdown penalties**: Penalize maximum portfolio drawdown
- **Transaction costs**: Include realistic costs in reward
- **Benchmark tracking**: Reward relative to benchmark performance

## Summary

Reward shaping is a powerful tool for accelerating RL training. Potential-based shaping is theoretically safe, while practical reward engineering requires careful testing to avoid unintended behaviors.
