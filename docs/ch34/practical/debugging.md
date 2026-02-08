# 34.6.4 Debugging Reinforcement Learning

## Introduction

RL algorithms are notoriously difficult to debug. Silent failures, non-stationarity, and high variance make it hard to distinguish bugs from normal training behavior. This section provides systematic debugging strategies.

## Common Failure Modes

### 1. Policy Collapse
The policy becomes deterministic too early, stopping exploration. Signs: entropy drops to zero, reward plateaus.
**Fix**: Increase entropy coefficient, reduce learning rate, check initialization.

### 2. Value Function Divergence
Critic predicts increasingly extreme values. Signs: value loss explodes, NaN gradients.
**Fix**: Reduce learning rate, clip gradients, check reward scale, use value clipping.

### 3. Reward Hacking
Agent finds unintended shortcuts. Signs: high reward but undesired behavior.
**Fix**: Redesign reward, add constraints, inspect agent behavior visually.

### 4. No Learning
Reward stays flat. Multiple potential causes:
- Bug in gradient computation (check surrogate loss sign)
- Learning rate too high/low
- Network too small/large
- Reward signal too sparse

## Systematic Debugging Checklist

### Phase 1: Verify Components
1. **Random policy baseline**: What reward does a random agent achieve?
2. **Known solution**: Can the algorithm solve a trivially easy version?
3. **Loss computation**: Is the loss decreasing? Is the sign correct?
4. **Gradient flow**: Are gradients non-zero and finite for all parameters?

### Phase 2: Monitor Training
Key metrics to track:
- **Episode reward** (mean, min, max, std)
- **Policy entropy**
- **Value function predictions vs actual returns**
- **KL divergence** between old and new policy (for PPO/TRPO)
- **Clip fraction** (for PPO)
- **Gradient norms** per layer
- **Explained variance** of value function

### Phase 3: Ablation
- Test with 1 environment before scaling to many
- Use known-good hyperparameters from papers
- Start with simple environments (CartPole, Pendulum)
- Compare against reference implementations

## Diagnostic Metrics

| Metric | Healthy Range | Problem Indicator |
|--------|--------------|-------------------|
| Entropy | Slowly decreasing | Drops to 0 quickly |
| KL divergence | 0.01-0.02 | > 0.1 (unstable) |
| Clip fraction | 0.1-0.3 | > 0.5 or 0 |
| Value explained var | > 0.5 | < 0 (worse than mean) |
| Gradient norm | Stable | Increasing/NaN |
| Approx KL | < 0.05 | > 0.1 |

## Summary

Systematic debugging using the checklist approach, combined with comprehensive metric tracking, is essential for successful RL development. Always start simple, verify components individually, and scale up gradually.
