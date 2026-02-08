# 34.4.4 SAC Implementation

## Introduction

This section provides a complete SAC implementation with automatic temperature tuning, squashed Gaussian policy, twin critics, and all engineering details needed for robust training.

## Implementation Checklist

1. **Squashed Gaussian policy** with reparameterized sampling and log-prob correction
2. **Twin Q-networks** with independent targets
3. **Automatic entropy tuning** via dual gradient descent on $\alpha$
4. **Replay buffer** with uniform sampling
5. **Soft target updates** via Polyak averaging
6. **Proper log-probability computation** with numerical stability

## Key Implementation Details

### Log-Probability with Tanh Squashing

The change-of-variables formula for $a = \tanh(u)$:

$$\log \pi(a|s) = \log \mathcal{N}(u; \mu, \sigma) - \sum_i \log(1 - \tanh^2(u_i))$$

Numerically stable computation:

$$\log(1 - \tanh^2(u)) = 2(\log 2 - u - \text{softplus}(-2u))$$

### Action Rescaling

For environments with action bounds $[a_\text{low}, a_\text{high}]$:

$$a_\text{env} = a_\text{low} + \frac{(a_\tanh + 1)}{2}(a_\text{high} - a_\text{low})$$

### Network Architecture

Standard SAC uses ReLU activations (not Tanh) in hidden layers, with 256 hidden units. The critic takes concatenated (state, action) as input.

## Training Tips

- **No target actor**: SAC uses the current policy for target computation (unlike TD3)
- **Gradient steps per env step**: 1:1 ratio is standard; increase for complex tasks
- **Warmup**: Random actions for 5K-25K steps before training begins
- **Initial temperature**: $\alpha = 0.2$ or learn from $\log \alpha = 0$

## Summary

SAC's implementation requires careful handling of the squashed Gaussian log-probability and automatic temperature tuning. The complete implementation below provides a production-ready SAC agent.
