# 34.6.2 Action Spaces

## Introduction

Action space design critically impacts learning efficiency and the quality of learned behaviors. The choice between discrete and continuous actions, action bounds, and multi-dimensional action structures requires careful consideration.

## Discrete Action Spaces

Suitable when actions are naturally categorical (buy/sell/hold, select asset). Policy outputs a categorical distribution via softmax. Simple but cannot represent fine-grained continuous control.

**Discretization of continuous spaces**: Convert continuous actions to discrete bins. Works for low-dimensional actions but scales exponentially with dimensions.

## Continuous Action Spaces

### Bounded Actions
Most physical and financial actions have natural bounds. Common parameterizations:
- **Tanh squashing**: $a = \tanh(\mu_\theta(s)) \in [-1, 1]$, rescale to actual bounds
- **Beta distribution**: Naturally bounded on $[0, 1]$
- **Clipping**: Sample from Gaussian, clip to bounds (introduces bias)

### Unbounded Actions
For naturally unbounded quantities (e.g., order sizes), use Gaussian policy without squashing. Ensure the environment handles extreme actions gracefully.

## Multi-Dimensional Actions

For portfolios with $N$ assets, the action is $a \in \mathbb{R}^N$ (weights). Constraints:
- **Simplex constraint**: Weights sum to 1 → use softmax output
- **Long-short constraint**: Weights sum to 0 → use centered softmax
- **Leverage constraint**: $\|a\|_1 \leq L$ → normalize if exceeded

## Action Transformations

### Portfolio Weight Actions
```
Raw output → Softmax → Portfolio weights (sum to 1)
Raw output → Tanh → Rescale → Position sizes
```

### Hierarchical Actions
Decompose complex actions: first choose asset class, then allocation within class.

## Summary

Match the action space to the problem structure. Use continuous actions for fine-grained control, discrete for categorical decisions, and proper constraints for portfolio problems.
