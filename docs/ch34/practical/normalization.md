# 34.6.3 Observation Normalization

## Introduction

Observation normalization is a critical but often overlooked component of RL pipelines. Neural networks perform best with inputs near zero mean and unit variance. Unnormalized observations with varying scales across dimensions cause gradient imbalances and slow learning.

## Running Mean-Variance Normalization

The standard approach maintains running statistics using Welford's online algorithm:

$$\hat{s}_t = \frac{s_t - \mu_\text{run}}{\sqrt{\sigma^2_\text{run} + \epsilon}}$$

Clipped to $[-c, c]$ (typically $c = 10$) to prevent extreme values.

## What to Normalize

- **Observations**: Always normalize. Different features may have vastly different scales.
- **Rewards**: Normalize by running standard deviation (not mean). Helps with varying reward scales.
- **Advantages**: Normalize per minibatch (standard in PPO). Keeps gradient magnitudes consistent.
- **Value targets**: Some implementations normalize value targets for the critic.

## Implementation Details

- **Initialization**: Use batch statistics from the first few episodes to warm up
- **Update frequency**: Update statistics at every step during training
- **Evaluation**: Use frozen statistics during evaluation (don't update)
- **Per-dimension**: Apply normalization independently to each observation dimension

## Common Pitfalls

1. **Normalizing with batch mean during training**: Creates dependency between transitions in a batch
2. **Not clipping**: Extreme normalized values can cause NaN gradients
3. **Forgetting to normalize at test time**: Policy expects normalized inputs
4. **Normalizing binary/categorical features**: Only normalize continuous features

## Summary

Always normalize observations and consider normalizing rewards. Running statistics with clipping provide robust normalization without assumptions about the data distribution.
