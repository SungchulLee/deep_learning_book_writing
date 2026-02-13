# NFNet

## Learning Objectives

By the end of this section, you will be able to:

- Understand the key innovations introduced by NFNet (2021)
- Identify how NFNet influenced subsequent architecture design

## Overview

**Year**: 2021 | **Parameters**: Varies | **Key Innovation**: Normalizer-free networks removing batch normalization

NFNet (Brock et al., 2021) achieves state-of-the-art accuracy **without batch normalization** by using Adaptive Gradient Clipping (AGC) and careful variance-preserving initialization.

## Motivation: Problems with Batch Normalization

While BN stabilizes training, it introduces issues:

1. **Batch size dependence**: Performance degrades with small batches
2. **Train/eval discrepancy**: Running statistics differ from batch statistics
3. **Memory overhead**: Must store batch statistics
4. **Distributed training**: Requires synchronized statistics across devices

## Adaptive Gradient Clipping (AGC)

AGC clips gradients based on the ratio of gradient norm to parameter norm:

$$G_i^{\text{clip}} = \begin{cases} \lambda \frac{\|W_i\|}{\|G_i\|} G_i & \text{if } \frac{\|G_i\|}{\|W_i\|} > \lambda \\ G_i & \text{otherwise} \end{cases}$$

This provides training stability without batch normalization's drawbacks.

NFNets demonstrate that batch normalization, while convenient, is not fundamental—alternative training stabilization techniques can achieve equal or better results.

## References

1. Brock, A., De, S., Smith, S. L., & Simonyan, K. (2021). High-Performance Large-Scale Image Recognition Without Normalization. ICML.
