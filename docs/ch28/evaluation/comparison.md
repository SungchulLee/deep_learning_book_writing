# Comparison with Discrete Models

## Overview

Neural ODEs offer continuous-depth dynamics, but how do they compare with standard discrete architectures (ResNets, RNNs) in terms of accuracy, efficiency, and applicability?

## ResNet as Discretized Neural ODE

A ResNet block $z_{n+1} = z_n + f_\theta(z_n)$ is an Euler discretization of $dz/dt = f_\theta(z)$ with step size $h=1$. Neural ODEs generalize this by allowing adaptive step sizes and shared parameters.

## Empirical Comparison

| Aspect | ResNet | Neural ODE |
|--------|--------|-----------|
| Parameters | $L \times P_f$ | $P_f$ (shared) |
| Forward cost | $L$ evaluations | NFE evaluations |
| Memory (training) | $O(L)$ activations | $O(1)$ (adjoint) or $O(\sqrt{L})$ (checkpoint) |
| Depth | Fixed | Adaptive |
| Irregular data | Not natural | Natural (Latent ODE) |
| Accuracy (image classification) | Higher | Slightly lower |
| Accuracy (irregular time series) | N/A | Higher |

## When Neural ODEs Win

- **Memory-constrained settings**: the adjoint method uses $O(1)$ memory
- **Irregularly-sampled data**: natural handling without imputation
- **Continuous-time modeling**: when the underlying process is truly continuous (physics, finance)
- **Parameter efficiency**: same dynamics network applied at all "depths"

## When Discrete Models Win

- **Speed**: fixed computational graph, no solver overhead
- **Standard benchmarks**: ResNets still outperform on ImageNet-like tasks
- **Simplicity**: easier to implement, debug, and deploy
- **Batch efficiency**: fixed computation across all samples in a batch
