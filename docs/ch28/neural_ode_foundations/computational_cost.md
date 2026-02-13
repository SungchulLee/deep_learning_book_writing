# Computational Cost Analysis

## Overview

Neural ODEs trade architectural flexibility for computational cost. Understanding the cost structure is essential for practical deployment.

## Cost Per Forward Pass

Each forward pass requires solving the ODE from $t=0$ to $t=T$, with cost proportional to the number of function evaluations (NFE):

$$\text{Cost}_{\text{forward}} = \text{NFE} \times \text{Cost}(f_\theta)$$

For adaptive solvers, NFE varies by input — complex inputs require more steps.

## NFE Statistics

| Model / Task | Typical NFE (Forward) | NFE (Backward) |
|-------------|----------------------|----------------|
| Simple toy ODE | 10–30 | 15–40 |
| FFJORD (density estimation) | 50–150 | 70–200 |
| Latent ODE (time series) | 20–60 | 30–80 |

## Comparison with Discrete Models

A Neural ODE layer with NFE=100 is roughly equivalent to a 100-layer ResNet in computational cost, but with shared parameters across all "layers" and continuous dynamics.

| Model | FLOPs | Parameters | Flexibility |
|-------|-------|-----------|-------------|
| ResNet-100 | $100 \times C_f$ | $100 \times P_f$ | Fixed depth |
| Neural ODE (NFE≈100) | $100 \times C_f$ | $1 \times P_f$ | Adaptive depth |

The parameter efficiency is a key advantage: Neural ODEs use the same $f_\theta$ at every "step."

## Reducing Cost

### Regularization of Dynamics
Penalize the complexity of the learned dynamics to reduce NFE:

$$\mathcal{L}_{\text{reg}} = \lambda \int_0^T \|f_\theta(z(t), t)\|^2 dt$$

Smoother dynamics require fewer solver steps.

### Fixed-Step Solvers
Use Euler or RK4 with a fixed number of steps for predictable cost, at the expense of accuracy.

### Distillation
Train a discrete model (ResNet) to mimic the Neural ODE, then deploy the faster discrete model.

## Training Cost

Training is ~2-3x more expensive than inference due to the backward ODE solve (adjoint method) or checkpointed backpropagation. Total training cost = forward NFE + backward NFE (~1.5x forward NFE).
