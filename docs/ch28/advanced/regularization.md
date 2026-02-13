# Regularization Techniques

## Overview

Neural ODEs can learn unnecessarily complex dynamics that require many ODE solver steps, increasing computational cost without improving accuracy. Regularization techniques encourage simpler, smoother dynamics.

## Kinetic Energy Regularization

Penalize the magnitude of the velocity field:

$$\mathcal{L}_{\text{KE}} = \int_0^T \|f_\theta(z(t), t)\|^2 dt$$

This encourages straight-line trajectories in latent space, reducing NFE.

## Jacobian Regularization

Penalize the Frobenius norm of the Jacobian:

$$\mathcal{L}_{\text{Jac}} = \int_0^T \left\|\frac{\partial f_\theta}{\partial z}\right\|_F^2 dt$$

This promotes contractive dynamics, improving generalization and reducing stiffness.

## STEER (Stochastic Regularization)

Add noise to the integration endpoint, encouraging robustness to numerical errors and smoother dynamics.

## Weight Decay on Dynamics Network

Standard L2 regularization on the parameters of $f_\theta$ limits the complexity of the learned dynamics. This is the simplest regularization and is always recommended.

## Practical Impact

| Regularization | NFE Reduction | Accuracy Impact |
|---------------|--------------|-----------------|
| None (baseline) | — | — |
| Kinetic energy ($\lambda=0.01$) | 30–50% | Minimal |
| Jacobian ($\lambda=0.01$) | 20–40% | Slight improvement |
| Weight decay ($\lambda=0.001$) | 10–20% | Neutral to positive |

Combining kinetic energy regularization with weight decay gives the best efficiency-accuracy tradeoff in practice.
