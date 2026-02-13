# Neural SDEs

## Overview

Neural Stochastic Differential Equations extend Neural ODEs by adding a learnable diffusion (noise) term, enabling the modeling of stochastic processes — essential for financial applications where uncertainty is intrinsic.

## Formulation

$$dz(t) = f_\theta(z(t), t) \, dt + g_\phi(z(t), t) \, dW(t)$$

where $f_\theta$ is the drift network, $g_\phi$ is the diffusion network, and $W(t)$ is a Wiener process.

## Comparison with Neural ODEs

| Aspect | Neural ODE | Neural SDE |
|--------|-----------|-----------|
| Dynamics | Deterministic | Stochastic |
| Output | Single trajectory | Distribution over trajectories |
| Suited for | Deterministic systems | Stochastic systems, uncertainty |
| Training | Adjoint method | SDE adjoint, variational methods |
| Finance relevance | Limited | Natural fit |

## Training Approaches

### SDE Adjoint
Extend the adjoint method to SDEs (Li et al., 2020). More complex than the ODE case due to the stochastic integral.

### Variational Approach
Treat the SDE as a latent variable model and optimize a variational lower bound, similar to latent ODEs but with stochastic dynamics.

### Score Matching
Connect to diffusion models: the SDE framework unifies score-based generative models (Song et al., 2021).

## Financial Applications

Neural SDEs naturally model:

- **Stochastic volatility**: $dS = \mu S \, dt + \sigma(S, t) S \, dW$ with learned $\sigma$
- **Correlated processes**: multi-dimensional SDEs for correlated asset dynamics
- **Option pricing**: risk-neutral dynamics learned from market data
- **Uncertainty quantification**: the diffusion term directly models prediction uncertainty

## Connection to Diffusion Models

Diffusion models (DDPM, score-based models) are SDEs with specific drift and diffusion structures. Neural SDEs generalize this by learning both terms from data, offering a flexible framework for generative modeling of continuous-time processes.
