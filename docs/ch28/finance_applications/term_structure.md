# Term Structure Dynamics

## Overview

Neural ODEs can model yield curve dynamics as continuous-time processes, learning the evolution of interest rate term structures from market data.

## Term Structure as Continuous Dynamics

The yield curve $y(\tau, t)$ (yield as a function of maturity $\tau$ at time $t$) evolves according to:

$$\frac{\partial y}{\partial t} = f_\theta(y(\cdot, t), t)$$

This is an infinite-dimensional ODE, discretized by representing the yield curve at a finite set of maturities.

## Latent Factor Approach

Encode the yield curve into a low-dimensional latent state:

$$z(t) = \text{Encoder}(y(\tau_1, t), \ldots, y(\tau_M, t))$$
$$\frac{dz}{dt} = f_\theta(z(t), t)$$
$$\hat{y}(\tau, t) = \text{Decoder}(z(t), \tau)$$

The latent space typically has 3–5 dimensions, corresponding to level, slope, and curvature factors (analogous to Nelson-Siegel).

## Advantages Over Classical Models

| Classical Model | Neural ODE Advantage |
|----------------|---------------------|
| Nelson-Siegel | No fixed functional form |
| Vasicek/CIR | Non-linear dynamics, multi-factor |
| HJM | Learned drift restriction (no-arbitrage learned from data) |
| Affine models | No restrictive affine assumption |

## Training

Train on historical yield curve snapshots:

$$\mathcal{L} = \sum_{t} \sum_{\tau} (y(\tau, t) - \hat{y}(\tau, t))^2$$

The ODE is integrated between observation times, handling irregular spacing naturally.

## Applications

- **Yield curve forecasting**: predict future term structure shapes
- **Scenario generation**: sample yield curve paths for risk management
- **Bond pricing**: price bonds and interest rate derivatives from the learned dynamics
- **Monetary policy analysis**: model the yield curve response to policy changes
