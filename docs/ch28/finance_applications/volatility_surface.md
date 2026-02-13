# Volatility Surface Modeling

## Overview

Neural ODEs can model the dynamics of implied volatility surfaces — the map from strike price and maturity to implied volatility — as continuous-time processes, respecting the fundamental constraints of no-arbitrage.

## The Volatility Surface

The implied volatility surface $\sigma(K, T)$ is a function of strike $K$ and maturity $T$. Its dynamics are crucial for option pricing and hedging.

## Neural ODE for Surface Dynamics

Model the evolution of the volatility surface as an ODE in a latent space:

$$\frac{dz}{dt} = f_\theta(z(t), t)$$

where $z(t)$ encodes the volatility surface state at time $t$, and a decoder maps $z(t)$ to the full surface $\sigma(K, T; t)$.

## No-Arbitrage Constraints

The volatility surface must satisfy no-arbitrage conditions:

- **Calendar spread**: call prices must be non-decreasing in maturity
- **Butterfly spread**: call prices must be convex in strike
- These translate to constraints on $\sigma(K, T)$

Neural ODEs can enforce these constraints through the architecture (monotonic networks) or soft penalties in the loss function.

## Applications

- **Surface interpolation**: predict volatility for strikes/maturities not directly quoted
- **Surface dynamics**: forecast how the surface will evolve
- **Exotic option pricing**: price path-dependent options using the learned dynamics
- **Hedging**: compute Greeks from the differentiable surface model
