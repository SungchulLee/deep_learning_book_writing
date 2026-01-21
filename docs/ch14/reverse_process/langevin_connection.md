# Connection to Langevin Dynamics

## Introduction

The connection between diffusion models and **Langevin dynamics** reveals the deep relationship between score-based generative modeling and classical Bayesian sampling methods.

## Langevin Dynamics Review

### Standard Langevin Equation

$$dx = \nabla_x \log p(x) \, dt + \sqrt{2} \, dw$$

This SDE has $p(x)$ as its stationary distribution.

### Discrete Langevin (ULA)

$$x_{t+1} = x_t + \epsilon \nabla_x \log p(x_t) + \sqrt{2\epsilon} z_t$$

## Connection to Diffusion

The reverse SDE can be viewed as **time-inhomogeneous Langevin dynamics**:
- The score changes with time: $\nabla_x \log p_t(x)$
- There's additional drift from the forward process
- The noise level varies with $t$

## Unifying Perspective

| Method | Score Function | Time Dependence |
|--------|---------------|-----------------|
| Standard Langevin | $\nabla_x \log p(x)$ | Stationary |
| Annealed Langevin | $\nabla_x \log p_{\sigma_i}(x)$ | Discrete levels |
| Reverse Diffusion | $\nabla_x \log p_t(x)$ | Continuous time |

## Key Insights

1. **Score is fundamental**: Both use $\nabla_x \log p(x)$
2. **Time-varying target**: Diffusion uses time-dependent distributions
3. **Annealing helps**: Multi-scale approaches work better
4. **SDE framework unifies**: All methods are special cases

## Summary

Diffusion models are **generalized Langevin samplers** where:
- Score matching trains the key component
- Time-dependent scores enable better mixing
- The mathematical framework unifies sampling theory

## Navigation

- **Previous**: [Probability Flow ODE](probability_flow_ode.md)
- **Next**: [DDPM](../ddpm/ddpm.md)
