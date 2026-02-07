# Chapter 18: Sampling and Inference

## Overview

When analytical Bayesian inference is intractable and variational approximations are insufficient, **sampling methods** provide a powerful alternative. This chapter develops the theory and practice of Monte Carlo methods for Bayesian computation, from basic Monte Carlo integration through advanced Markov Chain Monte Carlo (MCMC) algorithms and Langevin dynamics.

The progression follows a natural arc: Markov chain theory provides the mathematical foundation, basic Monte Carlo methods introduce sampling-based integration, MCMC algorithms combine both for posterior sampling, and Langevin dynamics connects sampling to gradient-based optimization.

---

## Computational Context: From Grid Approximation to MCMC

Before diving into sampling methods, it is instructive to understand **why** they are needed. For 1-2 parameters, **grid approximation** computes the posterior by evaluating the unnormalized posterior on a discrete grid and normalizing. This approach is exact in the limit of fine grids but suffers from the **curse of dimensionality**: a grid with $G$ points per dimension requires $G^d$ evaluations in $d$ dimensions.

| Dimensions | Grid Points ($G=100$) | Feasibility |
|------------|----------------------|-------------|
| 1 | $10^2$ | Trivial |
| 2 | $10^4$ | Easy |
| 5 | $10^{10}$ | Impractical |
| 10 | $10^{20}$ | Impossible |
| 100 | $10^{200}$ | Absurd |

This exponential scaling motivates the transition to **Monte Carlo methods**, which circumvent the curse of dimensionality by focusing computational effort on regions of high posterior probability.

---

## Chapter Structure

### 18.1 Markov Chains

The theoretical foundation for MCMC:

- **[Markov Chain Fundamentals](markov_chains/fundamentals.md)** — Transition kernels, Chapman-Kolmogorov equations, and classification of states
- **[Stationary Distribution](markov_chains/stationary.md)** — Existence, uniqueness, and detailed balance condition
- **[Ergodicity](markov_chains/ergodicity.md)** — Convergence theorems, mixing times, and ergodic averages
- **[Hidden Markov Models](markov_chains/hmm.md)** — Forward-backward algorithm, Viterbi decoding, and Baum-Welch learning

### 18.2 Monte Carlo Methods

Sampling-based integration and basic sampling algorithms:

- **[Monte Carlo Integration](monte_carlo/integration.md)** — Law of large numbers, CLT for Monte Carlo, variance reduction techniques, and convergence rates
- **[Rejection Sampling](monte_carlo/rejection.md)** — Accept-reject algorithm, efficiency analysis, and limitations in high dimensions
- **[Importance Sampling](monte_carlo/importance_sampling.md)** — Proposal distributions, importance weights, self-normalized estimators, and optimal proposals
- **[Effective Sample Size](monte_carlo/ess.md)** — ESS computation, weight degeneracy, and diagnostics for sampling quality

### 18.3 MCMC

Markov Chain Monte Carlo for posterior sampling:

- **[Metropolis-Hastings](mcmc/metropolis_hastings.md)** — General MCMC framework, acceptance probability, proposal design, and random-walk vs independence samplers
- **[Gibbs Sampling](mcmc/gibbs_sampling.md)** — Sampling from full conditionals, systematic and random scan, blocked Gibbs, and conjugate models
- **[Hamiltonian Monte Carlo](mcmc/hmc.md)** — Hamiltonian dynamics for efficient exploration, leapfrog integration, mass matrix tuning
- **[NUTS](mcmc/nuts.md)** — No-U-Turn Sampler: adaptive HMC that eliminates manual tuning of trajectory length
- **[Diagnostics](mcmc/diagnostics.md)** — $\hat{R}$ convergence diagnostic, trace plots, ESS, autocorrelation analysis, and practical workflow

### 18.4 Langevin Dynamics

Gradient-based sampling connecting optimization and MCMC:

- **[Langevin Fundamentals](langevin/fundamentals.md)** — Langevin diffusion, Fokker-Planck equation, and connections to score-based models
- **[Unadjusted Langevin](langevin/ula.md)** — ULA algorithm, discretization bias, and step size considerations
- **[MALA](langevin/mala.md)** — Metropolis-adjusted Langevin algorithm: correcting discretization error with accept-reject

---

## The Sampling Landscape

| Method | Exact? | Scalability | Tuning | Best For |
|--------|--------|-------------|--------|----------|
| Grid approximation | Yes (in limit) | $O(G^d)$ | Grid resolution | 1-2 parameters |
| Rejection sampling | Yes | Low-$d$ only | Envelope function | Simple distributions |
| Importance sampling | Weighted | Moderate | Proposal choice | Evidence estimation |
| Metropolis-Hastings | Asymptotic | Moderate | Proposal variance | General posteriors |
| Gibbs sampling | Asymptotic | Moderate | None (given conditionals) | Conjugate models |
| HMC/NUTS | Asymptotic | High-$d$ | Mass matrix, step size | Complex posteriors |
| SGLD | Asymptotic | Large data | Step size schedule | Big data Bayesian |

---

## Prerequisites

- Bayesian foundations: posteriors, evidence (Ch16)
- Probability distributions and expectations (Ch3)
- Gradient computation (Ch5) for HMC and Langevin sections

## Key Connections

| Topic | Chapter | Connection |
|-------|---------|------------|
| Bayesian Foundations | Ch16 | Posteriors that MCMC samples from |
| Approximate Inference | Ch19 | VI as alternative to MCMC |
| Stochastic Processes | Ch4 | Markov chains and diffusion processes |
| Langevin & Score Models | Ch18.4 | Foundation for score-based generative models |

---

## References

1. Robert, C. P., & Casella, G. (2004). *Monte Carlo Statistical Methods* (2nd ed.). Springer.
2. Brooks, S., et al. (2011). *Handbook of Markov Chain Monte Carlo*. CRC Press.
3. Neal, R. M. (2011). MCMC using Hamiltonian dynamics. In *Handbook of MCMC*, Chapter 5.
4. Hoffman, M. D., & Gelman, A. (2014). The No-U-Turn Sampler. *JMLR*, 15, 1593-1623.
