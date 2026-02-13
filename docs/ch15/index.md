# Chapter 18: Sampling and Inference

## Overview

When analytical Bayesian inference is intractable and variational approximations are insufficient, **sampling methods** provide a powerful alternative. This chapter develops the theory and practice of Monte Carlo methods for Bayesian computation, from basic Monte Carlo integration through advanced Markov Chain Monte Carlo (MCMC) algorithms, Langevin dynamics, and likelihood-free inference.

The progression follows a natural arc: Markov chain theory provides the mathematical foundation, basic Monte Carlo methods introduce sampling-based integration, MCMC algorithms combine both for posterior sampling, Langevin dynamics connects sampling to gradient-based optimization, approximate Bayesian computation extends inference to models with intractable likelihoods, and a systematic comparison guides method selection in practice.

In quantitative finance, these methods are indispensable. Portfolio models with non-conjugate priors, stochastic volatility models with latent state processes, regime-switching models requiring posterior inference over discrete states, and risk models demanding full predictive distributions rather than point estimates—all require sampling-based inference when closed-form solutions do not exist.

---

## Computational Context: From Grid Approximation to MCMC

Before diving into sampling methods, it is instructive to understand **why** they are needed. For 1–2 parameters, **grid approximation** computes the posterior by evaluating the unnormalized density on a discrete grid and normalizing. This approach is exact in the limit of fine grids but suffers from the **curse of dimensionality**: a grid with $G$ points per dimension requires $G^d$ evaluations in $d$ dimensions.

| Dimensions | Grid Points ($G = 100$) | Feasibility |
|------------|------------------------|-------------|
| 1 | $10^2$ | Trivial |
| 2 | $10^4$ | Easy |
| 5 | $10^{10}$ | Impractical |
| 10 | $10^{20}$ | Impossible |
| 100 | $10^{200}$ | Absurd |

This exponential scaling motivates the transition to **Monte Carlo methods**, which circumvent the curse of dimensionality by focusing computational effort on regions of high posterior probability. Rather than exhaustively evaluating every point in parameter space, sampling methods generate correlated draws that concentrate in the typical set of the target distribution.

---

## Chapter Structure

### 18.1 Markov Chains

The theoretical foundation for MCMC—understanding how sequences of dependent random variables can converge to a target distribution:

- **[Markov Chain Fundamentals](markov_chains/fundamentals.md)** — Transition kernels, Chapman-Kolmogorov equations, and classification of states (recurrent, transient, absorbing)
- **[Stationary Distribution](markov_chains/stationary.md)** — Existence and uniqueness conditions, detailed balance, and reversibility
- **[Ergodicity](markov_chains/ergodicity.md)** — Convergence theorems, mixing times, ergodic averages, and the law of large numbers for Markov chains
- **[Hidden Markov Models](markov_chains/hmm.md)** — Forward-backward algorithm, Viterbi decoding, and Baum-Welch learning for latent discrete-state models

### 18.2 Monte Carlo Methods

Sampling-based integration and fundamental sampling algorithms:

- **[Monte Carlo Integration](monte_carlo/integration.md)** — Law of large numbers, CLT for Monte Carlo estimators, variance reduction techniques, and convergence rates
- **[Rejection Sampling](monte_carlo/rejection.md)** — Accept-reject algorithm, efficiency analysis, envelope construction, and limitations in high dimensions
- **Importance Sampling** — A comprehensive treatment of reweighting-based estimation:
    - **[IS Fundamentals](monte_carlo/importance_sampling/fundamentals.md)** — Importance weights, the importance sampling identity, and bias-variance tradeoffs
    - **[Self-Normalized IS](monte_carlo/importance_sampling/self_normalized.md)** — Ratio estimators for unnormalized targets, consistency properties, and finite-sample bias
    - **[Proposal Design](monte_carlo/importance_sampling/proposal_design.md)** — Optimal proposal distributions, adaptive proposals, and mixture proposals for multimodal targets
    - **[Effective Sample Size](monte_carlo/importance_sampling/ess.md)** — ESS computation, weight degeneracy diagnostics, and the relationship between proposal quality and effective samples
- **[Effective Sample Size](monte_carlo/ess.md)** — Unified ESS framework across Monte Carlo and MCMC, autocorrelation-based ESS, and practical diagnostics for sampling quality

### 18.3 MCMC

Markov Chain Monte Carlo for posterior sampling—the workhorse of modern Bayesian computation:

- **[Metropolis-Hastings](mcmc/metropolis_hastings.md)** — General MCMC framework, acceptance probability derivation, proposal design, and random-walk versus independence samplers
- **Simulated Annealing** — Temperature-based optimization connecting MCMC to combinatorial and continuous optimization:
    - **[SA Fundamentals](mcmc/simulated_annealing/fundamentals.md)** — The annealing metaphor, Boltzmann distributions, and the SA algorithm
    - **[SA as Non-Stationary MCMC](mcmc/simulated_annealing/sa_as_mcmc.md)** — Viewing SA through the MCMC lens: time-varying target distributions and non-homogeneous Markov chains
    - **[Temperature Schedules](mcmc/simulated_annealing/schedules.md)** — Logarithmic, exponential, and adaptive cooling schedules with convergence implications
    - **[Convergence Theory](mcmc/simulated_annealing/convergence.md)** — Conditions for convergence to global optima and finite-time approximation guarantees
    - **[Temperature as Unifying Concept](mcmc/simulated_annealing/temperature_unifying.md)** — Temperature in statistical mechanics, softmax, energy-based models, and sampling—a cross-cutting perspective
    - **[Deterministic Annealing for EM](mcmc/simulated_annealing/annealed_em.md)** — Combining annealing with EM to escape local optima in mixture model fitting
- **[Gibbs Sampling](mcmc/gibbs_sampling.md)** — Sampling from full conditionals, systematic and random scan, blocked Gibbs, and conjugate model applications
- **Hamiltonian Monte Carlo** — Leveraging gradient information for efficient exploration of continuous parameter spaces:
    - **[HMC Overview](mcmc/hmc/overview.md)** — Motivation, intuition, and the role of Hamiltonian dynamics in sampling
    - **[Hamiltonian Dynamics](mcmc/hmc/hamiltonian_dynamics.md)** — Hamilton's equations, energy conservation, and symplectic structure
    - **[Phase Space](mcmc/hmc/phase_space.md)** — Position-momentum augmentation, the extended target distribution, and marginalizing auxiliary variables
    - **[Leapfrog Integrator](mcmc/hmc/leapfrog_integrator.md)** — Symplectic integration, volume preservation, time-reversibility, and discretization error analysis
    - **[The HMC Algorithm](mcmc/hmc/algorithm.md)** — Complete algorithm specification, momentum refreshment, acceptance probability, and implementation details
    - **[Mass Matrix](mcmc/hmc/mass_matrix.md)** — Preconditioning via mass matrix tuning, diagonal and dense adaptation, and the connection to covariance estimation
    - **[Geometric Interpretation](mcmc/hmc/geometric_interpretation.md)** — Level sets, trajectories on the typical set, and why HMC avoids random-walk behavior
- **[NUTS](mcmc/nuts.md)** — No-U-Turn Sampler: adaptive trajectory length via recursive doubling, eliminating the need to hand-tune leapfrog steps
- **[Diagnostics](mcmc/diagnostics.md)** — $\hat{R}$ convergence diagnostic, trace plots, ESS, autocorrelation analysis, divergent transitions, and practical diagnostic workflows

### 18.4 Langevin Dynamics

Gradient-based sampling connecting stochastic differential equations, optimization, and generative modeling:

- **[Langevin Fundamentals](langevin/fundamentals.md)** — Langevin diffusion as a continuous-time stochastic process, the Fokker-Planck equation, and convergence to the target distribution
- **[Unadjusted Langevin](langevin/ula.md)** — ULA algorithm, discretization bias analysis, step size selection, and convergence rate bounds
- **[MALA](langevin/mala.md)** — Metropolis-adjusted Langevin algorithm: correcting discretization error with an accept-reject step for exact targeting
- **[Score Matching and Diffusion Models](langevin/score_and_diffusion.md)** — Score functions, denoising score matching, and the connection from Langevin sampling to score-based generative models

### 18.5 Approximate Bayesian Computation

Likelihood-free inference for models where the likelihood function is intractable or computationally prohibitive:

- **[Likelihood-Free Inference](abc/likelihood_free.md)** — When and why standard methods fail: simulator-based models, implicit likelihoods, and the ABC paradigm
- **[ABC Rejection Sampling](abc/rejection_sampling.md)** — The basic ABC algorithm, summary statistics, distance metrics, and tolerance selection
- **[ABC-MCMC](abc/abc_mcmc.md)** — Markov chain Monte Carlo within the ABC framework for improved efficiency over pure rejection
- **[ABC-SMC](abc/abc_smc.md)** — Sequential Monte Carlo ABC with adaptive tolerance schedules and population-based sampling

### 18.6 MCMC Methods Comparison

Systematic comparison and practical guidance for method selection:

- **[Methods Overview](mcmc_comparison/overview.md)** — Taxonomy of sampling methods and their core assumptions
- **[Theoretical Comparison](mcmc_comparison/theoretical.md)** — Convergence rates, mixing properties, and theoretical efficiency bounds across methods
- **[Scaling with Dimension](mcmc_comparison/scaling.md)** — How Random Walk MH, HMC, MALA, and Gibbs scale as dimensionality increases
- **[Practical Method Selection](mcmc_comparison/method_selection.md)** — Decision framework for choosing the right sampler based on model structure, dimensionality, gradient availability, and computational budget

---

## The Sampling Landscape

| Method | Exact? | Scalability | Tuning Required | Best For |
|--------|--------|-------------|-----------------|----------|
| Grid approximation | Yes (in limit) | $O(G^d)$ | Grid resolution | 1–2 parameters |
| Rejection sampling | Yes | Low-$d$ only | Envelope function | Simple distributions |
| Importance sampling | Weighted | Moderate | Proposal choice | Evidence estimation, reweighting |
| Metropolis-Hastings | Asymptotic | Moderate | Proposal variance | General posteriors |
| Simulated annealing | Optimization | Moderate | Cooling schedule | Global optimization |
| Gibbs sampling | Asymptotic | Moderate | None (given conditionals) | Conjugate / conditional models |
| HMC | Asymptotic | High-$d$ | Step size, trajectory length | Continuous posteriors with gradients |
| NUTS | Asymptotic | High-$d$ | Step size only (auto-tuned) | Default choice for continuous models |
| ULA / MALA | Asymptotic | High-$d$ | Step size | Large-scale / streaming Bayesian |
| ABC | Approximate | Model-dependent | Tolerance, summary statistics | Simulator-based / intractable likelihood |

---

## Finance Applications

| Application | Method | Why |
|-------------|--------|-----|
| Stochastic volatility models | HMC / NUTS | High-dimensional latent volatility paths require gradient-based sampling |
| Regime-switching models | Gibbs sampling | Discrete latent states with conjugate conditional structure |
| Portfolio allocation under uncertainty | Importance sampling | Reweight posterior samples for different utility functions |
| Bayesian risk models | MCMC + diagnostics | Full posterior over VaR/CVaR requires careful convergence assessment |
| Agent-based market simulators | ABC | Intractable likelihood from complex simulation dynamics |
| Option pricing with jump-diffusion | MALA | Gradient-informed sampling over jump parameters and intensities |
| Model calibration | Simulated annealing | Global optimization over non-convex calibration surfaces |

---

## Connections to Other Chapters

| Topic | Chapter | Connection |
|-------|---------|------------|
| Bayesian Foundations | Ch 16 | Posteriors that MCMC samples from; priors and likelihood specification |
| Conjugate Models | Ch 17 | Gibbs sampling exploits conjugate full conditionals |
| Approximate Inference | Ch 19 | VI as a fast alternative to MCMC; ELBO vs. sampling-based evidence estimation |
| Stochastic Processes | Ch 4 | Markov chains, diffusion processes, and Brownian motion foundations |
| Optimization | Ch 5 | Simulated annealing for global optimization; gradient computation for HMC |
| Score-Based Models | Ch 19 / Ch 30+ | Langevin dynamics as the sampling backbone of diffusion generative models |

---

## Prerequisites

- Bayesian foundations: posteriors, evidence, prior specification (Ch 16)
- Probability distributions and expectations (Ch 3)
- Stochastic processes: Markov property, Brownian motion (Ch 4)
- Gradient computation and optimization (Ch 5) for HMC, Langevin, and SA sections

## Learning Objectives

After completing this chapter, you will be able to:

1. **Formalize** Markov chain convergence through stationarity, detailed balance, and ergodic theorems
2. **Implement** basic Monte Carlo methods including rejection sampling, importance sampling, and ESS diagnostics
3. **Design** importance sampling proposals and diagnose weight degeneracy
4. **Build** Metropolis-Hastings and Gibbs samplers for Bayesian posterior inference
5. **Apply** simulated annealing to global optimization problems and understand its MCMC interpretation
6. **Implement** Hamiltonian Monte Carlo with leapfrog integration and mass matrix adaptation
7. **Use** NUTS for automatic trajectory length tuning in production Bayesian workflows
8. **Diagnose** MCMC convergence using $\hat{R}$, ESS, trace plots, and divergent transition analysis
9. **Connect** Langevin dynamics to both MCMC sampling and score-based generative models
10. **Apply** ABC methods when the likelihood is intractable or available only through simulation
11. **Select** the appropriate sampling method based on model structure, dimensionality, and computational constraints

## References

1. Robert, C. P. & Casella, G. (2004). *Monte Carlo Statistical Methods* (2nd ed.). Springer.
2. Brooks, S., Gelman, A., Jones, G. L., & Meng, X.-L. (Eds.). (2011). *Handbook of Markov Chain Monte Carlo*. CRC Press.
3. Neal, R. M. (2011). MCMC Using Hamiltonian Dynamics. In *Handbook of Markov Chain Monte Carlo*, Chapter 5.
4. Hoffman, M. D. & Gelman, A. (2014). The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo. *Journal of Machine Learning Research*, 15, 1593–1623.
5. Roberts, G. O. & Rosenthal, J. S. (2004). General State Space Markov Chains and MCMC Algorithms. *Probability Surveys*, 1, 20–71.
6. Betancourt, M. (2018). A Conceptual Introduction to Hamiltonian Monte Carlo. *arXiv preprint arXiv:1701.02434*.
7. Sisson, S. A., Fan, Y., & Beaumont, M. A. (Eds.). (2018). *Handbook of Approximate Bayesian Computation*. CRC Press.
8. Marin, J.-M., Pudlo, P., Robert, C. P., & Ryder, R. J. (2012). Approximate Bayesian Computational Methods. *Statistics and Computing*, 22(6), 1167–1180.
9. Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983). Optimization by Simulated Annealing. *Science*, 220(4598), 671–680.
10. Roberts, G. O. & Tweedie, R. L. (1996). Exponential Convergence of Langevin Distributions and Their Discrete Approximations. *Bernoulli*, 2(4), 341–363.
11. Song, Y. & Ermon, S. (2019). Generative Modeling by Estimating Gradients of the Data Distribution. *Advances in Neural Information Processing Systems (NeurIPS)*.
12. Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P.-C. (2021). Rank-Normalization, Folding, and Localization: An Improved $\hat{R}$ for Assessing Convergence of MCMC. *Bayesian Analysis*, 16(2), 667–718.
