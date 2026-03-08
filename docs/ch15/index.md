<<<<<<< HEAD
# Chapter Overview

This chapter covers **Shortest Paths**.

# Reference

[Introduction to Algorithms (CLRS), Chapters 24-25](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
=======
# Chapter 15: Sampling and Inference

This chapter covers the theory and practice of sampling-based inference methods, from foundational Markov chain theory through Monte Carlo integration, MCMC algorithms, Langevin dynamics, and likelihood-free inference. These methods provide the computational machinery for Bayesian inference when analytical solutions are intractable, enabling posterior estimation in complex models with many parameters.

---

## Markov Chains

- [Fundamentals](markov_chains/fundamentals.md) -- Core definitions, the Markov property, transition matrices, and multi-step dynamics
- [Stationary Distribution](markov_chains/stationary.md) -- The long-run equilibrium distribution and its central role in connecting Markov chain theory to MCMC
- [Ergodicity](markov_chains/ergodicity.md) -- State classification, irreducibility, aperiodicity, recurrence, and convergence guarantees for MCMC correctness
- [Hidden Markov Models](markov_chains/hmm.md) -- Latent Markov chains with observation models, bridging chain theory and statistical inference

## Monte Carlo Methods

- [Monte Carlo Integration](monte_carlo/integration.md) -- Using random sampling to approximate intractable integrals, with variance reduction techniques
- [Effective Sample Size](monte_carlo/ess.md) -- Measuring the information content of weighted or correlated samples as a fundamental diagnostic
- [Rejection Sampling](monte_carlo/rejection.md) -- Generating samples from a target distribution using a bounding proposal distribution

### Importance Sampling

- [Fundamentals](monte_carlo/importance_sampling/fundamentals.md) -- Computing expectations under one distribution by sampling from another with importance weights
- [Effective Sample Size for IS](monte_carlo/importance_sampling/ess.md) -- Quantifying how many independent target samples the weighted importance samples are equivalent to
- [Self-Normalized Importance Sampling](monte_carlo/importance_sampling/self_normalized.md) -- Handling unnormalized target densities by estimating both numerator and denominator
- [Proposal Distribution Design](monte_carlo/importance_sampling/proposal_design.md) -- Principles and strategies for designing effective proposals including the optimal proposal derivation

## MCMC Methods

- [Metropolis-Hastings](mcmc/metropolis_hastings.md) -- The foundational MCMC algorithm for sampling from distributions known only up to a normalizing constant
- [Gibbs Sampling](mcmc/gibbs_sampling.md) -- Special case of Metropolis-Hastings with acceptance probability always equal to one using full conditionals
- [MCMC Diagnostics](mcmc/diagnostics.md) -- Assessing convergence, mixing quality, and reliability of posterior summaries from finite MCMC output
- [NUTS](mcmc/nuts.md) -- No-U-Turn Sampler that automatically tunes HMC trajectory length by detecting turnarounds

### Simulated Annealing

- [Fundamentals](mcmc/simulated_annealing/fundamentals.md) -- Non-stationary Metropolis-Hastings for global optimization using the Boltzmann distribution
- [Temperature Schedules](mcmc/simulated_annealing/schedules.md) -- Cooling schedule design controlling the trade-off between exploration and exploitation
- [Temperature as a Unifying Concept](mcmc/simulated_annealing/temperature_unifying.md) -- Connections between temperature in MCMC, softmax, diffusion models, and reinforcement learning
- [SA as Non-Stationary MCMC](mcmc/simulated_annealing/sa_as_mcmc.md) -- Understanding simulated annealing as a time-varying Metropolis-Hastings with changing target distributions
- [Convergence Theory](mcmc/simulated_annealing/convergence.md) -- Mathematical theory of when and why SA finds the global optimum, energy barriers, and practical implications
- [Deterministic Annealing for EM](mcmc/simulated_annealing/annealed_em.md) -- Applying temperature to the EM algorithm for escaping local optima in likelihood optimization

### Hamiltonian Monte Carlo

- [Overview](mcmc/hmc/overview.md) -- Complete theory of HMC: physics-guided proposals that dramatically outperform random-walk methods
- [Hamiltonian Dynamics](mcmc/hmc/hamiltonian_dynamics.md) -- Physics foundations from Lagrangian to Hamiltonian mechanics, symplectic structure, and conservation laws
- [Phase Space](mcmc/hmc/phase_space.md) -- The extended state space of position and momentum variables that enables deterministic dynamics
- [The HMC Algorithm](mcmc/hmc/algorithm.md) -- Complete algorithm with momentum augmentation, leapfrog integration, and Metropolis correction
- [Leapfrog Integrator](mcmc/hmc/leapfrog_integrator.md) -- The symplectic numerical integrator preserving volume and time-reversibility for HMC
- [Mass Matrix](mcmc/hmc/mass_matrix.md) -- Tuning how momentum translates to velocity, with geometric interpretation and estimation strategies
- [Geometric Interpretation](mcmc/hmc/geometric_interpretation.md) -- Understanding HMC through differential geometry, information geometry, and physical intuition

## Langevin Dynamics

- [Fundamentals](langevin/fundamentals.md) -- Continuous-time framework connecting MCMC sampling with gradient-based optimization via the Langevin SDE
- [Unadjusted Langevin Algorithm (ULA)](langevin/ula.md) -- Discretized Langevin dynamics without Metropolis correction, compatible with stochastic gradients
- [MALA](langevin/mala.md) -- Metropolis-Adjusted Langevin Algorithm combining gradient-informed proposals with accept-reject correction
- [Score Matching and Diffusion](langevin/score_and_diffusion.md) -- The score function as a unifying concept connecting Langevin dynamics, density estimation, and generative models

## Approximate Bayesian Computation

- [Likelihood-Free Inference](abc/likelihood_free.md) -- Introduction to simulation-based inference when likelihoods are unavailable but simulators exist
- [ABC Rejection Sampling](abc/rejection_sampling.md) -- The simplest likelihood-free algorithm using summary statistics and tolerance thresholds
- [ABC-MCMC](abc/abc_mcmc.md) -- Combining ABC with Markov chain Monte Carlo for more efficient parameter space exploration
- [ABC-SMC](abc/abc_smc.md) -- Sequential Monte Carlo ABC with adaptive tolerance selection and particle-based sampling

## MCMC Methods Comparison

- [Overview](mcmc_comparison/overview.md) -- Comprehensive comparison of MH, Gibbs, Langevin, and HMC with practical method selection guidance
- [Theoretical Comparison](mcmc_comparison/theoretical.md) -- Rigorous comparison covering convergence rates, spectral analysis, and optimal scaling theory
- [Scaling with Dimension](mcmc_comparison/scaling.md) -- How different MCMC methods behave as dimensionality increases and strategies for maintaining efficiency
- [Practical Method Selection](mcmc_comparison/method_selection.md) -- Decision framework based on differentiability, dimensionality, correlation structure, and computational budget
>>>>>>> 96f31bd (...)
