<<<<<<< HEAD
# Chapter Overview

This chapter covers **Computational Complexity**.

# Reference

[Introduction to the Theory of Computation (Sipser)](https://www.amazon.com/Introduction-Theory-Computation-Michael-Sipser/dp/113318779X)
=======
# Chapter 27: Energy-Based Models

Energy-Based Models (EBMs) define probability distributions through scalar energy functions, where lower energy corresponds to higher probability via the Boltzmann distribution. This flexible framework places essentially no constraints on the energy function architecture, making EBMs a unifying perspective across generative model families. This chapter covers the theoretical foundations from statistical physics, classical architectures like Boltzmann machines and Hopfield networks, training methods that circumvent the intractable partition function, modern neural EBMs, and applications to quantitative finance.

---

## EBM Foundations

- [Introduction to Energy-Based Models](ebm_foundations/introduction.md) -- Core principle of energy-based modeling, the energy-probability connection, and the central computational challenges.
- [Energy-Based Models Overview](ebm_foundations/energy_based_models_overview.md) -- Comprehensive tutorial package covering EBM theory, implementations, and connections to modern generative models.
- [Boltzmann Distribution](ebm_foundations/boltzmann.md) -- Derivation from maximum entropy principles, temperature effects, and connections to the exponential family.
- [Energy Functions](ebm_foundations/energy_functions.md) -- Formal definition, desirable properties, and neural network parameterization of energy functions.
- [Partition Function](ebm_foundations/partition_function.md) -- The intractable normalization constant, its role in encoding distributional information, and estimation methods.

## Classical EBMs

- [Boltzmann Machines](classical_ebm/boltzmann_machines.md) -- Stochastic binary units, Gibbs sampling, and the extension from Hopfield networks to generative models.
- [Restricted Boltzmann Machines](classical_ebm/rbm.md) -- Bipartite architecture with tractable conditional distributions and Contrastive Divergence training.
- [Deep Boltzmann Machines](classical_ebm/dbm.md) -- Multi-layer undirected models with hierarchical representations and mean-field variational inference.
- [Hopfield Networks](classical_ebm/hopfield.md) -- Energy-based associative memories using energy minimization for pattern storage and retrieval.

## EBM Training

- [Contrastive Divergence](ebm_training/contrastive_divergence.md) -- Practical approximation to the intractable maximum likelihood gradient using short-run MCMC from data.
- [Persistent Contrastive Divergence](ebm_training/pcd.md) -- Maintaining persistent Markov chains across updates for less biased negative phase samples.
- [Score Matching for EBMs](ebm_training/score_matching.md) -- Training energy functions by matching score functions, avoiding the partition function entirely.
- [Noise Contrastive Estimation](ebm_training/nce.md) -- Reducing density estimation to binary classification between data and noise, learning the partition function as a byproduct.

## Modern EBMs

- [Neural Energy-Based Models](modern_ebm/neural_ebm.md) -- Deep neural networks as energy functions trained with Langevin dynamics and replay buffers.
- [Joint Energy Models (JEM)](modern_ebm/jem.md) -- Unifying classification and generation by reinterpreting classifier logits as energy functions.
- [Cooperative Learning](modern_ebm/cooperative.md) -- Combining EBMs with autoencoders, VAEs, and GANs for hybrid systems leveraging complementary strengths.
- [EBM for Classification and Beyond](modern_ebm/classification.md) -- Out-of-distribution detection, compositional generation, and image denoising using energy scores.
- [Connection to Diffusion Models](modern_ebm/diffusion_connection.md) -- Mathematical unification of EBMs and diffusion models through score-based generative modeling.
- [EBM-Based Generation](modern_ebm/generation.md) -- Generating samples from learned energy landscapes via Langevin and Hamiltonian dynamics.

## Finance Applications

- [Anomaly Detection](finance_applications/anomaly_detection.md) -- Detecting unusual trading patterns and rare market events using energy-based anomaly scoring.
- [Density Estimation for Finance](finance_applications/density_estimation.md) -- Flexible non-parametric density estimation for financial returns capturing heavy tails and non-Gaussianity.
- [Energy-Based Portfolio Optimization](finance_applications/portfolio.md) -- Framing portfolio optimization as energy minimization with return objectives and risk penalties.
- [Credit Network Models](finance_applications/credit_networks.md) -- Modeling credit default dependencies using Boltzmann machine structure for systemic risk monitoring.
>>>>>>> 96f31bd (...)
