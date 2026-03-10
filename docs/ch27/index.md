# Chapter 27: Energy-Based Models


!!! warning "Incomplete page"
    This page is missing the required five-section structure (Concept Definition, Explanation, Diagram / Example). Content needs to be reorganized and expanded.

Energy-Based Models (EBMs) define probability distributions through scalar energy functions, where lower energy corresponds to higher probability via the Boltzmann distribution. This flexible framework places essentially no constraints on the energy function architecture, making EBMs a unifying perspective across generative model families. This chapter covers the theoretical foundations from statistical physics, classical architectures like Boltzmann machines and Hopfield networks, training methods that circumvent the intractable partition function, modern neural EBMs, and applications to quantitative finance.

---

## EBM Foundations

- Introduction to Energy-Based Models -- Core principle of energy-based modeling, the energy-probability connection, and the central computational challenges.
- Energy-Based Models Overview -- Comprehensive tutorial package covering EBM theory, implementations, and connections to modern generative models.
- [Boltzmann Distribution](ebm_foundations/boltzmann.md) -- Derivation from maximum entropy principles, temperature effects, and connections to the exponential family.
- Energy Functions -- Formal definition, desirable properties, and neural network parameterization of energy functions.
- [Partition Function](ebm_foundations/partition_function.md) -- The intractable normalization constant, its role in encoding distributional information, and estimation methods.

## Classical EBMs

- [Boltzmann Machines](classical_ebm/boltzmann_machines.md) -- Stochastic binary units, Gibbs sampling, and the extension from Hopfield networks to generative models.
- [Restricted Boltzmann Machines](classical_ebm/rbm.md) -- Bipartite architecture with tractable conditional distributions and Contrastive Divergence training.
- Deep Boltzmann Machines -- Multi-layer undirected models with hierarchical representations and mean-field variational inference.
- [Hopfield Networks](classical_ebm/hopfield.md) -- Energy-based associative memories using energy minimization for pattern storage and retrieval.

## EBM Training

- Contrastive Divergence -- Practical approximation to the intractable maximum likelihood gradient using short-run MCMC from data.
- Persistent Contrastive Divergence -- Maintaining persistent Markov chains across updates for less biased negative phase samples.
- Score Matching for EBMs -- Training energy functions by matching score functions, avoiding the partition function entirely.
- Noise Contrastive Estimation -- Reducing density estimation to binary classification between data and noise, learning the partition function as a byproduct.

## Modern EBMs

- Neural Energy-Based Models -- Deep neural networks as energy functions trained with Langevin dynamics and replay buffers.
- Joint Energy Models (JEM) -- Unifying classification and generation by reinterpreting classifier logits as energy functions.
- [Cooperative Learning](modern_ebm/cooperative.md) -- Combining EBMs with autoencoders, VAEs, and GANs for hybrid systems leveraging complementary strengths.
- EBM for Classification and Beyond -- Out-of-distribution detection, compositional generation, and image denoising using energy scores.
- Connection to Diffusion Models -- Mathematical unification of EBMs and diffusion models through score-based generative modeling.
- [EBM-Based Generation](modern_ebm/generation.md) -- Generating samples from learned energy landscapes via Langevin and Hamiltonian dynamics.

## Finance Applications

- Anomaly Detection -- Detecting unusual trading patterns and rare market events using energy-based anomaly scoring.
- Density Estimation for Finance -- Flexible non-parametric density estimation for financial returns capturing heavy tails and non-Gaussianity.
- Energy-Based Portfolio Optimization -- Framing portfolio optimization as energy minimization with return objectives and risk penalties.
- [Credit Network Models](finance_applications/credit_networks.md) -- Modeling credit default dependencies using Boltzmann machine structure for systemic risk monitoring.
