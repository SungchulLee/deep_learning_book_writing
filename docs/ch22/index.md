<<<<<<< HEAD
# Chapter Overview

This chapter covers the fundamental algorithms for string matching, one of the most important problems in computer science. Given a text $T$ of length $n$ and a pattern $P$ of length $m$, the goal is to find all occurrences of $P$ in $T$.
=======
# Chapter 22: Variational Autoencoders (VAE)

This chapter provides a rigorous treatment of Variational Autoencoders, bridging deep learning with Bayesian inference. Starting from probabilistic foundations and information theory, it develops the ELBO derivation, reparameterization trick, and encoder-decoder architecture. The chapter then covers major VAE variants, training strategies for posterior collapse, evaluation metrics, and applications to financial data generation and analysis.
>>>>>>> 96f31bd (...)

We organize the material into three major areas. **Exact single-pattern matching** includes the naive brute-force method, the Knuth-Morris-Pratt (KMP) algorithm with its failure function, the Boyer-Moore algorithm with its bad-character and good-suffix heuristics, the Rabin-Karp rolling-hash approach, and the Z-algorithm. **Multiple-pattern matching** covers the Aho-Corasick automaton together with its failure links and dictionary links. **Regular expression matching** introduces NFA construction via Thompson's algorithm, the subset construction for converting an NFA to a DFA, and DFA state minimization.

<<<<<<< HEAD
$$

\text{String Matching}
\left\{\begin{array}{lll}
\text{Exact Matching} & O(nm) \text{ naive}, \; O(n+m) \text{ KMP/BM/Z}\\
\\
\text{Multiple Pattern} & O(n + m + z) \text{ Aho-Corasick}\\
\\
\text{Regular Expression} & \text{Thompson's NFA} \to \text{DFA} \to \text{Minimized DFA}
\end{array}\right.

$$
=======
## Foundations

Probabilistic and information-theoretic foundations underlying Variational Autoencoders.

- [Introduction to VAEs](foundations/introduction.md) -- Why probabilistic generative models are needed and how VAEs bridge deep learning with Bayesian inference
- [Latent Variable Models](foundations/latent_variable.md) -- Probabilistic foundation of VAEs with latent variable formulation
- [Generative vs Discriminative Models](foundations/generative_discriminative.md) -- Two fundamental paradigms in probabilistic machine learning
- [Information Theory Foundations](foundations/information_theory.md) -- Entropy, KL divergence, and the mathematical language of compression
- [Mutual Information](foundations/mutual_information.md) -- Information flow between data and latent representations
>>>>>>> 96f31bd (...)

Throughout this chapter, we use 0-based indexing for strings unless otherwise noted.

<<<<<<< HEAD
# Reference

[Introduction to Algorithms (CLRS), Chapters 32](https://mitpress.mit.edu/books/introduction-to-algorithms-fourth-edition/)

[Algorithms on Strings, Trees and Sequences - Dan Gusfield](https://www.cambridge.org/core/books/algorithms-on-strings-trees-and-sequences/F0B095049C8347C7F1D2DC5F1D74AC5D)
=======
## Theory

Mathematical derivations of the VAE objective function and its components.

- [ELBO Derivation](theory/elbo_derivation.md) -- Deriving the Evidence Lower Bound from first principles
- [KL Divergence Term](theory/kl_term.md) -- Properties and computation of the KL regularization in VAEs
- [Reconstruction Term](theory/reconstruction.md) -- The likelihood component of the VAE objective
- [Reparameterization Trick](theory/reparameterization.md) -- Making stochastic sampling differentiable for backpropagation

---

## Architecture

Neural network components and implementation of the VAE architecture.

- [Encoder Network](architecture/encoder.md) -- Amortized variational inference mapping data to approximate posteriors
- [Decoder Network](architecture/decoder.md) -- Generative model mapping latent codes to data distributions
- [Prior Selection](architecture/prior.md) -- Choosing and designing the prior distribution for the latent space
- [Posterior Collapse](architecture/posterior_collapse.md) -- Understanding, diagnosing, and mitigating posterior collapse
- [PyTorch Implementation](architecture/implementation.md) -- Complete VAE implementation with training pipeline and visualization
- [Autoencoder Basics for VAEs](architecture/autoencoder_basics_overview.md) -- Foundational autoencoder tutorials as prerequisites for VAEs

---

## Variants

Major VAE variants with different latent representations and architectural innovations.

- [Beta-VAE](variants/beta_vae.md) -- Learning disentangled representations with weighted KL divergence
- [Conditional VAE](variants/cvae.md) -- Controlled generation through label conditioning
- [VQ-VAE](variants/vqvae.md) -- Discrete latent representations through vector quantization
- [VQ-VAE-2](variants/vqvae2.md) -- Multi-scale hierarchical discrete latent representations
- [Hierarchical VAE](variants/hierarchical.md) -- Multi-scale continuous latent representations at multiple levels
- [NVAE](variants/nvae.md) -- Deep hierarchical VAE achieving state-of-the-art generation quality

---

## Training

Practical training strategies and techniques for effective VAE optimization.

- [VAE Optimization](training/optimization.md) -- Practical strategies for training VAEs effectively
- [KL Annealing](training/kl_annealing.md) -- Gradually introducing KL penalty to prevent posterior collapse
- [Free Bits](training/free_bits.md) -- Guaranteeing minimum information per latent dimension
- [Batch Size Effects](training/batch_size.md) -- How batch size influences training dynamics and gradient estimation

---

## Evaluation

Metrics and methods for assessing VAE quality across reconstruction, generation, and latent space.

- [Reconstruction Quality](evaluation/reconstruction.md) -- Evaluating how well a trained VAE reconstructs input data
- [Generation Quality](evaluation/generation.md) -- Assessing quality and diversity of generated samples (FID, IS)
- [Latent Space Quality](evaluation/latent_space.md) -- Evaluating structure and properties of learned representations
- [Disentanglement Metrics](evaluation/disentanglement.md) -- Quantifying correspondence between latent dimensions and factors of variation

---

## Finance

Applications of VAEs to quantitative finance for data generation, imputation, and risk analysis.

- [Synthetic Data Generation](finance/synthetic_data.md) -- Generating realistic synthetic financial data for augmentation and privacy
- [Missing Data Imputation](finance/imputation.md) -- Filling in missing values using learned data distributions
- [Scenario Generation](finance/scenarios.md) -- Generating market scenarios for stress testing and risk management
- [Anomaly Detection and Denoising](finance/anomaly_detection.md) -- Using reconstruction error and latent space for anomaly detection
>>>>>>> 96f31bd (...)
