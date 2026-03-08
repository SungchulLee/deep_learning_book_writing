# Chapter Overview

<<<<<<< HEAD
This chapter covers **Backtracking**.

# Reference

[Algorithm Design Manual (Skiena), Chapter 9](https://www.algorist.com/)
=======
This chapter provides a comprehensive treatment of autoencoders for unsupervised learning and dimensionality reduction. Starting from the basic encoder-decoder framework, it covers architecture design, training procedures, loss functions, and major variants including sparse, denoising, contractive, and convolutional autoencoders. The chapter also explores latent space analysis and applications to quantitative finance.

---

## Autoencoder Fundamentals

Core concepts, architecture design, and training procedures for standard autoencoders.

- [Introduction to Autoencoders](ae/introduction.md) -- Unsupervised learning, dimensionality reduction, and the bridge from PCA to VAEs
- [Architecture](ae/architecture.md) -- Encoder-decoder design from basic to convolutional and deep variants
- [Training](ae/training.md) -- Training procedures, evaluation methods, and practical applications
- [Loss Functions](ae/loss_functions.md) -- Reconstruction losses, regularization objectives, and specialized criteria
- [Bottleneck Design](ae/bottleneck.md) -- Information compression, dimensionality selection, and capacity trade-offs
- [Reconstruction Analysis](ae/reconstruction_analysis.md) -- Assessing reconstruction quality with per-sample and feature-wise diagnostics

---

## Variants

Autoencoder variants with different regularization strategies and architectural designs.

- [Undercomplete and Overcomplete](variants/undercomplete.md) -- How latent vs input dimensionality shapes learning dynamics
- [Sparse Autoencoder](variants/sparse.md) -- Penalizing hidden unit activations for interpretable features
- [Denoising Autoencoder](variants/denoising.md) -- Reconstructing clean data from corrupted inputs for robust representations
- [Contractive Autoencoder](variants/contractive.md) -- Penalizing encoder sensitivity to input perturbations
- [Convolutional Autoencoder](variants/convolutional.md) -- Preserving spatial structure in image data with convolutional layers

---

## Representation Learning

Analyzing and understanding learned latent representations.

- [Latent Space](representation/latent_space.md) -- Geometry, structure, and information-theoretic properties of latent representations
- [Disentanglement](representation/disentanglement.md) -- Learning dimensions that correspond to independent factors of variation
- [Interpolation and Latent Arithmetic](representation/interpolation.md) -- Vector arithmetic, smooth interpolation, and semantic direction discovery

---

## Finance

Applications of autoencoders to quantitative finance problems.

- [Anomaly Detection](finance/anomaly_detection.md) -- Using reconstruction error to detect market anomalies and fraudulent activity
- [Factor Discovery](finance/factor_discovery.md) -- Unsupervised extraction of latent risk factors from market data
- [Portfolio Compression](finance/portfolio_compression.md) -- Dimensionality reduction for portfolio management and risk decomposition
>>>>>>> 96f31bd (...)
