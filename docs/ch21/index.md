# Chapter 21: Autoencoders

Autoencoders are a fundamental class of neural networks for **unsupervised learning** and **dimensionality reduction**. They learn efficient data representations by compressing input data into a lower-dimensional latent space and then reconstructing the original input.

Autoencoders serve as a bridge between classical dimensionality reduction methods (like PCA) and modern generative models (like VAEs). Understanding autoencoders deeply prepares you for the probabilistic extensions covered in Chapter 22 (Variational Autoencoders).

---

## Learning Objectives

By completing this chapter, you will be able to:

- Understand the encoder-decoder architecture and its mathematical foundations
- Implement various autoencoder types: vanilla, denoising, sparse, contractive, convolutional, and deep
- Derive and implement reconstruction loss functions (MSE, BCE) and regularized objectives
- Apply autoencoders to practical problems: dimensionality reduction, denoising, anomaly detection, and feature learning
- Understand the relationship between linear autoencoders and PCA
- Visualize and interpret learned latent representations
- Apply autoencoders to quantitative finance: factor discovery, anomaly detection, and portfolio compression

---

## Prerequisites

- **Section 1.2: Tensors** — tensor operations and manipulation
- **Section 1.4: Gradients** — autograd and backpropagation
- **Section 1.6: Maximum Likelihood Estimation** — connection to loss functions
- **Section 2.1: Loss Functions** — MSE, BCE
- **Section 2.2: Optimizers** — Adam
- **Section 2.3: Activation Functions** — ReLU, Sigmoid
- **Section 2.7: Feedforward Networks** — MLP architecture
- **Section 3.1: Convolutional Neural Networks** — for convolutional autoencoders

---

## Chapter Structure

### 21.1 Fundamentals

| Section | Topic | Description |
|---------|-------|-------------|
| 21.1.1 | [Introduction](ae/introduction.md) | Overview, mathematical foundations, taxonomy |
| 21.1.2 | [Architecture](ae/architecture.md) | Encoder-decoder design, FC/Conv/Deep variants, implementations |
| 21.1.3 | [Loss Functions](ae/loss_functions.md) | Reconstruction losses, regularization objectives |
| 21.1.4 | [Training](ae/training.md) | Training procedures, analysis, best practices |

### 21.2 Variants

| Section | Topic | Description |
|---------|-------|-------------|
| 21.2.1 | [Undercomplete](variants/undercomplete.md) | Undercomplete vs overcomplete, identity mapping problem |
| 21.2.2 | [Sparse Autoencoder](variants/sparse.md) | L1/KL divergence sparsity constraints |
| 21.2.3 | [Denoising Autoencoder](variants/denoising.md) | Learning robust representations via noise corruption |
| 21.2.4 | [Contractive Autoencoder](variants/contractive.md) | Jacobian penalty for robust features |
| 21.2.5 | [Convolutional Autoencoder](variants/convolutional.md) | Preserving spatial structure in images |

### 21.3 Representation Learning

| Section | Topic | Description |
|---------|-------|-------------|
| 21.3.1 | [Latent Space](representation/latent_space.md) | Latent space geometry, information bottleneck, visualization |
| 21.3.2 | [Disentanglement](representation/disentanglement.md) | Disentangled representations, factor isolation |
| 21.3.3 | [Interpolation](representation/interpolation.md) | Latent arithmetic, smooth interpolation, semantic directions |

### 21.4 Finance Applications

| Section | Topic | Description |
|---------|-------|-------------|
| 21.4.1 | [Factor Discovery](finance/factor_discovery.md) | Unsupervised factor extraction from market data |
| 21.4.2 | [Anomaly Detection](finance/anomaly_detection.md) | Regime detection, fraud detection via reconstruction error |
| 21.4.3 | [Portfolio Compression](finance/portfolio_compression.md) | Dimensionality reduction for portfolio management |

---

## Key Concepts Summary

### Architecture Components

```
Input Layer → Encoder → Bottleneck → Decoder → Output Layer
   (d_x)       ↓          (d_z)        ↓         (d_x)
           Compress              Reconstruct
```

### Autoencoder Types Comparison

| Type | Key Feature | Regularization | Best For |
|------|-------------|----------------|----------|
| **Vanilla** | Basic reconstruction | None (bottleneck only) | Compression, feature learning |
| **Denoising** | Corrupt input, reconstruct clean | Input noise | Robust features, denoising |
| **Sparse** | Few active neurons | L1 or KL penalty | Interpretable features |
| **Contractive** | Stable to perturbations | Jacobian penalty | Robust manifold learning |
| **Convolutional** | Preserve spatial structure | Weight sharing | Image data |
| **Deep/Stacked** | Hierarchical features | Batch norm, dropout | Complex patterns |

---

## Comparison with Related Methods

### Autoencoder vs PCA

| Aspect | PCA | Autoencoder |
|--------|-----|-------------|
| Linearity | Linear only | Can be nonlinear |
| Optimization | Closed-form (SVD) | Iterative (gradient descent) |
| Basis | Orthogonal | Not necessarily orthogonal |
| Scalability | $O(d^3)$ or $O(nd^2)$ | Flexible (batch training) |
| Interpretability | High (principal components) | Lower (distributed) |

### Autoencoder vs VAE

| Aspect | Autoencoder | VAE |
|--------|-------------|-----|
| Latent space | Deterministic | Probabilistic |
| Generation | Limited | Natural sampling |
| Regularization | Optional | Built-in (KL divergence) |
| Smoothness | No guarantee | Smooth by design |
| Training | MSE/BCE only | ELBO (reconstruction + KL) |

---

## References

### Foundational Papers

1. Hinton, G. E., & Salakhutdinov, R. R. (2006). "Reducing the dimensionality of data with neural networks." *Science*, 313(5786), 504-507.

2. Vincent, P., Larochelle, H., Bengio, Y., & Manzagol, P. A. (2008). "Extracting and composing robust features with denoising autoencoders." *ICML*.

3. Rifai, S., Vincent, P., Muller, X., Glorot, X., & Bengio, Y. (2011). "Contractive auto-encoders: Explicit invariance during feature extraction." *ICML*.

### Textbooks

- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. Chapter 14.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer. Chapter 12.
