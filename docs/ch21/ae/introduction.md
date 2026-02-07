# Introduction to Autoencoders

Autoencoders are a fundamental class of neural networks for **unsupervised learning** and **dimensionality reduction**. They learn efficient data representations by compressing input data into a lower-dimensional latent space and then reconstructing the original input.

Autoencoders serve as a bridge between classical dimensionality reduction methods (like PCA) and modern generative models (like VAEs). Understanding autoencoders deeply prepares you for the probabilistic extensions covered in Section 21.2 (Variational Autoencoders).

---

## Learning Objectives

By completing this section, you will be able to:

- Understand the encoder-decoder architecture and its mathematical foundations
- Implement various autoencoder types: vanilla, denoising, sparse, contractive, convolutional, and deep
- Derive and implement reconstruction loss functions (MSE, BCE) and regularized objectives
- Apply autoencoders to practical problems: dimensionality reduction, denoising, anomaly detection, and feature learning
- Understand the relationship between linear autoencoders and PCA
- Visualize and interpret learned latent representations
- Analyze autoencoder performance across different architectures

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

## Mathematical Foundation

### Core Concept

An autoencoder consists of two learned functions:

**Encoder:** $f_\theta: \mathcal{X} \to \mathcal{Z}$ maps input to latent representation

**Decoder:** $g_\phi: \mathcal{Z} \to \hat{\mathcal{X}}$ reconstructs input from latent representation

The composition $g_\phi \circ f_\theta$ forms an identity approximation under the constraint that $\mathcal{Z}$ has lower dimensionality than $\mathcal{X}$.

### Objective Function

The autoencoder minimizes reconstruction error:

$$\mathcal{L}(\theta, \phi) = \frac{1}{n} \sum_{i=1}^{n} \ell(x_i, g_\phi(f_\theta(x_i)))$$

where $\ell$ is typically:

| Loss Function | Formula | Use Case |
|--------------|---------|----------|
| MSE | $\|x - \hat{x}\|^2$ | Continuous data, Gaussian assumption |
| BCE | $-[x \log \hat{x} + (1-x) \log(1-\hat{x})]$ | Binary/normalized data |
| MAE | $\|x - \hat{x}\|_1$ | Robust to outliers |

### Dimensionality Reduction

The latent space $\mathcal{Z}$ typically has lower dimension than input space $\mathcal{X}$:

- **Input dimension:** $d_x$ (e.g., 784 for MNIST)
- **Latent dimension:** $d_z$ where $d_z < d_x$
- **Compression ratio:** $d_x / d_z$

This bottleneck forces the network to learn a compact, informative representation.

### Connection to PCA

Linear autoencoders (without activation functions) trained with MSE loss learn the same subspace as Principal Component Analysis, though not necessarily the same basis vectors. This connection is explored in detail in Section 21.0 (PCA).

---

## Architecture Overview

```
Input Layer → Encoder → Bottleneck → Decoder → Output Layer
   (d_x)       ↓          (d_z)        ↓         (d_x)
           Compress              Reconstruct
```

- **Input Layer:** Accepts original data (e.g., 28×28 images flattened to 784)
- **Encoder:** Progressive dimensionality reduction through hidden layers
- **Bottleneck/Latent Space:** Compressed representation (the learned features)
- **Decoder:** Progressive dimensionality expansion, mirroring encoder
- **Output Layer:** Same size as input for reconstruction

---

## Autoencoder Variants

| Type | Key Feature | Regularization | Best For |
|------|-------------|----------------|----------|
| **Vanilla** | Basic reconstruction | None (bottleneck only) | Compression, feature learning |
| **Denoising** | Corrupt input, reconstruct clean | Input noise | Robust features, denoising |
| **Sparse** | Few active neurons | L1 or KL penalty | Interpretable features |
| **Contractive** | Stable to perturbations | Jacobian penalty | Robust manifold learning |
| **Convolutional** | Preserve spatial structure | Weight sharing | Image data |
| **Deep/Stacked** | Hierarchical features | Batch norm, dropout | Complex patterns |

---

## Undercomplete vs Overcomplete

The relationship between latent dimension and input dimension defines two fundamental categories:

### Undercomplete Autoencoder

$$\text{dim}(z) < \text{dim}(x)$$

The standard configuration. The bottleneck forces compression and prevents the network from learning the trivial identity mapping. This provides **natural regularization** through the information bottleneck.

### Overcomplete Autoencoder

$$\text{dim}(z) \geq \text{dim}(x)$$

With more latent dimensions than input dimensions, the network can potentially learn the identity function $f(x) = x$, $g(z) = z$, achieving zero reconstruction error without learning anything useful. Overcomplete autoencoders **require explicit regularization** — sparsity constraints, denoising, or contractive penalties — to learn meaningful representations.

| Condition | Identity Risk | Mitigation |
|-----------|---------------|------------|
| `latent_dim >= input_dim` | High | Sparsity, denoising, contractive penalty |
| Linear activations | Very high | Use nonlinear activations |
| No regularization | High | Add explicit regularization |

---

## The Latent Space

### Definition

The **latent space** $\mathcal{Z}$ is the lower-dimensional representation learned by the encoder:

$$z = f_\theta(x) \in \mathbb{R}^k$$

where $k < d$ (latent dimension < input dimension).

### Properties of Good Latent Spaces

| Property | Description | Benefit |
|----------|-------------|---------|
| **Compactness** | Similar inputs map to nearby points | Generalization |
| **Smoothness** | Small changes in $z$ → small changes in $\hat{x}$ | Interpolation |
| **Disentanglement** | Different factors map to different dimensions | Interpretability |

### Information Bottleneck

The latent space acts as an **information bottleneck**: the encoder must discard irrelevant information while preserving what is needed for reconstruction. Smaller $k$ forces more compression and higher reconstruction error, creating a fundamental trade-off between compression and fidelity.

---

## Practical Applications

1. **Dimensionality Reduction:** Alternative to PCA with nonlinear capability
2. **Denoising:** Remove noise from images, audio, and other signals
3. **Anomaly Detection:** High reconstruction error indicates anomalies
4. **Feature Learning:** Pretrain representations for downstream supervised tasks
5. **Data Compression:** Lossy compression via learned encoder/decoder
6. **Data Generation:** Limited generative capability via latent space sampling

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

## Common Pitfalls and Solutions

| Problem | Symptoms | Solution |
|---------|----------|----------|
| **Identity mapping** | Perfect reconstruction, no compression | Ensure latent_dim < input_dim, or add regularization |
| **Poor reconstruction** | Blurry or distorted outputs | Increase capacity, adjust learning rate |
| **Overfitting** | Training loss low, validation high | Add dropout, denoising, or early stopping |
| **Dead neurons** | Some latent units never activate | Check activation functions, initialization |
| **Mode collapse** | All inputs map to similar latent codes | Reduce regularization, check architecture |

---

## References

### Foundational Papers

1. Hinton, G. E., & Salakhutdinov, R. R. (2006). "Reducing the dimensionality of data with neural networks." *Science*, 313(5786), 504-507.

2. Vincent, P., Larochelle, H., Bengio, Y., & Manzagol, P. A. (2008). "Extracting and composing robust features with denoising autoencoders." *ICML*.

3. Rifai, S., Vincent, P., Muller, X., Glorot, X., & Bengio, Y. (2011). "Contractive auto-encoders: Explicit invariance during feature extraction." *ICML*.

### Textbooks

- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. Chapter 14.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer. Chapter 12.

---

## Section Contents

| Section | Topic | Description |
|---------|-------|-------------|
| 21.1.1 | [Introduction](introduction.md) | Overview, mathematical foundations, taxonomy |
| 21.1.2 | [Architecture](architecture.md) | Encoder-decoder design, variants, implementations |
| 21.1.3 | [Loss Functions](loss_functions.md) | Reconstruction losses, regularization objectives |
| 21.1.4 | [Training](training.md) | Training procedures, analysis, applications |
