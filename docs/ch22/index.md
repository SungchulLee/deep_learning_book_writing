# Chapter 22: Variational Autoencoders

A rigorous treatment of VAEs: from probabilistic foundations to modern variants and quantitative finance applications.

---

## Chapter Overview

Variational Autoencoders (VAEs) represent a fundamental breakthrough in generative modeling, combining deep learning with Bayesian inference to learn meaningful latent representations of data. Unlike standard autoencoders that learn deterministic point encodings, VAEs learn probabilistic distributions over a latent space, enabling principled generation, interpolation, and uncertainty quantification.

This chapter provides a mathematically rigorous treatment of VAEs, starting from the probabilistic foundations of latent variable models and building up through the ELBO derivation, architectural considerations, modern variants, and practical applications in quantitative finance.

---

## Chapter Structure

### 22.1 Foundations

We begin by establishing the probabilistic framework that motivates VAEs, covering latent variable models, the distinction between generative and discriminative approaches, and why exact posterior inference is intractable for deep generative models.

### 22.2 VAE Theory

The theoretical core of the chapter derives the Evidence Lower Bound (ELBO) from first principles, introduces the reparameterization trick that makes stochastic gradient optimization possible, and decomposes the ELBO into its KL divergence and reconstruction components.

### 22.3 VAE Architecture

We examine the encoder and decoder networks in detail, discuss the role of the prior distribution, and analyze the posterior collapse problem that commonly arises during training.

### 22.4 VAE Variants

This section covers the most important VAE extensions: β-VAE for disentanglement, Conditional VAE for controlled generation, VQ-VAE and VQ-VAE-2 for discrete latent spaces, Hierarchical VAEs for multi-scale representations, and NVAE for state-of-the-art image generation.

### 22.5 Training

Practical training considerations including optimization strategies, KL annealing schedules, the free bits technique for preventing posterior collapse, and the effects of batch size on VAE training dynamics.

### 22.6 Evaluation

Metrics and methods for assessing VAE quality across reconstruction fidelity, generation quality, latent space structure, and disentanglement.

### 22.7 Finance Applications

Quantitative finance applications including synthetic data generation for augmenting limited datasets, missing data imputation for incomplete financial records, and scenario generation for stress testing and risk management.

---

## The Big Picture

### From Autoencoders to VAEs

```
Standard Autoencoder                Variational Autoencoder
─────────────────────               ───────────────────────

    Input x                             Input x
       │                                   │
       ▼                                   ▼
   [Encoder]                           [Encoder]
       │                                   │
       ▼                                   ▼
   z (point)              →        (μ, σ²) distribution
       │                                   │
       ▼                                   ▼
   [Decoder]                    z ~ N(μ, σ²) [sample]
       │                                   │
       ▼                                   ▼
  Reconstruction x̂                    [Decoder]
                                           │
                                           ▼
                                    Reconstruction x̂

Loss: ||x - x̂||²            Loss: Recon + KL(q||p)
```

VAEs introduce **stochasticity** and **regularization** into the latent space through three key insights: stochastic encoding maps inputs to distributions rather than points, KL regularization keeps the latent distribution close to a prior $\mathcal{N}(0, I)$, and generation becomes straightforward by sampling from the prior and decoding.

---

## Mathematical Notation

| Symbol | Meaning |
|--------|---------|
| $x$ | Observed data (input) |
| $z$ | Latent variable |
| $\theta$ | Decoder (generative model) parameters |
| $\phi$ | Encoder (inference model) parameters |
| $p_\theta(x\|z)$ | Decoder distribution (likelihood) |
| $p(z)$ | Prior distribution over latent space |
| $p_\theta(z\|x)$ | True posterior (intractable) |
| $q_\phi(z\|x)$ | Approximate posterior (encoder) |
| $\mathcal{L}$ | Evidence Lower Bound (ELBO) |
| $D_{KL}$ | Kullback-Leibler divergence |

---

## Key Equations

### The ELBO

$$\mathcal{L}(\theta, \phi; x) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) \| p(z))$$

### Gaussian KL Divergence

$$D_{KL}(q_\phi(z|x) \| p(z)) = -\frac{1}{2}\sum_{j=1}^{d}(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2)$$

### Reparameterization Trick

$$z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

### Fundamental Identity

$$\log p_\theta(x) = \mathcal{L}(\theta, \phi; x) + D_{KL}(q_\phi(z|x) \| p_\theta(z|x))$$

---

## Choosing the Right VAE Variant

```
Do you need...
│
├── Controlled generation? → Conditional VAE (22.4.2)
│
├── Disentangled representations? → β-VAE (22.4.1)
│
├── Sharp reconstructions? → VQ-VAE (22.4.3)
│
├── Multi-scale generation? → VQ-VAE-2 (22.4.4) or Hierarchical VAE (22.4.5)
│
├── State-of-the-art image quality? → NVAE (22.4.6)
│
└── General purpose? → Standard VAE (22.2–22.3)
```

---

## Prerequisites

Before starting this chapter, you should be familiar with:

- **Probability theory:** Random variables, expectations, Bayes' theorem
- **Neural networks:** Feedforward networks, backpropagation, PyTorch basics
- **Autoencoders:** Encoder-decoder architecture, reconstruction loss
- **Optimization:** Gradient descent, Adam optimizer

---

## References

### Foundational Papers

1. Kingma & Welling (2014). "Auto-Encoding Variational Bayes" — The original VAE paper
2. Rezende et al. (2014). "Stochastic Backpropagation and Approximate Inference in Deep Generative Models"
3. Higgins et al. (2017). "β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework"
4. van den Oord et al. (2017). "Neural Discrete Representation Learning" — VQ-VAE

### Reviews and Tutorials

- Doersch (2016). "Tutorial on Variational Autoencoders"
- Kingma & Welling (2019). "An Introduction to Variational Autoencoders"
