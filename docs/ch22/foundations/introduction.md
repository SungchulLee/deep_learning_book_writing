# Introduction to Variational Autoencoders

Why we need probabilistic generative models and how VAEs bridge deep learning with Bayesian inference.

---

## Learning Objectives

By the end of this section, you will be able to:

- Articulate the fundamental limitations of deterministic autoencoders
- Explain why probabilistic latent representations are desirable
- Describe the high-level VAE framework and its key components
- Understand the central role of variational inference in VAEs

---

## The Need for Generative Models

### Beyond Discriminative Learning

Most deep learning applications focus on **discriminative** tasks: given input $x$, predict output $y$. Image classifiers, language models for next-token prediction, and regression networks all learn conditional distributions $p(y|x)$. However, many important problems require understanding the data distribution itself.

Consider a quantitative finance setting where we have historical return data for 500 assets over 10 years. A discriminative model might predict tomorrow's returns given today's features. But what if we need to generate 10,000 plausible market scenarios for stress testing? Or fill in missing data for assets that were delisted? Or detect anomalous market conditions? These tasks require a **generative model** — one that captures the full joint distribution of the data.

### What Makes a Good Generative Model?

A useful generative model should provide:

| Capability | Description |
|------------|-------------|
| **Sampling** | Draw new data points from the learned distribution |
| **Density evaluation** | Assess how likely a given data point is |
| **Representation learning** | Discover meaningful latent structure |
| **Interpolation** | Smoothly transition between data points |

Standard autoencoders achieve representation learning and interpolation to some degree, but fail at principled sampling and density evaluation. VAEs address all four capabilities within a unified probabilistic framework.

---

## From Autoencoders to VAEs

### The Autoencoder Baseline

A standard autoencoder learns a compressed representation by minimizing reconstruction error:

$$\min_{\theta, \phi} \mathbb{E}_{x \sim p_{\text{data}}}[\|x - g_\theta(f_\phi(x))\|^2]$$

where $f_\phi$ is the encoder and $g_\theta$ is the decoder. The latent code $z = f_\phi(x)$ is a deterministic point in $\mathbb{R}^d$.

**Problem:** The latent space has no structure guaranteeing that arbitrary points decode to meaningful outputs. Sampling $z$ uniformly or from a Gaussian gives no assurance of quality.

### The VAE Insight

VAEs resolve this by making two fundamental changes:

1. **Probabilistic encoding:** Instead of mapping $x$ to a point $z$, the encoder outputs parameters of a distribution $q_\phi(z|x) = \mathcal{N}(\mu_\phi(x), \sigma^2_\phi(x))$
2. **Regularized latent space:** A KL divergence term encourages $q_\phi(z|x)$ to stay close to a prior $p(z) = \mathcal{N}(0, I)$

These changes transform the autoencoder into a proper generative model where sampling $z \sim \mathcal{N}(0, I)$ and decoding produces valid outputs.

---

## The VAE Framework

### Conceptual Overview

```
    DATA SPACE                  LATENT SPACE                DATA SPACE
    
    Input x         Encoder         z              Decoder      x̂
   ┌─────────┐    q_ϕ(z|x)    ┌─────────┐      p_θ(x|z)   ┌─────────┐
   │ ▓▓░░▓▓  │ ──────────► │  μ, σ²  │ ──────────►  │ ▓▓░░▓▓  │
   │ ░░▓▓░░  │  (infer      │ sample z │  (generate   │ ░░▓▓░░  │
   │ ▓▓░░▓▓  │   latent)    │  ~ N(μ,σ²)│   from code) │ ▓▓░░▓▓  │
   └─────────┘               └─────────┘               └─────────┘
   
   Observed                   Hidden                    Reconstructed
```

The VAE simultaneously learns:

- An **inference model** (encoder) $q_\phi(z|x)$ that maps data to latent distributions
- A **generative model** (decoder) $p_\theta(x|z)$ that maps latent codes to data
- Both are trained jointly by maximizing the Evidence Lower Bound (ELBO)

### The Training Objective

The VAE maximizes a lower bound on the log-likelihood of the data:

$$\log p_\theta(x) \geq \underbrace{\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]}_{\text{Reconstruction term}} - \underbrace{D_{KL}(q_\phi(z|x) \| p(z))}_{\text{Regularization term}} = \mathcal{L}(\theta, \phi; x)$$

The reconstruction term encourages faithful data reconstruction, while the KL divergence term regularizes the latent space to remain close to the prior. This elegant balance between reconstruction fidelity and latent space regularity is what makes VAEs powerful generative models.

---

## Why VAEs Matter for Quantitative Finance

VAEs are particularly well-suited to quantitative finance applications for several reasons.

**Uncertainty quantification** is built into the framework. The probabilistic latent space naturally captures uncertainty, and the decoder distribution $p_\theta(x|z)$ can express heteroscedastic noise — critical for financial data where volatility varies across regimes.

**Structured latent spaces** enable factor discovery. The latent dimensions of a trained VAE can correspond to interpretable market factors (momentum, value, volatility), providing a data-driven complement to traditional factor models.

**Conditional generation** via CVAEs allows scenario analysis. By conditioning on market regimes, stress conditions, or portfolio constraints, we can generate targeted scenarios for risk management.

**Missing data handling** comes naturally. The latent space provides a principled mechanism for imputing missing values by finding latent codes consistent with observed data.

---

## Chapter Roadmap

This chapter progresses through VAE concepts in order of mathematical dependency:

```
Foundations (22.1)
├── Latent Variable Models — Why latent variables and intractable inference
└── Generative vs Discriminative — Framework comparison

Theory (22.2)
├── ELBO Derivation — The core training objective
├── Reparameterization Trick — Making sampling differentiable
├── KL Divergence Term — Regularization analysis
└── Reconstruction Term — Likelihood and loss functions

Architecture (22.3)
├── Encoder Network — Amortized inference
├── Decoder Network — Generation
├── Prior Selection — Beyond standard Gaussian
└── Posterior Collapse — Failure mode analysis

Variants (22.4)
├── β-VAE — Disentanglement
├── Conditional VAE — Controlled generation
├── VQ-VAE / VQ-VAE-2 — Discrete latent spaces
├── Hierarchical VAE — Multi-scale structure
└── NVAE — State-of-the-art architecture

Training & Evaluation (22.5–22.6)
└── Optimization, annealing, metrics

Finance Applications (22.7)
├── Synthetic Data Generation
├── Missing Data Imputation
└── Scenario Generation
```

---

## Summary

| Concept | Key Insight |
|---------|-------------|
| **Autoencoders** | Learn compressed representations but lack principled generation |
| **VAE innovation** | Probabilistic encoder + KL regularization = generative model |
| **ELBO** | Training objective that lower-bounds the log-likelihood |
| **Finance relevance** | Uncertainty quantification, factor discovery, scenario generation |

---

## What's Next

The next section covers [Latent Variable Models](latent_variable.md), where we establish the probabilistic foundations and motivation for variational inference.
