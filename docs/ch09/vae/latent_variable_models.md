# Latent Variable Models

Understanding the probabilistic foundation of Variational Autoencoders.

---

## Learning Objectives

By the end of this section, you will be able to:

- Explain what latent variables are and why they are useful
- Describe the generative model framework
- Understand why exact inference is intractable
- Motivate the need for variational approximations

---

## What Are Latent Variables?

### Definition

A **latent variable** is an unobserved (hidden) variable that explains patterns in observed data. The term "latent" comes from Latin *latēre*, meaning "to lie hidden."

### Intuition

Consider a dataset of handwritten digits. We observe pixel values $x$, but there are underlying factors we don't directly observe:

- **Digit identity** (0-9)
- **Writing style** (slant, thickness)
- **Scale and position**

These hidden factors are **latent variables** $z$ that generate the observed data.

```
Latent Variables (z)          Observed Data (x)
┌─────────────────┐           ┌─────────────────┐
│  digit = "7"    │           │  ░░░░░▓▓▓░░░░░  │
│  slant = 15°    │  ──────►  │  ░░░░░▓░░░░░░░  │
│  thickness = 2  │           │  ░░░░▓░░░░░░░░  │
│  scale = 1.2    │           │  ░░░▓░░░░░░░░░  │
└─────────────────┘           └─────────────────┘
     Hidden                        Visible
```

---

## The Generative Model Perspective

### Forward Process: How Data Is Generated

A latent variable model assumes data is generated through a two-step process:

1. **Sample latent code:** $z \sim p(z)$ (prior distribution)
2. **Generate observation:** $x \sim p_\theta(x|z)$ (likelihood/decoder)

The joint distribution is:

$$p_\theta(x, z) = p_\theta(x|z) \cdot p(z)$$

### VAE Generative Model

In a VAE, we make specific choices:

| Component | Choice | Interpretation |
|-----------|--------|----------------|
| **Prior** $p(z)$ | $\mathcal{N}(0, I)$ | Standard Gaussian |
| **Likelihood** $p_\theta(x\|z)$ | Neural network decoder | Maps latent to data |

The decoder network $g_\theta$ parameterizes the likelihood:

$$p_\theta(x|z) = \mathcal{N}(x; g_\theta(z), \sigma^2 I) \quad \text{or} \quad \text{Bernoulli}(x; g_\theta(z))$$

### The Marginal Likelihood

The probability of observing data $x$ under our model:

$$p_\theta(x) = \int p_\theta(x|z) p(z) dz$$

This integrates over all possible latent codes that could have generated $x$.

---

## Why Latent Variable Models?

### Benefits

| Advantage | Explanation |
|-----------|-------------|
| **Interpretability** | Latent dimensions may capture meaningful factors |
| **Generation** | Sample $z \sim p(z)$, then decode to generate new data |
| **Compression** | Latent space is typically lower-dimensional |
| **Disentanglement** | With proper training, factors become independent |

### Applications

- **Image Generation:** Generate new faces, digits, objects
- **Representation Learning:** Learn useful features for downstream tasks
- **Anomaly Detection:** Unusual data has low likelihood
- **Data Compression:** Store latent codes instead of raw data

---

## The Inference Problem

### What We Want

Given observed data $x$, we want the **posterior distribution**:

$$p_\theta(z|x) = \text{"What latent codes could have generated this data?"}$$

### Bayes' Theorem

By Bayes' theorem:

$$p_\theta(z|x) = \frac{p_\theta(x|z) p(z)}{p_\theta(x)} = \frac{p_\theta(x|z) p(z)}{\int p_\theta(x|z') p(z') dz'}$$

### The Intractability Problem

The denominator $p_\theta(x) = \int p_\theta(x|z') p(z') dz'$ requires integrating over the entire latent space.

**Why is this hard?**

- The integral is over a high-dimensional space
- No closed-form solution for neural network decoders
- Monte Carlo estimation is computationally expensive

**Example:** For a 32-dimensional latent space with neural network decoder, we would need to evaluate the decoder at an astronomically large number of points.

---

## Solutions to Intractability

### Option 1: Approximate the Posterior

Use a simpler distribution $q_\phi(z|x)$ to approximate $p_\theta(z|x)$:

$$q_\phi(z|x) \approx p_\theta(z|x)$$

This is **variational inference** — the approach VAEs take.

### Option 2: Use MCMC

Sample from the posterior using Markov Chain Monte Carlo. Accurate but computationally expensive.

### Option 3: Simplify the Model

Use models where inference is tractable (e.g., mixture of Gaussians with few components). Limited expressiveness.

---

## Variational Inference: The VAE Approach

### The Variational Distribution

VAEs introduce an **encoder network** that outputs an approximate posterior:

$$q_\phi(z|x) = \mathcal{N}(\mu_\phi(x), \sigma_\phi^2(x) \cdot I)$$

where $\mu_\phi(x)$ and $\sigma_\phi(x)$ are neural network outputs.

### Why Gaussian?

| Property | Benefit |
|----------|---------|
| **Closed-form KL** | Easy to compute KL divergence to Gaussian prior |
| **Reparameterization** | Can backpropagate through sampling |
| **Simplicity** | Only need to output mean and variance |

### The Training Objective

Instead of maximizing the intractable $\log p_\theta(x)$, we maximize a **lower bound**:

$$\log p_\theta(x) \geq \underbrace{\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]}_{\text{Reconstruction}} - \underbrace{D_{KL}(q_\phi(z|x) \| p(z))}_{\text{Regularization}}$$

This is the **Evidence Lower Bound (ELBO)**.

---

## VAE Architecture

```
              ENCODER                           DECODER
         (Recognition Model)               (Generative Model)
         
    ┌──────────────────────────┐     ┌──────────────────────────┐
    │                          │     │                          │
x ──│──► Hidden Layers ──► μ  │     │   z ──► Hidden Layers ──►│──► x̂
    │                    ╲    │     │   ▲                      │
    │                     ╲   │     │   │                      │
    │                      z ─│─────│───┘                      │
    │                     ╱   │     │                          │
    │                    ╱    │     │                          │
    │              ──► σ     │     │                          │
    │                          │     │                          │
    └──────────────────────────┘     └──────────────────────────┘
    
         q_φ(z|x)                         p_θ(x|z)
    "Infer latent code"             "Generate from code"
```

---

## Comparison with Related Models

### vs. Standard Autoencoders

| Aspect | Autoencoder | VAE |
|--------|-------------|-----|
| **Latent space** | Deterministic point | Probabilistic distribution |
| **Training** | Reconstruction only | ELBO (reconstruction + KL) |
| **Generation** | Cannot sample meaningfully | Sample from prior |
| **Interpolation** | May have gaps | Smooth, continuous |

### vs. GANs

| Aspect | VAE | GAN |
|--------|-----|-----|
| **Training** | Maximize likelihood bound | Minimax game |
| **Inference** | Has encoder | No encoder (typically) |
| **Samples** | Often blurry | Sharp but may lack diversity |
| **Likelihood** | Can evaluate (bound) | Cannot evaluate |

### vs. Normalizing Flows

| Aspect | VAE | Normalizing Flow |
|--------|-----|------------------|
| **Posterior** | Approximate | Exact |
| **Likelihood** | Lower bound | Exact |
| **Architecture** | Flexible | Must be invertible |
| **Training** | Generally easier | Requires careful design |

---

## Key Terminology

| Term | Definition |
|------|------------|
| **Latent variable** | Unobserved variable that explains data |
| **Prior** $p(z)$ | Distribution over latent space before seeing data |
| **Likelihood** $p(x\|z)$ | Distribution of data given latent code |
| **Posterior** $p(z\|x)$ | Distribution of latent code given data |
| **Variational distribution** $q(z\|x)$ | Approximate posterior |
| **ELBO** | Evidence Lower Bound; training objective |

---

## Mathematical Notation Reference

| Symbol | Meaning |
|--------|---------|
| $x$ | Observed data |
| $z$ | Latent variable |
| $\theta$ | Decoder (generative model) parameters |
| $\phi$ | Encoder (inference model) parameters |
| $p_\theta(x\|z)$ | Decoder distribution |
| $p(z)$ | Prior distribution |
| $q_\phi(z\|x)$ | Encoder distribution (approximate posterior) |

---

## Summary

| Concept | Key Insight |
|---------|-------------|
| **Latent variables** | Hidden factors that generate observed data |
| **Generative model** | $p(x,z) = p(x\|z)p(z)$ — how data is created |
| **Posterior inference** | Finding $p(z\|x)$ — intractable for neural networks |
| **Variational inference** | Approximate with $q_\phi(z\|x)$ and optimize ELBO |
| **VAE architecture** | Encoder outputs distribution parameters; decoder generates |

---

## Exercises

### Exercise 1: Generative Process

Write pseudocode for generating a batch of 100 samples from a trained VAE.

### Exercise 2: Latent Dimensionality

For MNIST digits (28×28 = 784 dimensions), why might we choose a latent dimension of 10-32? What are the trade-offs of larger vs. smaller latent spaces?

### Exercise 3: Intractability

Explain why $\int p_\theta(x|z) p(z) dz$ is intractable when $p_\theta(x|z)$ is parameterized by a neural network, even though both $p_\theta(x|z)$ and $p(z)$ have known forms.

---

## What's Next

The next section covers the information-theoretic foundations that underpin VAEs, including entropy, KL divergence, and mutual information.
