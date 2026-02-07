# Generative vs Discriminative Models

Understanding the two fundamental paradigms in probabilistic machine learning and where VAEs fit.

---

## Learning Objectives

By the end of this section, you will be able to:

- Distinguish between generative and discriminative modeling approaches
- Explain why generative models are necessary for tasks beyond prediction
- Describe the taxonomy of generative models and position VAEs within it
- Articulate the trade-offs between different generative model families

---

## The Two Paradigms

### Discriminative Models

**Discriminative models** learn the conditional distribution $p(y|x)$ directly. They answer: "Given this input, what is the output?"

$$\text{Discriminative:} \quad p(y|x) = \frac{p(x|y)p(y)}{p(x)} \quad \text{(modeled directly)}$$

Examples include logistic regression, neural network classifiers, and support vector machines. These models excel at prediction tasks but cannot generate new data or evaluate the likelihood of observations.

### Generative Models

**Generative models** learn the joint distribution $p(x, y)$ or the data distribution $p(x)$ itself. They answer: "How is this data generated?"

$$\text{Generative:} \quad p(x, y) = p(x|y)p(y) \quad \text{or} \quad p(x) = \int p(x|z)p(z)dz$$

These models can generate new samples, evaluate likelihoods, and perform inference on latent variables.

### Comparison

| Aspect | Discriminative | Generative |
|--------|---------------|------------|
| **Models** | $p(y\|x)$ | $p(x)$ or $p(x, y)$ |
| **Prediction** | Direct, often more accurate | Via Bayes' rule |
| **Generation** | Not possible | Sample from $p(x)$ |
| **Likelihood** | Only conditional | Full data likelihood |
| **Missing data** | Requires complete inputs | Can handle naturally |
| **Unsupervised** | Not applicable | Core use case |

---

## Taxonomy of Generative Models

### The Generative Model Landscape

```
                        Generative Models
                              │
              ┌───────────────┼───────────────┐
              │               │               │
        Explicit           Implicit       Hybrid
        Density            Density
              │               │               │
     ┌────────┤          ┌────┤          ┌────┤
     │        │          │    │          │    │
  Tractable  Approximate GAN  ...     VAE-GAN ...
  Density    Density
     │        │
     │    ┌───┤
     │    │   │
  Flows  VAE  Diffusion
```

### Explicit Density Models

These models define and optimize an explicit probability distribution over the data.

**Tractable density** models allow exact likelihood computation. Autoregressive models (PixelCNN, WaveNet) decompose $p(x) = \prod_i p(x_i | x_{<i})$ into a product of conditionals. Normalizing flows use invertible transformations to map a simple base distribution to a complex data distribution with exact likelihood via the change-of-variables formula.

**Approximate density** models optimize a bound on the likelihood rather than the likelihood itself. VAEs maximize the Evidence Lower Bound (ELBO), which lower-bounds $\log p(x)$. Diffusion models learn to reverse a gradual noising process.

### Implicit Density Models

Generative Adversarial Networks (GANs) learn to generate samples without explicitly modeling the density. A generator $G$ maps noise $z$ to data space, and a discriminator $D$ distinguishes real from generated samples. GANs produce sharp samples but cannot evaluate likelihoods.

---

## Where VAEs Fit

### VAEs as Approximate Density Models

VAEs occupy a unique position in the generative model landscape. They are latent variable models that use variational inference to approximate an intractable posterior:

$$\log p_\theta(x) \geq \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) \| p(z)) = \mathcal{L}(\theta, \phi; x)$$

This provides a **lower bound** on the log-likelihood, not the exact value. The gap between the bound and the true log-likelihood equals $D_{KL}(q_\phi(z|x) \| p_\theta(z|x))$, which measures how well the encoder approximates the true posterior.

### VAE Strengths and Weaknesses

| Strength | Weakness |
|----------|----------|
| Principled probabilistic framework | Approximate (not exact) likelihood |
| Encoder provides inference | Samples often blurrier than GANs |
| Stable training | ELBO gap may be large |
| Smooth, structured latent space | Posterior collapse can occur |
| Natural uncertainty quantification | Gaussian assumption may be limiting |

---

## Generative Models Compared

### VAE vs GAN

| Aspect | VAE | GAN |
|--------|-----|-----|
| **Training** | Maximize ELBO | Minimax game |
| **Stability** | Stable | Can be unstable |
| **Inference** | Has encoder | No encoder (typically) |
| **Samples** | Often blurry | Sharp but may lack diversity |
| **Likelihood** | Lower bound available | Cannot evaluate |
| **Mode coverage** | Covers all modes | May have mode collapse |

### VAE vs Normalizing Flows

| Aspect | VAE | Normalizing Flow |
|--------|-----|------------------|
| **Posterior** | Approximate | Exact (if used as posterior) |
| **Likelihood** | Lower bound | Exact |
| **Architecture** | Flexible | Must be invertible |
| **Latent space** | Lower-dimensional | Same dimensionality |
| **Training** | Generally easier | Requires careful design |

### VAE vs Diffusion Models

| Aspect | VAE | Diffusion Model |
|--------|-----|-----------------|
| **Latent space** | Learned, low-dimensional | Fixed noise schedule |
| **Generation speed** | Single forward pass | Many denoising steps |
| **Sample quality** | Moderate | State-of-the-art |
| **Likelihood** | ELBO | ELBO (tighter) |
| **Inference** | Fast (single pass) | Fast (single pass) |

---

## The Generative Approach in Finance

### Why Generative Models for Quant Finance?

In quantitative finance, generative models address fundamental challenges that discriminative models cannot.

**Limited historical data** is a perennial problem. Markets have finite history, regime changes are rare, and extreme events are scarce. Generative models can synthesize plausible scenarios beyond what has been observed.

**Risk assessment** requires understanding tail behavior and joint distributions, not just conditional predictions. A generative model of asset returns enables coherent scenario analysis across the entire portfolio.

**Regulatory requirements** increasingly demand stress testing under hypothetical scenarios. Generative models provide a principled way to create these scenarios while maintaining realistic cross-asset dependencies.

### VAE Advantages for Finance

VAEs are particularly suited to financial applications because the latent space can encode interpretable market factors, the encoder enables rapid inference on new data, conditional variants allow regime-specific generation, and the probabilistic framework provides natural uncertainty estimates.

---

## Summary

| Concept | Key Insight |
|---------|-------------|
| **Discriminative models** | Learn $p(y\|x)$; excel at prediction |
| **Generative models** | Learn $p(x)$; enable generation, density estimation |
| **VAE position** | Approximate density model with latent variables |
| **Key trade-off** | VAEs trade sample sharpness for inference, stability, and coverage |

---

## What's Next

Having established the generative modeling context, the next section dives into the [ELBO Derivation](../theory/elbo_derivation.md), the mathematical core of VAE training.
