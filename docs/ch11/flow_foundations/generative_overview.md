# Generative Modeling Overview

## Introduction

Generative models learn to create new data samples that resemble a training distribution. Unlike discriminative models (which learn $p(y|x)$), generative models learn the data distribution $p(x)$ itself, enabling both sampling and density evaluation.

## The Generative Modeling Problem

### Goal

Given samples $\{x_1, x_2, \ldots, x_N\}$ from unknown distribution $p_{\text{data}}(x)$:

1. **Density Estimation**: Learn $p_\theta(x) \approx p_{\text{data}}(x)$
2. **Sampling**: Generate new samples $x_{\text{new}} \sim p_\theta(x)$

### Key Capabilities

| Capability | Description | Applications |
|------------|-------------|--------------|
| **Sampling** | Generate new data points | Image synthesis, data augmentation |
| **Density Evaluation** | Compute $p(x)$ for any $x$ | Anomaly detection, model comparison |
| **Latent Representation** | Encode data to latent space | Compression, interpolation |
| **Conditional Generation** | Sample $p(x|c)$ given context | Text-to-image, translation |

## Taxonomy of Generative Models

### By Likelihood Treatment

```
Generative Models
├── Explicit Density
│   ├── Tractable Density
│   │   ├── Autoregressive Models (PixelCNN, WaveNet)
│   │   └── Normalizing Flows ← This chapter
│   └── Approximate Density
│       └── Variational Autoencoders (VAE)
└── Implicit Density
    └── Generative Adversarial Networks (GAN)
```

### Comparison

| Model | Exact Likelihood | Fast Sampling | Latent Space | Training Stability |
|-------|-----------------|---------------|--------------|-------------------|
| **Normalizing Flows** | ✓ | ✓ | ✓ (bijective) | ✓ |
| **VAE** | ✗ (ELBO) | ✓ | ✓ | ✓ |
| **GAN** | ✗ | ✓ | ✓ | ✗ (mode collapse) |
| **Autoregressive** | ✓ | ✗ (sequential) | ✗ | ✓ |
| **Diffusion** | ✗ (approx) | ✗ (iterative) | ✓ | ✓ |

## Normalizing Flows: Core Idea

### The Transformation Approach

Normalizing flows transform a **simple base distribution** (e.g., Gaussian) into a **complex data distribution** through a sequence of invertible transformations.

```
Simple Distribution          Complex Distribution
     z ~ N(0, I)     ──f──>      x ~ p_data
     
     Gaussian            Invertible        Real Data
     (easy to sample)    Transform         (complex)
```

### Why "Normalizing Flow"?

- **Normalizing**: The transformation "normalizes" complex data back to a simple (normal) distribution
- **Flow**: The transformation can be viewed as a continuous flow of probability mass

### Mathematical Foundation

If $z \sim p_Z(z)$ and $x = f(z)$ where $f$ is invertible:

$$p_X(x) = p_Z(f^{-1}(x)) \left| \det \frac{\partial f^{-1}}{\partial x} \right|$$

This is the **change of variables formula** - the heart of normalizing flows.

## Advantages of Normalizing Flows

### 1. Exact Likelihood Computation

Unlike VAEs (which optimize a lower bound) or GANs (which have no likelihood), flows compute exact log-likelihoods:

$$\log p(x) = \log p_Z(z) + \log \left| \det J \right|$$

**Benefits**:
- Rigorous model comparison
- Proper uncertainty quantification
- Direct maximum likelihood training

### 2. Efficient Sampling

Single forward pass through the network:

```python
def sample(self, n_samples):
    z = self.base_dist.sample(n_samples)  # Sample Gaussian
    x = self.forward(z)                    # One forward pass
    return x
```

Unlike diffusion models (hundreds of steps) or autoregressive models (sequential generation).

### 3. Bijective Latent Space

Every data point has a unique latent representation:

- **Encoding**: $z = f^{-1}(x)$ (deterministic)
- **Decoding**: $x = f(z)$ (deterministic)

No stochasticity like VAE's encoder, enabling:
- Exact reconstruction
- Meaningful interpolation
- Latent space manipulation

### 4. Stable Training

Direct likelihood maximization without:
- Adversarial dynamics (GAN)
- KL divergence balancing (VAE)
- Score matching approximations (Diffusion)

## Challenges and Trade-offs

### 1. Architectural Constraints

Transformations must be:
- **Invertible**: Limits network architectures
- **Efficient Jacobian**: Can't use arbitrary layers

### 2. Dimensionality Preservation

Input and output dimensions must match:
$$\dim(x) = \dim(z)$$

Unlike VAEs which can have lower-dimensional latents.

### 3. Expressiveness vs. Efficiency

More expressive flows often have higher computational cost:

| Architecture | Expressiveness | Jacobian Cost |
|--------------|---------------|---------------|
| Planar Flow | Low | O(d) |
| Coupling Flow | Medium | O(d) |
| Autoregressive | High | O(d) but sequential |
| Continuous (ODE) | Very High | O(d) per step |

## Applications

### Computer Vision
- Image generation and manipulation
- Super-resolution
- Image compression

### Audio
- Speech synthesis
- Music generation
- Audio compression

### Scientific Computing
- Molecular dynamics
- Physics simulations
- Uncertainty quantification

### Finance (Focus of This Curriculum)
- Return distribution modeling
- Risk measurement (VaR, CVaR)
- Option pricing
- Scenario generation
- Portfolio optimization

## Flow vs. Other Generative Models

### When to Use Flows

✅ **Good fit**:
- Need exact likelihoods (model comparison, anomaly detection)
- Need fast sampling AND density evaluation
- Want deterministic encoding/decoding
- Stable training is priority

❌ **Consider alternatives**:
- Very high-dimensional data (images >256×256): Diffusion often better
- Only need sampling, not density: GAN may suffice
- Need compressed latent space: VAE more natural

### Hybrid Approaches

Flows combine well with other models:
- **Flow + VAE**: Use flow as VAE prior
- **Flow + Diffusion**: Flow matching
- **Flow + GAN**: Flow-GAN hybrids

## Summary

Normalizing flows offer a unique combination:

1. **Exact likelihood** - principled density estimation
2. **Fast sampling** - single forward pass  
3. **Invertible mapping** - bijective latent space
4. **Stable training** - direct MLE optimization

The key trade-off is architectural constraints required for invertibility and efficient Jacobian computation.

## References

1. Papamakarios, G., et al. (2021). Normalizing Flows for Probabilistic Modeling and Inference. *JMLR*.
2. Kobyzev, I., et al. (2020). Normalizing Flows: An Introduction and Review of Current Methods. *TPAMI*.
3. Rezende, D. J., & Mohamed, S. (2015). Variational Inference with Normalizing Flows. *ICML*.
