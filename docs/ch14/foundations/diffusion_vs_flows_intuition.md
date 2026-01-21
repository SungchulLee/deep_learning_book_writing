# Diffusion Models vs Normalizing Flows

A structured comparison of **Diffusion Models** and **Normalizing Flows**, focusing on their mathematical foundations, practical tradeoffs, and when to use each approach.

## Overview

| Aspect | Diffusion Models | Normalizing Flows |
|--------|------------------|-------------------|
| **Core Idea** | Learn to reverse a gradual noise process | Learn an invertible mapping to a simple distribution |
| **Generative Mechanism** | Iterative denoising steps | Single pass through invertible function |
| **Mathematical Foundation** | SDEs / discrete Markov chains | Change-of-variables with tractable Jacobian |

## Key Architectural Difference

### Normalizing Flows: Invertible by Design

Normalizing flows use **invertible neural network layers** with tractable Jacobian determinants:

$$x = f_\theta(z), \quad z = f_\theta^{-1}(x)$$
$$\log p(x) = \log p(z) - \log |\det J_f|$$

Common architectures:
- **Coupling layers** (RealNVP, Glow)
- **Autoregressive flows** (MAF, IAF)
- **Residual flows** (with spectral normalization)

**Constraint**: Every layer must be invertible with computable Jacobian.

### Diffusion Models: No Invertibility Constraint

Diffusion models learn the **reverse of a fixed forward process**:

$$\text{Forward: } x_0 \to x_1 \to \cdots \to x_T \quad \text{(add noise)}$$
$$\text{Reverse: } x_T \to x_{T-1} \to \cdots \to x_0 \quad \text{(denoise)}$$

The neural network $\epsilon_\theta(x_t, t)$ can be **any architecture** (typically U-Net):
- No invertibility requirement
- No Jacobian computation
- More expressive function classes allowed

## Backward Direction Comparison

### Flows: Exact Backward

Given an image $x$, a flow can compute the **exact** latent code:
$$z = f_\theta^{-1}(x)$$

This enables:
- Exact likelihood computation
- Deterministic encoding/decoding
- Interpolation in latent space

### Diffusion: Backward in Distribution

Diffusion models go backward **in distribution**, not in individual samples:

$$p_\theta(x_{t-1} | x_t) \approx q(x_{t-1} | x_t, x_0)$$

The reverse process is:
- **Stochastic** (DDPM) or **deterministic** (DDIM)
- Approximate (learned)
- Defined over the marginal distributions, not individual trajectories

**Key insight**: Diffusion models learn to match the *distribution* $q(x_{t-1}|x_t)$, not to invert a specific function.

## Advantages

### Diffusion Models

1. **High Sample Quality**
   - State-of-the-art in image, audio, video synthesis
   - Often outperform GANs in fidelity and diversity

2. **Stable Training**
   - Simple regression loss (predict noise)
   - No adversarial dynamics

3. **Expressivity**
   - No architectural constraints
   - Can use any neural network (U-Net, Transformer, etc.)

4. **Flexible Conditioning**
   - Easy to add text, class, or image conditioning
   - Classifier-free guidance for quality control

### Normalizing Flows

1. **Exact Likelihood**
   - Tractable $\log p(x)$ via change-of-variables
   - Useful for density estimation, anomaly detection

2. **Fast Sampling**
   - Single forward pass: $x = f_\theta(z)$
   - No iterative refinement needed

3. **Invertibility**
   - Exact encoding: $z = f_\theta^{-1}(x)$
   - Interpretable latent space

4. **Training Simplicity**
   - Standard maximum likelihood
   - No noise schedules or iterative sampling

## Limitations

### Diffusion Models

1. **Slow Sampling**
   - Hundreds/thousands of denoising steps
   - Mitigated by DDIM, distillation, consistency models

2. **Implicit Likelihood**
   - Cannot compute exact $\log p(x)$
   - ELBO provides lower bound

3. **Compute Intensive**
   - Long training and sampling times
   - Heavy GPU usage

### Normalizing Flows

1. **Limited Expressivity**
   - Invertibility constraint limits function class
   - Harder to model complex distributions

2. **Architectural Constraints**
   - Must maintain tractable Jacobian
   - Coupling layers, autoregressive structure required

3. **Training Instability on Complex Data**
   - Struggles with high-dimensional, multimodal data
   - Often requires very deep networks

4. **Parameter Inefficiency**
   - Need many parameters to match diffusion quality
   - Jacobian computation adds overhead

## Probability Flow Connection

Interestingly, diffusion models have a **probability flow ODE** that connects them to flows:

$$\frac{dx}{dt} = f(x,t) - \frac{1}{2}g(t)^2 \nabla_x \log p_t(x)$$

This ODE has the same marginal distributions as the SDE but is **deterministic**. This reveals:

- Diffusion models implicitly define a flow
- The flow is more expressive (time-dependent)
- Enables exact likelihood computation (with cost)

## When to Use Each

| Use Case | Recommended |
|----------|-------------|
| High-fidelity image/audio generation | **Diffusion** |
| Fast sampling / real-time applications | **Flows** |
| Density estimation / anomaly detection | **Flows** |
| Text-to-image / conditional synthesis | **Diffusion** |
| Interpretable latent manipulation | **Flows** |
| Research with limited compute | **Flows** |
| State-of-the-art quality (with compute) | **Diffusion** |

## Popularity in Practice

**Current landscape** (as of 2024):
- **Diffusion models dominate** for generative tasks (Stable Diffusion, DALL-E, Midjourney)
- **Flows remain important** for:
  - Density estimation
  - Audio (WaveGlow, some TTS)
  - Scientific applications (molecular generation)
  - Variational inference

**Why diffusion won for images**:
1. Quality gap is significant
2. Slow sampling is acceptable for many applications
3. Conditioning is more flexible
4. Architecture freedom enables better scaling

## Summary Table

| Property | Diffusion | Normalizing Flows |
|----------|-----------|-------------------|
| Sample quality | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Sampling speed | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| Training stability | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Exact likelihood | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| Architecture flexibility | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| Conditioning ease | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

## Navigation

- **Previous**: [What is a Diffusion Model?](overview.md)
- **Next**: [Score Function Definition](../score_based/score_function.md)
