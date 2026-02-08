# Reparameterization Trick

Making stochastic sampling differentiable for backpropagation.

---

## Learning Objectives

By the end of this section, you will be able to:

- Explain why sampling blocks gradients
- Implement the reparameterization trick
- Handle numerical stability considerations
- Apply reparameterization in VAE training

---

## The Problem: Backpropagating Through Sampling

### Standard Autoencoder Flow

In a standard autoencoder, gradients flow smoothly:

$$x \xrightarrow{\nabla} h \xrightarrow{\nabla} z \xrightarrow{\nabla} h' \xrightarrow{\nabla} \hat{x} \xrightarrow{\nabla} \mathcal{L}$$

Every operation is deterministic and differentiable.

### VAE with Naive Sampling

In a VAE, we sample from the encoder's distribution:

$$x \xrightarrow{\nabla} h \xrightarrow{\nabla} (\mu, \sigma) \xrightarrow{\color{red}{\times}} z \sim \mathcal{N}(\mu, \sigma^2) \xrightarrow{\nabla} \hat{x} \xrightarrow{\nabla} \mathcal{L}$$

**Problem:** The sampling operation $z \sim \mathcal{N}(\mu, \sigma^2)$ is **stochastic** and **not differentiable**!

---

## The Solution: Reparameterization

Instead of sampling directly, we write:

$$z = \mu + \sigma \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0,1)$$

---

## What's Next

The next section covers the **[KL Divergence Term](kl_term.md)**, which regularizes the latent distribution and completes the ELBO formulation.
