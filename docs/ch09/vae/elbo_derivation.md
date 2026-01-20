# Evidence Lower Bound (ELBO) Derivation

The mathematical foundation of variational inference and VAE training.

---

## Learning Objectives

By the end of this section, you will be able to:

- Derive the ELBO from first principles using multiple approaches
- Explain the relationship between ELBO and log-likelihood
- Connect ELBO maximization to the EM algorithm
- Understand why ELBO provides a tractable training objective

---

## The Problem: Intractable Likelihood

### Goal: Maximum Likelihood

We want to learn a generative model $p_\theta(x)$ that maximizes the likelihood of observed data:

$$\max_\theta \log p_\theta(x)$$

### The Latent Variable Complication

With latent variables $z$, the likelihood involves an integral:

$$p_\theta(x) = \int p_\theta(x, z) dz = \int p_\theta(x|z) p(z) dz$$

**Problem:** This integral is intractable for neural network decoders.

### Solution: Variational Lower Bound

Instead of maximizing $\log p_\theta(x)$ directly, we maximize a **lower bound** — the ELBO.

---

## Derivation 1: Jensen's Inequality

### Step 1: Introduce Variational Distribution

Multiply and divide by a variational distribution $q_\phi(z|x)$:

$$\log p_\theta(x) = \log \int p_\theta(x, z) dz = \log \int q_\phi(z|x) \frac{p_\theta(x, z)}{q_\phi(z|x)} dz$$

### Step 2: Recognize as Expectation

$$= \log \mathbb{E}_{q_\phi(z|x)}\left[\frac{p_\theta(x, z)}{q_\phi(z|x)}\right]$$

### Step 3: Apply Jensen's Inequality

For concave function $\log$:

$$\log \mathbb{E}[Y] \geq \mathbb{E}[\log Y]$$

Therefore:

$$\log p_\theta(x) \geq \mathbb{E}_{q_\phi(z|x)}\left[\log \frac{p_\theta(x, z)}{q_\phi(z|x)}\right]$$

### Step 4: Define ELBO

$$\mathcal{L}(\theta, \phi; x) = \mathbb{E}_{q_\phi(z|x)}\left[\log \frac{p_\theta(x, z)}{q_\phi(z|x)}\right]$$

This is the **Evidence Lower Bound (ELBO)**.

---

## Derivation 2: KL Divergence Decomposition

### Step 1: Start with KL Divergence

Consider the KL divergence from approximate to true posterior:

$$D_{KL}(q_\phi(z|x) \| p_\theta(z|x)) = \mathbb{E}_{q_\phi}\left[\log \frac{q_\phi(z|x)}{p_\theta(z|x)}\right]$$

### Step 2: Expand True Posterior

Using Bayes' theorem: $p_\theta(z|x) = \frac{p_\theta(x,z)}{p_\theta(x)}$

$$D_{KL}(q \| p) = \mathbb{E}_{q_\phi}\left[\log \frac{q_\phi(z|x) \cdot p_\theta(x)}{p_\theta(x,z)}\right]$$

$$= \mathbb{E}_{q_\phi}\left[\log q_\phi(z|x) - \log p_\theta(x,z) + \log p_\theta(x)\right]$$

### Step 3: Separate Terms

$$= \mathbb{E}_{q_\phi}\left[\log q_\phi(z|x) - \log p_\theta(x,z)\right] + \log p_\theta(x)$$

Note: $\log p_\theta(x)$ doesn't depend on $z$, so it comes out of the expectation.

### Step 4: Rearrange

$$\log p_\theta(x) = D_{KL}(q_\phi(z|x) \| p_\theta(z|x)) + \mathbb{E}_{q_\phi}\left[\log p_\theta(x,z) - \log q_\phi(z|x)\right]$$

$$\log p_\theta(x) = D_{KL}(q_\phi(z|x) \| p_\theta(z|x)) + \mathcal{L}(\theta, \phi; x)$$

### The Fundamental Identity

$$\boxed{\log p_\theta(x) = \mathcal{L}_{\text{ELBO}}(\theta, \phi; x) + D_{KL}(q_\phi(z|x) \| p_\theta(z|x))}$$

Since $D_{KL} \geq 0$:

$$\log p_\theta(x) \geq \mathcal{L}_{\text{ELBO}}(\theta, \phi; x)$$

---

## Standard VAE ELBO Form

### Expanding the ELBO

Starting from:

$$\mathcal{L} = \mathbb{E}_{q_\phi(z|x)}\left[\log \frac{p_\theta(x, z)}{q_\phi(z|x)}\right]$$

Expand $p_\theta(x, z) = p_\theta(x|z) p(z)$:

$$= \mathbb{E}_{q_\phi}\left[\log p_\theta(x|z) + \log p(z) - \log q_\phi(z|x)\right]$$

$$= \mathbb{E}_{q_\phi}\left[\log p_\theta(x|z)\right] + \mathbb{E}_{q_\phi}\left[\log \frac{p(z)}{q_\phi(z|x)}\right]$$

$$= \mathbb{E}_{q_\phi}\left[\log p_\theta(x|z)\right] - D_{KL}(q_\phi(z|x) \| p(z))$$

### The Standard Form

$$\boxed{\mathcal{L}_{\text{ELBO}} = \underbrace{\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]}_{\text{Reconstruction Term}} - \underbrace{D_{KL}(q_\phi(z|x) \| p(z))}_{\text{KL Regularization}}}$$

---

## Interpretation of ELBO Terms

### Term 1: Reconstruction

$$\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]$$

**Meaning:** Expected log-likelihood of data $x$ given latent samples from the encoder.

**In practice:**
- **Gaussian decoder:** $-\frac{1}{2\sigma^2}\|x - \hat{x}\|^2 + \text{const} \propto -\text{MSE}$
- **Bernoulli decoder:** $\sum_i [x_i \log \hat{x}_i + (1-x_i)\log(1-\hat{x}_i)] = -\text{BCE}$

### Term 2: KL Regularization

$$D_{KL}(q_\phi(z|x) \| p(z))$$

**Meaning:** How different is the encoder output from the prior?

**Effect:** Encourages encoder to produce distributions close to $\mathcal{N}(0, I)$:
- Keeps latent space organized
- Enables meaningful sampling from prior
- Acts as regularization to prevent overfitting

---

## ELBO and Maximum Likelihood

### The Relationship

From the fundamental identity:

$$\log p_\theta(x) = \mathcal{L}_{\text{ELBO}} + D_{KL}(q_\phi(z|x) \| p_\theta(z|x))$$

### When Is ELBO Tight?

The bound is tight (equality holds) when:

$$q_\phi(z|x) = p_\theta(z|x)$$

That is, when the approximate posterior equals the true posterior.

### Implications

1. **Maximizing ELBO w.r.t. $\phi$:** Minimizes $D_{KL}(q_\phi \| p_\theta)$, making $q$ closer to true posterior
2. **Maximizing ELBO w.r.t. $\theta$:** Increases $\log p_\theta(x)$ (at least by the ELBO amount)
3. **Joint optimization:** Simultaneously improves model and inference

---

## ELBO and the EM Algorithm

### EM Also Uses ELBO!

In the EM algorithm for latent variable models:

**E-step:** Set $q(z) = p_\theta(z|x)$ (exact posterior) → maximizes ELBO over $q$

**M-step:** Maximize ELBO over $\theta$:
$$\theta^{\text{new}} = \arg\max_\theta \mathbb{E}_{q}[\log p_\theta(x,z)]$$

### VAE vs. EM

| Aspect | EM Algorithm | VAE |
|--------|--------------|-----|
| **E-step** | Exact posterior | Approximate $q_\phi(z\|x)$ |
| **Posterior** | Computed exactly | Parameterized by neural net |
| **Updates** | Alternating coordinate | Joint gradient descent |
| **Guarantee** | Monotonic likelihood increase | No monotonic guarantee |
| **Inference** | Per data point | Amortized across all data |

### Variational EM Interpretation

VAE training can be viewed as **amortized variational EM**:

- "E-step-like": Update $\phi$ to make $q_\phi(z|x)$ closer to true posterior
- "M-step-like": Update $\theta$ to improve reconstruction

But both happen simultaneously via gradient descent!

---

## Why ELBO Works for Training

### Tractability

Unlike $\log p_\theta(x)$, the ELBO:
1. Has a closed-form KL term (for Gaussians)
2. Can be estimated via Monte Carlo (for reconstruction term)
3. Allows gradient computation via reparameterization

### Monte Carlo Estimate

$$\mathcal{L} \approx \frac{1}{L}\sum_{l=1}^{L}\log p_\theta(x|z^{(l)}) - D_{KL}(q_\phi(z|x) \| p(z))$$

where $z^{(l)} \sim q_\phi(z|x)$.

In practice, $L=1$ sample often suffices!

### Gradient Estimation

Using the reparameterization trick:

$$z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

Gradients flow through the deterministic $\mu$ and $\sigma$ functions.

---

## The ELBO Gap

### Quantifying Approximation Quality

The gap between log-likelihood and ELBO:

$$\text{Gap} = \log p_\theta(x) - \mathcal{L}_{\text{ELBO}} = D_{KL}(q_\phi(z|x) \| p_\theta(z|x))$$

### Factors Affecting the Gap

| Factor | Effect on Gap |
|--------|---------------|
| **Expressive $q_\phi$** | Smaller gap (better approximation) |
| **Complex true posterior** | Larger gap (harder to match) |
| **Multimodal $p_\theta(z\|x)$** | Gap inevitable with unimodal $q$ |

### Tightening the Bound

Methods to reduce the gap:
1. **Richer $q$ families:** Normalizing flows, autoregressive
2. **Importance weighting:** IWAE objective
3. **Hierarchical models:** Multiple stochastic layers

---

## Practical VAE Loss

### Final Training Objective

Minimize the negative ELBO:

$$\mathcal{L}_{\text{VAE}} = -\mathcal{L}_{\text{ELBO}} = \underbrace{-\mathbb{E}_{q}[\log p_\theta(x|z)]}_{\text{Reconstruction Loss}} + \underbrace{D_{KL}(q_\phi(z|x) \| p(z))}_{\text{KL Loss}}$$

### Concrete Loss Functions

**For Gaussian decoder (continuous data):**
$$\mathcal{L}_{\text{VAE}} = \frac{1}{2\sigma^2}\|x - \hat{x}\|^2 + \frac{1}{2}\sum_j(\mu_j^2 + \sigma_j^2 - 1 - \log\sigma_j^2)$$

**For Bernoulli decoder (binary data):**
$$\mathcal{L}_{\text{VAE}} = -\sum_i[x_i\log\hat{x}_i + (1-x_i)\log(1-\hat{x}_i)] + \frac{1}{2}\sum_j(\mu_j^2 + \sigma_j^2 - 1 - \log\sigma_j^2)$$

### PyTorch Implementation

```python
import torch
import torch.nn.functional as F

def elbo_loss(recon_x, x, mu, logvar, beta=1.0):
    """
    Compute negative ELBO (VAE loss).
    
    Args:
        recon_x: Reconstructed data from decoder
        x: Original data
        mu: Encoder mean
        logvar: Encoder log-variance
        beta: Weight for KL term (β-VAE)
    
    Returns:
        loss, reconstruction_loss, kl_loss
    """
    # Reconstruction loss
    # Option 1: BCE for binary data
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # Option 2: MSE for continuous data
    # recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL divergence: -0.5 * sum(1 + log(σ²) - μ² - σ²)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss
```

---

## Summary of Key Equations

| Concept | Equation |
|---------|----------|
| **ELBO Definition** | $\mathcal{L} = \mathbb{E}_q[\log p(x,z)] - \mathbb{E}_q[\log q(z\|x)]$ |
| **Standard Form** | $\mathcal{L} = \mathbb{E}_q[\log p(x\|z)] - D_{KL}(q(z\|x) \| p(z))$ |
| **Fundamental Identity** | $\log p(x) = \mathcal{L} + D_{KL}(q(z\|x) \| p(z\|x))$ |
| **Lower Bound** | $\log p(x) \geq \mathcal{L}$ |
| **Gaussian KL** | $D_{KL} = -\frac{1}{2}\sum_j(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2)$ |

---

## Exercises

### Exercise 1: Jensen's Inequality

Show that Jensen's inequality becomes equality when $q(z|x)$ is a point mass at the optimal $z^*$.

### Exercise 2: ELBO Computation

For a 1D example:
- $p(z) = \mathcal{N}(0, 1)$
- $p(x|z) = \mathcal{N}(z, 0.5)$
- $q(z|x) = \mathcal{N}(x/1.5, 0.33)$

Compute the ELBO for $x = 1$.

### Exercise 3: EM vs. VAE

Explain why VAE training doesn't guarantee monotonic improvement in the ELBO, unlike EM.

### Exercise 4: ELBO Decomposition

Show that the ELBO can be written as:

$$\mathcal{L} = \log p(x) - D_{KL}(q(z|x) \| p(z|x))$$

What does this tell us about the relationship between ELBO and log-likelihood?

---

## What's Next

The next section examines the reconstruction term in detail, including different likelihood choices and their implications.
