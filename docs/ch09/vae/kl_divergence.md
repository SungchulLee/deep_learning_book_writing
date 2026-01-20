# KL Divergence

The mathematical measure at the heart of variational inference.

---

## Learning Objectives

By the end of this section, you will be able to:

- Derive KL divergence from first principles
- Compute KL divergence for Gaussian distributions
- Explain why KL divergence appears in the VAE objective
- Implement KL computation efficiently in PyTorch

---

## Definition and Intuition

### Formal Definition

The Kullback-Leibler divergence from distribution $q$ to distribution $p$ is:

$$D_{KL}(p \| q) = \int p(x) \log \frac{p(x)}{q(x)} dx = \mathbb{E}_{p(x)}\left[\log \frac{p(x)}{q(x)}\right]$$

**Read as:** "KL divergence from $q$ to $p$" or "KL of $p$ relative to $q$"

### Intuitive Interpretations

**Interpretation 1: Extra Bits**

If you design a code optimal for $q$ but use it to encode samples from $p$:

$$D_{KL}(p \| q) = \text{Extra bits needed beyond optimal}$$

**Interpretation 2: Evidence Ratio**

$$D_{KL}(p \| q) = \mathbb{E}_{p}\left[\log \frac{p(x)}{q(x)}\right] = \text{Expected log evidence ratio}$$

How much more likely are samples under $p$ than under $q$?

**Interpretation 3: Information Gain**

$$D_{KL}(p \| q) = \text{Information gained when updating from } q \text{ to } p$$

---

## Fundamental Properties

### Non-Negativity (Gibbs' Inequality)

$$D_{KL}(p \| q) \geq 0$$

with equality if and only if $p = q$ almost everywhere.

**Proof using Jensen's inequality:**

$$D_{KL}(p \| q) = -\mathbb{E}_p\left[\log\frac{q(x)}{p(x)}\right] \geq -\log\mathbb{E}_p\left[\frac{q(x)}{p(x)}\right] = -\log\int q(x)dx = -\log(1) = 0$$

### Asymmetry

$$D_{KL}(p \| q) \neq D_{KL}(q \| p) \text{ in general}$$

This is crucial for understanding variational inference!

### Decomposition

$$D_{KL}(p \| q) = H(p, q) - H(p)$$

where $H(p, q)$ is cross-entropy and $H(p)$ is entropy.

### Additivity for Independent Distributions

If $p(x, y) = p(x)p(y)$ and $q(x, y) = q(x)q(y)$:

$$D_{KL}(p(x,y) \| q(x,y)) = D_{KL}(p(x) \| q(x)) + D_{KL}(p(y) \| q(y))$$

---

## Forward vs. Reverse KL

### Forward KL: $D_{KL}(p \| q)$

**Minimizing forward KL** pushes $q$ to cover all modes of $p$:

$$\min_q D_{KL}(p \| q) = \min_q \mathbb{E}_p[-\log q(x)]$$

This is **mean-seeking** — $q$ spreads out to cover $p$.

**Typical use:** Maximum likelihood estimation (we know $p$ from data).

### Reverse KL: $D_{KL}(q \| p)$

**Minimizing reverse KL** pushes $q$ to avoid regions where $p$ is small:

$$\min_q D_{KL}(q \| p) = \min_q \mathbb{E}_q[\log q(x) - \log p(x)]$$

This is **mode-seeking** — $q$ concentrates on one mode of $p$.

**Typical use:** Variational inference (we know $p$ but can't sample from it).

### Visualization

```
True Distribution p(x):     Forward KL q:          Reverse KL q:
(bimodal)                   (covers both modes)    (one mode only)

    ╱╲      ╱╲                  ╱────╲                  ╱╲
   ╱  ╲    ╱  ╲                ╱      ╲                ╱  ╲
  ╱    ╲  ╱    ╲              ╱        ╲              ╱    ╲
─────────────────           ───────────────         ───────────────
```

---

## KL Divergence for Gaussians

### General Formula

For $p = \mathcal{N}(\mu_p, \Sigma_p)$ and $q = \mathcal{N}(\mu_q, \Sigma_q)$:

$$D_{KL}(p \| q) = \frac{1}{2}\left[\log\frac{|\Sigma_q|}{|\Sigma_p|} - d + \text{tr}(\Sigma_q^{-1}\Sigma_p) + (\mu_q - \mu_p)^T\Sigma_q^{-1}(\mu_q - \mu_p)\right]$$

where $d$ is the dimensionality.

### Derivation for Univariate Case

Let $p = \mathcal{N}(\mu_1, \sigma_1^2)$ and $q = \mathcal{N}(\mu_2, \sigma_2^2)$.

$$D_{KL}(p \| q) = \int p(x) \log\frac{p(x)}{q(x)} dx$$

Expanding the log ratio:

$$\log\frac{p(x)}{q(x)} = \log\frac{\sigma_2}{\sigma_1} + \frac{(x-\mu_2)^2}{2\sigma_2^2} - \frac{(x-\mu_1)^2}{2\sigma_1^2}$$

Taking expectation under $p$:

$$D_{KL}(p \| q) = \log\frac{\sigma_2}{\sigma_1} + \frac{\mathbb{E}_p[(x-\mu_2)^2]}{2\sigma_2^2} - \frac{\mathbb{E}_p[(x-\mu_1)^2]}{2\sigma_1^2}$$

Using $\mathbb{E}_p[(x-\mu_1)^2] = \sigma_1^2$ and $\mathbb{E}_p[(x-\mu_2)^2] = \sigma_1^2 + (\mu_1-\mu_2)^2$:

$$D_{KL}(p \| q) = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1-\mu_2)^2}{2\sigma_2^2} - \frac{1}{2}$$

### VAE Special Case: Encoder vs. Prior

For encoder $q_\phi(z|x) = \mathcal{N}(\mu, \sigma^2)$ and prior $p(z) = \mathcal{N}(0, 1)$:

Set $\mu_1 = \mu$, $\sigma_1 = \sigma$, $\mu_2 = 0$, $\sigma_2 = 1$:

$$D_{KL}(q \| p) = \log\frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}$$

$$= -\log\sigma + \frac{\sigma^2 + \mu^2 - 1}{2}$$

$$= -\frac{1}{2}\left(1 + \log\sigma^2 - \mu^2 - \sigma^2\right)$$

**This is the KL term in the VAE loss!**

### Multivariate with Diagonal Covariance

For $q = \mathcal{N}(\mu, \text{diag}(\sigma_1^2, \ldots, \sigma_d^2))$ and $p = \mathcal{N}(0, I)$:

$$D_{KL}(q \| p) = \sum_{j=1}^{d} D_{KL}(\mathcal{N}(\mu_j, \sigma_j^2) \| \mathcal{N}(0, 1))$$

$$= -\frac{1}{2}\sum_{j=1}^{d}\left(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2\right)$$

---

## PyTorch Implementation

### Standard VAE KL Loss

```python
import torch

def kl_divergence_standard(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    KL divergence from q = N(mu, diag(exp(logvar))) to p = N(0, I).
    
    Formula: D_KL = -0.5 * sum(1 + log(σ²) - μ² - σ²)
    
    Args:
        mu: Mean of q [batch_size, latent_dim]
        logvar: Log variance of q [batch_size, latent_dim]
        
    Returns:
        KL divergence, summed over latent dimensions [batch_size]
    """
    # Element-wise: -0.5 * (1 + logvar - mu^2 - exp(logvar))
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    
    # Sum over latent dimensions
    return kl_per_dim.sum(dim=1)


def kl_divergence_batch(mu: torch.Tensor, logvar: torch.Tensor, 
                        reduction: str = 'sum') -> torch.Tensor:
    """
    KL divergence with configurable reduction.
    
    Args:
        mu: Mean [batch_size, latent_dim]
        logvar: Log variance [batch_size, latent_dim]
        reduction: 'sum', 'mean', or 'none'
        
    Returns:
        KL divergence with specified reduction
    """
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    
    if reduction == 'sum':
        return kl.sum()
    elif reduction == 'mean':
        return kl.mean()
    else:  # 'none'
        return kl.sum(dim=1)
```

### Numerical Stability: Why Log-Variance?

```python
# ❌ UNSTABLE: Using variance directly
def kl_unstable(mu, var):
    # Problem: var can become 0 or very large
    return -0.5 * (1 + torch.log(var) - mu**2 - var)

# ✅ STABLE: Using log-variance
def kl_stable(mu, logvar):
    # logvar is well-behaved; exp(logvar) is always positive
    return -0.5 * (1 + logvar - mu**2 - logvar.exp())
```

**Why log-variance?**
1. `logvar` can be any real number (output of neural net)
2. `exp(logvar)` is always positive (valid variance)
3. `log(σ²) = logvar` directly, no numerical issues with log of small numbers

### Full VAE Loss

```python
def vae_loss(recon_x: torch.Tensor, x: torch.Tensor,
             mu: torch.Tensor, logvar: torch.Tensor,
             beta: float = 1.0) -> tuple:
    """
    Complete VAE loss function.
    
    Loss = Reconstruction + β * KL
    
    Args:
        recon_x: Reconstructed data [batch_size, data_dim]
        x: Original data [batch_size, data_dim]
        mu: Encoder mean [batch_size, latent_dim]
        logvar: Encoder log-variance [batch_size, latent_dim]
        beta: KL weight (β-VAE)
        
    Returns:
        total_loss, reconstruction_loss, kl_loss
    """
    import torch.nn.functional as F
    
    # Reconstruction loss (BCE for binary data, MSE for continuous)
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss
```

---

## KL Divergence in the VAE Objective

### Why KL Appears in ELBO

Recall the ELBO derivation:

$$\log p(x) = \mathcal{L}_{\text{ELBO}} + D_{KL}(q(z|x) \| p(z|x))$$

Since we can't compute the true posterior $p(z|x)$, we instead use:

$$\mathcal{L}_{\text{ELBO}} = \mathbb{E}_q[\log p(x|z)] - D_{KL}(q(z|x) \| p(z))$$

The KL term measures how different our approximate posterior is from the prior.

### Role of KL in Training

| KL Value | Meaning | Effect |
|----------|---------|--------|
| **High** | Encoder outputs distributions far from $\mathcal{N}(0,I)$ | More information encoded, better reconstruction |
| **Low** | Encoder outputs distributions close to $\mathcal{N}(0,I)$ | Less information encoded, smoother latent space |
| **Zero** | All inputs map to prior | No information encoded, random outputs |

### The KL-Reconstruction Trade-off

```
              Reconstruction ◄─────────────────────► Regularization
              (want low error)                       (want low KL)
              
         β = 0.1              β = 1                β = 10
    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
    │ Good recon   │    │ Balanced     │    │ Smooth latent│
    │ Messy latent │    │              │    │ Blurry recon │
    └──────────────┘    └──────────────┘    └──────────────┘
```

---

## Analytical KL vs. Monte Carlo

### Closed-Form (Preferred)

For Gaussian $q$ and Gaussian prior $p$:

```python
kl_analytical = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
```

**Advantages:** Exact, no variance, efficient

### Monte Carlo Estimate

For non-Gaussian or complex distributions:

```python
def kl_monte_carlo(q_samples, log_q, log_p, n_samples=100):
    """
    Estimate D_KL(q||p) via Monte Carlo.
    
    D_KL = E_q[log q - log p] ≈ (1/N) Σ (log q(z_i) - log p(z_i))
    """
    return (log_q - log_p).mean()
```

**When needed:** Non-Gaussian posteriors, complex priors

---

## Common Variations

### Free Bits / Minimum KL

Prevent posterior collapse by enforcing minimum KL per dimension:

```python
def kl_free_bits(mu, logvar, free_bits=0.1):
    """KL with free bits constraint."""
    kl_per_dim = -0.5 * (1 + logvar - mu**2 - logvar.exp())
    # Clamp each dimension to have at least 'free_bits' of KL
    kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
    return kl_per_dim.sum(dim=1)
```

### KL Annealing

Gradually increase KL weight during training:

```python
def get_beta(epoch, warmup_epochs=10, max_beta=1.0):
    """Linear KL annealing."""
    if epoch < warmup_epochs:
        return max_beta * epoch / warmup_epochs
    return max_beta
```

---

## Summary

| Concept | Formula | Interpretation |
|---------|---------|----------------|
| **KL definition** | $D_{KL}(p\|q) = \mathbb{E}_p[\log p/q]$ | Expected log-likelihood ratio |
| **Non-negativity** | $D_{KL}(p\|q) \geq 0$ | Fundamental property |
| **Gaussian KL** | $-\frac{1}{2}\sum(1 + \log\sigma^2 - \mu^2 - \sigma^2)$ | VAE regularization term |
| **Forward vs. Reverse** | Forward is mean-seeking, reverse is mode-seeking | Explains VAE behavior |

---

## Exercises

### Exercise 1: Derivation

Derive the KL divergence between two multivariate Gaussians with full covariance matrices.

### Exercise 2: Numerical Experiment

```python
# Create two distributions
mu_q, sigma_q = 1.0, 0.5
mu_p, sigma_p = 0.0, 1.0

# Compute KL analytically
kl_analytical = ...

# Estimate KL with Monte Carlo (sample from q, compute ratio)
samples = torch.randn(10000) * sigma_q + mu_q
kl_mc = ...

# Compare and analyze variance of MC estimate
```

### Exercise 3: Asymmetry

Compute $D_{KL}(p \| q)$ and $D_{KL}(q \| p)$ for:
- $p = \mathcal{N}(0, 1)$
- $q = \mathcal{N}(2, 0.5)$

Explain the difference in terms of mean-seeking vs. mode-seeking.

---

## What's Next

The next section covers mutual information and its relationship to the VAE objective.
