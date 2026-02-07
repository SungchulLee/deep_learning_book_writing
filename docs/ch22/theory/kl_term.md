# KL Divergence Term

The mathematical properties and computational details of the KL divergence regularization in VAEs.

---

## Learning Objectives

By the end of this section, you will be able to:

- Define KL divergence and its key properties
- Derive the closed-form KL divergence for Gaussian distributions
- Explain forward vs reverse KL and why VAEs use reverse KL
- Understand the information-theoretic role of KL divergence in VAE training

---

## KL Divergence: Definition and Properties

### Definition

The **Kullback-Leibler divergence** from distribution $q$ to distribution $p$ is:

$$D_{KL}(q \| p) = \mathbb{E}_{q(x)}\left[\log \frac{q(x)}{p(x)}\right] = \int q(x) \log \frac{q(x)}{p(x)} dx$$

### Fundamental Properties

| Property | Statement | Implication |
|----------|-----------|-------------|
| **Non-negativity** | $D_{KL}(q \| p) \geq 0$ | Ensures ELBO is a lower bound |
| **Zero iff equal** | $D_{KL}(q \| p) = 0 \Leftrightarrow q = p$ | Perfect approximation means zero KL |
| **Asymmetry** | $D_{KL}(q \| p) \neq D_{KL}(p \| q)$ in general | Direction matters |
| **Not a metric** | Triangle inequality fails | Cannot be used as a distance |

### Non-Negativity Proof (Gibbs' Inequality)

Using Jensen's inequality with the convex function $-\log$:

$$D_{KL}(q \| p) = -\mathbb{E}_q\left[\log \frac{p(x)}{q(x)}\right] \geq -\log \mathbb{E}_q\left[\frac{p(x)}{q(x)}\right] = -\log \int p(x) dx = 0$$

---

## Relationship to Entropy and Cross-Entropy

### The Fundamental Relationship

$$\underbrace{H(q, p)}_{\text{Cross-entropy}} = \underbrace{H(q)}_{\text{Entropy}} + \underbrace{D_{KL}(q \| p)}_{\text{KL divergence}}$$

where:

- **Entropy:** $H(q) = -\mathbb{E}_q[\log q(x)]$ — irreducible uncertainty
- **Cross-entropy:** $H(q, p) = -\mathbb{E}_q[\log p(x)]$ — bits needed using code from $p$
- **KL divergence:** Extra bits from using $p$ instead of $q$

Since $D_{KL} \geq 0$, cross-entropy is always at least as large as entropy. Minimizing cross-entropy is equivalent to minimizing KL divergence when $q$ is fixed.

---

## Forward vs Reverse KL

### Forward KL: $D_{KL}(p \| q)$ — Mean-Seeking

$$D_{KL}(p \| q) = \mathbb{E}_p\left[\log \frac{p(x)}{q(x)}\right]$$

This penalizes $q$ for having low probability where $p$ has high probability. The result is a $q$ that **covers all modes** of $p$, even if it assigns probability to regions where $p$ is low.

### Reverse KL: $D_{KL}(q \| p)$ — Mode-Seeking

$$D_{KL}(q \| p) = \mathbb{E}_q\left[\log \frac{q(x)}{p(x)}\right]$$

This penalizes $q$ for having high probability where $p$ has low probability. The result is a $q$ that **concentrates on high-probability regions** of $p$, potentially missing some modes.

### VAEs Use Reverse KL

In the ELBO, we minimize $D_{KL}(q_\phi(z|x) \| p(z))$, which is reverse KL from the approximate posterior to the prior. This encourages $q_\phi(z|x)$ to avoid placing mass in regions where $p(z)$ is low, keeping the latent codes within the "support" of the prior.

More fundamentally, the ELBO gap is $D_{KL}(q_\phi(z|x) \| p_\theta(z|x))$ — reverse KL from approximate to true posterior — which means the encoder tends to concentrate on high-probability regions of the true posterior rather than spreading to cover all modes.

---

## KL Divergence for Gaussians

### Univariate Case

For $q = \mathcal{N}(\mu_1, \sigma_1^2)$ and $p = \mathcal{N}(\mu_2, \sigma_2^2)$:

$$D_{KL}(q \| p) = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2}$$

### VAE Special Case: $q_\phi(z|x)$ vs Standard Normal

For $q = \mathcal{N}(\mu, \sigma^2)$ and $p = \mathcal{N}(0, 1)$:

$$D_{KL}(q \| p) = -\frac{1}{2}\left(1 + \log\sigma^2 - \mu^2 - \sigma^2\right)$$

**Derivation:**

$$D_{KL} = \mathbb{E}_q\left[\log \frac{q(z)}{p(z)}\right] = \mathbb{E}_q[\log q(z)] - \mathbb{E}_q[\log p(z)]$$

$$= -\frac{1}{2}(1 + \log 2\pi\sigma^2) - \left(-\frac{1}{2}\mathbb{E}_q[z^2] - \frac{1}{2}\log 2\pi\right)$$

$$= -\frac{1}{2}\log\sigma^2 - \frac{1}{2} + \frac{1}{2}\mathbb{E}_q[z^2]$$

Since $\mathbb{E}_q[z^2] = \mu^2 + \sigma^2$ (second moment of a Gaussian):

$$= -\frac{1}{2}\log\sigma^2 - \frac{1}{2} + \frac{1}{2}(\mu^2 + \sigma^2) = -\frac{1}{2}(1 + \log\sigma^2 - \mu^2 - \sigma^2)$$

### Multivariate Case

For $q = \mathcal{N}(\mu, \text{diag}(\sigma_1^2, \ldots, \sigma_d^2))$ and $p = \mathcal{N}(0, I)$:

$$D_{KL}(q \| p) = -\frac{1}{2}\sum_{j=1}^{d}(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2)$$

This decomposes as a **sum over dimensions**, which is computationally convenient and allows per-dimension analysis.

---

## PyTorch Implementation

```python
import torch

def kl_divergence_standard_normal(mu, logvar):
    """
    KL divergence from N(mu, diag(exp(logvar))) to N(0, I).
    
    D_KL = -0.5 * sum(1 + log(σ²) - μ² - σ²)
    
    Args:
        mu: Mean [batch_size, latent_dim]
        logvar: Log variance [batch_size, latent_dim]
    
    Returns:
        KL divergence per sample [batch_size]
    """
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)


def kl_divergence_two_gaussians(mu1, logvar1, mu2, logvar2):
    """
    KL divergence between two diagonal Gaussians.
    
    D_KL(q || p) where q = N(mu1, exp(logvar1)), p = N(mu2, exp(logvar2))
    
    Args:
        mu1, logvar1: Parameters of q
        mu2, logvar2: Parameters of p
    
    Returns:
        KL divergence per sample [batch_size]
    """
    var1 = logvar1.exp()
    var2 = logvar2.exp()
    
    kl = 0.5 * (logvar2 - logvar1 + var1 / var2 
                 + (mu1 - mu2).pow(2) / var2 - 1)
    return kl.sum(dim=1)


def kl_per_dimension(mu, logvar):
    """
    KL divergence contribution from each latent dimension.
    Useful for diagnosing posterior collapse.
    
    Args:
        mu: Mean [batch_size, latent_dim]
        logvar: Log variance [batch_size, latent_dim]
    
    Returns:
        Mean KL per dimension [latent_dim]
    """
    kl_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return kl_dim.mean(dim=0)
```

---

## Analyzing the KL Term

### When KL Is Zero

$D_{KL}(q_\phi(z|x) \| p(z)) = 0$ when $q_\phi(z|x) = \mathcal{N}(0, I)$ for all $x$, meaning the encoder ignores the input entirely. This is **posterior collapse** — the encoder produces the same distribution regardless of input.

### When KL Is Large

Large KL means the encoder is using the latent space heavily, encoding significant information about each input $x$. While good for reconstruction, excessive KL means the latent space deviates substantially from the prior, potentially hurting generation quality.

### Per-Dimension Analysis

Examining $D_{KL}$ per latent dimension reveals which dimensions are "active" (carrying information about the data) and which are "inactive" (collapsed to the prior):

```python
def analyze_kl_dimensions(model, data_loader, device):
    """Identify active vs inactive latent dimensions."""
    model.eval()
    all_kl = []
    
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.view(data.size(0), -1).to(device)
            mu, logvar = model.encode(data)
            kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
            all_kl.append(kl.cpu())
    
    mean_kl = torch.cat(all_kl, dim=0).mean(dim=0)
    
    active = (mean_kl > 0.1).sum().item()
    total = mean_kl.shape[0]
    
    print(f"Active dimensions: {active}/{total}")
    print(f"Total KL: {mean_kl.sum():.2f}")
    
    return mean_kl
```

---

## Mutual Information Connection

The expected KL over the data distribution relates to mutual information:

$$\mathbb{E}_{p_{\text{data}}(x)}[D_{KL}(q_\phi(z|x) \| p(z))] = I_q(X; Z) + D_{KL}(q_\phi(z) \| p(z))$$

The KL term thus penalizes both the mutual information between data and latent codes and the mismatch between the aggregated posterior and the prior. This decomposition is key to understanding β-VAE and the total correlation decomposition used in disentanglement.

---

## Summary

| Concept | Formula | Role in VAE |
|---------|---------|-------------|
| **KL to standard normal** | $-\frac{1}{2}\sum(1 + \log\sigma^2 - \mu^2 - \sigma^2)$ | Regularization term |
| **Forward KL** | $D_{KL}(p \| q)$ | Mean-seeking (not used in VAEs) |
| **Reverse KL** | $D_{KL}(q \| p)$ | Mode-seeking (VAE training) |
| **KL = 0** | $q(z\|x) = p(z)$ | Posterior collapse |
| **Decomposition** | $\text{MI} + \text{Marginal KL}$ | Information-theoretic view |

---

## Exercises

### Exercise 1: Closed-Form Derivation

Derive the KL divergence formula $D_{KL}(\mathcal{N}(\mu, \sigma^2) \| \mathcal{N}(0, 1))$ from the definition, showing all steps.

### Exercise 2: Numerical Verification

Verify the closed-form KL against a Monte Carlo estimate using 100,000 samples.

### Exercise 3: Forward vs Reverse KL

For a bimodal target $p(x) = 0.5\mathcal{N}(-3, 1) + 0.5\mathcal{N}(3, 1)$ and Gaussian approximation $q(x) = \mathcal{N}(\mu, \sigma^2)$, numerically find the optimal $q$ under forward KL vs reverse KL. Visualize the difference.

---

## What's Next

The next section examines the [Reconstruction Term](reconstruction.md), covering the likelihood component of the ELBO and different decoder distribution choices.
