# Information Theory Foundations

The mathematical language of uncertainty, compression, and representation.

---

## Learning Objectives

By the end of this section, you will be able to:

- Define and compute entropy for discrete and continuous distributions
- Understand the meaning of cross-entropy and its role in machine learning
- Explain why information theory provides the natural framework for VAEs
- Connect information-theoretic concepts to neural network training

---

## Why Information Theory for VAEs?

VAEs are fundamentally about:

1. **Compression:** Encode high-dimensional data into low-dimensional latent codes
2. **Uncertainty:** Model distributions rather than point estimates
3. **Reconstruction:** Minimize information loss during encoding/decoding

Information theory provides the mathematical framework for all of these.

---

## Entropy: Measuring Uncertainty

### Discrete Entropy

For a discrete random variable $X$ with probability mass function $p(x)$:

$$H(X) = -\sum_x p(x) \log p(x) = \mathbb{E}_{p(x)}[-\log p(x)]$$

**Interpretation:** Average number of bits needed to encode samples from $X$.

### Example: Coin Flip

| Scenario | Distribution | Entropy |
|----------|--------------|---------|
| Fair coin | $p = 0.5$ | $H = 1$ bit |
| Biased coin | $p = 0.9$ | $H \approx 0.47$ bits |
| Certain outcome | $p = 1.0$ | $H = 0$ bits |

```python
import numpy as np

def entropy(p):
    """Compute entropy of a Bernoulli distribution."""
    if p == 0 or p == 1:
        return 0
    return -p * np.log2(p) - (1-p) * np.log2(1-p)

# Fair coin: maximum uncertainty
print(f"Fair coin (p=0.5): H = {entropy(0.5):.3f} bits")

# Biased coin: less uncertainty
print(f"Biased coin (p=0.9): H = {entropy(0.9):.3f} bits")
```

### Continuous (Differential) Entropy

For a continuous random variable $X$ with density $p(x)$:

$$h(X) = -\int p(x) \log p(x) dx = \mathbb{E}_{p(x)}[-\log p(x)]$$

**Note:** Differential entropy can be negative!

### Gaussian Entropy

For $X \sim \mathcal{N}(\mu, \sigma^2)$:

$$h(X) = \frac{1}{2} \log(2\pi e \sigma^2)$$

Key insight: Entropy depends only on variance, not mean.

For multivariate $X \sim \mathcal{N}(\mu, \Sigma)$:

$$h(X) = \frac{1}{2} \log\det(2\pi e \Sigma) = \frac{d}{2}(1 + \log 2\pi) + \frac{1}{2}\log\det(\Sigma)$$

---

## Cross-Entropy: Measuring Model Fit

### Definition

The cross-entropy between true distribution $p$ and model $q$:

$$H(p, q) = -\mathbb{E}_{p(x)}[\log q(x)] = -\sum_x p(x) \log q(x)$$

**Interpretation:** Average bits needed to encode samples from $p$ using a code optimized for $q$.

### Cross-Entropy as Loss Function

In machine learning, we typically:
- Have true labels $y$ (distribution $p$)
- Have model predictions $\hat{y}$ (distribution $q$)
- Minimize cross-entropy: $-\sum_i y_i \log \hat{y}_i$

### Connection to Maximum Likelihood

Minimizing cross-entropy is equivalent to maximizing likelihood:

$$\min_\theta H(p_{\text{data}}, p_\theta) \equiv \max_\theta \mathbb{E}_{p_{\text{data}}}[\log p_\theta(x)]$$

This is why cross-entropy is the standard classification loss.

---

## KL Divergence: Measuring Distribution Difference

### Definition

The **Kullback-Leibler divergence** from $q$ to $p$:

$$D_{KL}(p \| q) = \mathbb{E}_{p(x)}\left[\log \frac{p(x)}{q(x)}\right] = \sum_x p(x) \log \frac{p(x)}{q(x)}$$

### Properties

| Property | Implication |
|----------|-------------|
| **Non-negative** | $D_{KL}(p \| q) \geq 0$ always |
| **Zero iff equal** | $D_{KL}(p \| q) = 0$ iff $p = q$ |
| **Asymmetric** | $D_{KL}(p \| q) \neq D_{KL}(q \| p)$ generally |
| **Not a metric** | Doesn't satisfy triangle inequality |

### Fundamental Relationship

$$\underbrace{H(p, q)}_{\text{Cross-entropy}} = \underbrace{H(p)}_{\text{Entropy}} + \underbrace{D_{KL}(p \| q)}_{\text{KL divergence}}$$

Since $D_{KL} \geq 0$, cross-entropy is always at least as large as entropy.

### Asymmetry: Forward vs. Reverse KL

**Forward KL** $D_{KL}(p \| q)$ — "mean-seeking"
- Penalizes $q$ where $p$ has mass
- Makes $q$ cover all modes of $p$

**Reverse KL** $D_{KL}(q \| p)$ — "mode-seeking"
- Penalizes $q$ where $p$ has no mass
- Makes $q$ avoid low-probability regions of $p$

VAEs minimize **reverse KL** from approximate posterior to true posterior:

$$D_{KL}(q_\phi(z|x) \| p_\theta(z|x))$$

This encourages $q$ to concentrate on high-probability regions of the true posterior.

---

## KL Divergence for Gaussians

### Univariate Case

For $p = \mathcal{N}(\mu_1, \sigma_1^2)$ and $q = \mathcal{N}(\mu_2, \sigma_2^2)$:

$$D_{KL}(p \| q) = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2}$$

### VAE Special Case: $q$ vs Standard Normal

For $q = \mathcal{N}(\mu, \sigma^2)$ and $p = \mathcal{N}(0, 1)$:

$$D_{KL}(q \| p) = -\frac{1}{2}\left(1 + \log\sigma^2 - \mu^2 - \sigma^2\right)$$

This is the KL term in the VAE loss!

### Multivariate Case

For $q = \mathcal{N}(\mu, \Sigma)$ and $p = \mathcal{N}(0, I)$:

$$D_{KL}(q \| p) = \frac{1}{2}\left(\text{tr}(\Sigma) + \mu^T\mu - d - \log\det(\Sigma)\right)$$

For diagonal $\Sigma = \text{diag}(\sigma_1^2, \ldots, \sigma_d^2)$:

$$D_{KL}(q \| p) = \frac{1}{2}\sum_{j=1}^{d}\left(\sigma_j^2 + \mu_j^2 - 1 - \log\sigma_j^2\right)$$

```python
import torch

def kl_divergence_gaussian(mu, logvar):
    """
    KL divergence from N(mu, diag(exp(logvar))) to N(0, I).
    
    Args:
        mu: Mean [batch_size, latent_dim]
        logvar: Log variance [batch_size, latent_dim]
    
    Returns:
        KL divergence [batch_size]
    """
    # D_KL = -0.5 * sum(1 + log(σ²) - μ² - σ²)
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
```

---

## Mutual Information

### Definition

The mutual information between $X$ and $Z$:

$$I(X; Z) = D_{KL}(p(x, z) \| p(x)p(z)) = \mathbb{E}_{p(x,z)}\left[\log\frac{p(x,z)}{p(x)p(z)}\right]$$

### Equivalent Forms

$$I(X; Z) = H(X) - H(X|Z) = H(Z) - H(Z|X)$$

**Interpretation:** Reduction in uncertainty about $X$ given knowledge of $Z$.

### In VAEs

The mutual information between data $X$ and latent codes $Z$ under the VAE model:

$$I(X; Z) = \mathbb{E}_{p_{\text{data}}(x)}[D_{KL}(q_\phi(z|x) \| p(z))]$$

This is related to the KL term in the ELBO! The KL term regularizes mutual information.

---

## Information Theory and VAE Objective

### ELBO Decomposition

The ELBO can be written as:

$$\mathcal{L} = \underbrace{\mathbb{E}_q[\log p(x|z)]}_{\text{Reconstruction}} - \underbrace{D_{KL}(q(z|x) \| p(z))}_{\text{Rate}}$$

In rate-distortion theory terms:
- **Distortion:** Reconstruction error (how much information is lost)
- **Rate:** KL divergence (how much information is transmitted)

### The Rate-Distortion Trade-off

VAEs navigate a fundamental trade-off:

```
                    High KL (High Rate)
                           ▲
                           │
                           │  ● Perfect reconstruction
                           │    (but complex codes)
                           │
Low KL ◄───────────────────┼───────────────────► High KL
(Simple codes)             │                    (Complex codes)
                           │
                           │  ● Poor reconstruction
                           │    (but simple codes)
                           │
                           ▼
                    Low KL (Low Rate)
```

β-VAE explicitly controls this trade-off with $\beta D_{KL}$.

---

## Bits-Back Coding

### Connection to Compression

Information theory proves that VAEs achieve near-optimal compression:

1. **Encode:** Use $q(z|x)$ to get latent code $z$
2. **Transmit:** Send $z$ using a code for $p(z)$ 
3. **Decode:** Receiver uses $p(x|z)$ to reconstruct

The expected message length is bounded by the negative ELBO!

### Theoretical Guarantee

$$\mathbb{E}[\text{message length}] \leq -\mathcal{L}_{\text{ELBO}} + \text{const}$$

This justifies ELBO maximization from a compression perspective.

---

## Summary Table

| Concept | Formula | VAE Role |
|---------|---------|----------|
| **Entropy** | $H(p) = -\mathbb{E}_p[\log p]$ | Measures uncertainty in distributions |
| **Cross-entropy** | $H(p,q) = -\mathbb{E}_p[\log q]$ | Reconstruction loss (BCE/MSE origin) |
| **KL divergence** | $D_{KL}(p\|q) = \mathbb{E}_p[\log p/q]$ | Regularization term in ELBO |
| **Mutual information** | $I(X;Z) = H(Z) - H(Z\|X)$ | Information in latent codes |

---

## Exercises

### Exercise 1: Entropy Computation

Compute the entropy of:
a) A uniform distribution over {1, 2, 3, 4, 5, 6} (fair die)
b) A Gaussian with $\sigma = 2$
c) A mixture of two Gaussians (approximate numerically)

### Exercise 2: KL Divergence

Show that $D_{KL}(p \| q) \geq 0$ using Jensen's inequality.

### Exercise 3: Gaussian KL

Derive the closed-form KL divergence formula for two univariate Gaussians starting from the definition.

### Exercise 4: Mutual Information Bound

Show that $I(X; Z) \leq H(Z)$ and interpret this in the context of VAE latent codes.

---

## What's Next

The next section provides a detailed derivation of KL divergence properties and its specific forms used in VAE training.
