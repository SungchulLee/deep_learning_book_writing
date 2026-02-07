# Information Theory Foundations

The mathematical language of uncertainty, compression, and representation that underpins VAEs.

---

## Learning Objectives

By the end of this section, you will be able to:

- Define and compute entropy for discrete and continuous distributions
- Understand cross-entropy and its connection to maximum likelihood training
- Explain the fundamental relationship: Cross-Entropy = Entropy + KL Divergence
- Connect information-theoretic concepts to the VAE objective via rate-distortion theory

---

## Why Information Theory for VAEs?

VAEs are fundamentally about three things that information theory formalizes:

1. **Compression:** Encode high-dimensional data into low-dimensional latent codes
2. **Uncertainty:** Model distributions rather than point estimates
3. **Reconstruction:** Minimize information loss during encoding/decoding

Information theory provides the precise mathematical framework for quantifying all three.

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

### Key Properties of Entropy

| Property | Statement |
|----------|-----------|
| **Non-negativity** | $H(X) \geq 0$ for discrete $X$ |
| **Maximum** | $H(X) \leq \log |\mathcal{X}|$, achieved by uniform distribution |
| **Concavity** | $H$ is concave in the distribution $p$ |

### Continuous (Differential) Entropy

For a continuous random variable $X$ with density $p(x)$:

$$h(X) = -\int p(x) \log p(x) \, dx = \mathbb{E}_{p(x)}[-\log p(x)]$$

**Important:** Unlike discrete entropy, differential entropy can be negative! For example, a uniform distribution on $[0, 1/2]$ has $h(X) = -\log 2 = -1$ bit.

### Gaussian Entropy

For $X \sim \mathcal{N}(\mu, \sigma^2)$:

$$h(X) = \frac{1}{2} \log(2\pi e \sigma^2)$$

Key insight: Entropy depends only on variance, not mean. Among all distributions with a given variance, the Gaussian has **maximum entropy**.

For multivariate $X \sim \mathcal{N}(\mu, \Sigma)$:

$$h(X) = \frac{d}{2}(1 + \log 2\pi) + \frac{1}{2}\log\det(\Sigma)$$

This is directly relevant to VAEs: the entropy of the encoder distribution $q_\phi(z|x) = \mathcal{N}(\mu_\phi(x), \text{diag}(\sigma^2_\phi(x)))$ is:

$$h(q_\phi(z|x)) = \frac{1}{2}\sum_{j=1}^{d}(1 + \log 2\pi\sigma_j^2)$$

---

## Cross-Entropy: Measuring Model Fit

### Definition

The cross-entropy between true distribution $p$ and model $q$:

$$H(p, q) = -\mathbb{E}_{p(x)}[\log q(x)] = -\sum_x p(x) \log q(x)$$

**Interpretation:** Average bits needed to encode samples from $p$ using a code optimized for $q$. Since the code is suboptimal (designed for $q$, not $p$), cross-entropy is always at least as large as entropy.

### Cross-Entropy as Loss Function

In supervised learning:

- True labels $y$ define distribution $p$
- Model predictions $\hat{y}$ define distribution $q$
- Training minimizes: $-\sum_i y_i \log \hat{y}_i$

### Connection to Maximum Likelihood

Minimizing cross-entropy is equivalent to maximizing likelihood:

$$\min_\theta H(p_{\text{data}}, p_\theta) \equiv \max_\theta \mathbb{E}_{p_{\text{data}}}[\log p_\theta(x)]$$

This is why cross-entropy is the standard classification loss, and why the VAE reconstruction term $\mathbb{E}_q[\log p_\theta(x|z)]$ is a conditional cross-entropy.

### The Fundamental Relationship

$$\underbrace{H(p, q)}_{\text{Cross-entropy}} = \underbrace{H(p)}_{\text{Entropy}} + \underbrace{D_{KL}(p \| q)}_{\text{KL divergence}}$$

Since $D_{KL} \geq 0$, cross-entropy is always at least as large as entropy. Minimizing cross-entropy is equivalent to minimizing KL divergence when $H(p)$ is constant (which it is when $p$ is the fixed data distribution).

---

## Information Theory and the VAE Objective

### ELBO as Rate-Distortion

The ELBO has a natural information-theoretic interpretation through rate-distortion theory:

$$\mathcal{L} = \underbrace{\mathbb{E}_q[\log p(x|z)]}_{\text{-Distortion}} - \underbrace{D_{KL}(q(z|x) \| p(z))}_{\text{Rate}}$$

- **Distortion** measures reconstruction error — how much information is lost during encoding
- **Rate** measures the information transmitted through the latent channel — how many bits the code uses

### The Rate-Distortion Trade-off

VAEs navigate this fundamental trade-off:

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

β-VAE explicitly controls this trade-off: $\mathcal{L}_\beta = -\text{Distortion} - \beta \cdot \text{Rate}$.

---

## Bits-Back Coding

### Connection to Compression

Information theory proves that VAEs achieve near-optimal compression through a technique called **bits-back coding**:

1. **Encode:** Use $q(z|x)$ to get latent code $z$
2. **Transmit:** Send $z$ using a code optimized for $p(z)$
3. **Decode:** Receiver uses $p(x|z)$ to reconstruct

The expected message length is bounded by the negative ELBO:

$$\mathbb{E}[\text{message length}] \leq -\mathcal{L}_{\text{ELBO}} + \text{const}$$

This provides a **compression-theoretic justification** for ELBO maximization: a better ELBO means a more efficient code.

### Practical Implication

The bits-back argument shows that VAEs are not just generative models — they are also optimal (or near-optimal) **learned compression schemes**. The latent code $z$ is the compressed representation, and the ELBO measures compression efficiency.

---

## Summary

| Concept | Formula | VAE Role |
|---------|---------|----------|
| **Entropy** | $H(p) = -\mathbb{E}_p[\log p]$ | Measures uncertainty; Gaussian entropy depends on variance |
| **Cross-entropy** | $H(p,q) = -\mathbb{E}_p[\log q]$ | Reconstruction loss derives from cross-entropy |
| **KL divergence** | $D_{KL}(p\|q) = H(p,q) - H(p)$ | Regularization term (see [KL Divergence Term](../theory/kl_term.md)) |
| **Rate-distortion** | Rate vs distortion trade-off | ELBO decomposes as -Distortion - Rate |
| **Bits-back coding** | Message length $\leq$ -ELBO | Compression justification for ELBO |

---

## Exercises

### Exercise 1: Entropy Computation

Compute the entropy of:
a) A uniform distribution over {1, 2, 3, 4, 5, 6} (fair die)
b) A Gaussian with $\sigma = 2$
c) A mixture of two Gaussians $0.5\mathcal{N}(-2, 1) + 0.5\mathcal{N}(2, 1)$ (approximate numerically)

### Exercise 2: Cross-Entropy and KL

Verify numerically that $H(p, q) = H(p) + D_{KL}(p \| q)$ for two discrete distributions of your choice.

### Exercise 3: Rate-Distortion

For a trained VAE, plot reconstruction error (distortion) vs KL divergence (rate) for different β values. Compare with the theoretical rate-distortion curve.

### Exercise 4: Compression Efficiency

Compare the bits-per-dimension achieved by a trained VAE ($-\text{ELBO} / d$) against standard compression algorithms (gzip, PNG) on the same dataset.

---

## What's Next

The next section covers [Mutual Information](mutual_information.md), which formalizes information flow between data and latent representations.
