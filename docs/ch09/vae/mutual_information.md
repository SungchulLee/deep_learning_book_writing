# Mutual Information

Understanding information flow between data and latent representations.

---

## Learning Objectives

By the end of this section, you will be able to:

- Define mutual information and its key properties
- Explain the relationship between mutual information and the VAE objective
- Understand the information bottleneck perspective on VAEs
- Analyze representation quality through an information-theoretic lens

---

## What Is Mutual Information?

### Definition

The **mutual information** between random variables $X$ and $Z$ measures how much knowing one tells us about the other:

$$I(X; Z) = \mathbb{E}_{p(x,z)}\left[\log\frac{p(x, z)}{p(x)p(z)}\right]$$

### Equivalent Formulations

$$I(X; Z) = H(X) - H(X|Z) = H(Z) - H(Z|X) = H(X) + H(Z) - H(X, Z)$$

$$I(X; Z) = D_{KL}(p(x, z) \| p(x)p(z))$$

**Interpretation:** KL divergence from the joint to the product of marginals — how far from independence.

### Key Properties

| Property | Formula | Meaning |
|----------|---------|---------|
| **Symmetry** | $I(X; Z) = I(Z; X)$ | Mutual information is bidirectional |
| **Non-negativity** | $I(X; Z) \geq 0$ | Always non-negative |
| **Zero iff independent** | $I(X; Z) = 0 \Leftrightarrow X \perp Z$ | Zero only when independent |
| **Upper bounds** | $I(X; Z) \leq \min(H(X), H(Z))$ | Can't exceed marginal entropy |

---

## Mutual Information in VAEs

### The VAE Joint Distribution

In a VAE, we have:

- **Data distribution:** $p_{\text{data}}(x)$
- **Encoder:** $q_\phi(z|x)$
- **Aggregated posterior:** $q_\phi(z) = \mathbb{E}_{p_{\text{data}}(x)}[q_\phi(z|x)]$

The mutual information under the encoder is:

$$I_q(X; Z) = \mathbb{E}_{p_{\text{data}}(x)}[D_{KL}(q_\phi(z|x) \| q_\phi(z))]$$

### Connection to ELBO

The KL term in the ELBO can be decomposed:

$$\mathbb{E}_{p_{\text{data}}(x)}[D_{KL}(q_\phi(z|x) \| p(z))] = I_q(X; Z) + D_{KL}(q_\phi(z) \| p(z))$$

**Interpretation:**
- **First term:** Mutual information between data and latent codes
- **Second term:** Divergence of aggregated posterior from prior

### ELBO in Information-Theoretic Form

$$\mathcal{L}_{\text{ELBO}} = \underbrace{\mathbb{E}_{q}[\log p(x|z)]}_{\text{Reconstruction}} - \underbrace{I_q(X; Z)}_{\text{Mutual Info}} - \underbrace{D_{KL}(q(z) \| p(z))}_{\text{Marginal KL}}$$

This reveals that the ELBO penalizes high mutual information!

---

## The Information Bottleneck

### Framework

The information bottleneck (IB) framework seeks representations $Z$ that:

1. **Preserve relevant information:** High $I(Z; Y)$ where $Y$ is the target
2. **Compress:** Low $I(X; Z)$

The IB objective:

$$\max_q \left[I(Z; Y) - \beta \cdot I(X; Z)\right]$$

### VAEs as Information Bottleneck

VAEs implement a form of information bottleneck:

$$\max \mathbb{E}[\log p(x|z)] - \beta \cdot D_{KL}(q(z|x) \| p(z))$$

**Comparison:**

| IB Term | VAE Equivalent |
|---------|----------------|
| $I(Z; Y)$ | $\mathbb{E}[\log p(x\|z)]$ (reconstruction) |
| $I(X; Z)$ | Related to KL term |
| $\beta$ | β-VAE hyperparameter |

### Why Compression Helps

The information bottleneck forces the encoder to:

1. **Discard noise:** Irrelevant variations are expensive to encode
2. **Keep structure:** Important patterns are preserved for reconstruction
3. **Learn disentanglement:** Independent factors require less capacity

---

## Mutual Information Estimation

### Challenge

Unlike KL divergence, mutual information doesn't have a closed-form solution for VAEs. We need estimation techniques.

### Method 1: Variational Lower Bounds

**InfoNCE (Contrastive):**

$$I(X; Z) \geq \mathbb{E}\left[\frac{1}{N}\sum_{i=1}^{N}\log\frac{f(x_i, z_i)}{\frac{1}{N}\sum_{j=1}^{N}f(x_i, z_j)}\right]$$

where $f$ is a critic function (e.g., neural network).

### Method 2: Variational Upper Bounds

Using the decomposition:

$$I(X; Z) = H(Z) - H(Z|X)$$

We can bound $H(Z)$ from above and $H(Z|X)$ from below.

### Method 3: Difference of KL Divergences

$$I_q(X; Z) = \mathbb{E}_{p(x)}[D_{KL}(q(z|x) \| q(z))]$$

Estimate by:
1. Sample $x \sim p_{\text{data}}$
2. Sample $z \sim q(z|x)$
3. Estimate $\log q(z|x) - \log q(z)$

The challenge is estimating $q(z) = \int q(z|x)p(x)dx$.

### PyTorch Implementation

```python
import torch
import torch.nn.functional as F

def estimate_mutual_info_sampling(encoder, data_loader, n_samples=1000):
    """
    Estimate I(X; Z) using sampling.
    
    I(X; Z) = E_q(z|x)[log q(z|x)] - E_q(z)[log q(z)]
            = -H(Z|X) + H(Z)
    """
    encoder.eval()
    
    # Collect samples
    all_mu = []
    all_logvar = []
    all_z = []
    
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.view(data.size(0), -1)
            mu, logvar = encoder(data)
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
            
            all_mu.append(mu)
            all_logvar.append(logvar)
            all_z.append(z)
            
            if len(all_z) * data.size(0) >= n_samples:
                break
    
    mu = torch.cat(all_mu, dim=0)[:n_samples]
    logvar = torch.cat(all_logvar, dim=0)[:n_samples]
    z = torch.cat(all_z, dim=0)[:n_samples]
    
    # Compute H(Z|X): average entropy of q(z|x)
    # For Gaussian: H(Z|X) = 0.5 * sum(1 + log(2πσ²))
    h_z_given_x = 0.5 * (1 + logvar + torch.log(torch.tensor(2 * 3.14159))).sum(dim=1).mean()
    
    # Estimate H(Z) using kernel density estimation or assume Gaussian
    # Simple approximation: treat q(z) as Gaussian with empirical mean/variance
    z_mean = z.mean(dim=0)
    z_var = z.var(dim=0)
    h_z = 0.5 * (1 + torch.log(2 * 3.14159 * z_var)).sum()
    
    # I(X; Z) = H(Z) - H(Z|X)
    mi_estimate = h_z - h_z_given_x
    
    return mi_estimate.item()
```

---

## Mutual Information and Representation Quality

### Active vs. Inactive Dimensions

Mutual information reveals which latent dimensions are actually used:

```python
def analyze_latent_dimensions(encoder, data_loader, device):
    """
    Analyze which latent dimensions carry information.
    
    High variance in z_j = dimension j is active
    Low variance = dimension j is inactive (collapsed to prior)
    """
    encoder.eval()
    all_z = []
    
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.view(data.size(0), -1).to(device)
            mu, logvar = encoder(data)
            z = mu  # Use mean for analysis
            all_z.append(z)
    
    z = torch.cat(all_z, dim=0)
    
    # Variance per dimension
    var_per_dim = z.var(dim=0)
    
    # KL per dimension (if we have stored it)
    # Dimensions with high KL are "active"
    
    return var_per_dim


def count_active_dimensions(var_per_dim, threshold=0.01):
    """Count dimensions with variance above threshold."""
    return (var_per_dim > threshold).sum().item()
```

### The Posterior Collapse Problem

When KL weight is too high, the encoder may ignore input:

$$q_\phi(z|x) \approx p(z) = \mathcal{N}(0, I) \quad \text{for all } x$$

This means:
- $I(X; Z) \approx 0$
- All latent dimensions are inactive
- Decoder generates from noise (poor quality)

**Diagnosis:**

```python
def diagnose_posterior_collapse(mu, logvar):
    """
    Check for posterior collapse.
    
    Signs:
    - mu close to 0 for all x
    - logvar close to 0 for all x (σ² ≈ 1)
    """
    mu_variance = mu.var(dim=0).mean()  # Variance of means across data
    logvar_mean = logvar.mean()  # Average log-variance
    
    print(f"Mean of mu variance: {mu_variance:.4f}")
    print(f"Mean of logvar: {logvar_mean:.4f}")
    
    if mu_variance < 0.01 and abs(logvar_mean) < 0.1:
        print("⚠️ Possible posterior collapse detected!")
    else:
        print("✓ Latent space appears active")
```

---

## Mutual Information in β-VAE

### Effect of β

In β-VAE, the ELBO is:

$$\mathcal{L} = \mathbb{E}[\log p(x|z)] - \beta \cdot D_{KL}(q(z|x) \| p(z))$$

**Low β:** High $I(X; Z)$, good reconstruction, entangled representation  
**High β:** Low $I(X; Z)$, poor reconstruction, disentangled representation

### The Total Correlation Decomposition

The KL term can be decomposed (as in β-TCVAE):

$$D_{KL}(q(z|x) \| p(z)) = \underbrace{I_q(X; Z)}_{\text{Index-Code MI}} + \underbrace{TC(Z)}_{\text{Total Correlation}} + \underbrace{\sum_j D_{KL}(q(z_j) \| p(z_j))}_{\text{Dimension-wise KL}}$$

where **Total Correlation** measures dependence between latent dimensions:

$$TC(Z) = D_{KL}\left(q(z) \| \prod_j q(z_j)\right)$$

Penalizing TC encourages statistical independence → disentanglement.

---

## Data Processing Inequality

### Statement

For any Markov chain $X \to Z \to \hat{X}$:

$$I(X; \hat{X}) \leq I(X; Z)$$

**Meaning:** You can't recover more information about $X$ from $\hat{X}$ than was present in $Z$.

### Implication for VAEs

The reconstruction quality is fundamentally limited by the mutual information $I(X; Z)$:

$$\text{Reconstruction quality} \leq f(I(X; Z))$$

High-quality reconstruction requires sufficient information in the latent code.

---

## Summary

| Concept | Definition | VAE Role |
|---------|------------|----------|
| **Mutual information** | $I(X;Z) = H(Z) - H(Z\|X)$ | Information in latent codes |
| **Information bottleneck** | Trade-off: compression vs. prediction | Framework for understanding VAEs |
| **Posterior collapse** | $I(X;Z) \to 0$ | Encoder ignores input |
| **Total correlation** | Dependence between latent dims | Disentanglement measure |

---

## Exercises

### Exercise 1: Mutual Information Bounds

Show that:
a) $I(X; Z) \leq H(X)$
b) $I(X; Z) \leq H(Z)$
c) $I(X; Z) \geq 0$

### Exercise 2: Estimation

Implement a Monte Carlo estimator for $I(X; Z)$ in a trained VAE using the difference of log-probabilities approach.

### Exercise 3: Posterior Collapse Detection

Train a VAE with different β values (0.1, 1, 10, 100). For each:
a) Estimate $I(X; Z)$
b) Count active latent dimensions
c) Measure reconstruction quality

Plot the trade-off curve.

---

## What's Next

The next section provides a detailed derivation of the Evidence Lower Bound (ELBO), the central training objective for VAEs.
