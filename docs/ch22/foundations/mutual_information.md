# Mutual Information

Understanding information flow between data and latent representations.

---

## Learning Objectives

By the end of this section, you will be able to:

- Define mutual information and its key properties
- Explain the relationship between mutual information and the VAE objective
- Understand the information bottleneck perspective on VAEs
- Estimate mutual information in trained VAE models
- Apply the data processing inequality to understand VAE limitations

---

## What Is Mutual Information?

### Definition

The **mutual information** between random variables $X$ and $Z$ measures how much knowing one tells us about the other:

$$I(X; Z) = \mathbb{E}_{p(x,z)}\left[\log\frac{p(x, z)}{p(x)p(z)}\right]$$

### Equivalent Formulations

$$I(X; Z) = H(X) - H(X|Z) = H(Z) - H(Z|X) = H(X) + H(Z) - H(X, Z)$$

$$I(X; Z) = D_{KL}(p(x, z) \| p(x)p(z))$$

**Interpretation:** KL divergence from the joint to the product of marginals — how far $X$ and $Z$ are from independence.

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

In a VAE, the relevant distributions are:

- **Data distribution:** $p_{\text{data}}(x)$
- **Encoder:** $q_\phi(z|x)$
- **Aggregated posterior:** $q_\phi(z) = \mathbb{E}_{p_{\text{data}}(x)}[q_\phi(z|x)]$

The mutual information under the encoder is:

$$I_q(X; Z) = \mathbb{E}_{p_{\text{data}}(x)}[D_{KL}(q_\phi(z|x) \| q_\phi(z))]$$

### Connection to the ELBO

The KL term in the ELBO decomposes as:

$$\mathbb{E}_{p_{\text{data}}(x)}[D_{KL}(q_\phi(z|x) \| p(z))] = \underbrace{I_q(X; Z)}_{\text{Mutual information}} + \underbrace{D_{KL}(q_\phi(z) \| p(z))}_{\text{Marginal KL}}$$

This reveals that the ELBO penalizes both the mutual information between data and latent codes and the mismatch between the aggregated posterior and the prior.

### ELBO in Information-Theoretic Form

$$\mathcal{L}_{\text{ELBO}} = \underbrace{\mathbb{E}_{q}[\log p(x|z)]}_{\text{Reconstruction}} - \underbrace{I_q(X; Z)}_{\text{Mutual Info}} - \underbrace{D_{KL}(q(z) \| p(z))}_{\text{Marginal KL}}$$

The ELBO explicitly penalizes high mutual information between data and latent codes.

---

## The Information Bottleneck

### Framework

The information bottleneck (IB) framework (Tishby et al., 2000) seeks representations $Z$ that:

1. **Preserve relevant information:** High $I(Z; Y)$ where $Y$ is the target
2. **Compress:** Low $I(X; Z)$

The IB objective:

$$\max_q \left[I(Z; Y) - \beta \cdot I(X; Z)\right]$$

### VAEs as Information Bottleneck

VAEs implement a form of information bottleneck:

$$\max \mathbb{E}[\log p(x|z)] - \beta \cdot D_{KL}(q(z|x) \| p(z))$$

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

Unlike KL divergence for Gaussians, mutual information doesn't have a closed-form solution in VAEs. We need estimation techniques.

### Method 1: InfoNCE (Contrastive)

$$I(X; Z) \geq \mathbb{E}\left[\frac{1}{N}\sum_{i=1}^{N}\log\frac{f(x_i, z_i)}{\frac{1}{N}\sum_{j=1}^{N}f(x_i, z_j)}\right]$$

where $f$ is a critic function (e.g., neural network).

### Method 2: Difference of Entropies

$$I_q(X; Z) = H(Z) - H(Z|X)$$

$H(Z|X)$ is available analytically for Gaussian encoders; $H(Z)$ must be estimated.

### Method 3: Sampling-Based Estimation

```python
import torch

def estimate_mutual_info(encoder, data_loader, n_samples=2000, device='cpu'):
    """
    Estimate I(X; Z) = H(Z) - H(Z|X) using sampling.
    
    H(Z|X) is analytic for Gaussian encoder.
    H(Z) is estimated assuming Gaussian aggregated posterior.
    """
    encoder.eval()
    
    all_mu, all_logvar, all_z = [], [], []
    
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.view(data.size(0), -1).to(device)
            mu, logvar = encoder(data)
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
            
            all_mu.append(mu.cpu())
            all_logvar.append(logvar.cpu())
            all_z.append(z.cpu())
            
            if sum(m.size(0) for m in all_mu) >= n_samples:
                break
    
    mu = torch.cat(all_mu)[:n_samples]
    logvar = torch.cat(all_logvar)[:n_samples]
    z = torch.cat(all_z)[:n_samples]
    
    # H(Z|X): average entropy of q(z|x) — analytic for Gaussian
    # H(Z|X) = 0.5 * mean(sum(1 + log(2πσ²)))
    h_z_given_x = 0.5 * (1 + logvar + torch.log(torch.tensor(2 * 3.14159))).sum(dim=1).mean()
    
    # H(Z): entropy of aggregated posterior — approximate as Gaussian
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
def analyze_latent_activity(encoder, data_loader, device):
    """
    Analyze which latent dimensions carry information.
    
    High variance in mu across data → dimension is active.
    Low variance → dimension collapsed to prior.
    """
    encoder.eval()
    all_mu = []
    
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.view(data.size(0), -1).to(device)
            mu, _ = encoder(data)
            all_mu.append(mu.cpu())
    
    mu = torch.cat(all_mu)
    var_per_dim = mu.var(dim=0)
    
    active = (var_per_dim > 0.01).sum().item()
    total = var_per_dim.shape[0]
    
    print(f"Active dimensions: {active}/{total}")
    print(f"Top 5 most active: {var_per_dim.topk(5).indices.tolist()}")
    
    return var_per_dim


def diagnose_posterior_collapse(mu, logvar):
    """
    Quick check for posterior collapse.
    
    Signs: mu ≈ 0 and logvar ≈ 0 (σ² ≈ 1) for all inputs.
    """
    mu_variance = mu.var(dim=0).mean().item()
    logvar_mean = logvar.mean().item()
    
    if mu_variance < 0.01 and abs(logvar_mean) < 0.1:
        print("⚠️  Possible posterior collapse detected!")
        print(f"   mu variance across data: {mu_variance:.4f} (should be >> 0)")
        print(f"   mean logvar: {logvar_mean:.4f} (collapse → 0)")
    else:
        print(f"✓ Latent space appears active (mu_var={mu_variance:.4f})")
```

---

## The Total Correlation Decomposition

The KL term decomposes into three interpretable components (as used in β-TCVAE):

$$\mathbb{E}[D_{KL}(q(z|x) \| p(z))] = \underbrace{I_q(X; Z)}_{\text{Index-Code MI}} + \underbrace{TC(Z)}_{\text{Total Correlation}} + \underbrace{\sum_j D_{KL}(q(z_j) \| p(z_j))}_{\text{Dimension-wise KL}}$$

where **Total Correlation** measures statistical dependence between latent dimensions:

$$TC(Z) = D_{KL}\left(q(z) \| \prod_j q(z_j)\right)$$

Penalizing TC specifically encourages statistical independence between dimensions → **disentanglement**.

---

## Data Processing Inequality

### Statement

For any Markov chain $X \to Z \to \hat{X}$:

$$I(X; \hat{X}) \leq I(X; Z)$$

**Meaning:** You can't recover more information about $X$ from $\hat{X}$ than was present in $Z$.

### Implication for VAEs

The reconstruction quality is fundamentally limited by the mutual information in the latent code:

$$\text{Reconstruction quality} \leq f(I(X; Z))$$

This makes precise the intuition that high-quality reconstruction requires sufficient information in the bottleneck. When β is high (strong KL penalty), $I(X; Z)$ is low, and reconstruction quality is fundamentally bounded — no amount of decoder capacity can compensate.

---

## Summary

| Concept | Definition | VAE Role |
|---------|------------|----------|
| **Mutual information** | $I(X;Z) = H(Z) - H(Z\|X)$ | Information in latent codes |
| **Information bottleneck** | Compress while preserving relevance | Framework for understanding β-VAE |
| **Total correlation** | Dependence between latent dims | Disentanglement measure |
| **Data processing inequality** | $I(X;\hat{X}) \leq I(X;Z)$ | Fundamental reconstruction limit |
| **Posterior collapse** | $I(X;Z) \to 0$ | Encoder ignores input |

---

## Exercises

### Exercise 1: Mutual Information Bounds

Show that:
a) $I(X; Z) \leq H(X)$
b) $I(X; Z) \leq H(Z)$
c) $I(X; Z) \geq 0$

### Exercise 2: MI Estimation

Implement the sampling-based MI estimator above for a trained VAE. Compare estimated $I(X; Z)$ across different β values.

### Exercise 3: Posterior Collapse Detection

Train a VAE with β ∈ {0.1, 1, 10, 100}. For each:
a) Estimate $I(X; Z)$
b) Count active latent dimensions
c) Measure reconstruction quality
Plot the trade-off curve.

---

## What's Next

The next section provides the [ELBO Derivation](../theory/elbo_derivation.md), the central training objective for VAEs.
