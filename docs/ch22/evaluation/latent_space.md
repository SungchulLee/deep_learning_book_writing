# Latent Space Quality

Evaluating the structure and properties of the learned latent representation.

---

## Learning Objectives

By the end of this section, you will be able to:

- Visualize latent spaces using dimensionality reduction techniques
- Measure latent space smoothness and continuity
- Analyze active vs inactive latent dimensions
- Assess prior-posterior alignment

---

## Latent Space Visualization

### 2D Latent Space

For VAEs with `latent_dim=2`, direct visualization is possible:

```python
def plot_latent_space_2d(model, loader, device):
    """Scatter plot of 2D latent means colored by class."""
    model.eval()
    all_mu, all_labels = [], []
    
    with torch.no_grad():
        for data, labels in loader:
            data = data.view(data.size(0), -1).to(device)
            mu, _ = model.encode(data)
            all_mu.append(mu.cpu())
            all_labels.append(labels)
    
    mu = torch.cat(all_mu).numpy()
    labels = torch.cat(all_labels).numpy()
    
    plt.scatter(mu[:, 0], mu[:, 1], c=labels, cmap='tab10', alpha=0.5, s=2)
    plt.colorbar(label='Class')
    plt.xlabel('z₁'); plt.ylabel('z₂')
    plt.title('Latent Space')
```

### Higher-Dimensional Latent Spaces

For `latent_dim > 2`, use t-SNE or UMAP:

```python
from sklearn.manifold import TSNE

def plot_latent_tsne(model, loader, device, perplexity=30):
    """t-SNE visualization of high-dimensional latent space."""
    model.eval()
    all_mu, all_labels = [], []
    
    with torch.no_grad():
        for data, labels in loader:
            data = data.view(data.size(0), -1).to(device)
            mu, _ = model.encode(data)
            all_mu.append(mu.cpu())
            all_labels.append(labels)
    
    mu = torch.cat(all_mu).numpy()[:5000]  # subsample
    labels = torch.cat(all_labels).numpy()[:5000]
    
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    mu_2d = tsne.fit_transform(mu)
    
    plt.scatter(mu_2d[:, 0], mu_2d[:, 1], c=labels, cmap='tab10', alpha=0.5, s=2)
    plt.colorbar(label='Class')
    plt.title('t-SNE of Latent Space')
```

---

## Active Dimensions Analysis

Not all latent dimensions carry information. **Active dimensions** are those where $D_{KL,j} > \text{threshold}$, meaning the encoder uses that dimension to encode data-dependent information.

```python
def count_active_dims(model, loader, device, threshold=0.01):
    """Count active latent dimensions based on KL contribution."""
    model.eval()
    all_kl = []
    
    with torch.no_grad():
        for data, _ in loader:
            data = data.view(data.size(0), -1).to(device)
            mu, logvar = model.encode(data)
            kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
            all_kl.append(kl.cpu())
    
    mean_kl = torch.cat(all_kl).mean(dim=0)
    active = (mean_kl > threshold).sum().item()
    
    return active, mean_kl
```

A healthy VAE should have a reasonable number of active dimensions relative to the total. Too few active dimensions suggests posterior collapse; too many may indicate insufficient regularization.

---

## Smoothness and Continuity

A well-regularized latent space should produce smooth transitions when interpolating between points:

```python
def evaluate_smoothness(model, z1, z2, device, steps=50):
    """Measure reconstruction change rate along interpolation path."""
    model.eval()
    recons = []
    
    with torch.no_grad():
        for t in torch.linspace(0, 1, steps):
            z = ((1 - t) * z1 + t * z2).to(device)
            recons.append(model.decode(z).cpu())
    
    # Compute consecutive differences
    diffs = [F.mse_loss(recons[i], recons[i+1]).item() for i in range(len(recons)-1)]
    
    return {
        'mean_step_change': sum(diffs) / len(diffs),
        'max_step_change': max(diffs),
        'smoothness_ratio': max(diffs) / (sum(diffs) / len(diffs) + 1e-8)
    }
```

A smoothness ratio close to 1.0 indicates uniform change along the path (ideal). Large ratios indicate "jumps" — regions where small latent changes cause large output changes.

---

## Prior-Posterior Alignment

### Aggregated Posterior vs Prior

The aggregated posterior $q(z) = \mathbb{E}_{p_{\text{data}}}[q(z|x)]$ should match the prior $p(z) = \mathcal{N}(0, I)$ for good generation:

```python
def check_prior_alignment(model, loader, device):
    """Compare aggregated posterior statistics to N(0,I)."""
    model.eval()
    all_z = []
    
    with torch.no_grad():
        for data, _ in loader:
            data = data.view(data.size(0), -1).to(device)
            mu, logvar = model.encode(data)
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
            all_z.append(z.cpu())
    
    z = torch.cat(all_z)
    
    print(f"z mean: {z.mean(dim=0).abs().mean():.4f} (target: 0)")
    print(f"z std:  {z.std(dim=0).mean():.4f} (target: 1)")
```

---

## Summary

| Assessment | Method | Healthy Sign |
|-----------|--------|-------------|
| **Structure** | t-SNE/UMAP visualization | Clear class clusters |
| **Active dims** | Per-dimension KL analysis | Reasonable utilization |
| **Smoothness** | Interpolation evaluation | Uniform change rate |
| **Prior alignment** | Aggregated posterior statistics | Mean≈0, Std≈1 |

---

## Exercises

### Exercise 1

Train a VAE with `latent_dim=2` and visualize the latent space directly. Then train with `latent_dim=32` and use t-SNE. Compare the class structure.

### Exercise 2

For different β values, plot the number of active dimensions and the smoothness ratio.

---

## What's Next

The next section covers [Disentanglement Metrics](disentanglement.md) for evaluating representation quality.
