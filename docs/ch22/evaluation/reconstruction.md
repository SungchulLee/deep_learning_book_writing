# Reconstruction Quality

Evaluating how well a trained VAE reconstructs input data.

---

## Learning Objectives

By the end of this section, you will be able to:

- Compute and interpret reconstruction metrics for VAEs
- Understand the relationship between reconstruction quality and ELBO
- Compare reconstruction across different VAE configurations
- Visualize reconstruction quality diagnostically

---

## Reconstruction Metrics

### Reconstruction Loss (Negative Log-Likelihood)

The most direct measure is the reconstruction term of the ELBO:

$$\mathcal{L}_{\text{recon}} = -\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]$$

For Bernoulli decoders this is BCE; for Gaussian decoders this is proportional to MSE.

### Mean Squared Error

$$\text{MSE} = \frac{1}{d}\|x - \hat{x}\|^2$$

where $\hat{x}$ is the reconstruction using the posterior mean $z = \mu_\phi(x)$.

### Structural Similarity Index (SSIM)

SSIM captures perceptual similarity better than pixel-wise metrics by comparing luminance, contrast, and structure:

```python
from torchmetrics.image import StructuralSimilarityIndexMeasure

ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
score = ssim(recon_images, original_images)  # Higher is better
```

### Per-Class Analysis

Reconstruction quality often varies across classes. Analyzing per-class metrics reveals which data subpopulations are well-modeled:

```python
def per_class_reconstruction(model, loader, device, num_classes=10):
    """Compute reconstruction error per class."""
    model.eval()
    class_errors = {c: [] for c in range(num_classes)}
    
    with torch.no_grad():
        for data, labels in loader:
            data = data.view(data.size(0), -1).to(device)
            recon, mu, _ = model(data)
            
            errors = (data - recon).pow(2).sum(dim=1)
            for i, label in enumerate(labels):
                class_errors[label.item()].append(errors[i].item())
    
    return {c: sum(v)/len(v) for c, v in class_errors.items()}
```

---

## Deterministic vs Stochastic Reconstruction

During evaluation, we can reconstruct using the posterior mean (deterministic, sharper) or by sampling from the posterior (stochastic, more faithful to training). The mean reconstruction $\hat{x} = g_\theta(\mu_\phi(x))$ is typically sharper and used for visual comparison, while the sampled reconstruction $\hat{x} = g_\theta(z)$ where $z \sim q_\phi(z|x)$ better reflects the training objective.

---

## Summary

| Metric | Formula | Interpretation |
|--------|---------|---------------|
| **BCE/MSE** | Negative log-likelihood | Direct ELBO component |
| **SSIM** | Structural similarity | Perceptual quality |
| **Per-class error** | Class-conditional MSE | Identifies weak spots |

---

## Exercises

### Exercise 1

Compute MSE and SSIM for a trained VAE across different latent dimensions. How does latent dimension affect reconstruction quality?

### Exercise 2

Compare deterministic (mean) vs stochastic (sampled) reconstructions visually and quantitatively.

---

## What's Next

The next section covers [Generation Quality](generation.md) evaluation.
