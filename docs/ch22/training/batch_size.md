# Batch Size Effects

How batch size influences VAE training dynamics, gradient estimation, and final model quality.

---

## Learning Objectives

By the end of this section, you will be able to:

- Explain how batch size affects ELBO gradient estimation
- Understand the interaction between batch size and loss normalization
- Choose appropriate batch sizes for VAE training
- Adapt learning rates when changing batch size

---

## Batch Size and Gradient Quality

### Monte Carlo Estimation

The VAE loss involves an expectation over the approximate posterior, estimated with a single sample per data point via the reparameterization trick. The batch gradient averages over $B$ independent estimates:

$$\nabla \hat{\mathcal{L}} = \frac{1}{B} \sum_{i=1}^{B} \nabla \mathcal{L}_i$$

Larger batches reduce gradient variance, providing more stable updates but fewer parameter updates per epoch.

### Reconstruction Term

The reconstruction gradient is estimated from mini-batch samples. Larger batches give lower-variance estimates of the expected reconstruction error.

### KL Term

The KL term for Gaussian VAEs has a **closed-form** gradient — no Monte Carlo estimation needed. Its gradient variance comes only from the diversity of data in the batch, not from sampling noise. This means the KL gradient is relatively stable even with small batches.

---

## Loss Normalization Matters

### The Normalization Problem

With `reduction='sum'`, the loss scales linearly with batch size:

$$\mathcal{L}_{\text{sum}} = \sum_{i=1}^{B} \left[\text{recon}_i + \beta \cdot \text{kl}_i\right]$$

This means the effective learning rate changes with batch size. Doubling the batch size doubles the gradient magnitude.

### Best Practice: Per-Sample Normalization

```python
def vae_loss_normalized(recon_x, x, mu, logvar, beta=1.0):
    """VAE loss normalized per sample for batch-size independence."""
    batch_size = x.size(0)
    
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum') / batch_size
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
    
    return recon_loss + beta * kl_loss, recon_loss, kl_loss
```

With per-sample normalization, the same learning rate works across different batch sizes.

---

## Practical Guidelines

### Recommended Batch Sizes

| Dataset | Recommended Batch Size | Notes |
|---------|----------------------|-------|
| MNIST | 64–256 | Small images, fast training |
| CIFAR-10 | 64–128 | Moderate complexity |
| CelebA | 32–64 | Larger images need more memory |
| Financial time series | 32–128 | Often limited data |

### Linear Scaling Rule

When increasing batch size by factor $k$, increase learning rate by factor $k$ as well (Goyal et al., 2017):

$$B' = k \cdot B \implies \eta' = k \cdot \eta$$

This preserves the expected parameter update magnitude per epoch.

---

## Batch Size and Posterior Collapse

Smaller batches introduce more noise into the gradient estimates, which can act as regularization and help prevent posterior collapse. Very large batches may lead to sharper optimization landscapes where collapse is a deeper local minimum. When increasing batch size, consider strengthening anti-collapse techniques (longer KL annealing, higher free bits threshold).

---

## Summary

| Aspect | Small Batch (32–64) | Large Batch (256–512) |
|--------|---------------------|----------------------|
| **Gradient variance** | Higher | Lower |
| **Updates per epoch** | More | Fewer |
| **Memory** | Lower | Higher |
| **Collapse risk** | Lower (noise helps) | Higher |
| **Training speed** | Slower (wall-clock) | Faster (with GPU) |

---

## Exercises

### Exercise 1: Batch Size Sweep

Train the same VAE with batch sizes {32, 64, 128, 256, 512}. Compare final ELBO and training wall-clock time.

### Exercise 2: Learning Rate Scaling

Verify the linear scaling rule: train with (B=64, lr=1e-3) and (B=256, lr=4e-3). Do they converge to similar solutions?

---

## What's Next

The next section covers [Reconstruction Quality](../evaluation/reconstruction.md) evaluation for trained VAEs.
