# Free Bits

Preventing posterior collapse by guaranteeing minimum information per latent dimension.

---

## Learning Objectives

By the end of this section, you will be able to:

- Explain the free bits technique and its motivation
- Implement free bits in VAE training
- Compare free bits with KL annealing
- Choose appropriate free bits thresholds

---

## The Free Bits Technique

### Motivation

KL annealing addresses posterior collapse temporally (during early training). **Free bits** (Kingma et al., 2016) addresses it structurally by modifying the KL term so that each latent dimension is allowed to carry at least $\lambda$ nats of information without penalty.

### Formulation

Replace the standard KL term with:

$$D_{KL}^{\text{free}} = \sum_{j=1}^{d} \max\left(\lambda, \, D_{KL,j}(q_\phi(z_j|x) \| p(z_j))\right)$$

where $D_{KL,j}$ is the KL contribution from dimension $j$ and $\lambda$ is the free bits threshold. When $D_{KL,j} < \lambda$, the KL is held constant at $\lambda$, providing no gradient to push the dimension further toward the prior.

### Implementation

```python
import torch
import torch.nn.functional as F

def vae_loss_free_bits(recon_x, x, mu, logvar, free_bits=0.5):
    """
    VAE loss with free bits constraint.
    
    Args:
        recon_x: Reconstructed output
        x: Original input
        mu: Encoder mean [batch_size, latent_dim]
        logvar: Encoder log-variance [batch_size, latent_dim]
        free_bits: Minimum KL per dimension (in nats)
    
    Returns:
        total_loss, recon_loss, kl_loss
    """
    # Reconstruction loss
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL per dimension: [batch_size, latent_dim]
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    
    # Average over batch first, then apply free bits
    kl_mean_per_dim = kl_per_dim.mean(dim=0)  # [latent_dim]
    kl_free = torch.clamp(kl_mean_per_dim, min=free_bits)
    kl_loss = kl_free.sum() * x.size(0)  # scale back to batch
    
    total_loss = recon_loss + kl_loss
    return total_loss, recon_loss, kl_loss
```

---

## Choosing $\lambda$

| $\lambda$ Value | Effect |
|-----------------|--------|
| $\lambda = 0$ | Standard VAE (no free bits) |
| $\lambda = 0.1$ | Mild protection against collapse |
| $\lambda = 0.5$ | Moderate — good default |
| $\lambda = 2.0$ | Strong — every dimension must carry significant information |

The optimal $\lambda$ depends on the data complexity and latent dimension. For MNIST with 32 latent dimensions, $\lambda \in [0.1, 1.0]$ typically works well. For more complex data or higher latent dimensions, lower values may suffice.

---

## Free Bits vs KL Annealing

| Aspect | KL Annealing | Free Bits |
|--------|-------------|-----------|
| **Mechanism** | Temporal (schedule-based) | Structural (per-dimension threshold) |
| **Hyperparameter** | Warmup duration | Threshold $\lambda$ |
| **Posterior collapse** | Prevents during warmup | Prevents at all times |
| **Can combine** | Yes — often used together | Yes |

In practice, combining both techniques provides the most robust training.

---

## Summary

| Concept | Key Point |
|---------|-----------|
| **Free bits** | Minimum KL per dimension: $\max(\lambda, D_{KL,j})$ |
| **Effect** | Prevents individual dimensions from collapsing |
| **Default $\lambda$** | 0.1–1.0 nats depending on task |
| **Combination** | Use with KL annealing for best results |

---

## Exercises

### Exercise 1: Free Bits Sweep

Train VAEs with $\lambda \in \{0, 0.1, 0.5, 1.0, 2.0\}$. Plot active dimensions and reconstruction quality.

### Exercise 2: Combined Strategy

Compare: (a) annealing only, (b) free bits only, (c) both combined. Which achieves the best ELBO?

---

## What's Next

The next section examines [Batch Size Effects](batch_size.md) on VAE training dynamics.
