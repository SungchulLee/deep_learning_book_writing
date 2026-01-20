# Reconstruction Term

Understanding the likelihood component of the VAE objective.

---

## Learning Objectives

By the end of this section, you will be able to:

- Explain the probabilistic interpretation of reconstruction loss
- Choose appropriate reconstruction losses for different data types
- Derive MSE and BCE losses from maximum likelihood principles
- Understand the trade-offs between different decoder distributions

---

## The Reconstruction Term in ELBO

### Definition

The reconstruction term in the ELBO is:

$$\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]$$

**Interpretation:** Expected log-likelihood of the data $x$ given latent codes sampled from the encoder.

### Maximization Objective

Maximizing this term encourages the decoder to:
1. **Accurately reconstruct** input data from latent codes
2. **Assign high probability** to the true data under the decoder distribution

---

## Decoder Distribution Choices

### Common Options

| Decoder Distribution | Data Type | Reconstruction Loss |
|---------------------|-----------|---------------------|
| **Gaussian** | Continuous, unbounded | Mean Squared Error (MSE) |
| **Bernoulli** | Binary or [0,1] bounded | Binary Cross-Entropy (BCE) |
| **Categorical** | Discrete classes | Cross-Entropy |
| **Laplace** | Continuous, robust | Mean Absolute Error (L1) |

---

## Gaussian Decoder (MSE Loss)

### Probabilistic Model

Assume the decoder outputs a Gaussian distribution:

$$p_\theta(x|z) = \mathcal{N}(x; \mu_\theta(z), \sigma^2 I)$$

where $\mu_\theta(z)$ is the decoder network output and $\sigma^2$ is fixed variance.

### Deriving the Loss

The log-likelihood is:

$$\log p_\theta(x|z) = -\frac{d}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\|x - \mu_\theta(z)\|^2$$

Taking the expectation and ignoring constants:

$$-\mathbb{E}_{q}[\log p_\theta(x|z)] \propto \mathbb{E}_{q}\left[\|x - \mu_\theta(z)\|^2\right]$$

This is the **Mean Squared Error (MSE)**!

### Implementation

```python
import torch
import torch.nn.functional as F

def gaussian_reconstruction_loss(recon_x, x, reduction='sum'):
    """
    Gaussian decoder reconstruction loss (MSE).
    
    Args:
        recon_x: Decoder output (mean of Gaussian)
        x: Original data
        reduction: 'sum' or 'mean'
    
    Returns:
        MSE reconstruction loss
    """
    return F.mse_loss(recon_x, x, reduction=reduction)
```

### When to Use

✓ Continuous, real-valued data  
✓ Images normalized to arbitrary range  
✓ Regression-type outputs  

---

## Bernoulli Decoder (BCE Loss)

### Probabilistic Model

For binary or $[0, 1]$-bounded data:

$$p_\theta(x_i|z) = p_i^{x_i}(1-p_i)^{1-x_i}$$

where $p_i = \sigma(\text{decoder}(z)_i)$ through sigmoid activation.

### Deriving the Loss

$$\log p_\theta(x|z) = \sum_{i=1}^{d} \left[x_i \log p_i + (1-x_i) \log(1-p_i)\right]$$

This is the negative **Binary Cross-Entropy (BCE)**!

### Implementation

```python
def bernoulli_reconstruction_loss(recon_x, x, reduction='sum'):
    """
    Bernoulli decoder reconstruction loss (BCE).
    
    Args:
        recon_x: Decoder output (probabilities after sigmoid)
        x: Original data in [0, 1]
        reduction: 'sum' or 'mean'
    
    Returns:
        BCE reconstruction loss
    """
    return F.binary_cross_entropy(recon_x, x, reduction=reduction)


def bernoulli_loss_stable(logits, x, reduction='sum'):
    """BCE from logits - more numerically stable."""
    return F.binary_cross_entropy_with_logits(logits, x, reduction=reduction)
```

### When to Use

✓ Binary data (black/white images)  
✓ MNIST and similar datasets with values in [0, 1]  
✓ When treating pixel intensities as probabilities  

---

## Comparing MSE and BCE

| Aspect | MSE (Gaussian) | BCE (Bernoulli) |
|--------|----------------|-----------------|
| **Assumption** | Gaussian noise | Bernoulli process |
| **Output range** | Unbounded | [0, 1] via sigmoid |
| **Error penalty** | Quadratic | Log-based |
| **Gradient** | Linear in error | Depends on confidence |

### Practical Recommendation

- **MNIST, Fashion-MNIST:** BCE (data naturally in [0, 1])
- **Natural images (CelebA, CIFAR):** MSE often works better
- **When unsure:** Try both, compare reconstruction quality

---

## Balancing Reconstruction and KL

### The Trade-off

$$\mathcal{L}_{\text{VAE}} = \text{Reconstruction Loss} + \beta \cdot \text{KL Loss}$$

| $\beta$ Value | Effect |
|---------------|--------|
| **Low (< 1)** | Better reconstruction, less regularized latent space |
| **$\beta = 1$** | Standard VAE (theoretical ELBO) |
| **High (> 1)** | More regularized latent space, blurrier reconstruction |

### Normalization Considerations

Reconstruction loss scales with data dimension, KL scales with latent dimension:

```python
def normalized_vae_loss(recon_x, x, mu, logvar, data_dim, latent_dim):
    """Loss normalized by dimensions for fair comparison."""
    recon_loss = F.mse_loss(recon_x, x, reduction='sum') / data_dim
    kl_loss = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp()) / latent_dim
    return recon_loss + kl_loss
```

---

## Summary

| Decoder | Loss | Formula | Use Case |
|---------|------|---------|----------|
| **Gaussian** | MSE | $\|x - \hat{x}\|^2$ | Continuous data |
| **Bernoulli** | BCE | $-[x\log\hat{x} + (1-x)\log(1-\hat{x})]$ | Binary/[0,1] data |
| **Laplace** | L1 | $\|x - \hat{x}\|$ | Robust to outliers |

---

## Exercises

### Exercise 1: Derivation
Derive that minimizing MSE is equivalent to maximizing Gaussian log-likelihood.

### Exercise 2: Comparison
Train a VAE on MNIST with both MSE and BCE losses. Compare:
- Reconstruction quality (visual)
- Final loss values
- Latent space structure

### Exercise 3: Learned Variance
Implement a Gaussian decoder with learned variance and compare to fixed variance.

---

## What's Next

The next section covers the reparameterization trick that enables gradient-based training of VAEs.
