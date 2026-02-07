# Training and Evaluation

Normalizing flows are trained by **maximum likelihood estimation**—directly maximising the probability of training data under the model.  This section covers the training objective, practical optimisation details, base distribution choices, the dequantisation trick for discrete data, and evaluation metrics.

## Maximum Likelihood Training

### Objective

Given data $\{x_1, \ldots, x_N\}$, we maximise the average log-likelihood:

$$\mathcal{L}(\theta) = \frac{1}{N}\sum_{i=1}^{N}\log p_\theta(x_i)$$

For a flow with transformation $f_\theta$:

$$\log p_\theta(x) = \log p_Z\!\bigl(f_\theta^{-1}(x)\bigr) + \log\!\left|\det \frac{\partial f_\theta^{-1}}{\partial x}\right|$$

In practice we minimise the **negative log-likelihood** (NLL):

$$\text{Loss} = -\frac{1}{N}\sum_{i=1}^{N}\Bigl[\log p_Z(z_i) + \sum_{k}\log|\det J_k|\Bigr]$$

where $z_i = f_\theta^{-1}(x_i)$.

### Training Loop

```python
import torch
import torch.nn as nn
import numpy as np


def train_flow(model, data, epochs=200, batch_size=256, lr=1e-3):
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, epochs)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data),
        batch_size=batch_size, shuffle=True,
    )

    for epoch in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            loss = -model.log_prob(batch).mean()
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            epoch_loss += loss.item() * len(batch)
        scheduler.step()

        if (epoch + 1) % 20 == 0:
            avg = epoch_loss / len(data)
            print(f"Epoch {epoch+1:4d}  NLL: {avg:.4f}")
```

### Practical Considerations

**Gradient clipping** prevents unstable updates, especially early in training when the log-determinant terms can be large.

**Learning-rate scheduling** (cosine annealing or reduce-on-plateau) improves convergence.

**Warm-up** the learning rate for the first few hundred steps to stabilise ActNorm initialisation.

**Batch size** should be large enough for stable gradient estimates; 256–1024 is typical.

### KL Divergence Interpretation

Minimising NLL is equivalent to minimising $D_{KL}(p_{\text{data}} \| p_\theta)$, the forward KL divergence.  This means the model is incentivised to cover all modes of the data (mass-covering behaviour), avoiding the mode-dropping that can occur with reverse-KL training (as in some VAE formulations).

## Base Distribution Choice

### Standard Gaussian

The default choice is $p_Z(z) = \mathcal{N}(z; 0, I)$:

$$\log p_Z(z) = -\frac{1}{2}\sum_{i=1}^{d}\bigl(z_i^2 + \log 2\pi\bigr)$$

This is simple, well-understood, and works well in most cases.

### Learnable Gaussian

Allowing a learnable mean $\mu$ and diagonal covariance $\sigma^2$ can help when the flow has limited depth:

```python
class LearnableGaussian(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(dim))
        self.log_sigma = nn.Parameter(torch.zeros(dim))

    def log_prob(self, z):
        var = self.log_sigma.exp() ** 2
        return -0.5 * (((z - self.mu) ** 2) / var
                       + 2 * self.log_sigma + np.log(2 * np.pi)).sum(-1)

    def sample(self, n, device="cpu"):
        eps = torch.randn(n, len(self.mu), device=device)
        return self.mu + eps * self.log_sigma.exp()
```

### Mixture of Gaussians

A mixture base can capture multi-modality in the latent space, reducing the burden on the flow layers.  However, this increases the number of hyperparameters and complicates sampling slightly.

### Student-t Base

For financial applications where heavy tails are expected, a Student-t base with learnable degrees of freedom can improve tail modelling.

### Practical Guidance

In most settings the standard Gaussian is sufficient—let the flow learn the complexity.  Non-standard bases are most useful when the flow is shallow or the data has known structure (e.g., known heavy tails) that can be baked into the prior.

## Dequantisation

### The Problem with Discrete Data

Normalizing flows define continuous densities $p(x)$.  Discrete data (e.g., 8-bit pixel values $\{0, 1, \ldots, 255\}$) assigns zero probability to any single point under a continuous density, so $\log p(x)$ is undefined.

### Uniform Dequantisation

The standard solution adds uniform noise:

$$\tilde{x} = \frac{x + u}{256}, \qquad u \sim \text{Uniform}(0, 1)$$

This spreads each integer value over a unit bin, creating continuous data.  The resulting continuous log-likelihood provides a valid lower bound on the discrete log-likelihood:

$$\log P_{\text{discrete}}(x) \ge \mathbb{E}_{u}\!\bigl[\log p(\tilde{x})\bigr] + D\log 256$$

### Variational Dequantisation

A learned dequantisation distribution $q(u | x)$ (itself parameterised by a small flow) provides a tighter bound and better density estimates.  This was introduced by Ho et al. (2019) and is now standard for image density estimation benchmarks.

### When Dequantisation Is Needed

For continuous data (e.g., financial returns, sensor readings), dequantisation is unnecessary.  It is only needed for data with discrete support.

## Evaluation Metrics

### Log-Likelihood (Bits per Dimension)

The primary metric for density estimation.  For $d$-dimensional data:

$$\text{BPD} = -\frac{1}{d}\;\frac{1}{N}\sum_{i=1}^{N}\frac{\log_2 p_\theta(x_i)}{1}$$

Lower BPD means better density fit.  The division by $d$ normalises across different data dimensions.

### Computing Test Log-Likelihood

```python
@torch.no_grad()
def test_log_likelihood(model, test_loader, device="cpu"):
    total_ll = 0.0
    total_n = 0
    for (batch,) in test_loader:
        batch = batch.to(device)
        ll = model.log_prob(batch)
        total_ll += ll.sum().item()
        total_n += len(batch)
    return total_ll / total_n
```

### Sample Quality

While log-likelihood is the principled metric, visual inspection of samples and quantitative metrics (FID for images) complement it.  Flows can sometimes achieve good log-likelihoods while producing blurry samples, because the forward-KL objective prioritises coverage over sharpness.

### Calibration

For applications like risk modelling, the learned density should be **calibrated**: events assigned probability $p$ should occur with frequency $\approx p$.  This can be tested with probability integral transform (PIT) histograms or reliability diagrams.

### Two-Sample Tests

Compare samples from the flow against held-out test data using statistical tests (Kolmogorov-Smirnov, maximum mean discrepancy) to assess whether the model captures the data distribution beyond simple log-likelihood.

## Monitoring Training

### Diagnostics

Track these quantities during training:

- **Training and validation NLL** — watch for overfitting (validation NLL increasing).
- **Log-determinant magnitude** — extremely large or small values indicate instability.
- **Latent-space distribution** — the transformed data should resemble the base distribution; plot histograms of $z$ values.
- **Reconstruction error** — $\|x - f(f^{-1}(x))\|$ should be near machine precision.

### Common Failure Modes

**Exploding log-determinant** — the scale parameters in affine coupling grow without bound.  Fix with bounded scale (e.g., $\tanh$ clipping) or gradient clipping.

**Mode collapse in latent space** — all data maps to a small region.  This usually indicates too few flow layers or insufficient conditioner capacity.

**NaN loss** — typically caused by numerical overflow in $\exp(\cdot)$ or $\log(\cdot)$.  Add epsilon terms and clamp intermediate values.

## Summary

Training normalizing flows by maximum likelihood is conceptually simple: minimise NLL via gradient descent.  The practical details—gradient clipping, learning-rate scheduling, bounded scale parameterisations, appropriate base distributions, and dequantisation for discrete data—are what make the difference between a model that trains stably and one that diverges.  Evaluation should go beyond log-likelihood to include calibration and distributional tests, especially for risk-sensitive financial applications.

## References

1. Papamakarios, G., et al. (2021). Normalizing Flows for Probabilistic Modeling and Inference. *JMLR*.
2. Ho, J., et al. (2019). Flow++: Improving Flow-Based Generative Models with Variational Dequantization and Architecture Design. *ICML*.
3. Dinh, L., et al. (2017). Density Estimation Using Real-NVP. *ICLR*.
