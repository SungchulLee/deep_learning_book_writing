# KL Divergence for Gaussians

Gaussian distributions are the most important special case for KL divergence computation in deep learning, appearing in variational autoencoders, Bayesian neural networks, and normalizing flows. This section derives the closed-form KL divergence for univariate, multivariate, and diagonal-covariance Gaussians, culminating in the standard VAE formula.

## Univariate Gaussians

### Setup

Let $p = \mathcal{N}(\mu_1, \sigma_1^2)$ and $q = \mathcal{N}(\mu_2, \sigma_2^2)$ with densities:

$$p(x) = \frac{1}{\sqrt{2\pi\sigma_1^2}}\exp\!\left(-\frac{(x-\mu_1)^2}{2\sigma_1^2}\right), \qquad q(x) = \frac{1}{\sqrt{2\pi\sigma_2^2}}\exp\!\left(-\frac{(x-\mu_2)^2}{2\sigma_2^2}\right)$$

### Derivation

Starting from the definition:

$$D_{\text{KL}}(p \| q) = \int p(x)\log\frac{p(x)}{q(x)}\,dx = \mathbb{E}_p\!\left[\log\frac{p(x)}{q(x)}\right]$$

Expand the log-density ratio:

$$\log\frac{p(x)}{q(x)} = \log\frac{\sigma_2}{\sigma_1} + \frac{(x - \mu_2)^2}{2\sigma_2^2} - \frac{(x - \mu_1)^2}{2\sigma_1^2}$$

Now take the expectation under $p = \mathcal{N}(\mu_1, \sigma_1^2)$ term by term.

**Term 1:** $\mathbb{E}_p\!\left[\log\frac{\sigma_2}{\sigma_1}\right] = \log\frac{\sigma_2}{\sigma_1}$ (constant).

**Term 2:** $\mathbb{E}_p\!\left[\frac{(x - \mu_2)^2}{2\sigma_2^2}\right]$. Expand $(x - \mu_2)^2 = (x - \mu_1 + \mu_1 - \mu_2)^2$:

$$\mathbb{E}_p[(x - \mu_2)^2] = \mathbb{E}_p[(x - \mu_1)^2] + 2(\mu_1 - \mu_2)\underbrace{\mathbb{E}_p[x - \mu_1]}_{=0} + (\mu_1 - \mu_2)^2 = \sigma_1^2 + (\mu_1 - \mu_2)^2$$

So this term becomes $\frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2}$.

**Term 3:** $\mathbb{E}_p\!\left[\frac{(x - \mu_1)^2}{2\sigma_1^2}\right] = \frac{\sigma_1^2}{2\sigma_1^2} = \frac{1}{2}$.

### Result

Combining all three terms:

$$\boxed{D_{\text{KL}}(p \| q) = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2}}$$

### Interpretation of Each Term

- $\log\frac{\sigma_2}{\sigma_1}$: **Variance ratio penalty.** Positive when $\sigma_2 > \sigma_1$ ($q$ is wider than $p$) and negative when $\sigma_2 < \sigma_1$.

- $\frac{(\mu_1 - \mu_2)^2}{2\sigma_2^2}$: **Mean shift penalty.** Increases quadratically with the distance between means, scaled by the variance of $q$. A mean difference of 1 standard deviation of $q$ contributes $\frac{1}{2}$ nats.

- $\frac{\sigma_1^2}{2\sigma_2^2}$: **Variance mismatch.** Contributes $\frac{1}{2}$ when $\sigma_1 = \sigma_2$ (canceling the $-\frac{1}{2}$ term), and grows when $\sigma_1 > \sigma_2$.

### Numerical Verification

```python
import torch
import numpy as np

def kl_gaussian_1d(mu1, sigma1, mu2, sigma2):
    """Analytical KL divergence between univariate Gaussians."""
    return (np.log(sigma2 / sigma1)
            + (sigma1**2 + (mu1 - mu2)**2) / (2 * sigma2**2)
            - 0.5)

def kl_gaussian_mc(mu1, sigma1, mu2, sigma2, n_samples=100000):
    """Monte Carlo estimate of KL divergence."""
    samples = np.random.normal(mu1, sigma1, n_samples)
    log_p = -0.5 * np.log(2 * np.pi * sigma1**2) - (samples - mu1)**2 / (2 * sigma1**2)
    log_q = -0.5 * np.log(2 * np.pi * sigma2**2) - (samples - mu2)**2 / (2 * sigma2**2)
    return np.mean(log_p - log_q)

# Test case
mu1, sigma1 = 1.0, 0.5
mu2, sigma2 = 0.0, 1.0

kl_analytical = kl_gaussian_1d(mu1, sigma1, mu2, sigma2)
kl_mc = kl_gaussian_mc(mu1, sigma1, mu2, sigma2)

print(f"Analytical: {kl_analytical:.6f}")
print(f"Monte Carlo: {kl_mc:.6f}")
print(f"Difference: {abs(kl_analytical - kl_mc):.6f}")
```

## Multivariate Gaussians (Full Covariance)

### Setup

Let $p = \mathcal{N}(\mu_p, \Sigma_p)$ and $q = \mathcal{N}(\mu_q, \Sigma_q)$ in $\mathbb{R}^d$ with densities:

$$p(x) = \frac{1}{(2\pi)^{d/2}|\Sigma_p|^{1/2}}\exp\!\left(-\frac{1}{2}(x - \mu_p)^T\Sigma_p^{-1}(x - \mu_p)\right)$$

### Derivation

$$D_{\text{KL}}(p \| q) = \mathbb{E}_p\!\left[\log\frac{p(x)}{q(x)}\right]$$

Expand the log-density ratio:

$$\log\frac{p(x)}{q(x)} = \frac{1}{2}\log\frac{|\Sigma_q|}{|\Sigma_p|} + \frac{1}{2}(x - \mu_q)^T\Sigma_q^{-1}(x - \mu_q) - \frac{1}{2}(x - \mu_p)^T\Sigma_p^{-1}(x - \mu_p)$$

Take expectations under $p$ using two key identities.

**Identity 1.** For $x \sim \mathcal{N}(\mu_p, \Sigma_p)$:

$$\mathbb{E}_p[(x - \mu_p)^T A (x - \mu_p)] = \operatorname{tr}(A\Sigma_p)$$

This follows from $\mathbb{E}[(x-\mu)(x-\mu)^T] = \Sigma_p$ and the trace-cyclic property.

**Identity 2.** Expanding $(x - \mu_q) = (x - \mu_p) + (\mu_p - \mu_q)$:

$$\mathbb{E}_p[(x - \mu_q)^T\Sigma_q^{-1}(x - \mu_q)] = \operatorname{tr}(\Sigma_q^{-1}\Sigma_p) + (\mu_p - \mu_q)^T\Sigma_q^{-1}(\mu_p - \mu_q)$$

The cross term vanishes because $\mathbb{E}_p[x - \mu_p] = 0$.

### Result

$$\boxed{D_{\text{KL}}(p \| q) = \frac{1}{2}\!\left[\log\frac{|\Sigma_q|}{|\Sigma_p|} - d + \operatorname{tr}\!\left(\Sigma_q^{-1}\Sigma_p\right) + (\mu_q - \mu_p)^T\Sigma_q^{-1}(\mu_q - \mu_p)\right]}$$

### Interpretation of Each Term

| Term | Interpretation |
|------|---------------|
| $\log\frac{\lvert\Sigma_q\rvert}{\lvert\Sigma_p\rvert}$ | Log determinant ratio: measures volume mismatch between covariance ellipsoids |
| $-d$ | Normalization constant (ensures $D_{\text{KL}} = 0$ when $p = q$) |
| $\operatorname{tr}(\Sigma_q^{-1}\Sigma_p)$ | Covariance alignment: measures how well $\Sigma_p$ aligns with $\Sigma_q$ |
| $(\mu_q - \mu_p)^T\Sigma_q^{-1}(\mu_q - \mu_p)$ | Mahalanobis distance between means, weighted by $\Sigma_q$ |

### Verification: Reduces to Univariate

For $d = 1$ with $\Sigma_p = \sigma_1^2$, $\Sigma_q = \sigma_2^2$:

$$D_{\text{KL}} = \frac{1}{2}\!\left[\log\frac{\sigma_2^2}{\sigma_1^2} - 1 + \frac{\sigma_1^2}{\sigma_2^2} + \frac{(\mu_1 - \mu_2)^2}{\sigma_2^2}\right] = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2}$$

which matches the univariate formula. ✓

## VAE Special Case: Diagonal Covariance vs Standard Normal

### Setup

In a VAE, the encoder outputs $q_\phi(z|x) = \mathcal{N}(\mu, \text{diag}(\sigma_1^2, \ldots, \sigma_d^2))$ and the prior is $p(z) = \mathcal{N}(0, I)$.

### Derivation from the Multivariate Formula

Substituting $\mu_p = \mu$, $\Sigma_p = \text{diag}(\sigma_1^2, \ldots, \sigma_d^2)$, $\mu_q = 0$, $\Sigma_q = I$:

**Log determinant ratio:** $\log\frac{|I|}{|\Sigma_p|} = -\log|\Sigma_p| = -\sum_{j=1}^d \log\sigma_j^2$

**Trace term:** $\operatorname{tr}(I^{-1}\Sigma_p) = \operatorname{tr}(\Sigma_p) = \sum_{j=1}^d \sigma_j^2$

**Mahalanobis term:** $\mu^T I^{-1} \mu = \|\mu\|^2 = \sum_{j=1}^d \mu_j^2$

Combining:

$$D_{\text{KL}}(q \| p) = \frac{1}{2}\!\left[-\sum_{j=1}^d \log\sigma_j^2 - d + \sum_{j=1}^d \sigma_j^2 + \sum_{j=1}^d \mu_j^2\right]$$

### Result

$$\boxed{D_{\text{KL}}(q \| p) = -\frac{1}{2}\sum_{j=1}^{d}\!\left(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2\right)}$$

### Per-Dimension Decomposition

Since the covariance is diagonal and the prior is isotropic, the KL decomposes into a sum of independent univariate KL divergences:

$$D_{\text{KL}}(q \| p) = \sum_{j=1}^d D_{\text{KL}}\!\left(\mathcal{N}(\mu_j, \sigma_j^2) \| \mathcal{N}(0, 1)\right)$$

Each term $-\frac{1}{2}(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2)$ decomposes further into:

- **Mean penalty:** $\frac{1}{2}\mu_j^2$ — penalizes the encoder mean for deviating from zero
- **Variance penalty:** $\frac{1}{2}(\sigma_j^2 - 1 - \log\sigma_j^2)$ — penalizes the encoder variance for deviating from 1 (this is non-negative with minimum at $\sigma_j^2 = 1$)

```python
import torch
import matplotlib.pyplot as plt

# Visualize the per-dimension KL as a function of mu and sigma
mu = torch.linspace(-3, 3, 200)
sigma_sq = torch.linspace(0.01, 4, 200)

# KL as function of mu (sigma=1)
kl_vs_mu = 0.5 * mu**2  # only the mu penalty when sigma=1
# KL as function of sigma^2 (mu=0)
kl_vs_sigma = 0.5 * (sigma_sq - 1 - torch.log(sigma_sq))

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].plot(mu.numpy(), kl_vs_mu.numpy())
axes[0].set_xlabel('μ')
axes[0].set_ylabel('KL contribution')
axes[0].set_title('Mean penalty (σ²=1)')

axes[1].plot(sigma_sq.numpy(), kl_vs_sigma.numpy())
axes[1].axvline(x=1, color='r', linestyle='--', label='σ²=1 (minimum)')
axes[1].set_xlabel('σ²')
axes[1].set_ylabel('KL contribution')
axes[1].set_title('Variance penalty (μ=0)')
axes[1].legend()

plt.tight_layout()
plt.show()
```

## PyTorch Implementation

### VAE KL Term

```python
import torch

def kl_divergence_vae(mu: torch.Tensor, logvar: torch.Tensor,
                      reduction: str = 'sum') -> torch.Tensor:
    """KL(q || p) where q = N(mu, diag(exp(logvar))) and p = N(0, I).

    Args:
        mu: Encoder mean, shape (batch_size, latent_dim).
        logvar: Encoder log-variance, shape (batch_size, latent_dim).
        reduction: 'sum' (total), 'mean' (per element), 'batch' (per sample).

    Returns:
        KL divergence.
    """
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

    if reduction == 'sum':
        return kl_per_dim.sum()
    elif reduction == 'mean':
        return kl_per_dim.mean()
    else:  # 'batch': sum over latent dims, keep batch
        return kl_per_dim.sum(dim=1)
```

### General Gaussian KL

```python
def kl_gaussian_general(mu_p, cov_p, mu_q, cov_q):
    """KL(N(mu_p, cov_p) || N(mu_q, cov_q)) for full covariance matrices.

    Args:
        mu_p, mu_q: Mean vectors, shape (d,).
        cov_p, cov_q: Covariance matrices, shape (d, d).

    Returns:
        Scalar KL divergence.
    """
    d = mu_p.shape[0]
    cov_q_inv = torch.linalg.inv(cov_q)

    # Log determinant ratio
    log_det_ratio = torch.logdet(cov_q) - torch.logdet(cov_p)

    # Trace term
    trace_term = torch.trace(cov_q_inv @ cov_p)

    # Mahalanobis term
    diff = mu_q - mu_p
    mahal_term = diff @ cov_q_inv @ diff

    return 0.5 * (log_det_ratio - d + trace_term + mahal_term)

# Example: 2D Gaussians
mu_p = torch.tensor([1.0, 2.0])
cov_p = torch.tensor([[1.0, 0.3], [0.3, 0.5]])
mu_q = torch.tensor([0.0, 0.0])
cov_q = torch.eye(2)

kl = kl_gaussian_general(mu_p, cov_p, mu_q, cov_q)
print(f"KL divergence: {kl.item():.6f}")
```

### Numerical Stability Notes

When computing KL divergence for Gaussians in practice:

1. **Use `logvar` not `var`:** Neural networks output $\log\sigma^2$, avoiding the need for `torch.log` on potentially small variances.
2. **Use `torch.logdet` not `log(det)`:** `torch.logdet` uses the LU decomposition internally, which is numerically stable for ill-conditioned matrices.
3. **Clamp `logvar`:** Extreme values can cause overflow in `exp(logvar)`. A typical clamp range is $[-20, 20]$.

```python
# Stable implementation with clamping
def kl_stable(mu, logvar, clamp_range=20.0):
    logvar = torch.clamp(logvar, -clamp_range, clamp_range)
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)
```

## Exercises

### Exercise 1: Verify the Multivariate Formula

For $p = \mathcal{N}([1, 0]^T, I)$ and $q = \mathcal{N}([0, 0]^T, 2I)$, compute $D_{\text{KL}}(p \| q)$ analytically and verify with the `kl_gaussian_general` function.

### Exercise 2: Asymmetry for Gaussians

Compute $D_{\text{KL}}(p \| q)$ and $D_{\text{KL}}(q \| p)$ for $p = \mathcal{N}(0, 1)$ and $q = \mathcal{N}(2, 0.25)$. Explain the magnitude difference: which direction is larger, and why?

### Exercise 3: KL Landscape for VAE

Fix $\sigma^2 = 1$ and plot $D_{\text{KL}}(\mathcal{N}(\mu, 1) \| \mathcal{N}(0, 1))$ as a function of $\mu$. Then fix $\mu = 0$ and plot as a function of $\sigma^2$. Identify the minimum in each case and explain why the VAE KL term acts as a regularizer.

## Key Takeaways

The KL divergence between Gaussians admits a closed-form expression involving the log determinant ratio, the trace of the precision-weighted covariance, and the Mahalanobis distance between means. For the VAE special case with diagonal covariance and standard normal prior, the formula reduces to a sum over independent per-dimension terms, each decomposing into a mean penalty and a variance penalty. The log-variance parameterization is essential for numerical stability. These closed-form expressions eliminate the need for Monte Carlo estimation, providing exact gradients for training variational models.
