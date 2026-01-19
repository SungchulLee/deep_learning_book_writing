# DDPM Loss Function

The loss function is central to training diffusion models. DDPM introduced a **simplified objective** that predicts the noise added during the forward process.

## Variational Lower Bound (VLB)

### Derivation from ELBO

The evidence lower bound for diffusion models is:

$$
\log p_\theta(x_0) \geq \mathbb{E}_q\left[\log \frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}\right] = -\mathcal{L}_{\text{VLB}}
$$

### Decomposition

The VLB decomposes into interpretable terms:

$$
\mathcal{L}_{\text{VLB}} = \mathcal{L}_T + \sum_{t=2}^{T} \mathcal{L}_{t-1} + \mathcal{L}_0
$$

where:

$$
\begin{aligned}
\mathcal{L}_T &= D_{\text{KL}}(q(x_T|x_0) \| p(x_T)) \\
\mathcal{L}_{t-1} &= D_{\text{KL}}(q(x_{t-1}|x_t, x_0) \| p_\theta(x_{t-1}|x_t)) \\
\mathcal{L}_0 &= -\log p_\theta(x_0|x_1)
\end{aligned}
$$

### Interpretation

- $\mathcal{L}_T$: Prior matching (typically constant, no gradients)
- $\mathcal{L}_{t-1}$: Denoising matching at each step
- $\mathcal{L}_0$: Reconstruction quality

## KL Between Gaussians

Since both distributions are Gaussian, $\mathcal{L}_{t-1}$ has a closed form.

### General KL Formula

For $p = \mathcal{N}(\mu_1, \sigma_1^2)$ and $q = \mathcal{N}(\mu_2, \sigma_2^2)$:

$$
D_{\text{KL}}(p \| q) = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2}
$$

### Applied to DDPM

With the true posterior $q(x_{t-1}|x_t, x_0) = \mathcal{N}(\tilde{\mu}_t, \tilde{\beta}_t I)$ and model $p_\theta(x_{t-1}|x_t) = \mathcal{N}(\mu_\theta, \sigma_t^2 I)$:

$$
\mathcal{L}_{t-1} = \mathbb{E}_q\left[\frac{1}{2\sigma_t^2}\|\tilde{\mu}_t(x_t, x_0) - \mu_\theta(x_t, t)\|^2\right] + C
$$

where $C$ contains terms not depending on $\theta$.

## Noise Prediction Reparameterization

### From Mean to Noise

The true posterior mean is:

$$
\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}x_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}x_t
$$

Using $x_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}(x_t - \sqrt{1-\bar{\alpha}_t}\epsilon)$:

$$
\tilde{\mu}_t = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon\right)
$$

### Model Parameterization

Similarly, if the model predicts noise $\epsilon_\theta(x_t, t)$:

$$
\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t, t)\right)
$$

### Substituting into the Loss

$$
\|\tilde{\mu}_t - \mu_\theta\|^2 = \frac{(1-\alpha_t)^2}{\alpha_t(1-\bar{\alpha}_t)}\|\epsilon - \epsilon_\theta(x_t, t)\|^2
$$

## The VLB in Terms of Noise

$$
\mathcal{L}_{t-1} = \mathbb{E}_{x_0, \epsilon}\left[\frac{(1-\alpha_t)^2}{2\sigma_t^2\alpha_t(1-\bar{\alpha}_t)}\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]
$$

Define the weighting:

$$
\lambda_t = \frac{(1-\alpha_t)^2}{2\sigma_t^2\alpha_t(1-\bar{\alpha}_t)}
$$

Then:

$$
\mathcal{L}_{\text{VLB}} = \mathbb{E}_{t, x_0, \epsilon}\left[\lambda_t \|\epsilon - \epsilon_\theta(x_t, t)\|^2\right] + C
$$

## Simplified Objective

### Ho et al.'s Discovery

DDPM found that dropping the weighting $\lambda_t$ improves sample quality:

$$
\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]
$$

This is equivalent to setting $\lambda_t = 1$ for all $t$.

### Why Simplified Works Better

| Aspect | VLB Weighting | Uniform Weighting |
|--------|---------------|-------------------|
| High noise (large $t$) | Low weight | Equal weight |
| Low noise (small $t$) | High weight | Equal weight |
| Training focus | Fine details | Balanced |
| Sample quality | Lower FID | Higher FID |

The simplified objective:
1. Gives more weight to high-noise timesteps
2. Helps learn coarse structure
3. Implicitly emphasizes perceptually important features

## Alternative Parameterizations

### $x_0$ Prediction

Instead of $\epsilon$, predict $x_0$ directly:

$$
\mathcal{L}_{x_0} = \mathbb{E}_{t, x_0, \epsilon}\left[\|\hat{x}_0 - x_{0,\theta}(x_t, t)\|^2\right]
$$

**Pros**: More interpretable
**Cons**: Can be unstable at high noise levels

### Velocity Prediction

Predict the "velocity" $v_t = \sqrt{\bar{\alpha}_t}\epsilon - \sqrt{1-\bar{\alpha}_t}x_0$:

$$
\mathcal{L}_v = \mathbb{E}_{t, x_0, \epsilon}\left[\|v_t - v_\theta(x_t, t)\|^2\right]
$$

**Pros**: Balanced learning across noise levels
**Cons**: Requires modified sampling

### Equivalence

All parameterizations are mathematically equivalent:

| Predict | Recover $\epsilon$ | Recover $x_0$ |
|---------|-------------------|---------------|
| $\epsilon_\theta$ | Direct | $\frac{x_t - \sqrt{1-\bar{\alpha}_t}\epsilon_\theta}{\sqrt{\bar{\alpha}_t}}$ |
| $x_{0,\theta}$ | $\frac{x_t - \sqrt{\bar{\alpha}_t}x_{0,\theta}}{\sqrt{1-\bar{\alpha}_t}}$ | Direct |
| $v_\theta$ | $\sqrt{\bar{\alpha}_t}v_\theta + \sqrt{1-\bar{\alpha}_t}x_t$ | $\sqrt{\bar{\alpha}_t}x_t - \sqrt{1-\bar{\alpha}_t}v_\theta$ |

## Implementation

```python
import torch
import torch.nn.functional as F

def simple_loss(model, x_0, alphas_bar, T):
    """DDPM simplified loss."""
    batch_size = x_0.shape[0]
    device = x_0.device
    
    # Sample random timesteps
    t = torch.randint(0, T, (batch_size,), device=device)
    
    # Sample noise
    noise = torch.randn_like(x_0)
    
    # Create noisy samples
    alpha_bar_t = alphas_bar[t].view(-1, 1, 1, 1)
    x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise
    
    # Predict noise
    noise_pred = model(x_t, t)
    
    # Simple MSE loss (uniform weighting)
    loss = F.mse_loss(noise_pred, noise)
    
    return loss


def vlb_loss(model, x_0, alphas, alphas_bar, betas, T):
    """VLB loss with proper weighting."""
    batch_size = x_0.shape[0]
    device = x_0.device
    
    # Sample timesteps (skip t=0)
    t = torch.randint(1, T, (batch_size,), device=device)
    
    # Sample noise
    noise = torch.randn_like(x_0)
    
    # Create noisy samples
    alpha_bar_t = alphas_bar[t].view(-1, 1, 1, 1)
    x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise
    
    # Predict noise
    noise_pred = model(x_t, t)
    
    # Compute weighting
    alpha_t = alphas[t].view(-1, 1, 1, 1)
    beta_t = betas[t].view(-1, 1, 1, 1)
    sigma_t_sq = beta_t  # Simplified: use beta as variance
    
    weight = (1 - alpha_t) ** 2 / (2 * sigma_t_sq * alpha_t * (1 - alpha_bar_t))
    
    # Weighted MSE loss
    loss = (weight * (noise_pred - noise) ** 2).mean()
    
    return loss
```

## Loss Weighting Strategies

### SNR Weighting (Min-SNR)

Weight by signal-to-noise ratio:

$$
\lambda_t = \min\left(\text{SNR}(t), \gamma\right)
$$

where $\text{SNR}(t) = \frac{\bar{\alpha}_t}{1-\bar{\alpha}_t}$ and $\gamma$ is a clipping threshold (e.g., 5).

### P2 Weighting

Progressive weighting from Choi et al.:

$$
\lambda_t = \frac{1}{(1 + \text{SNR}(t))^k}
$$

with $k \approx 1$ emphasizing higher noise levels.

## Summary

The DDPM loss predicts the noise $\epsilon$ added during the forward process. While the VLB provides a principled objective with per-timestep weighting, the simplified loss (uniform weighting) often produces better samples. Alternative parameterizations (predicting $x_0$ or velocity) are mathematically equivalent but may have different training dynamics. Modern methods often use SNR-based weighting for improved results.
