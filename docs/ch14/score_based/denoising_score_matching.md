# Denoising Score Matching

**Denoising Score Matching (DSM)** is a practical technique that transforms score estimation into a simple regression problem by using noise-perturbed data.

## Motivation

Vanilla score matching has two main issues:

1. **Computational cost**: Computing the Jacobian trace is expensive
2. **Poor estimation**: Score estimates are unreliable in low-density regions

DSM solves both problems by considering the score of a **noise-perturbed** distribution instead of the original data distribution.

## Core Idea

Instead of learning $\nabla_x \log p_{\text{data}}(x)$, we learn the score of the smoothed distribution:

$$
p_\sigma(\tilde{x}) = \int p_{\text{data}}(x) \mathcal{N}(\tilde{x}; x, \sigma^2 I) dx
$$

This is the data distribution convolved with Gaussian noise.

## The Denoising Score Matching Objective

**Theorem (Vincent, 2011)**: The optimal score network for the perturbed distribution can be found by minimizing:

$$
\mathcal{L}_{\text{DSM}}(\theta) = \mathbb{E}_{x \sim p_{\text{data}}} \mathbb{E}_{\tilde{x} \sim \mathcal{N}(x, \sigma^2 I)} \left[ \frac{1}{2} \left\| s_\theta(\tilde{x}) - \nabla_{\tilde{x}} \log p(\tilde{x} | x) \right\|^2 \right]
$$

For Gaussian noise, the conditional score is:

$$
\nabla_{\tilde{x}} \log p(\tilde{x} | x) = \nabla_{\tilde{x}} \log \mathcal{N}(\tilde{x}; x, \sigma^2 I) = -\frac{\tilde{x} - x}{\sigma^2}
$$

## Simplified DSM Objective

Substituting the Gaussian conditional score:

$$
\mathcal{L}_{\text{DSM}}(\theta) = \mathbb{E}_{x, \epsilon} \left[ \frac{1}{2} \left\| s_\theta(x + \sigma \epsilon) + \frac{\epsilon}{\sigma} \right\|^2 \right]
$$

where $\epsilon \sim \mathcal{N}(0, I)$ and $\tilde{x} = x + \sigma \epsilon$.

## Why This Works

### Intuition

The perturbed score at $\tilde{x}$ points back toward the clean data $x$. By training the network to predict this direction, we learn how to "denoise" – which is exactly what we need for generation.

### Mathematical Justification

The theorem shows that minimizing DSM is equivalent to minimizing the Fisher divergence between $s_\theta(\tilde{x})$ and $\nabla_{\tilde{x}} \log p_\sigma(\tilde{x})$, up to a constant:

$$
\mathcal{L}_{\text{DSM}}(\theta) = \mathcal{L}_{\text{ISM}}(\theta; p_\sigma) + C
$$

## Connection to Diffusion Models

DSM is the foundation of diffusion model training. In DDPM:

1. The forward process adds noise: $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$
2. The network predicts the noise: $\epsilon_\theta(x_t, t)$
3. This is equivalent to learning the score: $s_\theta(x_t, t) = -\frac{\epsilon_\theta(x_t, t)}{\sqrt{1 - \bar{\alpha}_t}}$

The DDPM loss is exactly DSM with a specific noise schedule!

## Multi-Scale Denoising Score Matching

Using a single noise level $\sigma$ has limitations:

- **Too small**: Poor coverage of space between modes
- **Too large**: Destroys data structure

**Solution**: Learn scores at multiple noise scales $\{\sigma_i\}_{i=1}^L$:

$$
\mathcal{L}_{\text{NCSN}}(\theta) = \sum_{i=1}^L \lambda(\sigma_i) \mathbb{E}_{x, \epsilon} \left[ \left\| s_\theta(x + \sigma_i \epsilon, \sigma_i) + \frac{\epsilon}{\sigma_i} \right\|^2 \right]
$$

This is the **Noise Conditional Score Network (NCSN)** objective.

## Implementation

```python
import torch
import torch.nn as nn

def denoising_score_matching_loss(score_net, x, sigma):
    """
    Basic DSM loss for a single noise level.
    
    Args:
        score_net: Network s(x) mapping inputs to scores
        x: Clean data [batch_size, dim]
        sigma: Noise standard deviation
    
    Returns:
        DSM loss
    """
    # Add noise
    noise = torch.randn_like(x)
    x_noisy = x + sigma * noise
    
    # Predict score
    score = score_net(x_noisy)
    
    # Target: -noise/sigma = gradient of log p(x_noisy|x)
    target = -noise / sigma
    
    # MSE loss
    loss = 0.5 * ((score - target) ** 2).sum(dim=-1).mean()
    
    return loss


def multi_scale_dsm_loss(score_net, x, sigmas, weights=None):
    """
    Multi-scale DSM loss (NCSN-style).
    
    Args:
        score_net: Network s(x, sigma) with conditioning
        x: Clean data [batch_size, dim]
        sigmas: List of noise levels
        weights: Optional weighting per scale
    
    Returns:
        Weighted DSM loss across scales
    """
    if weights is None:
        weights = [1.0] * len(sigmas)
    
    total_loss = 0.0
    for sigma, weight in zip(sigmas, weights):
        noise = torch.randn_like(x)
        x_noisy = x + sigma * noise
        
        # Network takes (noisy_x, sigma) as input
        score = score_net(x_noisy, sigma)
        target = -noise / sigma
        
        loss = 0.5 * ((score - target) ** 2).sum(dim=-1).mean()
        total_loss += weight * loss
    
    return total_loss
```

## Weighting Strategies

The choice of $\lambda(\sigma_i)$ affects training dynamics:

| Weighting | Formula | Effect |
|-----------|---------|--------|
| Uniform | $\lambda(\sigma) = 1$ | Equal contribution from all scales |
| $\sigma^2$ weighting | $\lambda(\sigma) = \sigma^2$ | Balances gradient magnitudes |
| SNR weighting | $\lambda(\sigma) = 1/(1 + \sigma^2)$ | Emphasizes low-noise regime |

DDPM effectively uses $\lambda(t) \propto 1 - \bar{\alpha}_t$, which corresponds to $\sigma^2$ weighting.

## Advantages of DSM

1. **No Jacobian computation**: Simple regression objective
2. **Better coverage**: Noise fills low-density regions
3. **Stable training**: Smooth targets at all noise levels
4. **Scalable**: Works for high-dimensional data (images, audio)

## Limitations

1. **Biased estimates**: Learns score of $p_\sigma$, not $p_{\text{data}}$
2. **Requires noise schedule design**: Choice of $\sigma$ levels matters
3. **Annealing needed**: Must carefully reduce noise during sampling

## Summary

Denoising score matching transforms score estimation into predicting the noise direction from perturbed data. This simple regression objective scales to high dimensions and forms the theoretical foundation for diffusion model training. By using multiple noise scales, we obtain robust score estimates that enable effective generation through annealed Langevin dynamics or the diffusion reverse process.
