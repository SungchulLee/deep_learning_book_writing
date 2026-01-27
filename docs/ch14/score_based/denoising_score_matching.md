# Denoising Score Matching

## Learning Objectives

By the end of this section, you will be able to:

1. Derive the denoising score matching (DSM) objective from first principles
2. Understand why DSM is equivalent to explicit score matching as noise vanishes
3. Explain the geometric intuition behind learning to denoise
4. Implement DSM efficiently with various noise kernels
5. Apply multi-scale DSM and connect it to diffusion models
6. Select appropriate noise levels and weighting strategies

## Prerequisites

- Score matching fundamentals
- Gaussian distribution properties
- PyTorch training loops

---

## 1. Motivation

### 1.1 Two Problems with Vanilla Score Matching

Explicit Score Matching (ESM) has two critical issues:

| Problem | Description | Impact |
|---------|-------------|--------|
| **Computational cost** | Computing $\text{tr}(\nabla_{\mathbf{x}} \mathbf{s}_\theta)$ requires $D$ backward passes | Impractical for images ($D > 10^4$) |
| **Poor estimation** | Score estimates are unreliable in low-density regions | Sampling fails between modes |

DSM elegantly solves both problems by learning the score of a **noise-perturbed** distribution.

### 1.2 The Core Idea

Instead of learning $\nabla_{\mathbf{x}} \log p_{\text{data}}(\mathbf{x})$ directly, we learn the score of a smoothed distribution:

$$
p_\sigma(\tilde{\mathbf{x}}) = \int p_{\text{data}}(\mathbf{x}) \, \mathcal{N}(\tilde{\mathbf{x}}; \mathbf{x}, \sigma^2 \mathbf{I}) \, d\mathbf{x}
$$

This is the data distribution **convolved with Gaussian noise**—a smoothed version that fills in low-density regions.

### 1.3 The Denoising Insight

Vincent (2011) discovered that adding noise creates a **known target score**:

$$
q(\tilde{\mathbf{x}}|\mathbf{x}) = \mathcal{N}(\tilde{\mathbf{x}}|\mathbf{x}, \sigma^2\mathbf{I}) \implies \nabla_{\tilde{\mathbf{x}}} \log q(\tilde{\mathbf{x}}|\mathbf{x}) = -\frac{\tilde{\mathbf{x}} - \mathbf{x}}{\sigma^2}
$$

The perturbed score at $\tilde{\mathbf{x}}$ **points back toward the clean data** $\mathbf{x}$. By training the network to predict this direction, we learn how to "denoise"—which is exactly what we need for generation!

---

## 2. Mathematical Derivation

### 2.1 The Perturbed Distribution

Given clean data $\mathbf{x} \sim p_{\text{data}}(\mathbf{x})$ and Gaussian noise kernel:

$$
q(\tilde{\mathbf{x}}|\mathbf{x}) = \mathcal{N}(\tilde{\mathbf{x}}|\mathbf{x}, \sigma^2\mathbf{I}) = \frac{1}{(2\pi\sigma^2)^{D/2}} \exp\left(-\frac{\|\tilde{\mathbf{x}} - \mathbf{x}\|^2}{2\sigma^2}\right)
$$

The marginal distribution of noisy samples is:

$$
q_\sigma(\tilde{\mathbf{x}}) = \int p_{\text{data}}(\mathbf{x}) \, q(\tilde{\mathbf{x}}|\mathbf{x}) \, d\mathbf{x}
$$

### 2.2 The DSM Objective

**Theorem (Vincent, 2011)**: The optimal score network for the perturbed distribution can be found by minimizing:

$$
\mathcal{L}_{\text{DSM}}(\theta) = \frac{1}{2} \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}} \mathbb{E}_{\tilde{\mathbf{x}} \sim q(\cdot|\mathbf{x})} \left[\|\mathbf{s}_\theta(\tilde{\mathbf{x}}) - \nabla_{\tilde{\mathbf{x}}} \log q(\tilde{\mathbf{x}}|\mathbf{x})\|^2\right]
$$

### 2.3 Computing the Target Score

The log of the noise kernel:

$$
\log q(\tilde{\mathbf{x}}|\mathbf{x}) = -\frac{\|\tilde{\mathbf{x}} - \mathbf{x}\|^2}{2\sigma^2} - \frac{D}{2}\log(2\pi\sigma^2)
$$

Taking the gradient with respect to $\tilde{\mathbf{x}}$:

$$
\nabla_{\tilde{\mathbf{x}}} \log q(\tilde{\mathbf{x}}|\mathbf{x}) = -\frac{\tilde{\mathbf{x}} - \mathbf{x}}{\sigma^2}
$$

Since $\tilde{\mathbf{x}} = \mathbf{x} + \sigma\boldsymbol{\epsilon}$ where $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$:

$$
\nabla_{\tilde{\mathbf{x}}} \log q(\tilde{\mathbf{x}}|\mathbf{x}) = -\frac{\sigma\boldsymbol{\epsilon}}{\sigma^2} = -\frac{\boldsymbol{\epsilon}}{\sigma}
$$

### 2.4 Final Practical Loss

$$
\boxed{\mathcal{L}_{\text{DSM}}(\theta) = \frac{1}{2} \mathbb{E}_{\mathbf{x}, \boldsymbol{\epsilon}} \left[\left\|\mathbf{s}_\theta(\mathbf{x} + \sigma\boldsymbol{\epsilon}) + \frac{\boldsymbol{\epsilon}}{\sigma}\right\|^2\right]}
$$

where $\mathbf{x} \sim p_{\text{data}}$ and $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$.

!!! info "Key Insight"
    The target $-\boldsymbol{\epsilon}/\sigma$ is simply the **negative noise direction**, scaled by inverse noise level. Training becomes a simple regression problem!

---

## 3. Why This Works: Theoretical Justification

### 3.1 Equivalence to ESM

**Theorem**: The DSM and ESM objectives differ only by a constant:

$$
\mathcal{L}_{\text{DSM}}(\theta) = \mathcal{L}_{\text{ESM}}(\theta; p_\sigma) + C_\sigma
$$

where $\mathcal{L}_{\text{ESM}}(\theta; p_\sigma)$ is the ESM objective on the perturbed distribution.

### 3.2 Proof Sketch

The key is showing that minimizing DSM is equivalent to minimizing the **Fisher divergence** between $\mathbf{s}_\theta(\tilde{\mathbf{x}})$ and $\nabla_{\tilde{\mathbf{x}}} \log p_\sigma(\tilde{\mathbf{x}})$.

Starting from the Fisher divergence:

$$
D_F = \frac{1}{2}\mathbb{E}_{q_\sigma}[\|\mathbf{s}_\theta(\tilde{\mathbf{x}}) - \nabla_{\tilde{\mathbf{x}}} \log q_\sigma(\tilde{\mathbf{x}})\|^2]
$$

Expanding and using the law of total expectation:

$$
\mathbb{E}_{q_\sigma}[\mathbf{s}_\theta^\top \nabla \log q_\sigma] = \mathbb{E}_{p_{\text{data}}} \mathbb{E}_{q(\tilde{\mathbf{x}}|\mathbf{x})}[\mathbf{s}_\theta(\tilde{\mathbf{x}})^\top \nabla_{\tilde{\mathbf{x}}} \log q(\tilde{\mathbf{x}}|\mathbf{x})]
$$

This shows DSM's target is the correct gradient for learning scores on $q_\sigma$.

### 3.3 Limit as $\sigma \to 0$

As the noise level vanishes:

$$
\lim_{\sigma \to 0} q_\sigma(\tilde{\mathbf{x}}) = p_{\text{data}}(\tilde{\mathbf{x}})
$$

Therefore:

$$
\lim_{\sigma \to 0} \mathcal{L}_{\text{DSM}}(\theta) = \mathcal{L}_{\text{ESM}}(\theta) + C
$$

!!! success "Practical Implication"
    DSM with small $\sigma$ approximates ESM **without computing any Jacobians**!

---

## 4. Multi-Scale Denoising Score Matching

### 4.1 The Single-Scale Limitation

Using a single noise level $\sigma$ has problems:

| $\sigma$ | Issue |
|----------|-------|
| **Too small** | Poor coverage of space between modes; score undefined far from data |
| **Too large** | Destroys data structure; learns trivial scores |

### 4.2 Solution: Learn Scores at Multiple Scales

The **Noise Conditional Score Network (NCSN)** learns scores across a range of noise levels $\{\sigma_i\}_{i=1}^L$:

$$
\mathcal{L}_{\text{NCSN}}(\theta) = \sum_{i=1}^L \lambda(\sigma_i) \, \mathbb{E}_{\mathbf{x}, \boldsymbol{\epsilon}} \left[\left\|\mathbf{s}_\theta(\mathbf{x} + \sigma_i\boldsymbol{\epsilon}, \sigma_i) + \frac{\boldsymbol{\epsilon}}{\sigma_i}\right\|^2\right]
$$

**Benefits:**
- Large $\sigma$: Fills low-density regions, provides global structure
- Small $\sigma$: Captures fine details near data manifold
- Smooth interpolation between scales during sampling

### 4.3 Noise Schedule Design

**Geometric schedule** (most common):

$$
\sigma_i = \sigma_{\min} \cdot \left(\frac{\sigma_{\max}}{\sigma_{\min}}\right)^{\frac{i-1}{L-1}}, \quad i = 1, \ldots, L
$$

**Typical values:**
- $\sigma_{\max} \approx$ maximum pairwise distance in data (e.g., 50 for images)
- $\sigma_{\min} \approx$ noise floor (e.g., 0.01)
- $L \in [10, 1000]$ levels

---

## 5. Weighting Strategies

The choice of $\lambda(\sigma)$ significantly affects training dynamics:

| Weighting | Formula | Effect | Use Case |
|-----------|---------|--------|----------|
| **Uniform** | $\lambda(\sigma) = 1$ | Equal contribution from all scales | Baseline |
| **$\sigma^2$ weighting** | $\lambda(\sigma) = \sigma^2$ | Balances gradient magnitudes | NCSN default |
| **SNR weighting** | $\lambda(\sigma) = 1/(1 + \sigma^2)$ | Emphasizes low-noise regime | Fine details |
| **DDPM-style** | $\lambda(t) \propto 1 - \bar{\alpha}_t$ | Equivalent to $\sigma^2$ | Diffusion models |

### 5.1 Why $\sigma^2$ Weighting?

The score magnitude scales as $\|\mathbf{s}\| \sim 1/\sigma$, so the loss scales as:

$$
\|\mathbf{s}_\theta + \boldsymbol{\epsilon}/\sigma\|^2 \sim 1/\sigma^2
$$

Multiplying by $\sigma^2$ normalizes contributions across scales:

$$
\sigma^2 \cdot \left\|\mathbf{s}_\theta + \frac{\boldsymbol{\epsilon}}{\sigma}\right\|^2 = \|\sigma \cdot \mathbf{s}_\theta + \boldsymbol{\epsilon}\|^2
$$

This is equivalent to **predicting the noise** $\boldsymbol{\epsilon}$ directly!

---

## 6. Connection to Diffusion Models

### 6.1 DSM is DDPM Training

The DDPM training objective:

$$
\mathcal{L}_{\text{DDPM}}(t) = \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}}\left[\|\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) - \boldsymbol{\epsilon}\|^2\right]
$$

where $\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}$.

**This is exactly DSM** with:

| DDPM | DSM Equivalent |
|------|----------------|
| Noise predictor $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$ | Score: $\mathbf{s}_\theta = -\boldsymbol{\epsilon}_\theta / \sqrt{1-\bar{\alpha}_t}$ |
| Forward process $\mathbf{x}_t$ | Noisy sample $\tilde{\mathbf{x}} = \mathbf{x} + \sigma_t \boldsymbol{\epsilon}$ |
| Noise schedule $\bar{\alpha}_t$ | Noise level $\sigma_t = \sqrt{(1-\bar{\alpha}_t)/\bar{\alpha}_t}$ |

### 6.2 Unified View

$$
\text{DSM} \iff \text{DDPM Training} \iff \text{Score-based SDE}
$$

All three perspectives describe the same underlying mathematical framework!

!!! tip "Key Insight"
    Understanding DSM deeply means understanding the foundation of **all modern diffusion models**.

---

## 7. Alternative Noise Kernels

While Gaussian noise is standard, other kernels can be useful:

### 7.1 Gaussian Kernel (Standard)

$$
q(\tilde{\mathbf{x}}|\mathbf{x}) = \mathcal{N}(\tilde{\mathbf{x}}|\mathbf{x}, \sigma^2\mathbf{I})
$$

- **Target score**: $-(\tilde{\mathbf{x}} - \mathbf{x})/\sigma^2 = -\boldsymbol{\epsilon}/\sigma$
- **Advantages**: Simple, well-understood, connects to diffusion models

### 7.2 Laplace Kernel

$$
q(\tilde{\mathbf{x}}|\mathbf{x}) = \text{Laplace}(\tilde{\mathbf{x}}|\mathbf{x}, b)
$$

- **Target score**: $-\text{sign}(\tilde{\mathbf{x}} - \mathbf{x})/b$
- **Use case**: Heavy-tailed data, sparse perturbations

### 7.3 Student-t Kernel

$$
q(\tilde{\mathbf{x}}|\mathbf{x}) = \text{Student-}t_\nu(\tilde{\mathbf{x}}|\mathbf{x}, \sigma^2)
$$

- **Target score**: $-\frac{(\nu + D)(\tilde{\mathbf{x}} - \mathbf{x})}{\nu\sigma^2 + \|\tilde{\mathbf{x}} - \mathbf{x}\|^2}$
- **Use case**: Robustness to outliers

---

## 8. PyTorch Implementation

### 8.1 Basic DSM Loss

```python
import torch
import torch.nn as nn

def denoising_score_matching_loss(
    score_net: nn.Module, 
    x: torch.Tensor, 
    sigma: float
) -> torch.Tensor:
    """
    Basic DSM loss for a single noise level.
    
    Args:
        score_net: Network s(x) mapping inputs to scores
        x: Clean data [batch_size, dim]
        sigma: Noise standard deviation
    
    Returns:
        DSM loss (scalar)
    """
    # Add noise: x̃ = x + σε
    noise = torch.randn_like(x)
    x_noisy = x + sigma * noise
    
    # Predict score at noisy location
    score = score_net(x_noisy)
    
    # Target: -ε/σ = ∇log p(x̃|x)
    target = -noise / sigma
    
    # MSE loss
    loss = 0.5 * ((score - target) ** 2).sum(dim=-1).mean()
    
    return loss
```

### 8.2 Multi-Scale DSM (NCSN-style)

```python
def multi_scale_dsm_loss(
    score_net: nn.Module, 
    x: torch.Tensor, 
    sigmas: list[float],
    weights: list[float] = None
) -> torch.Tensor:
    """
    Multi-scale DSM loss (NCSN-style).
    
    Args:
        score_net: Network s(x, sigma) with noise conditioning
        x: Clean data [batch_size, dim]
        sigmas: List of noise levels
        weights: Optional weighting per scale (default: σ² weighting)
    
    Returns:
        Weighted DSM loss across scales
    """
    if weights is None:
        # Default: σ² weighting for balanced gradients
        weights = [sigma ** 2 for sigma in sigmas]
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
    
    total_loss = 0.0
    
    for sigma, weight in zip(sigmas, weights):
        noise = torch.randn_like(x)
        x_noisy = x + sigma * noise
        
        # Network takes (noisy_x, sigma) as input
        # Can pass sigma as scalar or expanded tensor
        sigma_tensor = torch.full((x.shape[0], 1), sigma, device=x.device)
        score = score_net(x_noisy, sigma_tensor)
        
        target = -noise / sigma
        
        loss = 0.5 * ((score - target) ** 2).sum(dim=-1).mean()
        total_loss += weight * loss
    
    return total_loss
```

### 8.3 Efficient Random Sigma Selection

```python
def dsm_loss_random_sigma(
    score_net: nn.Module,
    x: torch.Tensor,
    sigma_min: float = 0.01,
    sigma_max: float = 10.0
) -> torch.Tensor:
    """
    DSM loss with randomly sampled sigma per batch element.
    More efficient than iterating over all sigma levels.
    
    Args:
        score_net: Network s(x, sigma)
        x: Clean data [batch_size, dim]
        sigma_min, sigma_max: Noise level range
    
    Returns:
        DSM loss
    """
    batch_size = x.shape[0]
    device = x.device
    
    # Sample sigma uniformly in log space (geometric distribution)
    log_sigma = torch.rand(batch_size, device=device) * \
                (torch.log(torch.tensor(sigma_max)) - torch.log(torch.tensor(sigma_min))) + \
                torch.log(torch.tensor(sigma_min))
    sigma = torch.exp(log_sigma).view(-1, 1)
    
    # Add noise
    noise = torch.randn_like(x)
    x_noisy = x + sigma * noise
    
    # Predict score
    score = score_net(x_noisy, sigma)
    target = -noise / sigma
    
    # σ² weighted loss
    loss = 0.5 * (sigma ** 2 * (score - target) ** 2).sum(dim=-1).mean()
    
    return loss
```

### 8.4 Complete Training Example

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate Swiss Roll data
def generate_swiss_roll(n_samples: int = 5000) -> torch.Tensor:
    t = 1.5 * np.pi * (1 + 2 * np.random.rand(n_samples))
    x = t * np.cos(t) + 0.1 * np.random.randn(n_samples)
    y = t * np.sin(t) + 0.1 * np.random.randn(n_samples)
    data = np.stack([x, y], axis=1)
    data = (data - data.mean(0)) / data.std(0)
    return torch.tensor(data, dtype=torch.float32)

# Score network with sigma conditioning
class ScoreNet(nn.Module):
    def __init__(self, data_dim: int = 2, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(data_dim + 1, hidden_dim),  # +1 for sigma
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, data_dim)
        )
    
    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        # Concatenate x and log(sigma) for conditioning
        sigma_feat = torch.log(sigma) / 4.0  # Normalize log-sigma
        inp = torch.cat([x, sigma_feat], dim=-1)
        return self.net(inp)

# Training loop
samples = generate_swiss_roll(5000)
model = ScoreNet()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(5000):
    idx = torch.randperm(len(samples))[:256]
    batch = samples[idx]
    
    loss = dsm_loss_random_sigma(model, batch, sigma_min=0.01, sigma_max=5.0)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 1000 == 0:
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")
```

---

## 9. Noise Level Selection

### 9.1 The Bias-Variance Trade-off

| $\sigma$ | Bias | Variance | Coverage | Detail |
|----------|------|----------|----------|--------|
| Very small | Low | High | Only near data | High |
| Small | Low | Medium | Good local | Good |
| Medium | Medium | Low | Good global | Medium |
| Large | High | Very low | Everywhere | Lost |

### 9.2 Empirical Guidelines

1. **Normalize data first**: Zero mean, unit variance
2. **For single-scale DSM**: $\sigma \approx 0.5$ for normalized data
3. **Data-dependent**: $\sigma \approx 0.5 \times \text{median nearest-neighbor distance}$
4. **For multi-scale**: Geometric schedule from $\sigma_{\min} = 0.01$ to $\sigma_{\max} = \max \|\mathbf{x}_i - \mathbf{x}_j\|$

---

## 10. Advantages and Limitations

### 10.1 Advantages

| Advantage | Explanation |
|-----------|-------------|
| **No Jacobian computation** | Simple regression objective |
| **Better coverage** | Noise fills low-density regions |
| **Stable training** | Smooth targets at all noise levels |
| **Scalable** | Works for high-dimensional data (images, audio) |
| **Foundation for diffusion** | Same objective used in DDPM, score SDEs |

### 10.2 Limitations

| Limitation | Explanation |
|------------|-------------|
| **Biased estimates** | Learns score of $p_\sigma$, not $p_{\text{data}}$ |
| **Noise schedule design** | Choice of $\sigma$ levels matters |
| **Annealing needed** | Must carefully reduce noise during sampling |

---

## 11. Summary

| Aspect | Description |
|--------|-------------|
| **Objective** | $\mathcal{L} = \frac{1}{2}\mathbb{E}[\|\mathbf{s}_\theta(\tilde{\mathbf{x}}) + \boldsymbol{\epsilon}/\sigma\|^2]$ |
| **Target** | $-(\tilde{\mathbf{x}} - \mathbf{x})/\sigma^2 = -\boldsymbol{\epsilon}/\sigma$ |
| **Intuition** | Learn to point from noisy sample back to clean data |
| **Computation** | Single forward + backward pass |
| **Equivalence** | $\mathcal{L}_{\text{DSM}} \to \mathcal{L}_{\text{ESM}}$ as $\sigma \to 0$ |
| **Multi-scale** | NCSN: Learn scores at multiple $\sigma$ levels |
| **Diffusion connection** | DDPM training = DSM with specific schedule |

!!! tip "Key Takeaways"
    1. **DSM transforms score estimation into denoising**—predict the noise direction
    2. **The target is trivial to compute**: just $-\boldsymbol{\epsilon}/\sigma$
    3. **Multi-scale DSM (NCSN)** provides robust coverage from global to local
    4. **DSM is the foundation** of all modern diffusion models
    5. **$\sigma^2$ weighting** balances gradients and is equivalent to noise prediction

---

## Exercises

1. **Derive the Laplace kernel target**: For $q(\tilde{x}|x) = \frac{1}{2b}e^{-|x-\tilde{x}|/b}$, show that $\nabla_{\tilde{x}} \log q = -\text{sign}(\tilde{x}-x)/b$.

2. **Multi-noise experiment**: Train models with $\sigma \in \{0.1, 0.3, 0.5, 1.0\}$ on Swiss roll. Plot learned score fields and compare sample quality.

3. **Weighting study**: Compare uniform vs $\sigma^2$ weighting for multi-scale DSM. Measure final loss at each scale.

4. **DSM for images**: Implement DSM on MNIST (flatten to 784D). Use a simple MLP and visualize samples via Langevin dynamics.

5. **DDPM equivalence**: Starting from the DDPM objective, derive the equivalent DSM formulation. Show the relationship between $\bar{\alpha}_t$ and $\sigma_t$.

---

## References

1. Vincent, P. (2011). "A Connection Between Score Matching and Denoising Autoencoders." *Neural Computation*.
2. Song, Y., & Ermon, S. (2019). "Generative Modeling by Estimating Gradients of the Data Distribution." *NeurIPS*.
3. Song, Y., & Ermon, S. (2020). "Improved Techniques for Training Score-Based Generative Models." *NeurIPS*.
4. Ho, J., et al. (2020). "Denoising Diffusion Probabilistic Models." *NeurIPS*.
