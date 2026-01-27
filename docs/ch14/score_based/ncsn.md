# Noise Conditional Score Networks (NCSN)

## Introduction

**Noise Conditional Score Networks (NCSN)** address a fundamental challenge in score-based generative models: learning accurate scores across all regions of data space. By training on multiple noise levels simultaneously, NCSN learns a family of score functions that enable sampling via annealed Langevin dynamics.

NCSN provides the conceptual bridge from basic score matching to modern diffusion models—it's where the key insight of "multi-scale denoising" was first formalized for generative modeling.

!!! info "Key Innovation"
    Instead of learning a single score function $\mathbf{s}_\theta(\mathbf{x})$, NCSN learns $\mathbf{s}_\theta(\mathbf{x}, \sigma)$—a score function **conditioned on the noise level** $\sigma$.

## Learning Objectives

By the end of this section, you will be able to:

1. Understand why single-noise-level score matching fails for complex data
2. Derive the multi-scale noise conditioning approach
3. Design geometric noise schedules with appropriate $\sigma_{\min}$ and $\sigma_{\max}$
4. Implement NCSN with proper loss weighting
5. Generate samples using annealed Langevin dynamics
6. Connect NCSN to modern diffusion models (DDPM, score SDEs)

## Prerequisites

- Denoising score matching
- Langevin dynamics basics
- PyTorch neural network design

---

## 1. The Multi-Scale Score Problem

### 1.1 Single Noise Level Limitations

With a single noise level $\sigma$, score matching faces a fundamental dilemma:

| $\sigma$ | Problem |
|----------|---------|
| **Too small** | Score is accurate near data manifold but **undefined/unreliable** in low-density regions |
| **Too large** | Score is smooth everywhere but **too blurry** to capture fine data structure |

When training with a single $\sigma$, the model only sees data within $\approx 3\sigma$ of the data manifold. In low-density regions:

- Few or no training samples exist
- Score estimates are inaccurate or random
- Sampling gets stuck or diverges

!!! warning "The Core Problem"
    Score matching works well **near the data** but fails in **low-density regions** where samples must traverse during generation.

### 1.2 Visual Intuition

```
                 ← Very few samples here (low-density)
    Data manifold
         ↓
    ∙∙∙∙∙∙∙∙∙∙∙∙∙∙∙∙∙∙  (score well-estimated)
    ∙∙∙∙∙∙∙∙∙∙∙∙∙∙∙∙∙∙
         ↑
    Score accurate here
    
    Random initialization →  ×  (far from data)
                              ↓
                         Score unreliable!
                         How to reach data?
```

### 1.3 The Solution: Multi-Scale Noise

Train on a sequence of noise levels $\sigma_1 > \sigma_2 > \cdots > \sigma_L$:

| Noise Level | Purpose |
|-------------|---------|
| **Large $\sigma$** | Fills space, provides global coverage, guides from far away |
| **Medium $\sigma$** | Bridges global and local structure |
| **Small $\sigma$** | Accurate near data, captures fine details |

Learn $\mathbf{s}_\theta(\mathbf{x}, \sigma_i) \approx \nabla_{\mathbf{x}} \log p_{\sigma_i}(\mathbf{x})$ for all $i$.

---

## 2. NCSN Architecture

### 2.1 Noise-Conditioned Score Network

Instead of $\mathbf{s}_\theta(\mathbf{x})$, learn $\mathbf{s}_\theta(\mathbf{x}, \sigma)$:

$$
\mathbf{s}_\theta(\mathbf{x}, \sigma) \approx \nabla_{\mathbf{x}} \log q_\sigma(\mathbf{x})
$$

where the noise-perturbed distribution is:

$$
q_\sigma(\mathbf{x}) = \int p_{\text{data}}(\mathbf{y}) \mathcal{N}(\mathbf{x}|\mathbf{y}, \sigma^2\mathbf{I}) \, d\mathbf{y}
$$

### 2.2 Noise Schedule Design

**Geometric schedule** (most common and effective):

$$
\sigma_i = \sigma_{\max} \cdot \left(\frac{\sigma_{\min}}{\sigma_{\max}}\right)^{\frac{i-1}{L-1}}, \quad i = 1, \ldots, L
$$

Equivalently in log-space:

$$
\log \sigma_i = \log \sigma_{\max} + \frac{i-1}{L-1}(\log \sigma_{\min} - \log \sigma_{\max})
$$

**Guidelines for choosing parameters:**

| Parameter | Guideline | Typical Value |
|-----------|-----------|---------------|
| $\sigma_{\max}$ | Comparable to data diameter / max pairwise distance | 10-50 for images |
| $\sigma_{\min}$ | Small enough to approximate clean data | 0.01 |
| $L$ | More levels = smoother annealing | 10-1000 |

### 2.3 Training Objective

$$
\mathcal{L}_{\text{NCSN}}(\theta) = \frac{1}{L} \sum_{i=1}^L \lambda(\sigma_i) \, \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}} \mathbb{E}_{\tilde{\mathbf{x}} \sim q_{\sigma_i}(\cdot|\mathbf{x})} \left[\|\mathbf{s}_\theta(\tilde{\mathbf{x}}, \sigma_i) - \nabla_{\tilde{\mathbf{x}}} \log q_{\sigma_i}(\tilde{\mathbf{x}}|\mathbf{x})\|^2\right]
$$

Since the conditional score is known: $\nabla_{\tilde{\mathbf{x}}} \log q_{\sigma_i}(\tilde{\mathbf{x}}|\mathbf{x}) = -(\tilde{\mathbf{x}} - \mathbf{x})/\sigma_i^2 = -\boldsymbol{\epsilon}/\sigma_i$

$$
\boxed{\mathcal{L}_{\text{NCSN}}(\theta) = \frac{1}{L} \sum_{i=1}^L \lambda(\sigma_i) \, \mathbb{E}_{\mathbf{x}, \boldsymbol{\epsilon}} \left[\left\|\mathbf{s}_\theta(\mathbf{x} + \sigma_i \boldsymbol{\epsilon}, \sigma_i) + \frac{\boldsymbol{\epsilon}}{\sigma_i}\right\|^2\right]}
$$

### 2.4 Loss Weighting

The choice of $\lambda(\sigma)$ significantly affects training:

| Weighting | Formula | Effect |
|-----------|---------|--------|
| **Uniform** | $\lambda(\sigma) = 1$ | Equal contribution per level |
| **$\sigma^2$ (recommended)** | $\lambda(\sigma) = \sigma^2$ | Scale-invariant, balanced gradients |
| **Inverse variance** | $\lambda(\sigma) = 1/\sigma^2$ | Emphasizes small noise |

**Why $\sigma^2$ weighting works:**

The score magnitude scales as $\|\mathbf{s}\| \sim 1/\sigma$, so the loss scales as $\sim 1/\sigma^2$. Multiplying by $\sigma^2$ normalizes:

$$
\sigma^2 \cdot \left\|\mathbf{s}_\theta + \frac{\boldsymbol{\epsilon}}{\sigma}\right\|^2 = \|\sigma \cdot \mathbf{s}_\theta + \boldsymbol{\epsilon}\|^2
$$

This is equivalent to **predicting the noise** $\boldsymbol{\epsilon}$ directly—the same parameterization used in DDPM!

---

## 3. Annealed Langevin Dynamics

### 3.1 Standard Langevin (Fails)

Single-scale Langevin dynamics:

$$
\mathbf{x}_{t+1} = \mathbf{x}_t + \frac{\epsilon}{2} \mathbf{s}_\theta(\mathbf{x}_t) + \sqrt{\epsilon} \, \mathbf{z}_t, \quad \mathbf{z}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

**Problem**: Starting from random initialization far from data, the score is unreliable and samples may never reach the data manifold.

### 3.2 Annealed Langevin Dynamics (ALD)

**Key idea**: Start with large noise level (global coverage), gradually reduce to small noise (fine details).

**Algorithm:**

```
Initialize: x ~ N(0, σ_max² I)

for i = 1, ..., L:
    σ = σ_i                              # Current noise level
    α = ε × (σ / σ_min)²                 # Adaptive step size
    
    for t = 1, ..., T:
        z ~ N(0, I)
        x = x + (α/2) × s_θ(x, σ) + √α × z
        
return x
```

### 3.3 Step Size Scheduling

The step size should scale with $\sigma^2$:

$$
\alpha_i = \epsilon \cdot \left(\frac{\sigma_i}{\sigma_{\min}}\right)^2
$$

**Intuition**: Larger noise levels have larger score magnitudes ($\|\mathbf{s}\| \sim 1/\sigma$), so we need proportionally smaller relative step sizes to maintain stability. The $\sigma^2$ scaling ensures consistent "effective" step sizes across all noise levels.

### 3.4 Number of Steps per Level

Typical choices:
- **$T = 100$** for quick sampling
- **$T = 1000$** for higher quality
- **Total steps** = $L \times T$ (e.g., 10 levels × 100 steps = 1000 total)

---

## 4. PyTorch Implementation

### 4.1 Sinusoidal Position Embeddings

```python
import torch
import torch.nn as nn
import numpy as np
from typing import List

class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal embeddings for noise level conditioning."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        device = sigma.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = sigma.unsqueeze(-1) * embeddings.unsqueeze(0)
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings
```

### 4.2 NCSN Score Network

```python
class NCSNScoreNetwork(nn.Module):
    """
    Score network with noise level conditioning.
    
    The network takes (x, σ) and outputs s_θ(x, σ) ∈ R^D.
    Uses sinusoidal embeddings for σ conditioning.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 256, 256],
        sigma_embed_dim: int = 64
    ):
        super().__init__()
        
        self.input_dim = input_dim
        
        # Noise level embedding: σ → embedding
        self.sigma_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(sigma_embed_dim),
            nn.Linear(sigma_embed_dim, sigma_embed_dim),
            nn.SiLU(),
            nn.Linear(sigma_embed_dim, sigma_embed_dim)
        )
        
        # Main network: [x, σ_emb] → score
        layers = []
        dims = [input_dim + sigma_embed_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.extend([
                nn.Linear(dims[i], dims[i+1]),
                nn.SiLU()
            ])
        layers.append(nn.Linear(dims[-1], input_dim))
        
        self.net = nn.Sequential(*layers)
        
        # Initialize output layer near zero for stability
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
    
    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input data, shape (N, D)
            sigma: Noise levels, shape (N,) or scalar
        
        Returns:
            Score vectors, shape (N, D)
        """
        if sigma.dim() == 0:
            sigma = sigma.expand(x.shape[0])
        
        sigma_emb = self.sigma_embed(sigma)
        x_input = torch.cat([x, sigma_emb], dim=1)
        return self.net(x_input)
```

### 4.3 NCSN Training Loss

```python
def ncsn_loss(
    score_model: nn.Module,
    samples: torch.Tensor,
    sigmas: torch.Tensor,
    use_weighting: bool = True
) -> torch.Tensor:
    """
    NCSN training loss with σ² weighting.
    
    Args:
        score_model: Noise-conditioned score network
        samples: Clean data, shape (N, D)
        sigmas: Noise levels (descending), shape (L,)
        use_weighting: If True, weight by σ² for balanced loss
    
    Returns:
        Scalar loss
    """
    N = samples.shape[0]
    L = len(sigmas)
    device = samples.device
    
    # Randomly select noise level for each sample
    idx = torch.randint(0, L, (N,), device=device)
    sigma = sigmas[idx]  # (N,)
    
    # Add noise: x̃ = x + σε
    noise = torch.randn_like(samples)
    noisy_samples = samples + sigma.unsqueeze(1) * noise
    
    # Predict score
    pred_score = score_model(noisy_samples, sigma)
    
    # Target: -ε/σ
    target_score = -noise / sigma.unsqueeze(1)
    
    # Loss with optional σ² weighting
    if use_weighting:
        # σ²||s + ε/σ||² = ||σs + ε||² (scale-invariant)
        loss = torch.mean(
            (sigma.unsqueeze(1) ** 2) * (pred_score - target_score) ** 2
        )
    else:
        loss = torch.mean((pred_score - target_score) ** 2)
    
    return loss


def create_noise_schedule(
    sigma_min: float = 0.01,
    sigma_max: float = 50.0,
    num_levels: int = 10
) -> torch.Tensor:
    """
    Create geometric noise schedule (descending).
    
    Returns:
        Tensor of shape (num_levels,) with σ values from σ_max to σ_min
    """
    return torch.exp(
        torch.linspace(np.log(sigma_max), np.log(sigma_min), num_levels)
    )
```

### 4.4 Annealed Langevin Sampling

```python
@torch.no_grad()
def annealed_langevin_dynamics(
    score_model: nn.Module,
    sigmas: torch.Tensor,
    n_samples: int,
    data_dim: int,
    n_steps_per_sigma: int = 100,
    step_size: float = 1e-5,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Generate samples using Annealed Langevin Dynamics.
    
    Args:
        score_model: Trained NCSN model
        sigmas: Noise schedule (descending order)
        n_samples: Number of samples to generate
        data_dim: Dimension of data
        n_steps_per_sigma: Langevin steps per noise level
        step_size: Base step size ε
        device: Compute device
    
    Returns:
        Generated samples, shape (n_samples, data_dim)
    """
    score_model.eval()
    sigmas = sigmas.to(device)
    sigma_min = sigmas[-1]
    
    # Initialize from large-scale Gaussian
    x = torch.randn(n_samples, data_dim, device=device) * sigmas[0]
    
    for sigma in sigmas:
        # Adaptive step size: α = ε × (σ/σ_min)²
        alpha = step_size * (sigma / sigma_min) ** 2
        
        for _ in range(n_steps_per_sigma):
            # Langevin step
            noise = torch.randn_like(x)
            score = score_model(x, sigma.expand(n_samples))
            x = x + (alpha / 2) * score + torch.sqrt(alpha) * noise
    
    return x
```

### 4.5 Complete NCSN Trainer

```python
class NCSN:
    """
    Complete NCSN trainer and sampler.
    """
    
    def __init__(
        self,
        score_net: NCSNScoreNetwork,
        sigma_min: float = 0.01,
        sigma_max: float = 10.0,
        n_sigmas: int = 10,
        lr: float = 1e-3
    ):
        self.score_net = score_net
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.n_sigmas = n_sigmas
        
        # Geometric noise schedule
        self.sigmas = create_noise_schedule(sigma_min, sigma_max, n_sigmas)
        
        self.optimizer = torch.optim.Adam(score_net.parameters(), lr=lr)
    
    def train_step(self, x: torch.Tensor) -> dict:
        """Single training step."""
        self.score_net.train()
        self.optimizer.zero_grad()
        
        loss = ncsn_loss(self.score_net, x, self.sigmas.to(x.device))
        loss.backward()
        self.optimizer.step()
        
        return {'loss': loss.item()}
    
    @torch.no_grad()
    def sample(
        self,
        n_samples: int,
        n_steps_per_sigma: int = 100,
        step_size: float = 1e-5,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """Generate samples using ALD."""
        return annealed_langevin_dynamics(
            self.score_net,
            self.sigmas,
            n_samples,
            self.score_net.input_dim,
            n_steps_per_sigma,
            step_size,
            device
        )
```

---

## 5. Complete Training Example

### 5.1 Training on Swiss Roll

```python
import matplotlib.pyplot as plt

# Generate Swiss roll data
def swiss_roll(n: int) -> torch.Tensor:
    t = 1.5 * np.pi * (1 + 2 * np.random.rand(n))
    x = t * np.cos(t) + 0.1 * np.random.randn(n)
    y = t * np.sin(t) + 0.1 * np.random.randn(n)
    data = np.stack([x, y], axis=1)
    data = (data - data.mean(0)) / data.std(0)  # Normalize
    return torch.tensor(data, dtype=torch.float32)

# Create data and model
samples = swiss_roll(5000)
score_net = NCSNScoreNetwork(input_dim=2, hidden_dims=[128, 128, 128])
ncsn = NCSN(score_net, sigma_min=0.01, sigma_max=5.0, n_sigmas=10)

# Training loop
print("Training NCSN...")
for epoch in range(10000):
    idx = torch.randperm(len(samples))[:256]
    metrics = ncsn.train_step(samples[idx])
    
    if (epoch + 1) % 1000 == 0:
        print(f"Epoch {epoch+1}: Loss = {metrics['loss']:.4f}")

# Generate samples
print("Sampling...")
generated = ncsn.sample(1000, n_steps_per_sigma=100, step_size=1e-4)

# Plot comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=1)
axes[0].set_title('Training Data')
axes[1].scatter(generated[:, 0], generated[:, 1], alpha=0.5, s=1)
axes[1].set_title('NCSN Generated Samples')
for ax in axes:
    ax.set_aspect('equal')
plt.tight_layout()
```

### 5.2 Visualizing Multi-Scale Scores

```python
def plot_multiscale_scores(model, sigmas, xlim=(-3, 3), ylim=(-3, 3)):
    """Visualize score fields at different noise levels."""
    n_show = min(4, len(sigmas))
    indices = np.linspace(0, len(sigmas)-1, n_show).astype(int)
    
    fig, axes = plt.subplots(1, n_show, figsize=(4*n_show, 4))
    
    for ax, i in zip(axes, indices):
        sigma = sigmas[i]
        
        # Create grid
        x = torch.linspace(xlim[0], xlim[1], 15)
        y = torch.linspace(ylim[0], ylim[1], 15)
        X, Y = torch.meshgrid(x, y, indexing='xy')
        grid = torch.stack([X.flatten(), Y.flatten()], dim=1)
        
        # Compute scores
        with torch.no_grad():
            scores = model(grid, sigma.expand(len(grid)))
        
        U = scores[:, 0].reshape(15, 15).numpy()
        V = scores[:, 1].reshape(15, 15).numpy()
        
        ax.quiver(X.numpy(), Y.numpy(), U, V, alpha=0.7)
        ax.set_title(f'σ = {sigma.item():.2f}')
        ax.set_aspect('equal')
    
    plt.suptitle('Score Fields at Different Noise Levels')
    plt.tight_layout()
```

---

## 6. Connection to Diffusion Models

### 6.1 NCSN vs DDPM

| Aspect | NCSN | DDPM |
|--------|------|------|
| **Noise levels** | Discrete set $\{\sigma_i\}_{i=1}^L$ | Continuous time $t \in [0, 1]$ |
| **Forward process** | Independent noise at each level | Markov chain |
| **Parameterization** | Score $\mathbf{s}_\theta(\mathbf{x}, \sigma)$ | Noise $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$ |
| **Relationship** | $\mathbf{s} = -\boldsymbol{\epsilon}/\sigma$ | $\boldsymbol{\epsilon} = -\sigma \mathbf{s}$ |
| **Training** | DSM at each noise level | DSM with time conditioning |
| **Sampling** | Annealed Langevin dynamics | Ancestral sampling / reverse SDE |

### 6.2 Unified View

NCSN is essentially DDPM with:

- **Discrete** instead of continuous time
- **Explicit score** parameterization instead of noise prediction
- **Langevin** instead of ancestral sampling

Both optimize the **same underlying denoising score matching objective**!

### 6.3 Evolution to Modern Methods

```
NCSN (2019)
    ↓ Continuous time
NCSNv2 (2020) / Score SDE (2021)
    ↓ Different parameterization
DDPM (2020)
    ↓ Faster sampling
DDIM, Consistency Models, etc.
```

---

## 7. Summary

| Component | Description |
|-----------|-------------|
| **Multi-scale noise** | Geometric schedule: $\sigma_i = \sigma_{\max}(\sigma_{\min}/\sigma_{\max})^{(i-1)/(L-1)}$ |
| **Conditioning** | Score network takes $(\mathbf{x}, \sigma)$ as input |
| **Training** | DSM at each noise level with $\lambda(\sigma) = \sigma^2$ weighting |
| **Sampling** | Annealed Langevin: start high $\sigma$, decrease gradually |
| **Step size** | $\alpha_i = \epsilon \cdot (\sigma_i/\sigma_{\min})^2$ |

!!! tip "Key Takeaways"
    1. **Single noise level fails** in low-density regions
    2. **Multi-scale conditioning** provides coverage at all scales
    3. **$\sigma^2$ weighting** balances loss across scales (equivalent to noise prediction)
    4. **Annealed Langevin dynamics** leverages multi-scale structure for sampling
    5. **NCSN = discrete-time diffusion** with score parameterization

---

## Exercises

1. **Noise schedule design**: Compare geometric vs linear vs cosine noise schedules on Swiss roll. Measure sample quality and training stability.

2. **Step size tuning**: Implement adaptive step size in ALD using the score magnitude: $\alpha \propto 1/\|\mathbf{s}\|^2$.

3. **Conditional NCSN**: Extend NCSN to class-conditional generation on a 2D mixture of Gaussians.

4. **Score visualization**: Create an animation showing how the score field evolves as $\sigma$ decreases from $\sigma_{\max}$ to $\sigma_{\min}$.

5. **NCSN vs DDPM**: Train both on the same 2D dataset and compare sample quality, training time, and sampling efficiency.

---

## References

1. Song, Y., & Ermon, S. (2019). "Generative Modeling by Estimating Gradients of the Data Distribution." *NeurIPS*.
2. Song, Y., & Ermon, S. (2020). "Improved Techniques for Training Score-Based Generative Models." *NeurIPS*.
3. Ho, J., et al. (2020). "Denoising Diffusion Probabilistic Models." *NeurIPS*.
4. Song, Y., et al. (2021). "Score-Based Generative Modeling through Stochastic Differential Equations." *ICLR*.
