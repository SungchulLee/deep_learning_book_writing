# Noise Conditional Score Networks (NCSN)

## Introduction

**Noise Conditional Score Networks (NCSN)** address a fundamental challenge in score-based generative models: learning accurate scores across all regions of data space. By training on multiple noise levels simultaneously, NCSN learns a family of score functions that enable sampling via annealed Langevin dynamics.

!!! info "Key Innovation"
    Instead of learning a single score function, NCSN learns $s_\theta(x, \sigma)$—a score function conditioned on the noise level $\sigma$.

## The Multi-Scale Score Problem

### Single Noise Level Limitations

With a single noise level $\sigma$:

- **$\sigma$ too small**: Score is accurate near data manifold but undefined/unreliable elsewhere
- **$\sigma$ too large**: Score is smooth everywhere but too blurry to capture data structure

### Solution: Noise Conditioning

Train on a geometric sequence of noise levels:

$$\sigma_1 > \sigma_2 > \cdots > \sigma_L$$

Learn $s_\theta(x, \sigma_i) \approx \nabla_x \log p_{\sigma_i}(x)$ for all $i$.

## NCSN Training

### Objective Function

$$\mathcal{L}(\theta) = \frac{1}{L}\sum_{i=1}^L \lambda(\sigma_i) \mathbb{E}_{x \sim p_{\text{data}}} \mathbb{E}_{\tilde{x} \sim \mathcal{N}(x, \sigma_i^2 I)}\left[\left\|s_\theta(\tilde{x}, \sigma_i) + \frac{\tilde{x} - x}{\sigma_i^2}\right\|^2\right]$$

### Noise Level Weighting

Common choices for $\lambda(\sigma)$:

- **Uniform**: $\lambda(\sigma) = 1$
- **Inverse variance**: $\lambda(\sigma) = \sigma^2$ (makes loss scale-invariant)
- **Learned**: Optimize weighting during training

### Noise Schedule Design

Geometric sequence from $\sigma_{\max}$ to $\sigma_{\min}$:

$$\sigma_i = \sigma_{\max} \left(\frac{\sigma_{\min}}{\sigma_{\max}}\right)^{\frac{i-1}{L-1}}$$

**Guidelines:**
- $\sigma_{\max}$: Comparable to data diameter
- $\sigma_{\min}$: Small enough to approximate clean data
- $L$: Typically 10-1000 levels

## Annealed Langevin Dynamics

### Sampling Procedure

```
1. Initialize x ~ N(0, σ_max² I)
2. For i = 1 to L:
   a. Run T steps of Langevin dynamics at noise level σ_i
   b. Use learned score s_θ(x, σ_i)
3. Return final x
```

### Step Size Selection

For noise level $\sigma_i$, step size:

$$\epsilon_i = c \cdot \frac{\sigma_i^2}{\sigma_L^2}$$

where $c$ is a base step size (typically $10^{-5}$ to $10^{-4}$).

## PyTorch Implementation

```python
"""
Noise Conditional Score Networks (NCSN)
=======================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional


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


class NCSNScoreNetwork(nn.Module):
    """
    Score network with noise level conditioning.
    
    The network takes (x, σ) and outputs s_θ(x, σ).
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 256, 256],
        sigma_embed_dim: int = 64
    ):
        super().__init__()
        
        self.input_dim = input_dim
        
        # Noise level embedding
        self.sigma_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(sigma_embed_dim),
            nn.Linear(sigma_embed_dim, sigma_embed_dim),
            nn.SiLU(),
            nn.Linear(sigma_embed_dim, sigma_embed_dim)
        )
        
        # Main network
        layers = []
        dims = [input_dim + sigma_embed_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.extend([
                nn.Linear(dims[i], dims[i+1]),
                nn.SiLU()
            ])
        layers.append(nn.Linear(dims[-1], input_dim))
        
        self.net = nn.Sequential(*layers)
        
        # Initialize output near zero
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
    
    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input [batch_size, input_dim]
            sigma: Noise levels [batch_size] or scalar
        """
        if sigma.dim() == 0:
            sigma = sigma.expand(x.shape[0])
        
        sigma_emb = self.sigma_embed(sigma)
        x_input = torch.cat([x, sigma_emb], dim=1)
        return self.net(x_input)


class NCSN:
    """
    Noise Conditional Score Network trainer and sampler.
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
        self.sigmas = torch.exp(
            torch.linspace(
                np.log(sigma_max),
                np.log(sigma_min),
                n_sigmas
            )
        )
        
        self.optimizer = torch.optim.Adam(score_net.parameters(), lr=lr)
    
    def loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute NCSN training loss.
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Sample random noise levels
        sigma_idx = torch.randint(0, self.n_sigmas, (batch_size,))
        sigma = self.sigmas[sigma_idx].to(device)
        
        # Add noise
        eps = torch.randn_like(x)
        x_noisy = x + sigma.unsqueeze(-1) * eps
        
        # Predict score
        score = self.score_net(x_noisy, sigma)
        
        # Target: -eps / sigma
        target = -eps / sigma.unsqueeze(-1)
        
        # Loss with sigma² weighting (scale-invariant)
        loss = ((score - target) ** 2).sum(dim=-1)
        loss = (loss * sigma ** 2).mean()
        
        return loss
    
    def train_step(self, x: torch.Tensor) -> dict:
        self.optimizer.zero_grad()
        loss = self.loss(x)
        loss.backward()
        self.optimizer.step()
        return {'loss': loss.item()}
    
    @torch.no_grad()
    def sample(
        self,
        n_samples: int,
        n_steps_per_sigma: int = 100,
        step_size: float = 1e-5,
        device: torch.device = torch.device('cpu')
    ) -> torch.Tensor:
        """
        Sample using annealed Langevin dynamics.
        """
        dim = self.score_net.input_dim
        
        # Initialize from large noise
        x = torch.randn(n_samples, dim, device=device) * self.sigma_max
        
        # Anneal through noise levels
        for sigma in self.sigmas.to(device):
            # Adaptive step size
            eps = step_size * (sigma / self.sigmas[-1]) ** 2
            
            for _ in range(n_steps_per_sigma):
                score = self.score_net(x, sigma.expand(n_samples))
                noise = torch.randn_like(x)
                x = x + eps * score + np.sqrt(2 * eps) * noise
        
        return x


def demonstrate_ncsn():
    """Demonstrate NCSN on 2D data."""
    import matplotlib.pyplot as plt
    
    # Create checkerboard data
    def sample_checkerboard(n):
        x1 = torch.rand(n) * 4 - 2
        x2 = torch.rand(n) * 4 - 2
        mask = ((x1.int() + x2.int()) % 2 == 0).float()
        x1 = x1 + (torch.rand(n) - 0.5) * 0.5
        x2 = x2 + (torch.rand(n) - 0.5) * 0.5
        x = torch.stack([x1, x2], dim=1)
        return x[mask > 0][:n//2]
    
    data = sample_checkerboard(5000)
    
    # Train NCSN
    score_net = NCSNScoreNetwork(input_dim=2, hidden_dims=[128, 128, 128])
    ncsn = NCSN(score_net, sigma_min=0.01, sigma_max=2.0, n_sigmas=10)
    
    print("Training NCSN...")
    for epoch in range(2000):
        idx = torch.randint(0, len(data), (256,))
        metrics = ncsn.train_step(data[idx])
        if (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch+1}: Loss = {metrics['loss']:.4f}")
    
    # Sample
    print("Sampling...")
    samples = ncsn.sample(500, n_steps_per_sigma=100)
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].scatter(data[:, 0], data[:, 1], alpha=0.3, s=1)
    axes[0].set_title('Training Data')
    axes[1].scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=5)
    axes[1].set_title('NCSN Samples')
    for ax in axes:
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig('ncsn_demo.png', dpi=150)
    plt.show()
    
    return ncsn, samples


if __name__ == "__main__":
    ncsn, samples = demonstrate_ncsn()
```

## Connection to Diffusion Models

NCSN is a precursor to diffusion models:

| Aspect | NCSN | DDPM |
|--------|------|------|
| Noise levels | Discrete set $\{\sigma_i\}$ | Continuous time $t$ |
| Forward process | Independent noise | Markov chain |
| Training | DSM at each level | DSM with time conditioning |
| Sampling | Annealed Langevin | Reverse diffusion |

DDPM can be viewed as NCSN with a specific noise schedule derived from a diffusion process.

## Summary

NCSN provides the conceptual bridge from score matching to diffusion models:

1. **Multi-scale learning**: Train scores at multiple noise levels
2. **Annealed sampling**: Gradually reduce noise during generation
3. **Noise conditioning**: Network adapts to current noise level

## References

1. Song & Ermon (2019). Generative Modeling by Estimating Gradients of the Data Distribution.
2. Song & Ermon (2020). Improved Techniques for Training Score-Based Generative Models.

## Navigation

- **Previous**: [Sliced Score Matching](sliced_score_matching.md)
- **Next**: [Forward Diffusion Process](../diffusion_process/forward_diffusion.md)
