# Score Function Learning

## Introduction

**Score function learning** is the core of diffusion model training. We train a neural network $s_\theta(x, t)$ to approximate the score $\nabla_x \log p_t(x)$ at all noise levels.

## Training Objective

### Denoising Score Matching

The practical training objective:

$$\mathcal{L}(\theta) = \mathbb{E}_{t, x_0, \epsilon}\left[\lambda(t) \left\|s_\theta(x_t, t) - \nabla_{x_t} \log p(x_t | x_0)\right\|^2\right]$$

For Gaussian noise:
$$\nabla_{x_t} \log p(x_t | x_0) = -\frac{x_t - \sqrt{\bar{\alpha}_t} x_0}{1 - \bar{\alpha}_t} = -\frac{\epsilon}{\sqrt{1-\bar{\alpha}_t}}$$

### Noise Prediction Formulation

Reparameterize: $s_\theta(x_t, t) = -\frac{\epsilon_\theta(x_t, t)}{\sqrt{1-\bar{\alpha}_t}}$

$$\mathcal{L}(\theta) = \mathbb{E}_{t, x_0, \epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]$$

This is the **DDPM loss**: predict the noise that was added.

## Network Architecture

### Time Conditioning

The network must be conditioned on timestep $t$:

1. **Sinusoidal embedding**: Map $t$ to high-dimensional vector
2. **Concatenation**: Add to input features
3. **FiLM**: Modulate intermediate features
4. **Adaptive normalization**: Scale/shift after normalization

### U-Net for Images

Standard architecture for image diffusion:
- Encoder: Downsampling with residual blocks
- Bottleneck: Self-attention
- Decoder: Upsampling with skip connections
- Time embedding added at each resolution

## PyTorch Implementation

```python
"""
Score Function Learning
=======================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for time."""
    
    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -np.log(self.max_period) * torch.arange(half, device=t.device) / half
        )
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([args.cos(), args.sin()], dim=-1)


class TimeConditionedMLP(nn.Module):
    """Simple MLP with time conditioning for low-dimensional data."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [256, 256, 256],
        time_embed_dim: int = 128
    ):
        super().__init__()
        
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        # Main network with time conditioning
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.extend([
                nn.Linear(dims[i] + time_embed_dim if i == 0 else dims[i], dims[i+1]),
                nn.SiLU()
            ])
        layers.append(nn.Linear(dims[-1], input_dim))
        
        self.net = nn.ModuleList(layers)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Time embedding
        t_emb = self.time_embed(t)
        
        # Concatenate time embedding with input
        h = torch.cat([x, t_emb], dim=-1)
        
        for layer in self.net:
            h = layer(h)
        
        return h


class ScoreTrainer:
    """Train score network using denoising score matching."""
    
    def __init__(
        self,
        model: nn.Module,
        n_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        lr: float = 1e-3,
        prediction_type: str = 'noise'  # 'noise' or 'score'
    ):
        self.model = model
        self.n_timesteps = n_timesteps
        self.prediction_type = prediction_type
        
        # Noise schedule
        betas = torch.linspace(beta_start, beta_end, n_timesteps)
        alphas = 1 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    def loss(self, x_0: torch.Tensor) -> torch.Tensor:
        """Compute training loss."""
        batch_size = x_0.shape[0]
        device = x_0.device
        
        # Sample random timesteps
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=device)
        
        # Sample noise
        eps = torch.randn_like(x_0)
        
        # Create noisy samples
        alpha_t = self.alphas_cumprod[t].view(-1, 1).to(device)
        x_t = torch.sqrt(alpha_t) * x_0 + torch.sqrt(1 - alpha_t) * eps
        
        # Normalize time to [0, 1]
        t_normalized = t.float() / self.n_timesteps
        
        # Predict
        pred = self.model(x_t, t_normalized)
        
        if self.prediction_type == 'noise':
            target = eps
        else:  # score
            target = -eps / torch.sqrt(1 - alpha_t)
        
        return F.mse_loss(pred, target)
    
    def train_step(self, x_0: torch.Tensor) -> dict:
        self.optimizer.zero_grad()
        loss = self.loss(x_0)
        loss.backward()
        self.optimizer.step()
        return {'loss': loss.item()}
    
    def get_score(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Get score from trained model."""
        with torch.no_grad():
            pred = self.model(x, t)
            
            if self.prediction_type == 'noise':
                # Convert noise prediction to score
                t_int = (t * self.n_timesteps).long().clamp(0, self.n_timesteps - 1)
                alpha_t = self.alphas_cumprod[t_int].view(-1, 1)
                return -pred / torch.sqrt(1 - alpha_t)
            else:
                return pred


def demonstrate_score_learning():
    """Demonstrate score function learning."""
    import matplotlib.pyplot as plt
    
    # Create 2D data
    def sample_data(n):
        # Two moons
        theta = torch.linspace(0, np.pi, n // 2)
        x1 = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)
        x2 = torch.stack([1 - torch.cos(theta), 1 - torch.sin(theta) - 0.5], dim=1)
        x = torch.cat([x1, x2], dim=0)
        x += 0.1 * torch.randn_like(x)
        return x
    
    data = sample_data(2000)
    
    # Train score model
    model = TimeConditionedMLP(input_dim=2, hidden_dims=[128, 128, 128])
    trainer = ScoreTrainer(model, n_timesteps=100, lr=1e-3)
    
    print("Training score network...")
    for epoch in range(2000):
        idx = torch.randint(0, len(data), (256,))
        metrics = trainer.train_step(data[idx])
        if (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch+1}: Loss = {metrics['loss']:.4f}")
    
    # Visualize learned scores at different noise levels
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    x_range = torch.linspace(-2, 3, 15)
    y_range = torch.linspace(-2, 2, 15)
    X, Y = torch.meshgrid(x_range, y_range, indexing='ij')
    grid = torch.stack([X.flatten(), Y.flatten()], dim=1)
    
    for ax, t_val in zip(axes, [0.1, 0.5, 0.9]):
        t = torch.full((len(grid),), t_val)
        scores = trainer.get_score(grid, t)
        
        ax.quiver(
            grid[:, 0], grid[:, 1],
            scores[:, 0], scores[:, 1],
            color='red', alpha=0.7
        )
        ax.scatter(data[:, 0], data[:, 1], alpha=0.1, s=1)
        ax.set_title(f't = {t_val}')
        ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('score_learning.png', dpi=150)
    plt.show()
    
    return model, trainer


if __name__ == "__main__":
    model, trainer = demonstrate_score_learning()
```

## Training Tips

1. **Timestep sampling**: Uniform is common; importance sampling can help
2. **Loss weighting**: $\lambda(t)$ can prioritize certain noise levels
3. **EMA**: Exponential moving average of weights for stable sampling
4. **Gradient clipping**: Prevents training instabilities

## Summary

Score learning via denoising score matching:

1. **Add noise** to create $x_t$ from $x_0$
2. **Predict** either noise $\epsilon$ or score $\nabla_x \log p_t$
3. **Minimize MSE** between prediction and target
4. **Time conditioning** is essential for multi-scale learning

## Navigation

- **Previous**: [Reverse SDE](reverse_sde.md)
- **Next**: [Probability Flow ODE](probability_flow_ode.md)
