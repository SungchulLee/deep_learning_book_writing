# DDPM Training

## Introduction

Training DDPM models involves several practical considerations beyond the basic loss function. This section covers the complete training pipeline.

## Training Pipeline

### Algorithm

```
1. Initialize model θ
2. For each training iteration:
   a. Sample batch x_0 from dataset
   b. Sample t ~ Uniform(1, T)
   c. Sample ε ~ N(0, I)
   d. Compute x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε
   e. Compute loss = ||ε - ε_θ(x_t, t)||²
   f. Update θ with gradient descent
3. Maintain EMA of θ for sampling
```

## Key Training Components

### Exponential Moving Average (EMA)

Critical for stable sampling:

```python
def update_ema(ema_model, model, decay=0.9999):
    with torch.no_grad():
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(decay).add_(p.data, alpha=1 - decay)
```

### Gradient Clipping

Prevents training instabilities:
- Typically clip at norm 1.0
- Essential for high-resolution images

## PyTorch Implementation

```python
"""
DDPM Training Pipeline
======================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional
import copy


class DDPMTrainer:
    """Complete DDPM training pipeline."""
    
    def __init__(
        self,
        model: nn.Module,
        n_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        lr: float = 2e-4,
        ema_decay: float = 0.9999,
        grad_clip: float = 1.0,
        device: torch.device = torch.device('cpu')
    ):
        self.model = model.to(device)
        self.device = device
        self.n_timesteps = n_timesteps
        self.ema_decay = ema_decay
        self.grad_clip = grad_clip
        
        # EMA model
        self.ema_model = copy.deepcopy(model).to(device)
        self.ema_model.requires_grad_(False)
        
        # Noise schedule
        betas = torch.linspace(beta_start, beta_end, n_timesteps)
        alphas = 1 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0).to(device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100000
        )
        
        self.step = 0
    
    def compute_loss(self, x_0: torch.Tensor) -> torch.Tensor:
        """Compute training loss."""
        batch_size = x_0.shape[0]
        
        # Sample timesteps
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=self.device)
        
        # Sample noise
        eps = torch.randn_like(x_0)
        
        # Create noisy samples
        sqrt_alpha = self.alphas_cumprod[t].sqrt()
        sqrt_one_minus_alpha = (1 - self.alphas_cumprod[t]).sqrt()
        
        # Handle shape for images vs vectors
        while sqrt_alpha.dim() < x_0.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)
        
        x_t = sqrt_alpha * x_0 + sqrt_one_minus_alpha * eps
        
        # Predict noise
        eps_pred = self.model(x_t, t)
        
        return F.mse_loss(eps_pred, eps)
    
    def update_ema(self):
        """Update EMA model."""
        with torch.no_grad():
            for ema_p, p in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_p.data.mul_(self.ema_decay).add_(p.data, alpha=1 - self.ema_decay)
    
    def train_step(self, x_0: torch.Tensor) -> dict:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()
        
        loss = self.compute_loss(x_0)
        loss.backward()
        
        # Gradient clipping
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        
        self.optimizer.step()
        self.scheduler.step()
        self.update_ema()
        
        self.step += 1
        
        return {
            'loss': loss.item(),
            'lr': self.optimizer.param_groups[0]['lr'],
            'step': self.step
        }
    
    def train_epoch(self, dataloader: DataLoader) -> dict:
        """Train for one epoch."""
        total_loss = 0.0
        n_batches = 0
        
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                x_0 = batch[0]
            else:
                x_0 = batch
            
            x_0 = x_0.to(self.device)
            metrics = self.train_step(x_0)
            total_loss += metrics['loss']
            n_batches += 1
        
        return {'epoch_loss': total_loss / n_batches}
    
    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        torch.save({
            'step': self.step,
            'model_state': self.model.state_dict(),
            'ema_state': self.ema_model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.step = ckpt['step']
        self.model.load_state_dict(ckpt['model_state'])
        self.ema_model.load_state_dict(ckpt['ema_state'])
        self.optimizer.load_state_dict(ckpt['optimizer_state'])
        self.scheduler.load_state_dict(ckpt['scheduler_state'])
```

## Training Tips

### Hyperparameters

| Parameter | Typical Value | Notes |
|-----------|---------------|-------|
| Learning rate | 1e-4 to 2e-4 | AdamW |
| Batch size | 64-256 | Larger is better |
| EMA decay | 0.9999 | Don't change |
| Grad clip | 1.0 | Important for stability |
| Timesteps | 1000 | Standard |

### Common Issues

1. **Loss exploding**: Reduce learning rate, increase grad clip
2. **Blurry samples**: Train longer, check EMA
3. **Mode collapse**: Rare with diffusion, check data

## Summary

DDPM training is straightforward:
1. Simple MSE loss on noise prediction
2. EMA for stable sampling
3. Gradient clipping for stability
4. Standard deep learning practices



- **Previous**: [DDPM Loss Function](loss_function.md)
- **Next**: [DDPM Sampling](sampling.md)
