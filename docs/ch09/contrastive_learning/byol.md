# BYOL: Bootstrap Your Own Latent

BYOL (Bootstrap Your Own Latent) achieves strong self-supervised representations **without requiring negative samples**, challenging the conventional wisdom that contrastive learning needs negatives to prevent collapse.

## The Collapse Problem

Without negatives, what prevents the model from learning a trivial constant representation?

$$f(x) = c \quad \forall x$$

BYOL prevents collapse through an asymmetric architecture: an online network with a predictor trained against a slowly-evolving target network updated via exponential moving average (EMA).

## Architecture

```
Online Network (trained with gradients):
  Input → Encoder → Projector → Predictor → online prediction
  
Target Network (updated via EMA, no gradients):
  Input → Encoder → Projector → target projection
  
Loss: MSE between online prediction and target projection (both L2-normalized)
```

## Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import copy
from PIL import ImageFilter, ImageOps
import random


class GaussianBlur:
    """Gaussian blur augmentation."""
    def __init__(self, sigma=(0.1, 2.0)):
        self.sigma = sigma
    
    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        return x.filter(ImageFilter.GaussianBlur(radius=sigma))


class Solarization:
    """Solarization augmentation."""
    def __init__(self, threshold=128):
        self.threshold = threshold
    
    def __call__(self, x):
        return ImageOps.solarize(x, self.threshold)


class BYOLAugmentation:
    """
    BYOL uses asymmetric augmentations for two views.
    View 1: Always Gaussian blur
    View 2: Sometimes solarization, sometimes blur
    """
    def __init__(self, size=224):
        self.transform1 = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.transform2 = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur()], p=0.1),
            transforms.RandomApply([Solarization()], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def __call__(self, x):
        return self.transform1(x), self.transform2(x)


class MLP(nn.Module):
    """MLP for projection and prediction heads."""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)


class BYOL(nn.Module):
    """
    BYOL: Bootstrap Your Own Latent
    
    Key components:
    1. Online network: encoder + projector + predictor (trained)
    2. Target network: encoder + projector (EMA updated)
    3. Asymmetric loss: Online predicts target's projection
    
    Args:
        base_encoder: Backbone architecture
        projection_dim: Output dimension of projector
        hidden_dim: Hidden dimension of MLP
        momentum: Momentum for target network EMA
    """
    def __init__(
        self,
        base_encoder='resnet50',
        projection_dim=256,
        hidden_dim=4096,
        momentum=0.996
    ):
        super().__init__()
        self.momentum = momentum
        
        # Build online network
        if base_encoder == 'resnet50':
            self.online_encoder = models.resnet50(weights=None)
            encoder_dim = 2048
            self.online_encoder.fc = nn.Identity()
        else:
            raise ValueError(f"Unknown encoder: {base_encoder}")
        
        self.encoder_dim = encoder_dim
        self.online_projector = MLP(encoder_dim, hidden_dim, projection_dim)
        self.predictor = MLP(projection_dim, hidden_dim, projection_dim)
        
        # Build target network (no gradients)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_projector = copy.deepcopy(self.online_projector)
        
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def update_target_network(self):
        """EMA update: θ_target ← τ * θ_target + (1 - τ) * θ_online"""
        for online_params, target_params in zip(
            self.online_encoder.parameters(),
            self.target_encoder.parameters()
        ):
            target_params.data = (
                self.momentum * target_params.data +
                (1 - self.momentum) * online_params.data
            )
        
        for online_params, target_params in zip(
            self.online_projector.parameters(),
            self.target_projector.parameters()
        ):
            target_params.data = (
                self.momentum * target_params.data +
                (1 - self.momentum) * online_params.data
            )
    
    def forward(self, x1, x2):
        """Forward pass with symmetric loss."""
        # Online network
        online_proj1 = self.online_projector(self.online_encoder(x1))
        online_proj2 = self.online_projector(self.online_encoder(x2))
        online_pred1 = self.predictor(online_proj1)
        online_pred2 = self.predictor(online_proj2)
        
        # Target network (no gradient)
        with torch.no_grad():
            target_proj1 = self.target_projector(self.target_encoder(x1))
            target_proj2 = self.target_projector(self.target_encoder(x2))
        
        # Symmetric loss
        loss = self.regression_loss(online_pred1, target_proj2)
        loss += self.regression_loss(online_pred2, target_proj1)
        return loss / 2
    
    def regression_loss(self, pred, target):
        """Normalized MSE: L = 2 - 2 * cos(pred, target)"""
        pred = F.normalize(pred, dim=-1)
        target = F.normalize(target, dim=-1)
        return (2 - 2 * (pred * target).sum(dim=-1)).mean()
    
    def get_representation(self, x):
        return self.online_encoder(x)




## Training

def cosine_scheduler(base_value, final_value, epochs):
    """Cosine schedule for momentum parameter."""
    iters = torch.arange(epochs)
    schedule = final_value - 0.5 * (final_value - base_value) * (
        1 + torch.cos(torch.pi * iters / epochs)
    )
    return schedule


def train_byol_epoch(model, dataloader, optimizer, device, epoch, momentum_schedule):
    """Train BYOL for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    model.momentum = momentum_schedule[epoch].item()
    
    for batch_idx, (x1, x2) in enumerate(dataloader):
        x1, x2 = x1.to(device), x2.to(device)
        
        loss = model(x1, x2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        model.update_target_network()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % 100 == 0:
            print(f"  Batch {batch_idx}: Loss = {loss.item():.4f}")
    
    return total_loss / num_batches


def train_simsiam_epoch(model, dataloader, optimizer, device, epoch):
    """Train SimSiam for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (x1, x2) in enumerate(dataloader):
        x1, x2 = x1.to(device), x2.to(device)
        
        loss = model(x1, x2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % 100 == 0:
            print(f"  Batch {batch_idx}: Loss = {loss.item():.4f}")
    
    return total_loss / num_batches




## Why Don't They Collapse?

### Stop-Gradient Analysis

```
Without stop-gradient (COLLAPSES):
  ∂L/∂θ = ∂L/∂p · ∂p/∂θ + ∂L/∂z · ∂z/∂θ
  → Both branches push toward trivial solution f(x) = 0

With stop-gradient (WORKS):
  ∂L/∂θ = ∂L/∂p · ∂p/∂θ + 0
  → Only predictor branch optimized
  → Target z provides a "moving target"
  → Predictor must learn to PREDICT, not just COPY
```

### Key Insights

1. **Predictor creates asymmetry**: Without the predictor, both branches would have identical gradients, leading to collapse.

2. **Stop-gradient creates implicit target**: The detached branch provides a slowly evolving target that the predictor must chase.

3. **Batch normalization helps**: BN in the projector prevents representations from collapsing to a single point.

4. **Momentum (BYOL) provides stability**: The slowly-evolving target network provides consistent targets.



## Summary

BYOL demonstrates that negative samples aren't strictly necessary for self-supervised learning. The combination of momentum EMA updates and a predictor head creates sufficient asymmetry to prevent collapse, while providing stable training even with smaller batch sizes than SimCLR.

## References

1. Grill, J.B., et al. (2020). "Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning." *NeurIPS*.
2. Richemond, P.H., et al. (2020). "BYOL works even without batch statistics." arXiv.
