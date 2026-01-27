# Score-Based Image Generation

## Learning Objectives

By the end of this section, you will be able to:

1. Design U-Net architectures for score estimation on images
2. Implement complete image generation pipelines
3. Train score-based models on MNIST and CIFAR-10
4. Apply predictor-corrector sampling for high-quality images
5. Evaluate generated images using FID and other metrics

## Prerequisites

- Score-based SDEs
- Convolutional neural networks
- U-Net architecture basics

---

## 1. U-Net Score Network

### 1.1 Why U-Net?

Score networks for images need:

- **Multi-scale features**: Capture both fine details and global structure
- **Skip connections**: Preserve spatial information
- **Time conditioning**: Handle different noise levels

U-Net provides all these through its encoder-decoder structure with skip connections.

### 1.2 Architecture Overview

```
Input: x_t (noisy image) + t (time embedding)
                    ↓
    ┌───────────────────────────────────┐
    │          Time Embedding           │
    │     t → MLP → [temb_1, temb_2]    │
    └───────────────────────────────────┘
                    ↓
    ┌───────────────────────────────────┐
    │           Encoder                 │
    │  Conv → ResBlock → Downsample     │
    │           (×3-4)                  │
    └───────────────────────────────────┘
                    ↓
    ┌───────────────────────────────────┐
    │        Middle Block               │
    │   ResBlock → Attention → ResBlock │
    └───────────────────────────────────┘
                    ↓
    ┌───────────────────────────────────┐
    │           Decoder                 │
    │  Upsample → ResBlock → (+ skip)   │
    │           (×3-4)                  │
    └───────────────────────────────────┘
                    ↓
Output: s_θ(x_t, t) (predicted score)
```

---

## 2. PyTorch Implementation

### 2.1 Time Embedding

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embeddings for time."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class TimeEmbedding(nn.Module):
    """Time embedding MLP."""
    
    def __init__(self, time_dim: int, emb_dim: int):
        super().__init__()
        self.sinusoidal = SinusoidalPosEmb(time_dim)
        self.mlp = nn.Sequential(
            nn.Linear(time_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim)
        )
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = self.sinusoidal(t)
        return self.mlp(emb)
```

### 2.2 ResNet Block with Time Conditioning

```python
class ResBlock(nn.Module):
    """Residual block with time embedding."""
    
    def __init__(
        self, 
        in_ch: int, 
        out_ch: int, 
        time_emb_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch)
        )
        
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        
        self.dropout = nn.Dropout(dropout)
        
        # Skip connection
        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        # Add time embedding
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.skip(x)
```

### 2.3 Attention Block

```python
class AttentionBlock(nn.Module):
    """Self-attention for spatial features."""
    
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        
        # Attention
        scale = (C // self.num_heads) ** -0.5
        attn = torch.einsum('bhcn,bhcm->bhnm', q, k) * scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.einsum('bhnm,bhcm->bhcn', attn, v)
        out = out.reshape(B, C, H, W)
        
        return x + self.proj(out)
```

### 2.4 Complete U-Net

```python
class UNet(nn.Module):
    """U-Net for score estimation."""
    
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        channel_mults: tuple = (1, 2, 4),
        num_res_blocks: int = 2,
        time_emb_dim: int = 128,
        attention_resolutions: tuple = (8,),
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.base_channels = base_channels
        
        # Time embedding
        self.time_emb = TimeEmbedding(base_channels, time_emb_dim)
        
        # Initial convolution
        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # Encoder
        self.encoder = nn.ModuleList()
        self.downsample = nn.ModuleList()
        
        ch = base_channels
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            
            for _ in range(num_res_blocks):
                self.encoder.append(ResBlock(ch, out_ch, time_emb_dim, dropout))
                ch = out_ch
            
            if i < len(channel_mults) - 1:
                self.downsample.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
        
        # Middle
        self.middle = nn.ModuleList([
            ResBlock(ch, ch, time_emb_dim, dropout),
            AttentionBlock(ch),
            ResBlock(ch, ch, time_emb_dim, dropout)
        ])
        
        # Decoder
        self.decoder = nn.ModuleList()
        self.upsample = nn.ModuleList()
        
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            
            for j in range(num_res_blocks + 1):
                # Extra channel for skip connection
                self.decoder.append(ResBlock(ch + out_ch if j == 0 else ch, out_ch, time_emb_dim, dropout))
                ch = out_ch
            
            if i > 0:
                self.upsample.append(nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1))
        
        # Output
        self.out_norm = nn.GroupNorm(8, ch)
        self.out_conv = nn.Conv2d(ch, in_channels, 3, padding=1)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Time embedding
        t_emb = self.time_emb(t)
        
        # Initial conv
        h = self.init_conv(x)
        
        # Encoder with skip connections
        skips = [h]
        
        enc_idx = 0
        for i, _ in enumerate(self.downsample):
            for _ in range(2):  # num_res_blocks
                h = self.encoder[enc_idx](h, t_emb)
                skips.append(h)
                enc_idx += 1
            h = self.downsample[i](h)
            skips.append(h)
        
        # Handle remaining encoder blocks
        while enc_idx < len(self.encoder):
            h = self.encoder[enc_idx](h, t_emb)
            skips.append(h)
            enc_idx += 1
        
        # Middle
        for layer in self.middle:
            if isinstance(layer, ResBlock):
                h = layer(h, t_emb)
            else:
                h = layer(h)
        
        # Decoder with skip connections
        dec_idx = 0
        up_idx = 0
        
        for i in range(len(self.upsample) + 1):
            for j in range(3):  # num_res_blocks + 1
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = self.decoder[dec_idx](h, t_emb)
                dec_idx += 1
            
            if up_idx < len(self.upsample):
                h = self.upsample[up_idx](h)
                up_idx += 1
        
        # Output
        h = self.out_norm(h)
        h = F.silu(h)
        h = self.out_conv(h)
        
        return h
```

---

## 3. Training Pipeline

### 3.1 Data Preparation

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_mnist_loader(batch_size: int = 64, train: bool = True):
    """Get MNIST data loader with normalization to [-1, 1]."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Map to [-1, 1]
    ])
    
    dataset = datasets.MNIST(
        root='./data', 
        train=train, 
        download=True, 
        transform=transform
    )
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=train)


def get_cifar10_loader(batch_size: int = 64, train: bool = True):
    """Get CIFAR-10 data loader."""
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip() if train else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = datasets.CIFAR10(
        root='./data',
        train=train,
        download=True,
        transform=transform
    )
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=train)
```

### 3.2 Training Loop

```python
def train_score_model(
    model: nn.Module,
    sde: SDE,
    train_loader: DataLoader,
    n_epochs: int = 100,
    lr: float = 2e-4,
    device: str = 'cuda'
):
    """Train score model on image data."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    losses = []
    
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.to(device)
            B = x.shape[0]
            
            # Sample random time
            t = torch.rand(B, device=device) * (sde.T - 1e-5) + 1e-5
            
            # Get noisy samples
            mean, std = sde.marginal_params(x, t)
            noise = torch.randn_like(x)
            x_t = mean + std.view(B, 1, 1, 1) * noise
            
            # Predict score
            pred_score = model(x_t, t)
            
            # Target score
            target_score = -noise / std.view(B, 1, 1, 1)
            
            # Loss
            loss = F.mse_loss(pred_score, target_score)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.6f}")
    
    return losses
```

---

## 4. Sampling

### 4.1 Predictor-Corrector for Images

```python
@torch.no_grad()
def sample_images(
    model: nn.Module,
    sde: SDE,
    n_samples: int,
    image_shape: tuple,
    n_steps: int = 500,
    n_corrector: int = 1,
    snr: float = 0.16,
    device: str = 'cuda'
) -> torch.Tensor:
    """Generate images using predictor-corrector sampling."""
    model.eval()
    
    # Initialize from prior
    x = sde.sample_prior((n_samples, *image_shape), device)
    
    dt = sde.T / n_steps
    
    for i in range(n_steps):
        t = sde.T - i * sde.T / n_steps
        t_tensor = torch.full((n_samples,), t, device=device)
        
        # Predictor step
        g = sde.diffusion(t_tensor)
        score = model(x, t_tensor)
        
        drift = -g.view(-1, 1, 1, 1) ** 2 * score
        noise = torch.randn_like(x) * math.sqrt(dt)
        x = x - drift * dt + g.view(-1, 1, 1, 1) * noise
        
        # Corrector steps (Langevin)
        for _ in range(n_corrector):
            score = model(x, t_tensor)
            noise = torch.randn_like(x)
            
            grad_norm = score.view(n_samples, -1).norm(dim=1).mean()
            step_size = (snr * g.mean() / (grad_norm + 1e-8)) ** 2 * 2
            
            x = x + step_size * score + torch.sqrt(2 * step_size) * noise
    
    return x


def save_samples(samples: torch.Tensor, path: str, nrow: int = 8):
    """Save generated samples as grid image."""
    from torchvision.utils import save_image
    
    # Denormalize from [-1, 1] to [0, 1]
    samples = (samples + 1) / 2
    samples = samples.clamp(0, 1)
    
    save_image(samples, path, nrow=nrow)
```

---

## 5. Evaluation

### 5.1 FID Score

```python
# Requires: pip install pytorch-fid

def compute_fid(real_images: torch.Tensor, fake_images: torch.Tensor):
    """
    Compute Fréchet Inception Distance.
    
    Lower FID = better quality and diversity.
    """
    import numpy as np
    from scipy import linalg
    
    # This is a simplified version; use pytorch-fid for proper implementation
    
    def compute_statistics(images):
        # Flatten and compute mean/cov
        flat = images.view(images.shape[0], -1).numpy()
        mu = np.mean(flat, axis=0)
        sigma = np.cov(flat, rowvar=False)
        return mu, sigma
    
    mu1, sigma1 = compute_statistics(real_images)
    mu2, sigma2 = compute_statistics(fake_images)
    
    # FID formula
    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1 @ sigma2)
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    
    return fid
```

### 5.2 Sample Quality Inspection

```python
import matplotlib.pyplot as plt

def visualize_generation_process(
    model: nn.Module,
    sde: SDE,
    image_shape: tuple,
    device: str = 'cuda'
):
    """Visualize the denoising process."""
    model.eval()
    
    n_vis = 10
    x = sde.sample_prior((1, *image_shape), device)
    
    snapshots = [x.cpu()]
    timesteps = torch.linspace(sde.T, 0, n_vis)
    
    dt = sde.T / 100
    current_t = sde.T
    
    for target_t in timesteps[1:]:
        while current_t > target_t:
            t_tensor = torch.tensor([current_t], device=device)
            g = sde.diffusion(t_tensor)
            score = model(x, t_tensor)
            
            drift = -g ** 2 * score
            noise = torch.randn_like(x) * math.sqrt(dt)
            x = x - drift * dt + g * noise
            
            current_t -= dt
        
        snapshots.append(x.cpu())
    
    # Plot
    fig, axes = plt.subplots(1, n_vis, figsize=(2*n_vis, 2))
    
    for ax, snap, t in zip(axes, snapshots, timesteps):
        img = (snap[0].squeeze() + 1) / 2
        ax.imshow(img.numpy(), cmap='gray')
        ax.set_title(f't={t:.2f}')
        ax.axis('off')
    
    plt.tight_layout()
    return fig
```

---

## 6. Summary

| Component | Purpose |
|-----------|---------|
| **U-Net** | Multi-scale score estimation with skip connections |
| **Time Embedding** | Sinusoidal + MLP for conditioning on noise level |
| **ResBlock** | Core building block with time modulation |
| **Attention** | Capture long-range dependencies |
| **Predictor-Corrector** | High-quality sampling combining SDE + Langevin |

**Training tips:**

1. Use gradient clipping (max norm = 1.0)
2. Start with smaller images (28×28 MNIST)
3. Use EMA for model weights
4. Train for many epochs (100+ for MNIST, 500+ for CIFAR)

---

## References

1. Song, Y., & Ermon, S. (2019). "Generative Modeling by Estimating Gradients of the Data Distribution." *NeurIPS*.
2. Ho, J., et al. (2020). "Denoising Diffusion Probabilistic Models." *NeurIPS*.
3. Song, Y., et al. (2021). "Score-Based Generative Modeling through Stochastic Differential Equations." *ICLR*.
