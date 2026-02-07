# BigGAN

BigGAN (Brock et al., 2019) demonstrated that scaling up GANs dramatically improves image quality, achieving state-of-the-art class-conditional image generation on ImageNet.

## Core Principle: Scale Everything

BigGAN's key insight: larger batch sizes, more parameters, and careful regularization dramatically improve quality.

## Scaling Strategies

### 1. Larger Batch Sizes

```python
# Standard GAN: batch_size = 64
# BigGAN: batch_size = 2048

# Larger batches provide:
# - More stable gradient estimates
# - Better coverage of class distribution
# - More effective BatchNorm statistics
```

### 2. Channel Multiplier

Scale all channel widths:

```python
class BigGANGenerator(nn.Module):
    def __init__(self, ch=96, n_classes=1000):  # ch was 64 in earlier GANs
        super().__init__()
        self.linear = nn.Linear(128 + 128, 16 * ch * 4 * 4)  # z + class embed
        
        self.blocks = nn.ModuleList([
            GBlock(16*ch, 16*ch, n_classes),  # 4→8
            GBlock(16*ch, 8*ch, n_classes),   # 8→16
            GBlock(8*ch, 4*ch, n_classes),    # 16→32
            GBlock(4*ch, 2*ch, n_classes),    # 32→64
            GBlock(2*ch, ch, n_classes),      # 64→128
        ])
```

### 3. Shared Class Embedding

Efficiently share class conditioning:

```python
class SharedEmbedding(nn.Module):
    def __init__(self, n_classes, embed_dim, n_layers):
        super().__init__()
        self.embed = nn.Embedding(n_classes, embed_dim)
        self.linears = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(n_layers)
        ])
    
    def forward(self, y, layer_idx):
        e = self.embed(y)
        return self.linears[layer_idx](e)
```

## Class-Conditional Architecture

### Generator Block with Conditioning

```python
class GBlock(nn.Module):
    """BigGAN generator block with class conditioning."""
    
    def __init__(self, in_ch, out_ch, n_classes):
        super().__init__()
        
        self.bn1 = ConditionalBatchNorm(in_ch, n_classes)
        self.bn2 = ConditionalBatchNorm(out_ch, n_classes)
        
        self.conv1 = spectral_norm(nn.Conv2d(in_ch, out_ch, 3, 1, 1))
        self.conv2 = spectral_norm(nn.Conv2d(out_ch, out_ch, 3, 1, 1))
        
        self.skip = spectral_norm(nn.Conv2d(in_ch, out_ch, 1))
    
    def forward(self, x, y):
        h = F.relu(self.bn1(x, y))
        h = F.interpolate(h, scale_factor=2)
        h = self.conv1(h)
        h = F.relu(self.bn2(h, y))
        h = self.conv2(h)
        
        x = F.interpolate(x, scale_factor=2)
        x = self.skip(x)
        
        return h + x
```

### Conditional Batch Normalization

```python
class ConditionalBatchNorm(nn.Module):
    def __init__(self, channels, n_classes):
        super().__init__()
        self.bn = nn.BatchNorm2d(channels, affine=False)
        self.gamma = nn.Embedding(n_classes, channels)
        self.beta = nn.Embedding(n_classes, channels)
        
        nn.init.ones_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)
    
    def forward(self, x, y):
        out = self.bn(x)
        gamma = self.gamma(y).view(-1, x.size(1), 1, 1)
        beta = self.beta(y).view(-1, x.size(1), 1, 1)
        return gamma * out + beta
```

## Training Techniques

### 1. Truncation Trick

Sample z from truncated normal for better quality:

```python
def truncated_normal(size, threshold=1.0):
    """Sample from truncated normal distribution."""
    values = torch.randn(size)
    while True:
        mask = values.abs() > threshold
        if not mask.any():
            break
        values[mask] = torch.randn(mask.sum())
    return values

# During inference
z = truncated_normal((batch_size, 128), threshold=0.5)  # Lower = better quality
```

### 2. Orthogonal Regularization

Encourage diverse features:

```python
def orthogonal_regularization(model, strength=1e-4):
    """Penalize non-orthogonal weight matrices."""
    loss = 0
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            w = param.view(param.size(0), -1)
            wt = w.t()
            identity = torch.eye(w.size(0), device=w.device)
            loss += ((w @ wt - identity) ** 2).sum()
    return strength * loss
```

### 3. Spectral Normalization (Both G and D)

```python
# Apply spectral normalization to all layers
for module in model.modules():
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        spectral_norm(module)
```

## Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Batch size | 2048 |
| z dimension | 128 |
| Channel multiplier | 96 |
| Learning rate (G) | 1e-4 |
| Learning rate (D) | 4e-4 |
| D steps per G step | 2 |

## Results

- 128×128 ImageNet: FID 8.7 (vs 18.65 for previous SOTA)
- 256×256 ImageNet: FID 9.6
- 512×512 ImageNet: FID 11.5

## Summary

| Innovation | Impact |
|------------|--------|
| Large batch size | Stable training |
| Spectral norm everywhere | Prevents mode collapse |
| Class conditioning | High-quality conditional generation |
| Truncation trick | Quality/diversity trade-off |
| Orthogonal regularization | Diverse features |
