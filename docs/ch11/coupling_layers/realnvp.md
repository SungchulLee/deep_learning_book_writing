# RealNVP: Real-valued Non-Volume Preserving Flows

## Introduction

RealNVP (Real-valued Non-Volume Preserving) is a landmark normalizing flow architecture that made density estimation with flows practical for complex, high-dimensional data like images. By introducing affine coupling layers with efficient masking strategies, RealNVP achieves both tractable likelihood computation and expressive transformations.

## Architecture Overview

### Core Components

RealNVP consists of:

1. **Affine coupling layers**: Transform part of the input based on the other part
2. **Masking patterns**: Alternating strategies to ensure all dimensions are transformed
3. **Multi-scale architecture**: Process data at different resolutions (for images)
4. **Batch normalization**: Stabilize training between layers

### The RealNVP Transform

For input $\mathbf{x} = (\mathbf{x}_{1:d}, \mathbf{x}_{d+1:D})$:

**Forward**:
$$\mathbf{y}_{1:d} = \mathbf{x}_{1:d}$$
$$\mathbf{y}_{d+1:D} = \mathbf{x}_{d+1:D} \odot \exp(s(\mathbf{x}_{1:d})) + t(\mathbf{x}_{1:d})$$

**Inverse**:
$$\mathbf{x}_{1:d} = \mathbf{y}_{1:d}$$
$$\mathbf{x}_{d+1:D} = (\mathbf{y}_{d+1:D} - t(\mathbf{y}_{1:d})) \odot \exp(-s(\mathbf{y}_{1:d}))$$

**Log-determinant**:
$$\log|\det J| = \sum_{j=d+1}^{D} s_j(\mathbf{x}_{1:d})$$

## Implementation

### Basic RealNVP Coupling Layer

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RealNVPCouplingLayer(nn.Module):
    """
    Single RealNVP affine coupling layer.
    """
    
    def __init__(self, dim, mask, hidden_dims=[256, 256]):
        super().__init__()
        
        self.dim = dim
        self.register_buffer('mask', mask.float())
        
        # Dimensions
        self.d_cond = int(mask.sum().item())  # Conditioning dimensions
        self.d_trans = dim - self.d_cond      # Transformed dimensions
        
        # Scale and translation network
        self.st_net = self._build_st_net(hidden_dims)
    
    def _build_st_net(self, hidden_dims):
        """Build network that outputs scale and translation."""
        layers = []
        prev_dim = self.d_cond
        
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU()
            ])
            prev_dim = h_dim
        
        # Output: scale and translation for each transformed dimension
        layers.append(nn.Linear(prev_dim, self.d_trans * 2))
        
        net = nn.Sequential(*layers)
        
        # Initialize for near-identity transform
        nn.init.zeros_(net[-1].weight)
        nn.init.zeros_(net[-1].bias)
        
        return net
    
    def forward(self, x):
        """Forward: x -> y"""
        batch_size = x.shape[0]
        
        # Extract conditioning and transformed parts
        x_cond = x[:, self.mask.bool()]
        x_trans = x[:, ~self.mask.bool()]
        
        # Compute scale and translation
        st = self.st_net(x_cond)
        s, t = st[:, :self.d_trans], st[:, self.d_trans:]
        
        # Bound scale for stability
        s = torch.tanh(s) * 2
        
        # Transform
        y_trans = x_trans * torch.exp(s) + t
        
        # Reconstruct full output
        y = torch.zeros_like(x)
        y[:, self.mask.bool()] = x_cond
        y[:, ~self.mask.bool()] = y_trans
        
        # Log determinant
        log_det = s.sum(dim=-1)
        
        return y, log_det
    
    def inverse(self, y):
        """Inverse: y -> x"""
        # Extract parts
        y_cond = y[:, self.mask.bool()]
        y_trans = y[:, ~self.mask.bool()]
        
        # Compute scale and translation
        st = self.st_net(y_cond)
        s, t = st[:, :self.d_trans], st[:, self.d_trans:]
        s = torch.tanh(s) * 2
        
        # Invert transform
        x_trans = (y_trans - t) * torch.exp(-s)
        
        # Reconstruct
        x = torch.zeros_like(y)
        x[:, self.mask.bool()] = y_cond
        x[:, ~self.mask.bool()] = x_trans
        
        # Log determinant of inverse
        log_det = -s.sum(dim=-1)
        
        return x, log_det
```

### Complete RealNVP Model

```python
class RealNVP(nn.Module):
    """
    Complete RealNVP flow model.
    """
    
    def __init__(self, dim, n_coupling_layers=8, hidden_dims=[256, 256], 
                 use_batch_norm=True):
        super().__init__()
        
        self.dim = dim
        self.n_layers = n_coupling_layers
        
        # Build coupling layers with alternating masks
        self.coupling_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        
        for i in range(n_coupling_layers):
            # Alternate mask pattern
            mask = self._create_mask(dim, i % 2 == 0)
            self.coupling_layers.append(
                RealNVPCouplingLayer(dim, mask, hidden_dims)
            )
            
            # Batch normalization between layers
            if use_batch_norm and i < n_coupling_layers - 1:
                self.batch_norms.append(BatchNormFlow(dim))
    
    def _create_mask(self, dim, first_half):
        """Create binary mask."""
        mask = torch.zeros(dim)
        if first_half:
            mask[:dim // 2] = 1
        else:
            mask[dim // 2:] = 1
        return mask
    
    def forward(self, x):
        """
        Forward pass: data -> latent.
        Used for density evaluation.
        """
        log_det_total = torch.zeros(x.shape[0], device=x.device)
        z = x
        
        for i, layer in enumerate(self.coupling_layers):
            z, log_det = layer(z)
            log_det_total += log_det
            
            # Apply batch norm
            if self.batch_norms is not None and i < len(self.batch_norms):
                z, bn_log_det = self.batch_norms[i](z)
                log_det_total += bn_log_det
        
        return z, log_det_total
    
    def inverse(self, z):
        """
        Inverse pass: latent -> data.
        Used for sampling.
        """
        log_det_total = torch.zeros(z.shape[0], device=z.device)
        x = z
        
        for i in reversed(range(self.n_layers)):
            # Inverse batch norm
            if self.batch_norms is not None and i < len(self.batch_norms):
                x, bn_log_det = self.batch_norms[i].inverse(x)
                log_det_total += bn_log_det
            
            x, log_det = self.coupling_layers[i].inverse(x)
            log_det_total += log_det
        
        return x, log_det_total
    
    def log_prob(self, x):
        """Compute log p(x)."""
        z, log_det = self.forward(x)
        
        # Standard Gaussian base distribution
        log_pz = -0.5 * (z ** 2 + np.log(2 * np.pi)).sum(dim=-1)
        
        return log_pz + log_det
    
    def sample(self, n_samples, device='cpu'):
        """Generate samples."""
        z = torch.randn(n_samples, self.dim, device=device)
        x, _ = self.inverse(z)
        return x


class BatchNormFlow(nn.Module):
    """Batch normalization as a flow layer."""
    
    def __init__(self, dim, momentum=0.1, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.momentum = momentum
        self.eps = eps
        
        # Learnable parameters
        self.log_gamma = nn.Parameter(torch.zeros(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        
        # Running statistics
        self.register_buffer('running_mean', torch.zeros(dim))
        self.register_buffer('running_var', torch.ones(dim))
    
    def forward(self, x):
        if self.training:
            # Use batch statistics
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)
            
            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var
        
        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # Scale and shift
        y = x_norm * torch.exp(self.log_gamma) + self.beta
        
        # Log determinant
        log_det = (self.log_gamma - 0.5 * torch.log(var + self.eps)).sum()
        log_det = log_det.expand(x.shape[0])
        
        return y, log_det
    
    def inverse(self, y):
        mean = self.running_mean
        var = self.running_var
        
        # Inverse scale and shift
        x_norm = (y - self.beta) * torch.exp(-self.log_gamma)
        
        # Inverse normalize
        x = x_norm * torch.sqrt(var + self.eps) + mean
        
        # Log determinant of inverse
        log_det = (-self.log_gamma + 0.5 * torch.log(var + self.eps)).sum()
        log_det = log_det.expand(y.shape[0])
        
        return x, log_det
```

## Masking Strategies

### For Vectors (MNIST flattened)

**Simple half-split**:
```python
def half_mask(dim, upper=True):
    mask = torch.zeros(dim)
    if upper:
        mask[:dim // 2] = 1
    else:
        mask[dim // 2:] = 1
    return mask
```

### For Images

**Checkerboard mask**:
```python
def checkerboard_mask(height, width, channels, even=True):
    """
    Checkerboard pattern: alternating pixels.
    """
    mask = torch.zeros(channels, height, width)
    for i in range(height):
        for j in range(width):
            if (i + j) % 2 == (0 if even else 1):
                mask[:, i, j] = 1
    return mask
```

**Channel-wise mask**:
```python
def channel_mask(channels, height, width, first_half=True):
    """
    Split by channels.
    """
    mask = torch.zeros(channels, height, width)
    if first_half:
        mask[:channels // 2, :, :] = 1
    else:
        mask[channels // 2:, :, :] = 1
    return mask
```

### Masking Schedule

RealNVP for images typically uses:

1. **Checkerboard layers** (spatial mixing)
2. **Squeeze operation** (reshape spatial to channels)
3. **Channel-wise layers** (channel mixing)
4. Repeat at multiple scales

```python
class RealNVPImage(nn.Module):
    """RealNVP for images with multi-scale architecture."""
    
    def __init__(self, in_channels, hidden_channels=64, n_blocks=4):
        super().__init__()
        
        self.blocks = nn.ModuleList()
        channels = in_channels
        
        for _ in range(n_blocks):
            # Checkerboard coupling layers
            self.blocks.append(
                CheckerboardBlock(channels, hidden_channels)
            )
            
            # Squeeze: (C, H, W) -> (4C, H/2, W/2)
            self.blocks.append(Squeeze())
            channels *= 4
            
            # Channel-wise coupling layers
            self.blocks.append(
                ChannelWiseBlock(channels, hidden_channels)
            )
```

## Multi-Scale Architecture

### Motivation

Processing high-resolution images directly is expensive. Multi-scale architecture:
- Reduces computation
- Captures features at different scales
- Allows early "factoring out" of variables

### Squeeze Operation

Reshape spatial dimensions into channels:

```python
class Squeeze(nn.Module):
    """Squeeze operation: trade spatial for channel dimensions."""
    
    def forward(self, x):
        """(B, C, H, W) -> (B, 4C, H/2, W/2)"""
        B, C, H, W = x.shape
        x = x.view(B, C, H // 2, 2, W // 2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(B, C * 4, H // 2, W // 2)
        return x, 0  # No log-det change
    
    def inverse(self, x):
        """(B, 4C, H/2, W/2) -> (B, C, H, W)"""
        B, C, H, W = x.shape
        x = x.view(B, C // 4, 2, 2, H, W)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(B, C // 4, H * 2, W * 2)
        return x, 0
```

### Factor Out

At each scale, "factor out" half the channels as final latent variables:

```python
class FactorOut(nn.Module):
    """Factor out half the channels."""
    
    def forward(self, x):
        # Split channels
        z_out, x_continue = x.chunk(2, dim=1)
        return z_out, x_continue
    
    def inverse(self, z_out, x_continue):
        # Concatenate
        return torch.cat([z_out, x_continue], dim=1)
```

### Full Multi-Scale Model

```python
class MultiScaleRealNVP(nn.Module):
    """Multi-scale RealNVP for images."""
    
    def __init__(self, image_shape, n_scales=3, n_coupling_per_scale=4):
        super().__init__()
        C, H, W = image_shape
        
        self.n_scales = n_scales
        self.flows = nn.ModuleList()
        self.factor_outs = nn.ModuleList()
        
        for scale in range(n_scales):
            # Coupling layers at this scale
            scale_flows = nn.ModuleList()
            
            for i in range(n_coupling_per_scale):
                if i % 2 == 0:
                    # Checkerboard
                    mask = checkerboard_mask(H, W, C, even=(i // 2) % 2 == 0)
                else:
                    # Channel-wise
                    mask = channel_mask(C, H, W, first_half=(i // 2) % 2 == 0)
                
                scale_flows.append(ConvCouplingLayer(C, mask))
            
            self.flows.append(scale_flows)
            
            # Squeeze and factor out (except last scale)
            if scale < n_scales - 1:
                C, H, W = C * 4, H // 2, W // 2  # After squeeze
                C = C // 2  # After factor out
                self.factor_outs.append(FactorOut())
    
    def forward(self, x):
        log_det_total = 0
        z_list = []  # Factored out latents
        
        for scale in range(self.n_scales):
            # Apply coupling layers
            for flow in self.flows[scale]:
                x, log_det = flow(x)
                log_det_total += log_det
            
            # Squeeze and factor out
            if scale < self.n_scales - 1:
                x = squeeze(x)
                z_out, x = self.factor_outs[scale](x)
                z_list.append(z_out)
        
        z_list.append(x)
        return z_list, log_det_total
    
    def log_prob(self, x):
        z_list, log_det = self.forward(x)
        
        # Sum log-prob over all factored-out latents
        log_pz = 0
        for z in z_list:
            log_pz += -0.5 * (z ** 2 + np.log(2 * np.pi)).sum(dim=[1, 2, 3])
        
        return log_pz + log_det
```

## Training RealNVP

### Data Preprocessing

```python
def preprocess(x):
    """
    Preprocess images for RealNVP.
    - Add uniform noise (dequantization)
    - Scale to [0, 1]
    - Apply logit transform
    """
    # Add uniform noise to dequantize discrete pixel values
    x = x + torch.rand_like(x) / 256.0
    
    # Scale to [0, 1]
    x = x / 256.0
    
    # Logit transform with bounds
    alpha = 0.05
    x = alpha + (1 - 2 * alpha) * x
    x = torch.log(x) - torch.log(1 - x)
    
    # Account for preprocessing in log-likelihood
    ldj = np.log(1 - 2 * alpha) - F.softplus(x) - F.softplus(-x)
    ldj = ldj.sum(dim=[1, 2, 3])
    
    return x, ldj

def postprocess(x):
    """Inverse of preprocess for sampling."""
    alpha = 0.05
    x = torch.sigmoid(x)
    x = (x - alpha) / (1 - 2 * alpha)
    x = torch.clamp(x * 256, 0, 255)
    return x
```

### Training Loop

```python
def train_realnvp(model, train_loader, n_epochs=100, lr=1e-4):
    """Train RealNVP with maximum likelihood."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        total_samples = 0
        
        for batch in train_loader:
            x = batch[0]
            
            # Preprocess
            x, preprocess_ldj = preprocess(x)
            
            optimizer.zero_grad()
            
            # Forward pass (data -> latent)
            z, log_det = model.forward(x)
            
            # Log probability
            log_pz = -0.5 * (z ** 2 + np.log(2 * np.pi)).sum(dim=-1)
            log_px = log_pz + log_det + preprocess_ldj
            
            # Loss: negative log-likelihood
            loss = -log_px.mean()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item() * len(x)
            total_samples += len(x)
        
        scheduler.step()
        
        # Compute bits per dimension
        avg_nll = total_loss / total_samples
        bpd = avg_nll / (np.log(2) * np.prod(x.shape[1:]))
        
        print(f"Epoch {epoch+1}, BPD: {bpd:.3f}")
```

## Results and Applications

### Typical Performance

| Dataset | Bits per Dimension |
|---------|-------------------|
| MNIST | ~1.06 |
| CIFAR-10 | ~3.49 |
| ImageNet 32x32 | ~4.28 |
| ImageNet 64x64 | ~3.98 |

### Strengths

1. **Exact likelihood**: No variational bounds
2. **Fast sampling**: Single forward pass
3. **Fast density**: Single inverse pass
4. **Interpolation**: Meaningful latent space
5. **Stable training**: Maximum likelihood objective

### Limitations

1. **Parameter efficiency**: Many parameters for competitive results
2. **Sample quality**: Generally below GANs for images
3. **Architecture constraints**: Must maintain invertibility
4. **Computational cost**: Expensive for high resolution

## Summary

RealNVP introduced key innovations that made normalizing flows practical:

1. **Affine coupling layers**: $y_B = x_B \odot \exp(s) + t$
2. **Masking strategies**: Checkerboard and channel-wise patterns
3. **Multi-scale architecture**: Efficient high-resolution processing
4. **Batch normalization**: Stable training dynamics

These components became the foundation for subsequent flow architectures like Glow and Neural Spline Flows.

## References

1. Dinh, L., Sohl-Dickstein, J., & Bengio, S. (2017). Density Estimation Using Real-NVP. *ICLR*.
2. Dinh, L., Krueger, D., & Bengio, Y. (2015). NICE: Non-linear Independent Components Estimation. *ICLR Workshop*.
3. Papamakarios, G., et al. (2021). Normalizing Flows for Probabilistic Modeling and Inference. *JMLR*.
