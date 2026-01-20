# Glow: Generative Flow with Invertible 1×1 Convolutions

## Introduction

Glow extends RealNVP with three key innovations: activation normalization (ActNorm), invertible 1×1 convolutions, and a more systematic multi-scale architecture. These improvements enable high-quality image generation with exact likelihood computation, achieving results competitive with GANs while maintaining the benefits of flow-based models.

## From RealNVP to Glow

### What Glow Inherits

From RealNVP, Glow keeps:
- Affine coupling layers
- Multi-scale architecture with squeeze and factor-out
- Maximum likelihood training

### What Glow Improves

| Component | RealNVP | Glow |
|-----------|---------|------|
| Normalization | Batch norm | **ActNorm** |
| Channel mixing | Fixed permutations | **Invertible 1×1 conv** |
| Architecture | Ad-hoc | **Systematic flow steps** |

## Architecture Overview

### The Glow Block

Each "step of flow" in Glow consists of three components:

```
Input → ActNorm → Invertible 1×1 Conv → Affine Coupling → Output
```

These are stacked multiple times per scale, then squeeze/factor-out operations are applied.

### Full Architecture

```
For each scale:
    For K steps:
        1. ActNorm
        2. Invertible 1×1 convolution
        3. Affine coupling layer
    
    If not final scale:
        Squeeze (spatial → channels)
        Factor out half the channels
```

## Component 1: ActNorm (Activation Normalization)

### Motivation

Batch normalization in flows is problematic:
- Depends on other samples in the batch
- Different behavior at train/test time
- Complicates invertibility

ActNorm provides normalization without batch statistics.

### Definition

Per-channel affine transform:

$$y_{c,h,w} = s_c \cdot x_{c,h,w} + b_c$$

where $s_c$ and $b_c$ are learnable per-channel parameters.

### Data-Dependent Initialization

Initialize using first batch statistics:

1. Compute per-channel mean $\mu_c$ and std $\sigma_c$
2. Set $s_c = 1/\sigma_c$ and $b_c = -\mu_c/\sigma_c$
3. After initialization, $s_c$ and $b_c$ become regular trainable parameters

### Implementation

```python
class ActNorm(nn.Module):
    """
    Activation Normalization layer.
    Data-dependent initialization, then regular training.
    """
    
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels
        
        # Learnable parameters
        self.log_s = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        
        # Track initialization
        self.register_buffer('initialized', torch.tensor(False))
    
    def initialize(self, x):
        """Data-dependent initialization using first batch."""
        with torch.no_grad():
            # Compute per-channel statistics
            # x: (B, C, H, W)
            mean = x.mean(dim=[0, 2, 3], keepdim=True)
            std = x.std(dim=[0, 2, 3], keepdim=True)
            
            # Initialize to normalize
            self.log_s.data = -torch.log(std + 1e-6)
            self.b.data = -mean / (std + 1e-6)
            
            self.initialized.fill_(True)
    
    def forward(self, x):
        """Forward: x -> y"""
        if not self.initialized:
            self.initialize(x)
        
        # y = s * x + b = exp(log_s) * x + b
        y = torch.exp(self.log_s) * x + self.b
        
        # Log determinant: H * W * sum(log_s)
        _, _, H, W = x.shape
        log_det = H * W * self.log_s.sum()
        log_det = log_det.expand(x.shape[0])
        
        return y, log_det
    
    def inverse(self, y):
        """Inverse: y -> x"""
        # x = (y - b) / s = (y - b) * exp(-log_s)
        x = (y - self.b) * torch.exp(-self.log_s)
        
        _, _, H, W = y.shape
        log_det = -H * W * self.log_s.sum()
        log_det = log_det.expand(y.shape[0])
        
        return x, log_det
```

## Component 2: Invertible 1×1 Convolution

### Motivation

RealNVP uses fixed permutations to mix channels between coupling layers. This limits the model's ability to learn optimal channel interactions.

Invertible 1×1 convolutions provide **learned channel mixing** with tractable Jacobian.

### Mathematical Formulation

A 1×1 convolution with weight $\mathbf{W} \in \mathbb{R}^{C \times C}$ applies a linear transform across channels at each spatial location:

$$y_{:,h,w} = \mathbf{W} x_{:,h,w}$$

For invertibility, $\mathbf{W}$ must be invertible (non-singular).

### Log-Determinant

The Jacobian is block diagonal with $\mathbf{W}$ repeated $H \times W$ times:

$$\log|\det J| = H \cdot W \cdot \log|\det \mathbf{W}|$$

### Implementation Strategies

**Direct parameterization** (simple but can become singular):
```python
class Invertible1x1ConvSimple(nn.Module):
    """Simple invertible 1x1 conv."""
    
    def __init__(self, num_channels):
        super().__init__()
        # Initialize with random orthogonal matrix
        W = torch.qr(torch.randn(num_channels, num_channels))[0]
        self.W = nn.Parameter(W)
    
    def forward(self, x):
        B, C, H, W_spatial = x.shape
        
        # Apply 1x1 conv
        y = F.conv2d(x, self.W.view(C, C, 1, 1))
        
        # Log determinant
        log_det = H * W_spatial * torch.slogdet(self.W)[1]
        log_det = log_det.expand(B)
        
        return y, log_det
    
    def inverse(self, y):
        B, C, H, W_spatial = y.shape
        
        W_inv = torch.inverse(self.W)
        x = F.conv2d(y, W_inv.view(C, C, 1, 1))
        
        log_det = -H * W_spatial * torch.slogdet(self.W)[1]
        log_det = log_det.expand(B)
        
        return x, log_det
```

**LU decomposition** (more stable, O(C) log-det):
```python
class Invertible1x1ConvLU(nn.Module):
    """
    Invertible 1x1 conv with LU decomposition for efficient computation.
    W = P @ L @ U where P is permutation, L lower triangular, U upper triangular.
    """
    
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels
        
        # Initialize with random orthogonal matrix
        W = torch.qr(torch.randn(num_channels, num_channels))[0]
        
        # LU decomposition
        P, L, U = torch.linalg.lu(W)
        
        # Store components
        # P is fixed (permutation)
        self.register_buffer('P', P)
        
        # L is lower triangular with ones on diagonal
        # Store the strict lower part
        L_mask = torch.tril(torch.ones_like(L), diagonal=-1)
        self.register_buffer('L_mask', L_mask)
        self.L = nn.Parameter(L * L_mask)
        
        # U is upper triangular
        # Store diagonal separately for log-det
        U_mask = torch.triu(torch.ones_like(U), diagonal=1)
        self.register_buffer('U_mask', U_mask)
        self.U = nn.Parameter(U * U_mask)
        
        # Diagonal of U (sign and log magnitude)
        diag_U = torch.diag(U)
        self.log_diag = nn.Parameter(torch.log(torch.abs(diag_U)))
        self.sign_diag = nn.Parameter(torch.sign(diag_U))
        self.sign_diag.requires_grad = False
    
    def _get_W(self):
        """Reconstruct W from LU components."""
        L = self.L * self.L_mask + torch.eye(self.num_channels, device=self.L.device)
        U = self.U * self.U_mask + torch.diag(self.sign_diag * torch.exp(self.log_diag))
        W = self.P @ L @ U
        return W
    
    def forward(self, x):
        B, C, H, W_spatial = x.shape
        
        W = self._get_W()
        y = F.conv2d(x, W.view(C, C, 1, 1))
        
        # Log det is just sum of log diagonal of U
        log_det = H * W_spatial * self.log_diag.sum()
        log_det = log_det.expand(B)
        
        return y, log_det
    
    def inverse(self, y):
        B, C, H, W_spatial = y.shape
        
        W = self._get_W()
        W_inv = torch.inverse(W)
        x = F.conv2d(y, W_inv.view(C, C, 1, 1))
        
        log_det = -H * W_spatial * self.log_diag.sum()
        log_det = log_det.expand(B)
        
        return x, log_det
```

## Component 3: Affine Coupling (Same as RealNVP)

Glow uses the same affine coupling as RealNVP, typically with channel-wise split:

```python
class GlowCouplingLayer(nn.Module):
    """Affine coupling layer for Glow."""
    
    def __init__(self, num_channels, hidden_channels=512):
        super().__init__()
        
        # Split channels in half
        self.split_idx = num_channels // 2
        
        # Conditioner network
        self.net = nn.Sequential(
            nn.Conv2d(self.split_idx, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, (num_channels - self.split_idx) * 2, 3, padding=1)
        )
        
        # Zero initialization
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
    
    def forward(self, x):
        # Split
        x_a, x_b = x[:, :self.split_idx], x[:, self.split_idx:]
        
        # Get scale and translation
        params = self.net(x_a)
        log_s, t = params.chunk(2, dim=1)
        
        # Bound scale
        log_s = torch.tanh(log_s) * 2
        
        # Transform
        y_b = x_b * torch.exp(log_s) + t
        y = torch.cat([x_a, y_b], dim=1)
        
        # Log determinant
        log_det = log_s.sum(dim=[1, 2, 3])
        
        return y, log_det
    
    def inverse(self, y):
        # Split
        y_a, y_b = y[:, :self.split_idx], y[:, self.split_idx:]
        
        # Get scale and translation (y_a = x_a)
        params = self.net(y_a)
        log_s, t = params.chunk(2, dim=1)
        log_s = torch.tanh(log_s) * 2
        
        # Invert
        x_b = (y_b - t) * torch.exp(-log_s)
        x = torch.cat([y_a, x_b], dim=1)
        
        log_det = -log_s.sum(dim=[1, 2, 3])
        
        return x, log_det
```

## Complete Glow Model

```python
class GlowStep(nn.Module):
    """Single step of flow: ActNorm -> 1x1Conv -> Coupling"""
    
    def __init__(self, num_channels, hidden_channels=512):
        super().__init__()
        
        self.actnorm = ActNorm(num_channels)
        self.inv_conv = Invertible1x1ConvLU(num_channels)
        self.coupling = GlowCouplingLayer(num_channels, hidden_channels)
    
    def forward(self, x):
        log_det = 0
        
        x, ld = self.actnorm(x)
        log_det += ld
        
        x, ld = self.inv_conv(x)
        log_det += ld
        
        x, ld = self.coupling(x)
        log_det += ld
        
        return x, log_det
    
    def inverse(self, y):
        log_det = 0
        
        y, ld = self.coupling.inverse(y)
        log_det += ld
        
        y, ld = self.inv_conv.inverse(y)
        log_det += ld
        
        y, ld = self.actnorm.inverse(y)
        log_det += ld
        
        return y, log_det


class Glow(nn.Module):
    """
    Complete Glow model with multi-scale architecture.
    
    Args:
        image_shape: (C, H, W) tuple
        K: Number of flow steps per scale
        L: Number of scales
        hidden_channels: Hidden channels in coupling networks
    """
    
    def __init__(self, image_shape, K=32, L=3, hidden_channels=512):
        super().__init__()
        
        self.K = K
        self.L = L
        C, H, W = image_shape
        
        self.flows = nn.ModuleList()
        
        for l in range(L):
            # After squeeze: (4C, H/2, W/2)
            C_curr = C * (4 ** (l + 1)) // (2 ** l)  # Account for factor-out
            
            # K flow steps at this scale
            scale_flows = nn.ModuleList([
                GlowStep(C_curr, hidden_channels) for _ in range(K)
            ])
            self.flows.append(scale_flows)
        
        # Track shapes for each scale
        self.output_shapes = self._compute_output_shapes(image_shape)
    
    def _compute_output_shapes(self, shape):
        """Compute output shapes at each scale."""
        C, H, W = shape
        shapes = []
        
        for l in range(self.L):
            # Squeeze
            C, H, W = C * 4, H // 2, W // 2
            
            # Factor out (except last scale)
            if l < self.L - 1:
                shapes.append((C // 2, H, W))
                C = C // 2
            else:
                shapes.append((C, H, W))
        
        return shapes
    
    def forward(self, x):
        """Forward: image -> latent"""
        log_det_total = 0
        z_list = []
        
        for l in range(self.L):
            # Squeeze
            x = self._squeeze(x)
            
            # Flow steps
            for flow in self.flows[l]:
                x, log_det = flow(x)
                log_det_total += log_det
            
            # Factor out (except last scale)
            if l < self.L - 1:
                z_out, x = x.chunk(2, dim=1)
                z_list.append(z_out)
        
        z_list.append(x)
        
        return z_list, log_det_total
    
    def inverse(self, z_list):
        """Inverse: latent -> image"""
        log_det_total = 0
        
        x = z_list[-1]
        
        for l in reversed(range(self.L)):
            # Un-factor
            if l < self.L - 1:
                x = torch.cat([z_list[l], x], dim=1)
            
            # Inverse flow steps
            for flow in reversed(self.flows[l]):
                x, log_det = flow.inverse(x)
                log_det_total += log_det
            
            # Unsqueeze
            x = self._unsqueeze(x)
        
        return x, log_det_total
    
    def _squeeze(self, x):
        """(B, C, H, W) -> (B, 4C, H/2, W/2)"""
        B, C, H, W = x.shape
        x = x.view(B, C, H // 2, 2, W // 2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(B, C * 4, H // 2, W // 2)
        return x
    
    def _unsqueeze(self, x):
        """(B, 4C, H/2, W/2) -> (B, C, H, W)"""
        B, C, H, W = x.shape
        x = x.view(B, C // 4, 2, 2, H, W)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(B, C // 4, H * 2, W * 2)
        return x
    
    def log_prob(self, x):
        """Compute log p(x)."""
        z_list, log_det = self.forward(x)
        
        # Sum log-prob over all latents
        log_pz = 0
        for z in z_list:
            log_pz += -0.5 * (z ** 2 + np.log(2 * np.pi)).sum(dim=[1, 2, 3])
        
        return log_pz + log_det
    
    def sample(self, n_samples, temperature=1.0, device='cpu'):
        """Generate samples."""
        z_list = []
        
        for shape in self.output_shapes:
            C, H, W = shape
            z = torch.randn(n_samples, C, H, W, device=device) * temperature
            z_list.append(z)
        
        x, _ = self.inverse(z_list)
        return x
```

## Training Glow

### Preprocessing

```python
def preprocess_glow(x, n_bits=8):
    """
    Preprocess images for Glow.
    - Dequantize
    - Scale to [0, 1]
    - Optional: reduce bits
    """
    n_bins = 2 ** n_bits
    
    # Add uniform noise for dequantization
    x = x + torch.rand_like(x) / n_bins
    
    # Scale to [0, 1]
    x = x / n_bins
    
    # No logit transform in Glow (uses different bounds)
    return x

def compute_loss(model, x, n_bits=8):
    """Compute bits per dimension loss."""
    n_bins = 2 ** n_bits
    n_pixels = np.prod(x.shape[1:])
    
    log_prob = model.log_prob(x)
    
    # Add log-det for dequantization
    log_prob = log_prob - np.log(n_bins) * n_pixels
    
    # Convert to bits per dimension
    bpd = -log_prob / (np.log(2) * n_pixels)
    
    return bpd.mean()
```

### Training Loop

```python
def train_glow(model, train_loader, n_epochs=100, lr=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(n_epochs):
        model.train()
        total_bpd = 0
        n_batches = 0
        
        for batch in train_loader:
            x = batch[0]
            x = preprocess_glow(x)
            
            optimizer.zero_grad()
            
            bpd = compute_loss(model, x)
            bpd.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            total_bpd += bpd.item()
            n_batches += 1
        
        avg_bpd = total_bpd / n_batches
        print(f"Epoch {epoch+1}, BPD: {avg_bpd:.3f}")
```

## Applications

### Latent Space Interpolation

```python
def interpolate(model, x1, x2, n_steps=10):
    """Interpolate between two images in latent space."""
    # Encode
    z1_list, _ = model.forward(x1)
    z2_list, _ = model.forward(x2)
    
    # Interpolate each latent
    images = []
    for alpha in torch.linspace(0, 1, n_steps):
        z_interp = [
            (1 - alpha) * z1 + alpha * z2 
            for z1, z2 in zip(z1_list, z2_list)
        ]
        x_interp, _ = model.inverse(z_interp)
        images.append(x_interp)
    
    return images
```

### Attribute Manipulation

```python
def find_attribute_direction(model, images_with_attr, images_without_attr):
    """Find direction in latent space corresponding to an attribute."""
    # Encode both sets
    z_with = model.forward(images_with_attr)[0]
    z_without = model.forward(images_without_attr)[0]
    
    # Compute mean difference
    direction = [
        z_w.mean(dim=0) - z_wo.mean(dim=0)
        for z_w, z_wo in zip(z_with, z_without)
    ]
    
    return direction

def apply_attribute(model, x, direction, strength=1.0):
    """Apply attribute direction to an image."""
    z_list, _ = model.forward(x)
    
    z_modified = [
        z + strength * d.unsqueeze(0)
        for z, d in zip(z_list, direction)
    ]
    
    x_modified, _ = model.inverse(z_modified)
    return x_modified
```

## Summary

Glow improves upon RealNVP with:

1. **ActNorm**: Data-dependent initialization, no batch statistics
2. **Invertible 1×1 conv**: Learned channel mixing with efficient log-det
3. **Systematic architecture**: Clean flow step structure

Key formula for Glow step:

$$\text{Output} = \text{Coupling}(\text{Inv1x1}(\text{ActNorm}(\text{Input})))$$

Glow demonstrated that flow-based models can generate high-quality images while maintaining exact likelihood computation—a capability that GANs lack.

## References

1. Kingma, D. P., & Dhariwal, P. (2018). Glow: Generative Flow with Invertible 1×1 Convolutions. *NeurIPS*.
2. Dinh, L., et al. (2017). Density Estimation Using Real-NVP. *ICLR*.
3. Papamakarios, G., et al. (2021). Normalizing Flows for Probabilistic Modeling and Inference. *JMLR*.
