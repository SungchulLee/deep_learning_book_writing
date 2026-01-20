# Affine Coupling Layers

## Introduction

Affine coupling layers are the most widely used coupling transformation in normalizing flows. They apply an element-wise affine transformation (scale and shift) to one part of the input, with parameters computed from the other part. This simple yet powerful design enables tractable density estimation while maintaining sufficient expressiveness for complex distributions.

## Mathematical Formulation

### Forward Transformation

Given input $\mathbf{x} = (\mathbf{x}_A, \mathbf{x}_B)$:

$$\mathbf{y}_A = \mathbf{x}_A$$
$$\mathbf{y}_B = \mathbf{x}_B \odot \exp(\mathbf{s}(\mathbf{x}_A)) + \mathbf{t}(\mathbf{x}_A)$$

where:
- $\mathbf{s}(\cdot)$: Scale network outputting log-scale factors
- $\mathbf{t}(\cdot)$: Translation network outputting shift values
- $\odot$: Element-wise multiplication

### Inverse Transformation

$$\mathbf{x}_A = \mathbf{y}_A$$
$$\mathbf{x}_B = (\mathbf{y}_B - \mathbf{t}(\mathbf{y}_A)) \odot \exp(-\mathbf{s}(\mathbf{y}_A))$$

The inverse is equally efficient—no iterative procedures required.

### Jacobian and Log-Determinant

The Jacobian matrix has block structure:

$$J = \begin{pmatrix} I & 0 \\ \frac{\partial \mathbf{y}_B}{\partial \mathbf{x}_A} & \text{diag}(\exp(\mathbf{s}(\mathbf{x}_A))) \end{pmatrix}$$

The determinant of a block triangular matrix:

$$\det(J) = \det(I) \cdot \det(\text{diag}(\exp(\mathbf{s}))) = \prod_i \exp(s_i) = \exp\left(\sum_i s_i\right)$$

Therefore:

$$\log|\det(J)| = \sum_i s_i(\mathbf{x}_A)$$

This is simply the **sum of scale outputs**—O(D) computation.

## Implementation

### Basic Affine Coupling Layer

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class AffineCouplingLayer(nn.Module):
    """
    Affine coupling layer for normalizing flows.
    
    Args:
        dim: Input dimension
        mask: Binary mask indicating which dimensions are unchanged
        hidden_dims: Hidden layer dimensions for s and t networks
    """
    
    def __init__(self, dim, mask, hidden_dims=[256, 256]):
        super().__init__()
        
        self.dim = dim
        self.register_buffer('mask', mask.float())
        
        # Count dimensions in each partition
        self.d_a = int(mask.sum().item())
        self.d_b = dim - self.d_a
        
        # Scale and translation networks
        # Input: d_a dimensions, Output: d_b * 2 (for s and t)
        self.net = self._build_network(self.d_a, self.d_b * 2, hidden_dims)
    
    def _build_network(self, input_dim, output_dim, hidden_dims):
        """Build the conditioner network."""
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        net = nn.Sequential(*layers)
        
        # Initialize last layer for near-identity transform
        nn.init.zeros_(net[-1].weight)
        nn.init.zeros_(net[-1].bias)
        
        return net
    
    def _get_s_t(self, x_a):
        """Compute scale and translation from conditioning input."""
        # Extract conditioning dimensions
        batch_size = x_a.shape[0]
        x_a_flat = x_a.view(batch_size, -1)
        
        # Get parameters
        params = self.net(x_a_flat)
        s, t = params.chunk(2, dim=-1)
        
        # Optionally clamp scale for stability
        s = torch.tanh(s) * 2  # Range: [-2, 2]
        
        return s, t
    
    def forward(self, x):
        """
        Forward pass: x -> y
        
        Returns:
            y: Transformed tensor
            log_det: Log determinant of Jacobian
        """
        # Split using mask
        x_a = x * self.mask
        x_b = x * (1 - self.mask)
        
        # Get scale and translation
        # Only pass the masked (unchanged) dimensions
        x_a_input = x[:, self.mask.bool()]
        s, t = self._get_s_t(x_a_input)
        
        # Apply affine transform to x_b
        # s and t have shape (batch, d_b)
        y_b = torch.zeros_like(x)
        y_b[:, ~self.mask.bool()] = x[:, ~self.mask.bool()] * torch.exp(s) + t
        
        # Combine
        y = x_a + y_b
        
        # Log determinant
        log_det = s.sum(dim=-1)
        
        return y, log_det
    
    def inverse(self, y):
        """
        Inverse pass: y -> x
        
        Returns:
            x: Original tensor
            log_det: Log determinant of inverse Jacobian
        """
        # Split
        y_a = y * self.mask
        
        # Get scale and translation (same as forward since y_a = x_a)
        y_a_input = y[:, self.mask.bool()]
        s, t = self._get_s_t(y_a_input)
        
        # Invert affine transform
        x_b = torch.zeros_like(y)
        x_b[:, ~self.mask.bool()] = (y[:, ~self.mask.bool()] - t) * torch.exp(-s)
        
        # Combine
        x = y_a + x_b
        
        # Log determinant of inverse
        log_det = -s.sum(dim=-1)
        
        return x, log_det
```

### Masked Implementation (More Flexible)

```python
class MaskedAffineCoupling(nn.Module):
    """
    Affine coupling using explicit masking operations.
    More flexible for different mask patterns.
    """
    
    def __init__(self, dim, hidden_dims=[256, 256]):
        super().__init__()
        self.dim = dim
        
        # Single network that operates on full dimension
        # Mask is applied inside
        self.scale_net = self._build_net(dim, dim, hidden_dims)
        self.trans_net = self._build_net(dim, dim, hidden_dims)
    
    def _build_net(self, input_dim, output_dim, hidden_dims):
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        
        net = nn.Sequential(*layers)
        nn.init.zeros_(net[-1].weight)
        nn.init.zeros_(net[-1].bias)
        return net
    
    def forward(self, x, mask):
        """
        Args:
            x: Input tensor (batch, dim)
            mask: Binary mask (dim,) - 1 for unchanged, 0 for transformed
        """
        # Masked input (zeros where mask is 0)
        x_masked = x * mask
        
        # Compute scale and translation
        s = self.scale_net(x_masked) * (1 - mask)  # Only for transformed dims
        t = self.trans_net(x_masked) * (1 - mask)
        
        # Clamp scale
        s = torch.tanh(s) * 2
        
        # Transform
        y = x * mask + (x * torch.exp(s) + t) * (1 - mask)
        
        # Log det (only non-masked dimensions contribute)
        log_det = (s * (1 - mask)).sum(dim=-1)
        
        return y, log_det
    
    def inverse(self, y, mask):
        y_masked = y * mask
        
        s = self.scale_net(y_masked) * (1 - mask)
        t = self.trans_net(y_masked) * (1 - mask)
        s = torch.tanh(s) * 2
        
        x = y * mask + ((y - t) * torch.exp(-s)) * (1 - mask)
        log_det = -(s * (1 - mask)).sum(dim=-1)
        
        return x, log_det
```

## Scale Function Parameterization

### Why Use exp(s)?

The transformation $y = x \cdot \exp(s) + t$ uses $\exp(s)$ rather than $s$ directly because:

1. **Positivity**: $\exp(s) > 0$ ensures the transform is monotonic
2. **Unbounded range**: Can represent any positive scale
3. **Numerical stability**: Log-det is simply $\sum s_i$

### Alternative Parameterizations

**Sigmoid scaling**:
```python
def sigmoid_scale(s, min_scale=0.001, max_scale=10):
    """Scale in bounded range."""
    return min_scale + (max_scale - min_scale) * torch.sigmoid(s)
```

**Softplus scaling**:
```python
def softplus_scale(s):
    """Smooth positive scaling."""
    return F.softplus(s)
```

**Tanh-bounded log-scale**:
```python
def bounded_exp(s, log_scale_bound=2.0):
    """Bound the log-scale to prevent extreme values."""
    s_clamped = torch.tanh(s / log_scale_bound) * log_scale_bound
    return torch.exp(s_clamped)
```

### Stability Considerations

```python
class StableAffineCoupling(nn.Module):
    """Affine coupling with numerical stability improvements."""
    
    def __init__(self, dim, mask, scale_bound=2.0):
        super().__init__()
        self.scale_bound = scale_bound
        # ... rest of initialization
    
    def _get_s_t(self, x_a):
        params = self.net(x_a)
        s_raw, t = params.chunk(2, dim=-1)
        
        # Bounded scale for stability
        s = torch.tanh(s_raw) * self.scale_bound
        
        return s, t
    
    def forward(self, x):
        # ... split x
        s, t = self._get_s_t(x_a)
        
        # Use log-sum-exp trick for very large/small scales
        y_b = x_b * torch.exp(s) + t
        
        # Check for NaN/Inf
        if torch.isnan(y_b).any() or torch.isinf(y_b).any():
            raise ValueError("Numerical instability in affine coupling")
        
        # ...
```

## Convolutional Affine Coupling

For image data, use convolutional conditioner networks:

```python
class ConvAffineCoupling(nn.Module):
    """Affine coupling for image data with conv networks."""
    
    def __init__(self, in_channels, hidden_channels=64):
        super().__init__()
        
        # Convolutional s and t networks
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, in_channels * 2, 3, padding=1)
        )
        
        # Zero init
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
    
    def forward(self, x, mask):
        """
        Args:
            x: Image tensor (batch, channels, height, width)
            mask: Binary mask (channels, height, width)
        """
        # Masked input
        x_masked = x * mask
        
        # Get s and t
        params = self.net(x_masked)
        s, t = params.chunk(2, dim=1)
        
        # Apply only to non-masked locations
        s = s * (1 - mask)
        t = t * (1 - mask)
        s = torch.tanh(s) * 2
        
        # Transform
        y = x * mask + (x * torch.exp(s) + t) * (1 - mask)
        
        # Log det: sum over all spatial locations and channels
        log_det = s.sum(dim=[1, 2, 3])
        
        return y, log_det
    
    def inverse(self, y, mask):
        y_masked = y * mask
        
        params = self.net(y_masked)
        s, t = params.chunk(2, dim=1)
        s = s * (1 - mask)
        t = t * (1 - mask)
        s = torch.tanh(s) * 2
        
        x = y * mask + ((y - t) * torch.exp(-s)) * (1 - mask)
        log_det = -s.sum(dim=[1, 2, 3])
        
        return x, log_det
```

## Building a Complete Flow

### Stacking with Alternating Masks

```python
class AffineFlow(nn.Module):
    """Complete flow with stacked affine coupling layers."""
    
    def __init__(self, dim, n_layers=8, hidden_dims=[256, 256]):
        super().__init__()
        self.dim = dim
        
        self.layers = nn.ModuleList()
        
        for i in range(n_layers):
            # Alternate between two mask patterns
            if i % 2 == 0:
                mask = torch.cat([
                    torch.ones(dim // 2),
                    torch.zeros(dim - dim // 2)
                ])
            else:
                mask = torch.cat([
                    torch.zeros(dim // 2),
                    torch.ones(dim - dim // 2)
                ])
            
            self.layers.append(
                AffineCouplingLayer(dim, mask, hidden_dims)
            )
    
    def forward(self, x):
        """Transform data to latent space."""
        log_det_total = torch.zeros(x.shape[0], device=x.device)
        
        for layer in self.layers:
            x, log_det = layer(x)
            log_det_total += log_det
        
        return x, log_det_total
    
    def inverse(self, z):
        """Transform latent to data space."""
        log_det_total = torch.zeros(z.shape[0], device=z.device)
        
        for layer in reversed(self.layers):
            z, log_det = layer.inverse(z)
            log_det_total += log_det
        
        return z, log_det_total
    
    def log_prob(self, x):
        """Compute log probability of data."""
        z, log_det = self.forward(x)
        
        # Standard Gaussian base distribution
        log_pz = -0.5 * (z ** 2 + np.log(2 * np.pi)).sum(dim=-1)
        
        return log_pz + log_det
    
    def sample(self, n_samples, device='cpu'):
        """Generate samples."""
        z = torch.randn(n_samples, self.dim, device=device)
        x, _ = self.inverse(z)
        return x
```

### Training Loop

```python
def train_affine_flow(model, data, n_epochs=100, batch_size=256, lr=1e-3):
    """Train flow via maximum likelihood."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data),
        batch_size=batch_size, shuffle=True
    )
    
    for epoch in range(n_epochs):
        total_loss = 0
        
        for batch, in loader:
            optimizer.zero_grad()
            
            # Negative log-likelihood
            loss = -model.log_prob(batch).mean()
            
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            total_loss += loss.item() * len(batch)
        
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(data)
            print(f"Epoch {epoch+1}, NLL: {avg_loss:.4f}")
    
    return model
```

## Expressiveness Analysis

### Single Layer Limitations

One affine coupling layer can only:
- Scale dimensions in $\mathbf{x}_B$
- Shift dimensions in $\mathbf{x}_B$
- Leave $\mathbf{x}_A$ unchanged

This is a relatively weak transformation family.

### Stacking Increases Expressiveness

With multiple layers and alternating masks:
- Each dimension gets transformed multiple times
- Nonlinear dependencies accumulate
- Universal approximation possible (with enough layers)

### Depth vs Width Trade-off

| More Layers | Wider Networks |
|-------------|----------------|
| More transformations | More complex per-layer |
| Better gradient flow | Larger memory |
| Typically preferred | Diminishing returns |

## Summary

Affine coupling layers provide:

1. **Simple formula**: $y_B = x_B \cdot \exp(s) + t$
2. **Easy inverse**: $x_B = (y_B - t) \cdot \exp(-s)$
3. **Trivial log-det**: $\sum_i s_i$
4. **Flexible conditioner**: Any neural network for $s$ and $t$
5. **Efficient computation**: O(D) for all operations

The affine coupling layer is the workhorse of practical normalizing flows, forming the basis for RealNVP, Glow, and many other architectures. Its simplicity and efficiency make it the default choice for most flow-based models.

## References

1. Dinh, L., et al. (2017). Density Estimation Using Real-NVP. *ICLR*.
2. Kingma, D. P., & Dhariwal, P. (2018). Glow: Generative Flow with Invertible 1×1 Convolutions. *NeurIPS*.
3. Papamakarios, G., et al. (2021). Normalizing Flows for Probabilistic Modeling and Inference. *JMLR*.
