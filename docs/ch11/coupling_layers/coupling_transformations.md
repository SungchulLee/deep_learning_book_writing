# Coupling Transformations

## Introduction

Coupling transformations are the backbone of modern normalizing flow architectures. The key idea is remarkably simple: split the input into two parts, transform one part using information from the other, and leave the other part unchanged. This design provides invertibility by construction and yields triangular Jacobians with O(D) determinant computation.

## The Coupling Principle

### Basic Structure

Given input $\mathbf{x} \in \mathbb{R}^D$, split it into two parts:

$$\mathbf{x} = (\mathbf{x}_A, \mathbf{x}_B)$$

where $\mathbf{x}_A \in \mathbb{R}^d$ and $\mathbf{x}_B \in \mathbb{R}^{D-d}$.

The coupling transformation is:

$$\mathbf{y}_A = \mathbf{x}_A$$
$$\mathbf{y}_B = g(\mathbf{x}_B; \theta(\mathbf{x}_A))$$

where:
- $\mathbf{x}_A$ passes through **unchanged**
- $\mathbf{x}_B$ is transformed by function $g$
- The transformation parameters $\theta$ are computed from $\mathbf{x}_A$
- $g$ must be invertible with respect to $\mathbf{x}_B$ for fixed $\theta$

### Why This Works

**Invertibility**: Since $\mathbf{x}_A = \mathbf{y}_A$, we can recover the conditioning information. Then:

$$\mathbf{x}_B = g^{-1}(\mathbf{y}_B; \theta(\mathbf{y}_A))$$

**Triangular Jacobian**: The Jacobian has block structure:

$$J = \begin{pmatrix} I_d & 0 \\ \frac{\partial \mathbf{y}_B}{\partial \mathbf{x}_A} & \frac{\partial \mathbf{y}_B}{\partial \mathbf{x}_B} \end{pmatrix}$$

**Efficient Determinant**: 

$$\det(J) = \det(I_d) \cdot \det\left(\frac{\partial \mathbf{y}_B}{\partial \mathbf{x}_B}\right) = \det\left(\frac{\partial g}{\partial \mathbf{x}_B}\right)$$

The determinant only depends on how $g$ transforms $\mathbf{x}_B$, not on the complex function $\theta(\mathbf{x}_A)$.

## Splitting Strategies

### Dimension-based Splitting

The simplest approach: split by index.

```python
def split_dimensions(x, d):
    """Split into first d and remaining dimensions."""
    return x[:, :d], x[:, d:]

def merge_dimensions(x_a, x_b):
    """Merge split dimensions."""
    return torch.cat([x_a, x_b], dim=-1)
```

### Masking

More flexible: use binary masks.

```python
class MaskedCoupling(nn.Module):
    """Coupling layer with arbitrary binary mask."""
    
    def __init__(self, dim, mask):
        super().__init__()
        self.register_buffer('mask', mask)  # Binary mask
        
        # Count dimensions in each part
        self.d_a = mask.sum().int().item()
        self.d_b = dim - self.d_a
        
        # Network to compute transformation parameters
        self.conditioner = self._build_conditioner()
    
    def forward(self, x):
        # Split using mask
        x_a = x * self.mask
        x_b = x * (1 - self.mask)
        
        # Get transformation parameters from x_a
        # (only non-masked values matter)
        params = self.conditioner(x_a)
        
        # Transform x_b
        y_b, log_det = self.transform(x_b, params)
        
        # Combine
        y = x_a + y_b  # Works because masks are complementary
        
        return y, log_det
```

### Common Mask Patterns

**Checkerboard (for images)**:
```python
def checkerboard_mask(height, width, channels, even=True):
    """Create checkerboard pattern mask."""
    mask = torch.zeros(channels, height, width)
    for i in range(height):
        for j in range(width):
            if (i + j) % 2 == (0 if even else 1):
                mask[:, i, j] = 1
    return mask
```

**Channel-wise (for images)**:
```python
def channel_mask(channels, channel_type='upper'):
    """Mask half the channels."""
    mask = torch.zeros(channels)
    if channel_type == 'upper':
        mask[:channels // 2] = 1
    else:
        mask[channels // 2:] = 1
    return mask
```

**Alternating (for vectors)**:
```python
def alternating_mask(dim, even=True):
    """Every other dimension."""
    mask = torch.zeros(dim)
    start = 0 if even else 1
    mask[start::2] = 1
    return mask
```

## The Conditioner Network

### Role of the Conditioner

The conditioner $\theta(\mathbf{x}_A)$ computes parameters for transforming $\mathbf{x}_B$. It can be **any neural network** since:

1. It doesn't need to be invertible
2. It doesn't contribute to the Jacobian determinant
3. It only affects expressiveness, not tractability

### Architectural Choices

**MLP (for vectors)**:
```python
class MLPConditioner(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[256, 256]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
        
        # Initialize last layer to near-identity transform
        nn.init.zeros_(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)
    
    def forward(self, x):
        return self.net(x)
```

**CNN (for images)**:
```python
class ConvConditioner(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, out_channels, 3, padding=1)
        )
        
        # Zero initialization for near-identity
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
    
    def forward(self, x):
        return self.net(x)
```

**ResNet blocks (for better gradient flow)**:
```python
class ResNetConditioner(nn.Module):
    def __init__(self, channels, n_blocks=4):
        super().__init__()
        self.blocks = nn.ModuleList([
            ResBlock(channels) for _ in range(n_blocks)
        ])
        self.output = nn.Conv2d(channels, channels * 2, 3, padding=1)
        
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
    
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.output(x)
```

## Transform Functions

The transform function $g$ determines the coupling layer's expressiveness. Common choices:

### Additive Coupling

$$y_B = x_B + t(x_A)$$

- **Jacobian**: Identity matrix
- **Log-det**: 0
- **Pros**: Simple, volume-preserving
- **Cons**: Limited expressiveness

```python
class AdditiveCoupling(nn.Module):
    def __init__(self, dim, mask):
        super().__init__()
        self.mask = mask
        self.t_net = MLPConditioner(mask.sum().int(), (1-mask).sum().int())
    
    def forward(self, x):
        x_a = x * self.mask
        t = self.t_net(x_a[self.mask.bool()].view(x.shape[0], -1))
        
        y = x.clone()
        y[:, ~self.mask.bool()] = x[:, ~self.mask.bool()] + t
        
        log_det = torch.zeros(x.shape[0], device=x.device)
        return y, log_det
    
    def inverse(self, y):
        y_a = y * self.mask
        t = self.t_net(y_a[self.mask.bool()].view(y.shape[0], -1))
        
        x = y.clone()
        x[:, ~self.mask.bool()] = y[:, ~self.mask.bool()] - t
        
        return x, torch.zeros(y.shape[0], device=y.device)
```

### Affine Coupling

$$y_B = x_B \odot \exp(s(x_A)) + t(x_A)$$

- **Jacobian**: Diagonal with entries $\exp(s_i)$
- **Log-det**: $\sum_i s_i$
- **Pros**: More expressive than additive
- **Cons**: Potential numerical issues with $\exp$

### Monotonic Transformations

More general invertible transforms like:
- Spline-based (Neural Spline Flows)
- Mixture of CDFs
- Rational quadratic splines

## Stacking Coupling Layers

### The Problem: Unchanged Dimensions

A single coupling layer leaves $\mathbf{x}_A$ unchanged. To transform all dimensions, we need multiple layers with **alternating masks**.

### Alternating Pattern

```python
class CouplingFlow(nn.Module):
    def __init__(self, dim, n_layers=8):
        super().__init__()
        
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            # Alternate masks
            if i % 2 == 0:
                mask = torch.cat([torch.ones(dim // 2), 
                                  torch.zeros(dim - dim // 2)])
            else:
                mask = torch.cat([torch.zeros(dim // 2), 
                                  torch.ones(dim - dim // 2)])
            
            self.layers.append(AffineCouplingLayer(dim, mask))
    
    def forward(self, x):
        log_det_total = 0
        for layer in self.layers:
            x, log_det = layer(x)
            log_det_total += log_det
        return x, log_det_total
    
    def inverse(self, y):
        log_det_total = 0
        for layer in reversed(self.layers):
            y, log_det = layer.inverse(y)
            log_det_total += log_det
        return y, log_det_total
```

### More Sophisticated Mixing

Beyond alternating masks, we can add:

1. **Permutations**: Shuffle dimensions between layers
2. **1×1 Convolutions**: Learned linear mixing (Glow)
3. **Random projections**: Fixed random rotations

```python
class CouplingWithPermutation(nn.Module):
    def __init__(self, dim, n_layers=8):
        super().__init__()
        
        self.layers = nn.ModuleList()
        self.permutations = nn.ParameterList()
        
        for i in range(n_layers):
            mask = self._create_mask(dim, i % 2 == 0)
            self.layers.append(AffineCouplingLayer(dim, mask))
            
            # Random permutation between layers
            if i < n_layers - 1:
                perm = torch.randperm(dim)
                self.register_buffer(f'perm_{i}', perm)
                self.register_buffer(f'inv_perm_{i}', torch.argsort(perm))
    
    def forward(self, x):
        log_det_total = 0
        
        for i, layer in enumerate(self.layers):
            x, log_det = layer(x)
            log_det_total += log_det
            
            # Apply permutation
            if i < len(self.layers) - 1:
                perm = getattr(self, f'perm_{i}')
                x = x[:, perm]
        
        return x, log_det_total
```

## Computational Complexity

### Forward Pass

| Operation | Complexity |
|-----------|------------|
| Split | O(D) |
| Conditioner | O(network) |
| Transform | O(D) |
| Log-det | O(D) |

### Inverse Pass

Same as forward—this is a key advantage of coupling layers.

### Comparison with Autoregressive

| Aspect | Coupling | Autoregressive |
|--------|----------|----------------|
| Forward | O(1) parallel | O(1) or O(D) |
| Inverse | O(1) parallel | O(1) or O(D) |
| Per-layer expressiveness | Lower | Higher |
| Layers needed | More | Fewer |

## Design Principles

### 1. Initialize Near Identity

Start with transforms close to identity for stable training:

```python
# Zero-initialize the last layer of conditioner
nn.init.zeros_(conditioner[-1].weight)
nn.init.zeros_(conditioner[-1].bias)
```

### 2. Bound the Scale

Prevent exploding/vanishing transforms:

```python
def get_scale(s_raw):
    # Soft clipping to reasonable range
    return torch.tanh(s_raw) * 2  # Range: [-2, 2]
    # Or: sigmoid for [0, 1] range
```

### 3. Sufficient Depth

Each layer transforms only half the dimensions:
- Minimum: 4 layers (each dimension transformed twice)
- Typical: 8-16 layers for toy problems
- Complex data: 32+ layers

### 4. Rich Conditioners

The conditioner's capacity limits expressiveness:
- Use deep networks
- Add residual connections
- Consider attention mechanisms

## Summary

Coupling transformations provide:

1. **Guaranteed invertibility** through the split-transform-merge structure
2. **Efficient Jacobian** computation via triangular structure
3. **Flexible expressiveness** through arbitrary conditioner networks
4. **Parallel computation** in both forward and inverse directions

The coupling principle—transforming one part based on another—is the foundation for RealNVP, Glow, and Neural Spline Flows. The specific choice of transform function (additive, affine, spline) and conditioner architecture determines the flow's expressiveness and computational properties.

## References

1. Dinh, L., et al. (2015). NICE: Non-linear Independent Components Estimation. *ICLR Workshop*.
2. Dinh, L., et al. (2017). Density Estimation Using Real-NVP. *ICLR*.
3. Kingma, D. P., & Dhariwal, P. (2018). Glow: Generative Flow with Invertible 1×1 Convolutions. *NeurIPS*.
4. Durkan, C., et al. (2019). Neural Spline Flows. *NeurIPS*.
