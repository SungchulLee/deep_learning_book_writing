# Neural Spline Flows

## Introduction

Neural Spline Flows (NSF) replace simple affine transformations in coupling layers with more expressive **monotonic spline functions**. By using rational quadratic splines, NSF achieves significantly greater expressiveness per layer while maintaining exact invertibility and tractable Jacobians, often resulting in better density estimation with fewer parameters.

## Motivation: Beyond Affine Transforms

### Limitations of Affine Coupling

Affine coupling transforms each dimension with:

$$y = x \cdot \exp(s) + t$$

This is a **linear function** of $x$ (for fixed $s, t$). To model complex, nonlinear relationships, many layers are required.

### The Spline Solution

Replace affine with a **monotonic spline**:

$$y = \text{Spline}(x; \theta)$$

where $\theta$ are spline parameters predicted by a neural network.

**Benefits**:
- More expressive per layer
- Fewer layers needed
- Better density estimation with similar parameter count

## Rational Quadratic Splines

### Why Rational Quadratic?

Among spline choices, **rational quadratic splines** offer:
- Smooth and infinitely differentiable
- Analytically invertible (closed-form)
- Tractable derivatives for log-det
- Flexible shape with few parameters

### Definition

A rational quadratic spline is defined by:
- **K bins** on the x-axis: widths $w_k$
- **K bins** on the y-axis: heights $h_k$  
- **K+1 derivatives** at knot points: $d_k$

Within bin $k$, the transformation is:

$$y = y_k + \frac{(y_{k+1} - y_k)[s_k\xi^2 + d_k\xi(1-\xi)]}{s_k + (d_{k+1} + d_k - 2s_k)\xi(1-\xi)}$$

where $\xi = (x - x_k)/(x_{k+1} - x_k)$ and $s_k = (y_{k+1} - y_k)/(x_{k+1} - x_k)$.

### Ensuring Monotonicity

For invertibility, enforce:
1. **Positive widths and heights**: Use softmax
2. **Positive derivatives**: Use softplus

```python
def make_spline_params(raw_widths, raw_heights, raw_derivatives):
    """Convert raw network outputs to valid spline parameters."""
    # Widths and heights via softmax
    widths = F.softmax(raw_widths, dim=-1)
    heights = F.softmax(raw_heights, dim=-1)
    
    # Derivatives via softplus
    derivatives = F.softplus(raw_derivatives) + 1e-3
    
    return widths, heights, derivatives
```

## Implementation

### Rational Quadratic Spline Function

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def rational_quadratic_spline(
    inputs,
    widths,
    heights, 
    derivatives,
    bounds=(-3.0, 3.0),
    inverse=False
):
    """
    Apply rational quadratic spline transformation.
    
    Args:
        inputs: Input values (batch, dim)
        widths: Bin widths (batch, dim, K)
        heights: Bin heights (batch, dim, K)
        derivatives: Knot derivatives (batch, dim, K+1)
        bounds: (left, right) bounds for the spline
        inverse: Whether to compute inverse
        
    Returns:
        outputs: Transformed values
        log_det: Log absolute Jacobian determinant
    """
    left, right = bounds
    bottom, top = bounds
    
    num_bins = widths.shape[-1]
    
    # Normalize to ensure they sum to interval width
    widths = F.softmax(widths, dim=-1) * (right - left)
    heights = F.softmax(heights, dim=-1) * (top - bottom)
    derivatives = F.softplus(derivatives) + 1e-4
    
    # Compute cumulative positions (knots)
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, (1, 0), value=0.0) + left
    
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, (1, 0), value=0.0) + bottom
    
    # Handle values outside the spline range (identity outside)
    inside_mask = (inputs >= left) & (inputs <= right)
    
    # Find bin indices
    if inverse:
        bin_idx = torch.searchsorted(cumheights[..., 1:-1], inputs.unsqueeze(-1))
    else:
        bin_idx = torch.searchsorted(cumwidths[..., 1:-1], inputs.unsqueeze(-1))
    bin_idx = bin_idx.squeeze(-1)
    
    # Gather parameters for each input's bin
    widths_k = widths.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
    heights_k = heights.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
    
    cumwidths_k = cumwidths.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
    cumheights_k = cumheights.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
    
    delta_k = heights_k / widths_k
    
    dk = derivatives.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
    dk1 = derivatives.gather(-1, (bin_idx + 1).unsqueeze(-1)).squeeze(-1)
    
    if inverse:
        # Solve quadratic for xi given y
        y = inputs
        y_rel = y - cumheights_k
        
        a = heights_k * (delta_k - dk) + y_rel * (dk1 + dk - 2*delta_k)
        b = heights_k * dk - y_rel * (dk1 + dk - 2*delta_k)
        c = -delta_k * y_rel
        
        discriminant = b**2 - 4*a*c
        xi = (-b + torch.sqrt(discriminant.clamp(min=1e-8))) / (2*a + 1e-8)
        xi = xi.clamp(0, 1)
        
        outputs = xi * widths_k + cumwidths_k
        
        # Derivative for log-det
        xi_1mxi = xi * (1 - xi)
        denom = delta_k + (dk1 + dk - 2*delta_k) * xi_1mxi
        deriv = delta_k**2 * (dk1*xi**2 + 2*delta_k*xi_1mxi + dk*(1-xi)**2) / (denom**2 + 1e-8)
        
        log_det = -torch.log(deriv + 1e-8)
        
    else:
        # Forward: compute y from x
        xi = (inputs - cumwidths_k) / widths_k
        xi = xi.clamp(0, 1)
        
        xi_1mxi = xi * (1 - xi)
        
        numerator = heights_k * (delta_k * xi**2 + dk * xi_1mxi)
        denominator = delta_k + (dk1 + dk - 2*delta_k) * xi_1mxi
        
        outputs = cumheights_k + numerator / (denominator + 1e-8)
        
        # Derivative for log-det
        deriv = delta_k**2 * (dk1*xi**2 + 2*delta_k*xi_1mxi + dk*(1-xi)**2) / (denominator**2 + 1e-8)
        log_det = torch.log(deriv + 1e-8)
    
    # Identity outside bounds
    outputs = torch.where(inside_mask, outputs, inputs)
    log_det = torch.where(inside_mask, log_det, torch.zeros_like(log_det))
    
    return outputs, log_det
```

### Neural Spline Coupling Layer

```python
class NeuralSplineCouplingLayer(nn.Module):
    """
    Coupling layer using rational quadratic splines.
    """
    
    def __init__(self, dim, mask, num_bins=8, hidden_dims=[256, 256], bounds=(-3, 3)):
        super().__init__()
        
        self.dim = dim
        self.num_bins = num_bins
        self.bounds = bounds
        self.register_buffer('mask', mask.float())
        
        # Dimensions
        self.d_cond = int(mask.sum().item())
        self.d_trans = dim - self.d_cond
        
        # Output: widths, heights (K each), derivatives (K+1) for each transformed dim
        output_dim = self.d_trans * (3 * num_bins + 1)
        
        # Conditioner network
        self.net = self._build_net(self.d_cond, output_dim, hidden_dims)
    
    def _build_net(self, input_dim, output_dim, hidden_dims):
        layers = []
        prev = input_dim
        
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        
        layers.append(nn.Linear(prev, output_dim))
        
        # Initialize for near-identity
        nn.init.zeros_(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)
        
        return nn.Sequential(*layers)
    
    def _get_spline_params(self, x_cond):
        """Get spline parameters from conditioning input."""
        batch_size = x_cond.shape[0]
        
        params = self.net(x_cond)
        params = params.view(batch_size, self.d_trans, -1)
        
        # Split into widths, heights, derivatives
        widths = params[..., :self.num_bins]
        heights = params[..., self.num_bins:2*self.num_bins]
        derivatives = params[..., 2*self.num_bins:]
        
        return widths, heights, derivatives
    
    def forward(self, x):
        """Forward: x -> y"""
        # Split
        x_cond = x[:, self.mask.bool()]
        x_trans = x[:, ~self.mask.bool()]
        
        # Get spline parameters
        widths, heights, derivatives = self._get_spline_params(x_cond)
        
        # Apply spline
        y_trans, log_det = rational_quadratic_spline(
            x_trans, widths, heights, derivatives,
            bounds=self.bounds, inverse=False
        )
        
        # Reconstruct
        y = torch.zeros_like(x)
        y[:, self.mask.bool()] = x_cond
        y[:, ~self.mask.bool()] = y_trans
        
        return y, log_det.sum(dim=-1)
    
    def inverse(self, y):
        """Inverse: y -> x"""
        # Split
        y_cond = y[:, self.mask.bool()]
        y_trans = y[:, ~self.mask.bool()]
        
        # Get spline parameters (same as forward since conditioning unchanged)
        widths, heights, derivatives = self._get_spline_params(y_cond)
        
        # Invert spline
        x_trans, log_det = rational_quadratic_spline(
            y_trans, widths, heights, derivatives,
            bounds=self.bounds, inverse=True
        )
        
        # Reconstruct
        x = torch.zeros_like(y)
        x[:, self.mask.bool()] = y_cond
        x[:, ~self.mask.bool()] = x_trans
        
        return x, log_det.sum(dim=-1)
```

### Complete Neural Spline Flow

```python
class NeuralSplineFlow(nn.Module):
    """
    Complete Neural Spline Flow model.
    """
    
    def __init__(self, dim, n_layers=8, num_bins=8, hidden_dims=[256, 256]):
        super().__init__()
        
        self.dim = dim
        
        self.layers = nn.ModuleList()
        
        for i in range(n_layers):
            # Alternate masks
            mask = torch.zeros(dim)
            if i % 2 == 0:
                mask[:dim // 2] = 1
            else:
                mask[dim // 2:] = 1
            
            self.layers.append(
                NeuralSplineCouplingLayer(dim, mask, num_bins, hidden_dims)
            )
    
    def forward(self, x):
        log_det_total = torch.zeros(x.shape[0], device=x.device)
        
        for layer in self.layers:
            x, log_det = layer(x)
            log_det_total += log_det
        
        return x, log_det_total
    
    def inverse(self, z):
        log_det_total = torch.zeros(z.shape[0], device=z.device)
        
        for layer in reversed(self.layers):
            z, log_det = layer.inverse(z)
            log_det_total += log_det
        
        return z, log_det_total
    
    def log_prob(self, x):
        z, log_det = self.forward(x)
        log_pz = -0.5 * (z ** 2 + np.log(2 * np.pi)).sum(dim=-1)
        return log_pz + log_det
    
    def sample(self, n_samples, device='cpu'):
        z = torch.randn(n_samples, self.dim, device=device)
        x, _ = self.inverse(z)
        return x
```

## Autoregressive Neural Spline Flows

NSF can also be used in autoregressive architectures:

```python
class AutoregressiveNSF(nn.Module):
    """
    Neural Spline Flow with autoregressive structure.
    Uses MADE-style masking for single-pass density evaluation.
    """
    
    def __init__(self, dim, num_bins=8, hidden_dims=[256, 256]):
        super().__init__()
        
        self.dim = dim
        self.num_bins = num_bins
        
        # MADE-style network
        # Output: for each dimension, predict (K widths, K heights, K+1 derivatives)
        output_per_dim = 3 * num_bins + 1
        
        self.made = MADE(dim, hidden_dims, output_per_dim)
    
    def forward(self, x):
        """Forward: data -> latent (parallel)."""
        # Get all spline parameters at once
        params = self.made(x)  # (batch, dim, output_per_dim)
        
        widths = params[..., :self.num_bins]
        heights = params[..., self.num_bins:2*self.num_bins]
        derivatives = params[..., 2*self.num_bins:]
        
        # Apply spline to each dimension
        z, log_det = rational_quadratic_spline(
            x, widths, heights, derivatives, inverse=False
        )
        
        return z, log_det.sum(dim=-1)
    
    def inverse(self, z):
        """Inverse: latent -> data (sequential)."""
        x = torch.zeros_like(z)
        log_det_total = torch.zeros(z.shape[0], device=z.device)
        
        for d in range(self.dim):
            # Get parameters based on x computed so far
            params = self.made(x)
            
            widths_d = params[:, d:d+1, :self.num_bins]
            heights_d = params[:, d:d+1, self.num_bins:2*self.num_bins]
            derivatives_d = params[:, d:d+1, 2*self.num_bins:]
            
            # Invert for dimension d
            x_d, log_det_d = rational_quadratic_spline(
                z[:, d:d+1], widths_d, heights_d, derivatives_d, inverse=True
            )
            
            x[:, d] = x_d.squeeze(-1)
            log_det_total += log_det_d.squeeze(-1)
        
        return x, log_det_total
```

## Comparison with Affine Flows

### Expressiveness per Layer

| Transform | Parameters per dim | Expressiveness |
|-----------|-------------------|----------------|
| Affine | 2 (scale, shift) | Linear |
| Linear Spline | K+1 | Piecewise linear |
| Quadratic Spline | 2K+1 | Piecewise quadratic |
| **RQ Spline** | 3K+1 | Smooth, flexible |

### Empirical Results

On standard density estimation benchmarks:

| Model | POWER | GAS | HEPMASS | MINIBOONE |
|-------|-------|-----|---------|-----------|
| MAF | 0.14 | 8.47 | -15.09 | -10.08 |
| RealNVP | 0.17 | 8.33 | -18.71 | -13.84 |
| **NSF** | **0.06** | **11.96** | **-14.01** | **-9.22** |

(Values are negative log-likelihood in nats; lower is better)

## Practical Considerations

### Number of Bins

- More bins = more flexible but more parameters
- Typical: 4-16 bins
- Start with 8, tune if needed

### Bounds

- Spline is identity outside bounds
- Set bounds to cover ~3σ of data
- Can learn bounds or use fixed [-3, 3] after standardization

### Numerical Stability

```python
# Add small epsilon to prevent division by zero
denominator = delta + (dk1 + dk - 2*delta) * xi_1mxi + 1e-8

# Clamp xi to [0, 1]
xi = xi.clamp(0, 1)

# Use log-space for derivatives when possible
log_deriv = 2*torch.log(delta) + torch.log(numerator) - 2*torch.log(denominator)
```

## Summary

Neural Spline Flows provide:

1. **Greater expressiveness** through monotonic splines
2. **Closed-form inverse** via rational quadratic formulation
3. **Tractable Jacobian** with analytic derivatives
4. **State-of-the-art density estimation** on tabular data
5. **Drop-in replacement** for affine coupling in existing architectures

The rational quadratic spline strikes an excellent balance between flexibility and computational tractability, making NSF the go-to choice for many density estimation applications.

## References

1. Durkan, C., et al. (2019). Neural Spline Flows. *NeurIPS*.
2. Gregory, J. A., & Delbourgo, R. (1982). Piecewise Rational Quadratic Interpolation to Monotonic Data. *IMA Journal of Numerical Analysis*.
3. Müller, T., et al. (2019). Neural Importance Sampling. *ACM TOG*.
4. Papamakarios, G., et al. (2021). Normalizing Flows for Probabilistic Modeling and Inference. *JMLR*.
