# Masked Autoregressive Flow (MAF)

## Introduction

Masked Autoregressive Flow (MAF) combines the density estimation efficiency of MADE with the expressive power of normalizing flows. By using MADE as a building block, MAF achieves **fast density evaluation** while providing the flexibility of invertible transformations. MAF has become a cornerstone architecture for density estimation tasks where exact likelihood computation is essential.

## From MADE to MAF

### MADE Recap

MADE computes all autoregressive conditionals in one pass:

$$p(x_d | x_{<d}) = \mathcal{N}(x_d | \mu_d(x_{<d}), \sigma_d^2(x_{<d}))$$

This gives a density estimator, but the distribution family is restricted by the choice of conditionals.

### MAF's Innovation

MAF uses MADE not as a density estimator, but as a **transformation layer**:

$$z_d = \frac{x_d - \mu_d(x_{<d})}{\sigma_d(x_{<d})}$$

This transformation:
- Maps complex data $\mathbf{x}$ to simpler latent $\mathbf{z}$
- Is invertible (given $\mathbf{z}$, can recover $\mathbf{x}$)
- Has tractable Jacobian determinant
- Can be stacked for greater expressiveness

## Mathematical Formulation

### Forward Transformation (Encoding)

Given data $\mathbf{x}$, compute latent $\mathbf{z}$:

$$z_d = \frac{x_d - \mu_d(x_1, \ldots, x_{d-1})}{\exp(s_d(x_1, \ldots, x_{d-1}))}$$

where $\mu_d$ and $s_d$ are outputs of a MADE network.

**Key property**: All $z_d$ can be computed in **parallel** because the conditioning is on $\mathbf{x}$, which is fully available.

### Inverse Transformation (Sampling)

Given latent $\mathbf{z}$, recover data $\mathbf{x}$:

$$x_d = \mu_d(x_1, \ldots, x_{d-1}) + \exp(s_d(x_1, \ldots, x_{d-1})) \cdot z_d$$

**Problem**: Computing $x_d$ requires $x_1, \ldots, x_{d-1}$, which must be computed first. This is **sequential**.

### Log-Determinant

The Jacobian is lower triangular with diagonal elements $1/\exp(s_d)$:

$$\log \left| \det \frac{\partial \mathbf{z}}{\partial \mathbf{x}} \right| = -\sum_{d=1}^{D} s_d(x_{<d})$$

For the inverse direction:

$$\log \left| \det \frac{\partial \mathbf{x}}{\partial \mathbf{z}} \right| = \sum_{d=1}^{D} s_d(x_{<d})$$

## Implementation

### Single MAF Layer

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MAFLayer(nn.Module):
    """
    Single Masked Autoregressive Flow layer.
    
    Forward: x -> z (parallel, for density evaluation)
    Inverse: z -> x (sequential, for sampling)
    """
    
    def __init__(self, dim, hidden_dims=[64, 64], reverse=False):
        super().__init__()
        self.dim = dim
        self.reverse = reverse  # Reverse ordering for alternating layers
        
        # Build MADE network
        self.made = self._build_made(dim, hidden_dims)
        
        # Create ordering
        if reverse:
            self.ordering = torch.arange(dim - 1, -1, -1)
        else:
            self.ordering = torch.arange(dim)
    
    def _build_made(self, dim, hidden_dims):
        """Build MADE with proper masks."""
        layers = nn.ModuleList()
        masks = []
        
        # Assign degrees
        m_input = np.arange(1, dim + 1)
        m_hidden = [np.random.randint(1, dim, size=h) for h in hidden_dims]
        m_output = np.concatenate([np.arange(dim), np.arange(dim)])  # mu and s
        
        # Input -> first hidden
        layers.append(nn.Linear(dim, hidden_dims[0]))
        mask = (m_hidden[0][:, None] >= m_input[None, :]).astype(np.float32)
        masks.append(torch.from_numpy(mask))
        
        # Hidden -> hidden
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            mask = (m_hidden[i+1][:, None] >= m_hidden[i][None, :]).astype(np.float32)
            masks.append(torch.from_numpy(mask))
        
        # Last hidden -> output
        layers.append(nn.Linear(hidden_dims[-1], dim * 2))
        mask = (m_output[:, None] > m_hidden[-1][None, :]).astype(np.float32)
        masks.append(torch.from_numpy(mask))
        
        # Register masks
        for i, mask in enumerate(masks):
            self.register_buffer(f'mask_{i}', mask)
        
        return layers
    
    def _forward_made(self, x):
        """Forward pass through MADE."""
        h = x
        for i, layer in enumerate(self.made[:-1]):
            mask = getattr(self, f'mask_{i}')
            h = F.relu(F.linear(h, layer.weight * mask, layer.bias))
        
        # Output layer
        mask = getattr(self, f'mask_{len(self.made)-1}')
        out = F.linear(h, self.made[-1].weight * mask, self.made[-1].bias)
        
        mu, s = out.chunk(2, dim=-1)
        return mu, s
    
    def forward(self, x):
        """
        Forward pass: x -> z (parallel).
        Used for density evaluation.
        
        Returns:
            z: Transformed values
            log_det: Log determinant of Jacobian
        """
        # Apply ordering
        x_ordered = x[:, self.ordering]
        
        # Get transformation parameters
        mu, s = self._forward_made(x_ordered)
        
        # Transform: z = (x - mu) / exp(s)
        z = (x_ordered - mu) * torch.exp(-s)
        
        # Undo ordering
        inv_ordering = torch.argsort(self.ordering)
        z = z[:, inv_ordering]
        
        # Log determinant
        log_det = -s.sum(dim=-1)
        
        return z, log_det
    
    def inverse(self, z):
        """
        Inverse pass: z -> x (sequential).
        Used for sampling.
        
        Returns:
            x: Original values
            log_det: Log determinant of Jacobian
        """
        # Apply ordering to z
        z_ordered = z[:, self.ordering]
        
        batch_size = z.shape[0]
        x = torch.zeros_like(z_ordered)
        log_det = torch.zeros(batch_size, device=z.device)
        
        for d in range(self.dim):
            # Get parameters based on x computed so far
            mu, s = self._forward_made(x)
            
            # Invert: x_d = mu_d + exp(s_d) * z_d
            x[:, d] = mu[:, d] + torch.exp(s[:, d]) * z_ordered[:, d]
            log_det += s[:, d]
        
        # Undo ordering
        inv_ordering = torch.argsort(self.ordering)
        x = x[:, inv_ordering]
        
        return x, log_det
```

### Full MAF Model

```python
class MAF(nn.Module):
    """
    Masked Autoregressive Flow with multiple layers.
    """
    
    def __init__(self, dim, hidden_dims=[64, 64], n_layers=5):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        
        # Alternate orderings between layers
        self.layers = nn.ModuleList([
            MAFLayer(dim, hidden_dims, reverse=(i % 2 == 1))
            for i in range(n_layers)
        ])
        
        # Batch normalization between layers (optional but helpful)
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(dim, affine=False)
            for _ in range(n_layers - 1)
        ])
    
    def forward(self, x):
        """
        Forward pass: x -> z (fast density evaluation).
        """
        log_det_total = torch.zeros(x.shape[0], device=x.device)
        z = x
        
        for i, layer in enumerate(self.layers):
            z, log_det = layer(z)
            log_det_total += log_det
            
            # Batch norm between layers
            if i < len(self.batch_norms):
                z = self.batch_norms[i](z)
        
        return z, log_det_total
    
    def inverse(self, z):
        """
        Inverse pass: z -> x (slow sampling).
        """
        log_det_total = torch.zeros(z.shape[0], device=z.device)
        x = z
        
        for i in reversed(range(self.n_layers)):
            # Inverse batch norm
            if i < len(self.batch_norms):
                # For inference, use running stats
                bn = self.batch_norms[i]
                x = x * torch.sqrt(bn.running_var + bn.eps) + bn.running_mean
            
            x, log_det = self.layers[i].inverse(x)
            log_det_total += log_det
        
        return x, log_det_total
    
    def log_prob(self, x):
        """Compute log p(x)."""
        z, log_det = self.forward(x)
        
        # Log prob under standard Gaussian base
        log_pz = -0.5 * (z ** 2 + np.log(2 * np.pi)).sum(dim=-1)
        
        return log_pz + log_det
    
    def sample(self, n_samples):
        """Generate samples."""
        z = torch.randn(n_samples, self.dim)
        x, _ = self.inverse(z)
        return x
```

## Training MAF

```python
def train_maf(model, data, n_epochs=100, batch_size=256, lr=1e-3):
    """Train MAF via maximum likelihood."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    
    dataset = torch.utils.data.TensorDataset(data)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    
    losses = []
    
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        
        for batch, in loader:
            optimizer.zero_grad()
            
            # Negative log-likelihood
            loss = -model.log_prob(batch).mean()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item() * len(batch)
        
        scheduler.step()
        
        avg_loss = epoch_loss / len(data)
        losses.append(avg_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}, NLL: {avg_loss:.4f}")
    
    return losses
```

## Conditional MAF

MAF can be conditioned on external variables:

```python
class ConditionalMAF(nn.Module):
    """MAF conditioned on external context."""
    
    def __init__(self, dim, context_dim, hidden_dims=[64, 64], n_layers=5):
        super().__init__()
        self.dim = dim
        self.context_dim = context_dim
        
        self.layers = nn.ModuleList([
            ConditionalMAFLayer(dim, context_dim, hidden_dims, reverse=(i % 2 == 1))
            for i in range(n_layers)
        ])
    
    def forward(self, x, context):
        """Forward with context."""
        log_det_total = torch.zeros(x.shape[0], device=x.device)
        z = x
        
        for layer in self.layers:
            z, log_det = layer(z, context)
            log_det_total += log_det
        
        return z, log_det_total
    
    def log_prob(self, x, context):
        """Conditional log probability."""
        z, log_det = self.forward(x, context)
        log_pz = -0.5 * (z ** 2 + np.log(2 * np.pi)).sum(dim=-1)
        return log_pz + log_det


class ConditionalMAFLayer(nn.Module):
    """MAF layer with context conditioning."""
    
    def __init__(self, dim, context_dim, hidden_dims, reverse=False):
        super().__init__()
        self.dim = dim
        
        # Context is concatenated with input
        # But context should influence all outputs (no masking)
        self.context_net = nn.Sequential(
            nn.Linear(context_dim, hidden_dims[0]),
            nn.ReLU()
        )
        
        # MADE operates on x, with context added to hidden layers
        self.made = self._build_made(dim, hidden_dims)
        
        self.ordering = torch.arange(dim - 1, -1, -1) if reverse else torch.arange(dim)
    
    def _build_made(self, dim, hidden_dims):
        # Similar to before, but first hidden layer receives context
        pass
    
    def forward(self, x, context):
        """Forward with context conditioning."""
        # Process context
        context_h = self.context_net(context)
        
        # Get MADE parameters with context
        mu, s = self._forward_made_with_context(x, context_h)
        
        z = (x - mu) * torch.exp(-s)
        log_det = -s.sum(dim=-1)
        
        return z, log_det
```

## Density Estimation Example

```python
import matplotlib.pyplot as plt

# Generate complex 2D data
def make_moons_data(n_samples=5000, noise=0.05):
    from sklearn.datasets import make_moons
    data, _ = make_moons(n_samples=n_samples, noise=noise)
    return torch.from_numpy(data).float()

# Train MAF
data = make_moons_data()
model = MAF(dim=2, hidden_dims=[128, 128], n_layers=8)
losses = train_maf(model, data, n_epochs=200)

# Evaluate
model.eval()
with torch.no_grad():
    # Density on grid
    x = torch.linspace(-2, 3, 100)
    y = torch.linspace(-1.5, 2, 100)
    xx, yy = torch.meshgrid(x, y, indexing='xy')
    grid = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    
    log_prob = model.log_prob(grid)
    prob = torch.exp(log_prob).reshape(100, 100)
    
    # Generate samples
    samples = model.sample(1000)

# Plot
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].scatter(data[:, 0], data[:, 1], alpha=0.3, s=1)
axes[0].set_title('Training Data')

axes[1].contourf(xx, yy, prob, levels=50, cmap='viridis')
axes[1].set_title('Learned Density')

axes[2].scatter(samples[:, 0], samples[:, 1], alpha=0.3, s=1)
axes[2].set_title('Generated Samples')

plt.tight_layout()
plt.show()
```

## Comparison with IAF

| Aspect | MAF | IAF |
|--------|-----|-----|
| **Forward** | x → z (parallel) | z → x (parallel) |
| **Inverse** | z → x (sequential) | x → z (sequential) |
| **Density** | Fast | Slow |
| **Sampling** | Slow | Fast |
| **Best for** | Density estimation | VAE posteriors |
| **Training** | Direct MLE | Requires tricks |

### Duality

MAF and IAF are **exact duals**: the forward pass of one is the inverse of the other. They use the same parameterization but apply it in opposite directions.

## Architectural Improvements

### 1. Permutations

Alternate orderings between layers:

```python
# Even layers: natural order (1, 2, ..., D)
# Odd layers: reverse order (D, D-1, ..., 1)
```

This ensures all dimensions interact.

### 2. Batch Normalization

Add batch norm between layers for stable training:

```python
class MAFWithBatchNorm(nn.Module):
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x, log_det = layer(x)
            if i < len(self.layers) - 1:
                x, bn_log_det = self.batch_norm(x)
                log_det += bn_log_det
```

### 3. Residual Connections

For deep networks:

```python
class ResidualMAFLayer(nn.Module):
    def forward(self, x):
        z_base, log_det = self.maf_layer(x)
        z = x + self.alpha * (z_base - x)  # Interpolate
        return z, log_det
```

### 4. Mixture of Gaussians Base

More flexible base distribution:

```python
class GMMBase(nn.Module):
    def __init__(self, dim, n_components=10):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(n_components) / n_components)
        self.means = nn.Parameter(torch.randn(n_components, dim))
        self.log_stds = nn.Parameter(torch.zeros(n_components, dim))
    
    def log_prob(self, z):
        # Mixture of Gaussians log probability
        pass
```

## MAF for Time Series

MAF can model temporal dependencies:

```python
class TemporalMAF(nn.Module):
    """MAF for time series with temporal context."""
    
    def __init__(self, dim, seq_len, hidden_dims):
        super().__init__()
        
        # Process temporal context
        self.temporal_encoder = nn.LSTM(
            dim, hidden_dims[0], batch_first=True
        )
        
        # MAF for each timestep, conditioned on history
        self.maf = ConditionalMAF(
            dim=dim,
            context_dim=hidden_dims[0],
            hidden_dims=hidden_dims
        )
    
    def log_prob(self, x_seq):
        """
        x_seq: (batch, seq_len, dim)
        """
        batch_size, seq_len, dim = x_seq.shape
        
        total_log_prob = torch.zeros(batch_size, device=x_seq.device)
        
        # Initial hidden state
        h = torch.zeros(1, batch_size, self.temporal_encoder.hidden_size)
        c = torch.zeros_like(h)
        
        for t in range(seq_len):
            x_t = x_seq[:, t, :]
            
            # Context from history
            if t == 0:
                context = torch.zeros(batch_size, h.shape[-1], device=x_seq.device)
            else:
                context = h.squeeze(0)
            
            # Log prob for this timestep
            log_prob_t = self.maf.log_prob(x_t, context)
            total_log_prob += log_prob_t
            
            # Update temporal context
            _, (h, c) = self.temporal_encoder(x_t.unsqueeze(1), (h, c))
        
        return total_log_prob
```

## Summary

Masked Autoregressive Flow:

1. **Uses MADE** for efficient single-pass density evaluation
2. **Fast forward** (density): All transformations computed in parallel
3. **Slow inverse** (sampling): Sequential O(D) computation
4. **Stacked layers** with alternating orderings for expressiveness
5. **Conditional** variants for supervised/semi-supervised learning

MAF excels at density estimation tasks where evaluating $\log p(x)$ is the primary operation. For applications requiring fast sampling, consider IAF or coupling flows instead.

## References

1. Papamakarios, G., et al. (2017). Masked Autoregressive Flow for Density Estimation. *NeurIPS*.
2. Germain, M., et al. (2015). MADE: Masked Autoencoder for Distribution Estimation. *ICML*.
3. Kingma, D. P., et al. (2016). Improving Variational Inference with Inverse Autoregressive Flow. *NeurIPS*.
4. Papamakarios, G., et al. (2021). Normalizing Flows for Probabilistic Modeling and Inference. *JMLR*.
