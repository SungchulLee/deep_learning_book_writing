# Base Distribution Choice

## Introduction

The **base distribution** (also called the prior or latent distribution) is where normalizing flows begin their transformation. While the standard Gaussian is most common, the choice of base distribution can significantly impact model performance and expressiveness.

## Standard Gaussian Base

### Definition

The standard multivariate Gaussian:

$$p_Z(z) = \mathcal{N}(z; 0, I) = \frac{1}{(2\pi)^{d/2}} \exp\left(-\frac{1}{2} z^T z\right)$$

### Log-Probability

$$\log p_Z(z) = -\frac{d}{2} \log(2\pi) - \frac{1}{2} \|z\|^2 = -\frac{1}{2} \sum_{i=1}^d \left(z_i^2 + \log(2\pi)\right)$$

### Implementation

```python
class GaussianBase:
    """Standard Gaussian base distribution."""
    
    def __init__(self, dim: int):
        self.dim = dim
        self.log_2pi = np.log(2 * np.pi)
    
    def sample(self, n_samples: int, device: str = 'cpu') -> torch.Tensor:
        """Sample from N(0, I)."""
        return torch.randn(n_samples, self.dim, device=device)
    
    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """Compute log p(z)."""
        return -0.5 * (z ** 2 + self.log_2pi).sum(dim=-1)
```

### Why Gaussian?

1. **Simple**: Easy to sample and compute log-probabilities
2. **Well-understood**: Extensive mathematical properties
3. **Unbounded**: Supports entire $\mathbb{R}^d$
4. **Reparameterizable**: Enables gradient-based training

## Alternative Base Distributions

### Uniform Distribution

For data bounded to a specific range:

$$p_Z(z) = \prod_{i=1}^d \mathbf{1}_{[a_i, b_i]}(z_i) / (b_i - a_i)$$

```python
class UniformBase:
    """Uniform base distribution on [low, high]^d."""
    
    def __init__(self, dim: int, low: float = 0.0, high: float = 1.0):
        self.dim = dim
        self.low = low
        self.high = high
        self.log_prob_const = -dim * np.log(high - low)
    
    def sample(self, n_samples: int, device: str = 'cpu') -> torch.Tensor:
        """Sample uniformly."""
        return torch.rand(n_samples, self.dim, device=device) * (self.high - self.low) + self.low
    
    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """Compute log p(z)."""
        # Check if in bounds
        in_bounds = ((z >= self.low) & (z <= self.high)).all(dim=-1)
        log_p = torch.where(
            in_bounds,
            torch.full_like(z[:, 0], self.log_prob_const),
            torch.full_like(z[:, 0], float('-inf'))
        )
        return log_p
```

**Use case**: When data is naturally bounded (e.g., probabilities, normalized features).

### Student-t Distribution

Heavier tails than Gaussian, robust to outliers:

$$p_Z(z) = \frac{\Gamma\left(\frac{\nu + d}{2}\right)}{\Gamma\left(\frac{\nu}{2}\right) (\nu\pi)^{d/2}} \left(1 + \frac{\|z\|^2}{\nu}\right)^{-\frac{\nu+d}{2}}$$

```python
class StudentTBase:
    """Student-t base distribution."""
    
    def __init__(self, dim: int, df: float = 4.0):
        """
        Args:
            dim: Dimensionality
            df: Degrees of freedom (lower = heavier tails)
        """
        self.dim = dim
        self.df = df
        self.distribution = torch.distributions.StudentT(df=df)
    
    def sample(self, n_samples: int, device: str = 'cpu') -> torch.Tensor:
        """Sample from t-distribution."""
        return self.distribution.sample((n_samples, self.dim)).to(device)
    
    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """Compute log p(z)."""
        # Independent t-distribution per dimension
        return self.distribution.log_prob(z).sum(dim=-1)
```

**Use case**: Financial data with heavy tails, robust modeling.

### Mixture of Gaussians

For multimodal latent spaces:

$$p_Z(z) = \sum_{k=1}^K \pi_k \mathcal{N}(z; \mu_k, \Sigma_k)$$

```python
class GaussianMixtureBase:
    """Mixture of Gaussians base distribution."""
    
    def __init__(self, dim: int, n_components: int = 8):
        self.dim = dim
        self.n_components = n_components
        
        # Mixture weights (uniform)
        self.weights = torch.ones(n_components) / n_components
        
        # Component means (arranged in a circle for 2D)
        if dim == 2:
            angles = torch.linspace(0, 2 * np.pi, n_components + 1)[:-1]
            self.means = torch.stack([
                2 * torch.cos(angles),
                2 * torch.sin(angles)
            ], dim=1)
        else:
            # Random means for higher dimensions
            self.means = torch.randn(n_components, dim) * 2
        
        # Component standard deviations
        self.log_stds = torch.zeros(n_components, dim)
    
    def sample(self, n_samples: int, device: str = 'cpu') -> torch.Tensor:
        """Sample from mixture."""
        # Sample component indices
        indices = torch.multinomial(
            self.weights.to(device), 
            n_samples, 
            replacement=True
        )
        
        # Sample from selected components
        means = self.means[indices].to(device)
        stds = self.log_stds[indices].exp().to(device)
        
        noise = torch.randn(n_samples, self.dim, device=device)
        return means + stds * noise
    
    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """Compute log p(z) using log-sum-exp."""
        batch_size = z.shape[0]
        device = z.device
        
        # Compute log prob for each component
        log_probs = []
        for k in range(self.n_components):
            mean = self.means[k].to(device)
            log_std = self.log_stds[k].to(device)
            std = log_std.exp()
            
            # Log probability of Gaussian
            log_p = -0.5 * (
                ((z - mean) / std) ** 2 + 
                2 * log_std + 
                np.log(2 * np.pi)
            ).sum(dim=-1)
            
            # Add log weight
            log_p = log_p + np.log(self.weights[k].item())
            log_probs.append(log_p)
        
        # Log-sum-exp over components
        log_probs = torch.stack(log_probs, dim=1)
        return torch.logsumexp(log_probs, dim=1)
```

**Use case**: Data with multiple distinct modes.

## Learned Base Distributions

### Trainable Gaussian Parameters

Learn mean and variance of base distribution:

```python
class LearnedGaussianBase(nn.Module):
    """Gaussian with learned mean and diagonal covariance."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        # Learnable parameters
        self.mean = nn.Parameter(torch.zeros(dim))
        self.log_std = nn.Parameter(torch.zeros(dim))
    
    def sample(self, n_samples: int, device: str = 'cpu') -> torch.Tensor:
        """Sample from learned Gaussian."""
        std = self.log_std.exp()
        noise = torch.randn(n_samples, self.dim, device=device)
        return self.mean + std * noise
    
    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """Compute log p(z)."""
        std = self.log_std.exp()
        
        log_p = -0.5 * (
            ((z - self.mean) / std) ** 2 + 
            2 * self.log_std + 
            np.log(2 * np.pi)
        )
        return log_p.sum(dim=-1)
```

### Flow-Based Base (Hierarchical)

Use another flow as the base distribution:

```python
class FlowBase(nn.Module):
    """Use a simpler flow as base distribution."""
    
    def __init__(self, dim: int, n_layers: int = 2):
        super().__init__()
        self.dim = dim
        
        # Standard Gaussian as ultimate base
        self.gaussian = GaussianBase(dim)
        
        # Simple flow to add flexibility
        self.base_flow = build_simple_flow(dim, n_layers)
    
    def sample(self, n_samples: int, device: str = 'cpu') -> torch.Tensor:
        """Sample through base flow."""
        z0 = self.gaussian.sample(n_samples, device)
        z, _ = self.base_flow.forward(z0)
        return z
    
    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """Compute log p(z) through inverse."""
        z0, log_det = self.base_flow.inverse(z)
        log_p0 = self.gaussian.log_prob(z0)
        return log_p0 + log_det
```

## Choosing the Right Base Distribution

### Decision Framework

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Characteristics                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Q: Is data bounded?                                         │
│    YES → Consider Uniform or transformed Gaussian           │
│    NO  → Continue                                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Q: Heavy tails expected?                                    │
│    YES → Student-t or Gaussian Mixture                      │
│    NO  → Continue                                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Q: Multiple distinct modes?                                 │
│    YES → Gaussian Mixture                                   │
│    NO  → Standard Gaussian (default)                        │
└─────────────────────────────────────────────────────────────┘
```

### Practical Guidelines

| Data Type | Recommended Base |
|-----------|------------------|
| Images | Standard Gaussian |
| Financial returns | Student-t (df=4-8) |
| Bounded data | Logit-transformed Gaussian or Uniform |
| Multimodal | Mixture of Gaussians |
| Unknown | Start with Gaussian, adjust if needed |

## Impact on Training

### Base Distribution Mismatch

If the flow struggles to transform the base to the data distribution:

**Symptoms**:
- Poor sample quality
- High training loss plateau
- Latent codes don't match base distribution

**Solutions**:
1. Use more flow layers
2. Try a different base distribution
3. Use a learned base distribution

### Latent Space Regularization

The choice of base affects the latent space structure:

```python
def analyze_base_fit(
    flow_model: nn.Module,
    base_dist,
    test_data: torch.Tensor
):
    """Analyze how well latents match base distribution."""
    
    # Encode data to latent space
    z, _ = flow_model.inverse(test_data)
    
    # Compute statistics
    z_mean = z.mean(dim=0)
    z_std = z.std(dim=0)
    
    # Sample from base
    z_base = base_dist.sample(len(test_data))
    
    # Compare distributions
    print(f"Encoded latents:")
    print(f"  Mean: {z_mean.mean():.3f} (expected: ~0)")
    print(f"  Std: {z_std.mean():.3f} (expected: ~1)")
    
    # KL divergence (approximate)
    kl = 0.5 * (z_std**2 + z_mean**2 - 1 - 2*torch.log(z_std)).sum()
    print(f"  Approximate KL: {kl.item():.3f}")
```

## Advanced: Conditional Base Distributions

For conditional generation, the base can depend on conditioning:

```python
class ConditionalGaussianBase(nn.Module):
    """Gaussian with parameters conditioned on input."""
    
    def __init__(self, dim: int, context_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.dim = dim
        
        # Networks to predict mean and log-std
        self.mean_net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )
        
        self.log_std_net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )
    
    def sample(self, context: torch.Tensor) -> torch.Tensor:
        """Sample conditioned on context."""
        mean = self.mean_net(context)
        log_std = self.log_std_net(context)
        std = log_std.exp()
        
        noise = torch.randn_like(mean)
        return mean + std * noise
    
    def log_prob(self, z: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Compute conditional log p(z|context)."""
        mean = self.mean_net(context)
        log_std = self.log_std_net(context)
        std = log_std.exp()
        
        log_p = -0.5 * (
            ((z - mean) / std) ** 2 + 
            2 * log_std + 
            np.log(2 * np.pi)
        )
        return log_p.sum(dim=-1)
```

## Summary

### Key Takeaways

1. **Standard Gaussian** is the default choice for most applications
2. **Student-t** is better for heavy-tailed data like financial returns
3. **Mixture of Gaussians** helps with multimodal data
4. **Learned base** distributions add flexibility with minimal overhead
5. Always verify latent codes match the base distribution after training

### Quick Reference

```python
# Standard choice (most applications)
base = GaussianBase(dim=784)

# Financial data with heavy tails
base = StudentTBase(dim=100, df=5)

# Multimodal data
base = GaussianMixtureBase(dim=2, n_components=8)

# Maximum flexibility
base = LearnedGaussianBase(dim=784)
```

## References

1. Papamakarios, G., et al. (2021). Normalizing Flows for Probabilistic Modeling and Inference. *JMLR*.
2. Durkan, C., et al. (2019). Neural Spline Flows. *NeurIPS*.
3. Jaini, P., et al. (2020). Tails of Lipschitz Triangular Flows. *ICML*.
