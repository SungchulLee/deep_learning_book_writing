# Radial Flows

## Introduction

**Radial flows**, introduced alongside planar flows by Rezende & Mohamed (2015), transform distributions by expanding or contracting space radially around a reference point. They are particularly effective at modeling distributions with radial symmetry or concentrations around specific locations.

## Definition

A radial flow transforms input $z$ as:

$$f(z) = z + \beta h(\alpha, r)(z - z_0)$$

where:
- $z_0 \in \mathbb{R}^d$ — center point (reference location)
- $\alpha > 0$ — controls the scale of the transformation
- $\beta \in \mathbb{R}$ — controls magnitude and direction
- $r = \|z - z_0\|$ — distance from center
- $h(\alpha, r) = \frac{1}{\alpha + r}$ — radial function

### Geometric Interpretation

1. Points are pushed away from ($\beta > 0$) or pulled toward ($\beta < 0$) the center $z_0$
2. The effect is stronger near the center (where $r$ is small)
3. Far from the center, the transformation approaches identity

### Alternative Formulation

Writing it more explicitly:
$$f(z) = z + \frac{\beta}{\alpha + r}(z - z_0)$$

Or in terms of the unit radial direction $\hat{r} = \frac{z - z_0}{r}$:
$$f(z) = z_0 + \left(1 + \frac{\beta}{\alpha + r}\right)(z - z_0)$$

## Jacobian and Log-Determinant

### Jacobian Derivation

Let $\hat{r} = \frac{z - z_0}{r}$ (unit vector toward $z$).

The Jacobian of a radial flow has the form:

$$\frac{\partial f}{\partial z} = (1 + \beta h) I + \beta h' \cdot (z - z_0)(z - z_0)^T / r$$

where $h = h(\alpha, r)$ and $h' = \frac{\partial h}{\partial r} = -\frac{1}{(\alpha + r)^2}$.

### Log-Determinant Formula

For radial flows, the log-determinant simplifies to:

$$\log \left| \det \frac{\partial f}{\partial z} \right| = (d-1) \log |1 + \beta h| + \log |1 + \beta h + \beta h' r|$$

where:
- First term: contribution from $d-1$ directions perpendicular to radial
- Second term: contribution from radial direction

### Expanded Form

With $h = \frac{1}{\alpha + r}$ and $h' = -\frac{1}{(\alpha + r)^2}$:

$$\log |\det J| = (d-1) \log \left|1 + \frac{\beta}{\alpha + r}\right| + \log \left|1 + \frac{\beta}{\alpha + r} - \frac{\beta r}{(\alpha + r)^2}\right|$$

**Cost**: $O(d)$ — only need to compute $r$ and evaluate simple expressions.

## Invertibility Constraints

### Condition for Invertibility

For the transformation to be invertible, we need:
1. $1 + \beta h > 0$ for all $r \geq 0$
2. $1 + \beta h + \beta h' r > 0$ for all $r \geq 0$

### Sufficient Condition

If $\beta \geq -\alpha$, then the flow is invertible.

**Derivation**:
- When $r = 0$: $1 + \beta h = 1 + \beta/\alpha > 0$ requires $\beta > -\alpha$
- When $r \to \infty$: both terms approach 1 (always positive)

### Enforcing Constraint

```python
def constrain_beta(beta_unconstrained, alpha):
    """Ensure beta >= -alpha for invertibility."""
    # beta = -alpha + softplus(beta_unconstrained)
    return -alpha + torch.nn.functional.softplus(beta_unconstrained)
```

## Implementation

```python
import torch
import torch.nn as nn
import numpy as np

class RadialFlow(nn.Module):
    """
    Radial flow: f(z) = z + beta * h(alpha, r) * (z - z0)
    
    Transforms space by radial expansion/contraction around center z0.
    """
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        # Parameters
        self.z0 = nn.Parameter(torch.zeros(dim))
        self.log_alpha = nn.Parameter(torch.zeros(1))  # alpha > 0
        self.beta_unconstrained = nn.Parameter(torch.zeros(1))
    
    @property
    def alpha(self):
        return torch.exp(self.log_alpha)
    
    @property
    def beta(self):
        """Constrained beta >= -alpha."""
        return -self.alpha + torch.nn.functional.softplus(self.beta_unconstrained)
    
    def forward(self, z: torch.Tensor):
        """
        Forward transformation: z -> x
        
        Args:
            z: Input tensor (batch_size, dim)
        
        Returns:
            x: Transformed tensor
            log_det: Log absolute determinant of Jacobian
        """
        # Compute distance from center
        diff = z - self.z0  # (batch, dim)
        r = torch.norm(diff, dim=1, keepdim=True)  # (batch, 1)
        
        # Radial function h = 1 / (alpha + r)
        h = 1.0 / (self.alpha + r)  # (batch, 1)
        
        # Transform
        x = z + self.beta * h * diff  # (batch, dim)
        
        # Log determinant
        # h' = -1 / (alpha + r)^2
        h_prime = -1.0 / (self.alpha + r) ** 2
        
        # det = (1 + beta*h)^(d-1) * (1 + beta*h + beta*h'*r)
        term1 = 1.0 + self.beta * h  # (batch, 1)
        term2 = 1.0 + self.beta * h + self.beta * h_prime * r  # (batch, 1)
        
        log_det = (self.dim - 1) * torch.log(torch.abs(term1)) + torch.log(torch.abs(term2))
        log_det = log_det.squeeze(1)  # (batch,)
        
        return x, log_det
    
    def inverse(self, x: torch.Tensor, n_iter: int = 100, tol: float = 1e-6):
        """
        Inverse transformation: x -> z
        
        No analytical inverse; use fixed-point iteration.
        
        Args:
            x: Transformed tensor
            n_iter: Maximum iterations
            tol: Convergence tolerance
        
        Returns:
            z: Original tensor
            log_det: Log determinant (negative of forward)
        """
        # Initialize with x
        z = x.clone()
        
        for _ in range(n_iter):
            diff = z - self.z0
            r = torch.norm(diff, dim=1, keepdim=True)
            h = 1.0 / (self.alpha + r)
            
            # Fixed point: z = x - beta * h(z) * (z - z0)
            z_new = x - self.beta * h * diff
            
            # Check convergence
            if (z_new - z).abs().max() < tol:
                break
            z = z_new
        
        # Compute log det at converged z
        _, log_det_forward = self.forward(z)
        log_det = -log_det_forward
        
        return z, log_det


class RadialFlowSequence(nn.Module):
    """Stack of radial flows."""
    
    def __init__(self, dim: int, n_flows: int):
        super().__init__()
        self.flows = nn.ModuleList([RadialFlow(dim) for _ in range(n_flows)])
    
    def forward(self, z):
        log_det_total = torch.zeros(z.shape[0], device=z.device)
        
        for flow in self.flows:
            z, log_det = flow.forward(z)
            log_det_total += log_det
        
        return z, log_det_total
    
    def inverse(self, x, **kwargs):
        log_det_total = torch.zeros(x.shape[0], device=x.device)
        
        for flow in reversed(self.flows):
            x, log_det = flow.inverse(x, **kwargs)
            log_det_total += log_det
        
        return x, log_det_total
```

## Visualization

### 2D Demonstration

```python
import matplotlib.pyplot as plt

def visualize_radial_flow(flow, n_samples=2000):
    """Visualize how radial flow transforms 2D Gaussian."""
    
    # Sample from standard Gaussian
    z = torch.randn(n_samples, 2)
    
    # Transform
    with torch.no_grad():
        x, _ = flow.forward(z)
    
    z_np = z.numpy()
    x_np = x.numpy()
    z0 = flow.z0.detach().numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Before
    axes[0].scatter(z_np[:, 0], z_np[:, 1], alpha=0.3, s=1, c='blue')
    axes[0].scatter(z0[0], z0[1], c='red', s=100, marker='x', label='Center $z_0$')
    axes[0].set_title("Before Transformation")
    axes[0].set_xlim(-4, 4)
    axes[0].set_ylim(-4, 4)
    axes[0].legend()
    axes[0].set_aspect('equal')
    
    # After
    axes[1].scatter(x_np[:, 0], x_np[:, 1], alpha=0.3, s=1, c='blue')
    axes[1].scatter(z0[0], z0[1], c='red', s=100, marker='x', label='Center $z_0$')
    axes[1].set_title(f"After (β={flow.beta.item():.2f}, α={flow.alpha.item():.2f})")
    axes[1].set_xlim(-4, 4)
    axes[1].set_ylim(-4, 4)
    axes[1].legend()
    axes[1].set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('radial_flow_viz.png', dpi=150)
    plt.close()


def visualize_radial_effect():
    """Show expansion vs contraction with different beta values."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    z = torch.randn(1000, 2)
    
    configs = [
        ("Expansion (β > 0)", 1.0),
        ("Identity (β = 0)", 0.0),
        ("Contraction (β < 0)", -0.5),
    ]
    
    for ax, (title, beta_val) in zip(axes, configs):
        flow = RadialFlow(2)
        with torch.no_grad():
            flow.beta_unconstrained.fill_(beta_val)  # Approximate
            x, _ = flow.forward(z)
        
        ax.scatter(x[:, 0].numpy(), x[:, 1].numpy(), alpha=0.3, s=1)
        ax.scatter(0, 0, c='red', s=100, marker='x')
        ax.set_title(title)
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('radial_effects.png', dpi=150)
    plt.close()
```

## Planar vs Radial Flows

### Comparison

| Aspect | Planar Flow | Radial Flow |
|--------|-------------|-------------|
| **Transformation** | Linear along hyperplane | Radial from point |
| **Best for** | Linear/planar features | Radial/concentrated features |
| **Parameters** | $w, u, b$ ($2d + 1$) | $z_0, \alpha, \beta$ ($d + 2$) |
| **Log-det cost** | $O(d)$ | $O(d)$ |
| **Constraint** | $u^T w \geq -1$ | $\beta \geq -\alpha$ |

### When to Use Radial Flows

✅ **Good for**:
- Distributions with radial symmetry
- Multimodal distributions (multiple centers)
- "Blob-like" concentrations

❌ **Less suitable for**:
- Linear/planar structure
- Highly anisotropic distributions
- Complex multi-scale patterns

## Multiple Centers

Stack radial flows with different centers to model complex distributions:

```python
def build_multi_center_radial_flow(dim, n_flows, centers=None):
    """
    Build radial flow with multiple centers.
    
    Args:
        dim: Dimensionality
        n_flows: Number of radial layers
        centers: Optional list of center initializations
    
    Returns:
        RadialFlowSequence
    """
    flows = RadialFlowSequence(dim, n_flows)
    
    if centers is not None:
        with torch.no_grad():
            for i, center in enumerate(centers[:n_flows]):
                flows.flows[i].z0.copy_(torch.tensor(center))
    
    return flows


# Example: Initialize centers at different locations
centers = [
    [2.0, 2.0],
    [-2.0, 2.0],
    [0.0, -2.0],
    [0.0, 0.0],
]
flow = build_multi_center_radial_flow(dim=2, n_flows=4, centers=centers)
```

## Training Example

```python
def train_radial_flow(target_samples, n_flows=8, n_epochs=1000, lr=1e-3):
    """
    Train radial flows to match target distribution.
    
    Args:
        target_samples: Tensor of samples from target
        n_flows: Number of radial flow layers
        n_epochs: Training epochs
        lr: Learning rate
    
    Returns:
        Trained flow, losses
    """
    dim = target_samples.shape[1]
    
    # Model
    flows = RadialFlowSequence(dim, n_flows)
    base_dist = torch.distributions.MultivariateNormal(
        torch.zeros(dim), torch.eye(dim)
    )
    
    optimizer = torch.optim.Adam(flows.parameters(), lr=lr)
    losses = []
    
    for epoch in range(n_epochs):
        # Batch
        idx = torch.randint(0, len(target_samples), (256,))
        x = target_samples[idx]
        
        # Inverse: x -> z
        z, log_det = flows.inverse(x)
        
        # Log probability
        log_pz = base_dist.log_prob(z)
        log_px = log_pz + log_det
        
        # Loss
        loss = -log_px.mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    return flows, losses
```

## Limitations

1. **Single mode per layer**: Each radial flow has one center
2. **Iterative inverse**: No analytical solution
3. **Limited expressiveness**: Radial symmetry assumption
4. **Slow convergence**: May need many iterations for inverse

## Summary

Radial flows:
- **Transformation**: $f(z) = z + \beta h(\alpha, r)(z - z_0)$
- **Effect**: Radial expansion/contraction around $z_0$
- **Log-det**: $O(d)$ closed-form expression
- **Constraint**: $\beta \geq -\alpha$ for invertibility
- **Use case**: Distributions with radial structure, variational inference

Combined with planar flows, radial flows provide complementary modeling capabilities—planar for linear structures, radial for point-centered structures.

## References

1. Rezende, D. J., & Mohamed, S. (2015). Variational Inference with Normalizing Flows. *ICML*.
2. Papamakarios, G., et al. (2021). Normalizing Flows for Probabilistic Modeling and Inference. *JMLR*.
