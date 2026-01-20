# Planar Flows

## Introduction

**Planar flows** were among the first normalizing flow architectures, introduced by Rezende & Mohamed (2015). They apply simple transformations that "bend" space along hyperplanes. While not the most expressive, they provide excellent intuition for how flows transform distributions.

## Definition

A planar flow transforms input $z$ as:

$$f(z) = z + u \cdot h(w^T z + b)$$

where:
- $u \in \mathbb{R}^d$ — direction of transformation
- $w \in \mathbb{R}^d$ — normal to the hyperplane
- $b \in \mathbb{R}$ — bias (offset of hyperplane)
- $h: \mathbb{R} \to \mathbb{R}$ — smooth activation function (e.g., tanh)

### Geometric Interpretation

1. $w^T z + b$ computes signed distance from hyperplane with normal $w$
2. $h(\cdot)$ applies nonlinearity to this distance
3. $u \cdot h(\cdot)$ shifts points in direction $u$ based on their distance from the hyperplane

The transformation "bends" space around the hyperplane defined by $w^T z + b = 0$.

## Jacobian and Log-Determinant

### Jacobian Computation

$$\frac{\partial f}{\partial z} = I + u \cdot h'(w^T z + b) \cdot w^T$$

This is the identity plus a **rank-1 matrix** $u w^T$ scaled by $h'$.

### Matrix Determinant Lemma

For invertible $A$ and vectors $u, v$:
$$\det(A + uv^T) = \det(A)(1 + v^T A^{-1} u)$$

With $A = I$:
$$\det(I + uv^T) = 1 + v^T u$$

### Log-Determinant Formula

$$\det \frac{\partial f}{\partial z} = 1 + h'(w^T z + b) \cdot u^T w$$

Let $\psi(z) = h'(w^T z + b) \cdot w$, then:

$$\log \left| \det \frac{\partial f}{\partial z} \right| = \log |1 + u^T \psi(z)|$$

**Cost**: $O(d)$ — just dot products!

## Invertibility Constraint

### The Problem

For $f$ to be invertible, we need:
$$1 + h'(w^T z + b) \cdot u^T w \neq 0 \quad \forall z$$

If $h' \geq 0$ everywhere (like tanh), we need:
$$u^T w \geq -1$$

### Enforcing Invertibility

**For $h = \tanh$** (where $h'(x) = 1 - \tanh^2(x) \in (0, 1]$):

If $u^T w < -1$, replace $u$ with:
$$\tilde{u} = u + \left( m(w^T u) - w^T u \right) \frac{w}{\|w\|^2}$$

where $m(x) = -1 + \log(1 + e^x)$ ensures $m(x) \geq -1$.

### Implementation

```python
import torch
import torch.nn as nn
import numpy as np

class PlanarFlow(nn.Module):
    """
    Planar flow: f(z) = z + u * h(w^T z + b)
    
    Transforms space by bending along hyperplanes.
    """
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        # Parameters
        self.w = nn.Parameter(torch.randn(dim) * 0.01)
        self.u = nn.Parameter(torch.randn(dim) * 0.01)
        self.b = nn.Parameter(torch.zeros(1))
    
    def _get_u_hat(self):
        """
        Compute constrained u to ensure invertibility.
        
        For tanh: need u^T w >= -1
        """
        wTu = torch.dot(self.w, self.u)
        
        # m(x) = -1 + softplus(x) = -1 + log(1 + exp(x))
        m_wTu = -1 + torch.nn.functional.softplus(wTu)
        
        # Constrained u
        u_hat = self.u + (m_wTu - wTu) * self.w / (torch.dot(self.w, self.w) + 1e-8)
        
        return u_hat
    
    def forward(self, z: torch.Tensor):
        """
        Forward transformation: z -> x
        
        Args:
            z: Input tensor (batch_size, dim)
        
        Returns:
            x: Transformed tensor
            log_det: Log absolute determinant of Jacobian
        """
        u_hat = self._get_u_hat()
        
        # Compute activation
        activation = torch.tanh(z @ self.w + self.b)  # (batch,)
        
        # Transform
        x = z + u_hat.unsqueeze(0) * activation.unsqueeze(1)  # (batch, dim)
        
        # Log determinant
        # psi = h'(w^T z + b) * w
        h_prime = 1 - activation ** 2  # tanh derivative
        psi = h_prime.unsqueeze(1) * self.w.unsqueeze(0)  # (batch, dim)
        
        # det = 1 + u^T psi
        det = 1 + psi @ u_hat  # (batch,)
        log_det = torch.log(torch.abs(det) + 1e-8)
        
        return x, log_det
    
    def inverse(self, x: torch.Tensor, n_iter: int = 100, tol: float = 1e-6):
        """
        Inverse transformation: x -> z
        
        Planar flows don't have analytical inverse.
        Use fixed-point iteration: z = x - u * h(w^T z + b)
        
        Args:
            x: Transformed tensor
            n_iter: Maximum iterations
            tol: Convergence tolerance
        
        Returns:
            z: Original tensor
            log_det: Log determinant (negative of forward)
        """
        u_hat = self._get_u_hat()
        
        # Initialize with x
        z = x.clone()
        
        for _ in range(n_iter):
            activation = torch.tanh(z @ self.w + self.b)
            z_new = x - u_hat.unsqueeze(0) * activation.unsqueeze(1)
            
            # Check convergence
            if (z_new - z).abs().max() < tol:
                break
            z = z_new
        
        # Compute log det at converged z
        _, log_det_forward = self.forward(z)
        log_det = -log_det_forward
        
        return z, log_det
```

## Composing Planar Flows

Single planar flow has limited expressiveness. Stack multiple layers:

```python
class PlanarFlowSequence(nn.Module):
    """Stack of planar flows."""
    
    def __init__(self, dim: int, n_flows: int):
        super().__init__()
        self.flows = nn.ModuleList([PlanarFlow(dim) for _ in range(n_flows)])
    
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

def visualize_planar_flow(flow, n_samples=2000, title="Planar Flow"):
    """Visualize how planar flow transforms 2D Gaussian."""
    
    # Sample from standard Gaussian
    z = torch.randn(n_samples, 2)
    
    # Transform
    with torch.no_grad():
        x, _ = flow.forward(z)
    
    z = z.numpy()
    x = x.numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Before transformation
    axes[0].scatter(z[:, 0], z[:, 1], alpha=0.5, s=1)
    axes[0].set_title("Before (Standard Gaussian)")
    axes[0].set_xlim(-4, 4)
    axes[0].set_ylim(-4, 4)
    axes[0].set_aspect('equal')
    
    # After transformation
    axes[1].scatter(x[:, 0], x[:, 1], alpha=0.5, s=1)
    axes[1].set_title(f"After ({title})")
    axes[1].set_xlim(-4, 4)
    axes[1].set_ylim(-4, 4)
    axes[1].set_aspect('equal')
    
    # Draw hyperplane
    w = flow.w.detach().numpy()
    b = flow.b.detach().item()
    
    # Hyperplane: w^T z + b = 0
    # z_2 = -(w_1 * z_1 + b) / w_2
    if abs(w[1]) > 1e-6:
        z1_line = np.linspace(-4, 4, 100)
        z2_line = -(w[0] * z1_line + b) / w[1]
        mask = (z2_line > -4) & (z2_line < 4)
        axes[0].plot(z1_line[mask], z2_line[mask], 'r-', linewidth=2, label='Hyperplane')
        axes[0].legend()
    
    plt.tight_layout()
    plt.savefig('planar_flow_viz.png', dpi=150)
    plt.close()


def visualize_flow_evolution(flows, n_samples=2000):
    """Visualize progressive transformation through multiple flows."""
    
    n_flows = len(flows)
    fig, axes = plt.subplots(1, n_flows + 1, figsize=(3 * (n_flows + 1), 3))
    
    z = torch.randn(n_samples, 2)
    
    # Initial distribution
    axes[0].scatter(z[:, 0].numpy(), z[:, 1].numpy(), alpha=0.5, s=1)
    axes[0].set_title("z_0 (Gaussian)")
    axes[0].set_xlim(-4, 4)
    axes[0].set_ylim(-4, 4)
    
    # After each flow
    for i, flow in enumerate(flows):
        with torch.no_grad():
            z, _ = flow.forward(z)
        
        axes[i + 1].scatter(z[:, 0].numpy(), z[:, 1].numpy(), alpha=0.5, s=1)
        axes[i + 1].set_title(f"z_{i+1}")
        axes[i + 1].set_xlim(-4, 4)
        axes[i + 1].set_ylim(-4, 4)
    
    plt.tight_layout()
    plt.savefig('flow_evolution.png', dpi=150)
    plt.close()
```

## Training Example

```python
def train_planar_flow_2d(target_samples, n_flows=8, n_epochs=1000, lr=1e-3):
    """
    Train planar flows to match target 2D distribution.
    
    Args:
        target_samples: Tensor of samples from target distribution
        n_flows: Number of planar flow layers
        n_epochs: Training epochs
        lr: Learning rate
    
    Returns:
        Trained flow model
    """
    dim = 2
    
    # Build model
    flows = PlanarFlowSequence(dim, n_flows)
    base_dist = torch.distributions.MultivariateNormal(
        torch.zeros(dim), torch.eye(dim)
    )
    
    optimizer = torch.optim.Adam(flows.parameters(), lr=lr)
    
    losses = []
    
    for epoch in range(n_epochs):
        # Get batch
        idx = torch.randint(0, len(target_samples), (256,))
        x = target_samples[idx]
        
        # Inverse pass: x -> z
        z, log_det = flows.inverse(x)
        
        # Log probability
        log_pz = base_dist.log_prob(z)
        log_px = log_pz + log_det
        
        # Loss: negative log-likelihood
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

### Expressiveness

1. **Rank-1 update**: Each layer only adds a rank-1 perturbation
2. **Many layers needed**: Complex distributions require many stacked flows
3. **No analytical inverse**: Must use iterative methods

### Comparison with Other Simple Flows

| Flow Type | Jacobian | Analytical Inverse | Expressiveness |
|-----------|----------|-------------------|----------------|
| Planar | Rank-1 update | No | Low |
| Radial | Rank-1 update | No | Low |
| Sylvester | Rank-M update | No | Medium |
| Coupling | Triangular | Yes | High |

## Summary

Planar flows:
- **Transformation**: $f(z) = z + u \cdot h(w^T z + b)$
- **Log-det**: $O(d)$ via matrix determinant lemma
- **Constraint**: $u^T w \geq -1$ for invertibility with tanh
- **Inverse**: Iterative (no closed form)
- **Use case**: Educational, simple 2D examples, variational inference

Modern architectures (RealNVP, Glow) have largely superseded planar flows, but they remain valuable for building intuition about how normalizing flows transform distributions.

## References

1. Rezende, D. J., & Mohamed, S. (2015). Variational Inference with Normalizing Flows. *ICML*.
2. Papamakarios, G., et al. (2021). Normalizing Flows for Probabilistic Modeling and Inference. *JMLR*.
