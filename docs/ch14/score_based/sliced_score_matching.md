# Sliced Score Matching

## Introduction

**Sliced Score Matching (SSM)** is an alternative to explicit score matching that avoids computing full Jacobians by using random projections. It provides a scalable, **unbiased** way to train score networks when denoising score matching's noise perturbation is not acceptable.

!!! info "Core Idea"
    Instead of computing the expensive full Jacobian trace ($D$ backward passes), project both the model and data scores onto random directions and match these 1D projections using Hutchinson's trace estimator.

## Learning Objectives

By the end of this section, you will be able to:

1. Understand the computational bottleneck of explicit score matching
2. Derive sliced score matching using Hutchinson's trace estimator
3. Implement SSM efficiently using vector-Jacobian products (VJP)
4. Compare SSM with ESM and DSM, understanding when to use each
5. Choose appropriate projection distributions and number of projections

## Prerequisites

- Explicit score matching objective
- Vector calculus (Jacobians, trace)
- PyTorch autograd mechanics

---

## 1. The Trace Estimation Problem

### 1.1 ESM Requires Expensive Jacobian Computation

The explicit score matching objective:

$$
\mathcal{L}_{\text{ESM}}(\theta) = \mathbb{E}_{p_{\text{data}}}\left[\frac{1}{2}\|\mathbf{s}_\theta(\mathbf{x})\|^2 + \text{tr}(\nabla_{\mathbf{x}} \mathbf{s}_\theta(\mathbf{x}))\right]
$$

The trace term requires computing diagonal elements of the Jacobian:

$$
\text{tr}(\nabla_{\mathbf{x}} \mathbf{s}_\theta) = \sum_{i=1}^D \frac{\partial s_{\theta,i}}{\partial x_i}
$$

This requires **$D$ backward passes**—prohibitively expensive for high-dimensional data!

| Data Type | Dimension $D$ | ESM Backward Passes |
|-----------|---------------|---------------------|
| 2D toy | 2 | 2 |
| Tabular | 100 | 100 |
| MNIST | 784 | 784 |
| CIFAR-10 | 3,072 | 3,072 |

### 1.2 The Solution: Random Projection

Instead of computing all $D$ diagonal elements, **estimate the trace** using random projections. This is the key insight of SSM.

---

## 2. Hutchinson's Trace Estimator

### 2.1 The Theorem

For any square matrix $\mathbf{A}$ and random vector $\mathbf{v}$ with $\mathbb{E}[\mathbf{v}\mathbf{v}^\top] = \mathbf{I}$:

$$
\boxed{\text{tr}(\mathbf{A}) = \mathbb{E}_{\mathbf{v}}[\mathbf{v}^\top \mathbf{A} \mathbf{v}]}
$$

**Proof:**
$$
\mathbb{E}[\mathbf{v}^\top \mathbf{A} \mathbf{v}] = \mathbb{E}\left[\sum_{i,j} v_i A_{ij} v_j\right] = \sum_{i,j} A_{ij} \mathbb{E}[v_i v_j] = \sum_{i,j} A_{ij} \delta_{ij} = \sum_i A_{ii} = \text{tr}(\mathbf{A})
$$

### 2.2 Valid Projection Distributions

Any distribution satisfying $\mathbb{E}[\mathbf{v}\mathbf{v}^\top] = \mathbf{I}$:

| Distribution | Formula | Variance | Recommendation |
|--------------|---------|----------|----------------|
| **Rademacher** | $v_i \in \{-1, +1\}$ uniformly | Lower | ✅ Preferred |
| **Gaussian** | $\mathbf{v} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ | Higher | Good default |
| **Uniform on sphere** | $\mathbf{v} \sim \text{Uniform}(\mathbb{S}^{D-1})$ | Lowest | More complex |

Rademacher vectors typically give lower variance estimates and are computationally cheaper.

---

## 3. The SSM Objective

### 3.1 Derivation

Applying Hutchinson's estimator to the ESM trace term:

$$
\text{tr}(\nabla_{\mathbf{x}} \mathbf{s}_\theta) \approx \mathbf{v}^\top \nabla_{\mathbf{x}} \mathbf{s}_\theta(\mathbf{x}) \mathbf{v}
$$

Substituting into ESM:

$$
\boxed{\mathcal{L}_{\text{SSM}}(\theta) = \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}} \mathbb{E}_{\mathbf{v}}\left[\frac{1}{2}(\mathbf{v}^\top \mathbf{s}_\theta(\mathbf{x}))^2 + \mathbf{v}^\top \nabla_{\mathbf{x}} \mathbf{s}_\theta(\mathbf{x}) \, \mathbf{v}\right]}
$$

### 3.2 Efficient Computation via VJP

The key computational insight:

$$
\mathbf{v}^\top \nabla_{\mathbf{x}} \mathbf{s}_\theta(\mathbf{x}) \, \mathbf{v} = \mathbf{v}^\top \nabla_{\mathbf{x}}(\mathbf{v}^\top \mathbf{s}_\theta(\mathbf{x}))
$$

The right-hand side is a **vector-Jacobian product (VJP)**:
1. Compute the scalar $\mathbf{v}^\top \mathbf{s}_\theta(\mathbf{x})$
2. Take its gradient with respect to $\mathbf{x}$
3. Dot product with $\mathbf{v}$

This requires only **one backward pass** per projection vector!

### 3.3 Multiple Projections

Using $M$ projection vectors reduces variance:

$$
\mathcal{L}_{\text{SSM}}(\theta) \approx \frac{1}{M} \sum_{m=1}^M \left[\frac{1}{2}(\mathbf{v}_m^\top \mathbf{s}_\theta)^2 + \mathbf{v}_m^\top \nabla_{\mathbf{x}} \mathbf{s}_\theta \, \mathbf{v}_m\right]
$$

| Projections $M$ | Variance | Backward Passes | Typical Use |
|-----------------|----------|-----------------|-------------|
| 1 | High | 1 | Fast training |
| 2-4 | Medium | 2-4 | Good balance |
| 8-16 | Low | 8-16 | High accuracy |

---

## 4. PyTorch Implementation

### 4.1 Basic SSM Loss

```python
import torch
import torch.nn as nn
from typing import Literal

def sample_rademacher(shape: tuple, device: torch.device) -> torch.Tensor:
    """Sample Rademacher random vectors (+1 or -1 with equal probability)."""
    return torch.randint(0, 2, shape, device=device).float() * 2 - 1


def ssm_loss(
    score_model: nn.Module,
    samples: torch.Tensor,
    n_projections: int = 1,
    projection_type: Literal['rademacher', 'gaussian'] = 'rademacher'
) -> torch.Tensor:
    """
    Sliced Score Matching loss.
    
    L_SSM = E_x,v [(v·s(x))²/2 + v·∇_x(v·s(x))]
    
    Args:
        score_model: Score network s_θ: R^D → R^D
        samples: Data samples, shape (N, D)
        n_projections: Number of random projections M
        projection_type: 'rademacher' (recommended) or 'gaussian'
    
    Returns:
        Scalar loss
    """
    samples = samples.requires_grad_(True)
    N, D = samples.shape
    device = samples.device
    
    # Compute scores once
    scores = score_model(samples)  # (N, D)
    
    total_loss = 0.0
    
    for _ in range(n_projections):
        # Sample random projection vector
        if projection_type == 'rademacher':
            v = sample_rademacher((N, D), device)
        else:
            v = torch.randn(N, D, device=device)
        
        # Term 1: (v·s(x))² / 2
        score_proj = torch.sum(v * scores, dim=1)  # (N,)
        squared_term = 0.5 * score_proj ** 2
        
        # Term 2: v·∇_x(v·s(x)) via VJP
        # Gradient of scalar (v·s) w.r.t. x gives v·∇s
        vjp = torch.autograd.grad(
            outputs=score_proj.sum(),
            inputs=samples,
            create_graph=True,
            retain_graph=True
        )[0]  # (N, D)
        
        # v·(v·∇s) = v·∇_x(v·s)
        trace_term = torch.sum(vjp * v, dim=1)  # (N,)
        
        total_loss += torch.mean(squared_term + trace_term)
    
    return total_loss / n_projections
```

### 4.2 Memory-Efficient Version

```python
def ssm_loss_memory_efficient(
    score_model: nn.Module,
    samples: torch.Tensor,
    n_projections: int = 1
) -> torch.Tensor:
    """
    Memory-efficient SSM that recomputes scores for each projection.
    Use when GPU memory is limited.
    """
    samples = samples.requires_grad_(True)
    N, D = samples.shape
    device = samples.device
    
    total_loss = 0.0
    
    for _ in range(n_projections):
        # Recompute scores (trades compute for memory)
        scores = score_model(samples)
        
        # Rademacher projection
        v = sample_rademacher((N, D), device)
        
        # Projected score
        score_proj = torch.sum(v * scores, dim=1)
        
        # SSM terms
        squared_term = 0.5 * score_proj ** 2
        
        vjp = torch.autograd.grad(
            score_proj.sum(), samples,
            create_graph=True
        )[0]
        trace_term = torch.sum(vjp * v, dim=1)
        
        total_loss += torch.mean(squared_term + trace_term)
        
        # Clear intermediate tensors
        del scores, score_proj, vjp
    
    return total_loss / n_projections
```

### 4.3 SSM Trainer Class

```python
class SlicedScoreMatchingTrainer:
    """Complete trainer for sliced score matching."""
    
    def __init__(
        self,
        score_net: nn.Module,
        lr: float = 1e-3,
        n_projections: int = 1,
        projection_type: str = 'rademacher'
    ):
        self.score_net = score_net
        self.optimizer = torch.optim.Adam(score_net.parameters(), lr=lr)
        self.n_projections = n_projections
        self.projection_type = projection_type
        
    def train_step(self, x: torch.Tensor) -> dict:
        """Single training step."""
        self.score_net.train()
        self.optimizer.zero_grad()
        
        loss = ssm_loss(
            self.score_net, x, 
            self.n_projections, 
            self.projection_type
        )
        loss.backward()
        self.optimizer.step()
        
        return {'loss': loss.item()}
    
    def train(
        self, 
        data: torch.Tensor, 
        n_epochs: int = 1000,
        batch_size: int = 256
    ) -> list:
        """Full training loop."""
        losses = []
        N = len(data)
        
        for epoch in range(n_epochs):
            idx = torch.randperm(N)[:batch_size]
            batch = data[idx]
            
            metrics = self.train_step(batch)
            losses.append(metrics['loss'])
            
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}: Loss = {metrics['loss']:.6f}")
        
        return losses
```

---

## 5. Comparison with Other Methods

### 5.1 Computational Comparison

| Method | Jacobian Cost | Backward Passes | Random Vectors |
|--------|---------------|-----------------|----------------|
| **ESM** | Full diagonal | $O(D)$ | No |
| **SSM** | Projected | $O(M)$ | Yes ($M$ vectors) |
| **DSM** | None | $O(1)$ | No |

### 5.2 Statistical Properties

| Aspect | ESM | SSM | DSM |
|--------|-----|-----|-----|
| **Bias** | None | None | $O(\sigma^2)$ |
| **Variance** | N/A | Higher (depends on $M$) | Lower |
| **Consistency** | Yes | Yes | Yes (as $\sigma \to 0$) |

### 5.3 When to Use Each Method

| Scenario | Recommended Method |
|----------|-------------------|
| Low-dimensional ($D < 10$) | ESM (exact) |
| Moderate-dimensional ($D \sim 10$-$100$) | SSM or DSM |
| High-dimensional (images) | DSM ✅ |
| Noise perturbation unacceptable | SSM |
| Need exact score matching guarantees | SSM |
| Training efficiency critical | DSM |
| General use | DSM ✅ |

!!! tip "Practical Recommendation"
    **Use DSM for most applications.** SSM is primarily useful when you cannot add noise to data (e.g., discrete data, certain physics applications) or need unbiased estimates.

---

## 6. Variance Reduction Techniques

### 6.1 More Projections

Increasing $M$ reduces variance proportionally to $1/M$:

$$
\text{Var}[\hat{\mathcal{L}}_{\text{SSM}}] \propto \frac{1}{M}
$$

### 6.2 Antithetic Sampling

Use paired projections $(\mathbf{v}, -\mathbf{v})$ to reduce variance:

```python
def ssm_loss_antithetic(score_model, samples, n_projections=1):
    """SSM with antithetic sampling for variance reduction."""
    samples = samples.requires_grad_(True)
    scores = score_model(samples)
    
    total_loss = 0.0
    
    for _ in range(n_projections):
        v = sample_rademacher(samples.shape, samples.device)
        
        # Forward projection
        score_proj_pos = torch.sum(v * scores, dim=1)
        vjp_pos = torch.autograd.grad(score_proj_pos.sum(), samples, 
                                       create_graph=True, retain_graph=True)[0]
        loss_pos = 0.5 * score_proj_pos**2 + torch.sum(vjp_pos * v, dim=1)
        
        # Antithetic projection (-v)
        score_proj_neg = torch.sum(-v * scores, dim=1)
        vjp_neg = torch.autograd.grad(score_proj_neg.sum(), samples,
                                       create_graph=True, retain_graph=True)[0]
        loss_neg = 0.5 * score_proj_neg**2 + torch.sum(vjp_neg * (-v), dim=1)
        
        # Average reduces variance
        total_loss += torch.mean((loss_pos + loss_neg) / 2)
    
    return total_loss / n_projections
```

### 6.3 Control Variates

For very low variance, use control variates based on known score functions (advanced technique).

---

## 7. Example: Training on 2D Data

```python
import matplotlib.pyplot as plt
import numpy as np

# Generate mixture of Gaussians
def sample_mog(n, n_components=4):
    """Sample from mixture of Gaussians."""
    angles = np.linspace(0, 2*np.pi, n_components, endpoint=False)
    centers = 2.0 * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    
    idx = np.random.randint(0, n_components, n)
    samples = centers[idx] + 0.3 * np.random.randn(n, 2)
    return torch.tensor(samples, dtype=torch.float32)

# Simple score network
class ScoreNet(nn.Module):
    def __init__(self, dim=2, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim)
        )
    
    def forward(self, x):
        return self.net(x)

# Train with SSM
data = sample_mog(5000)
model = ScoreNet()
trainer = SlicedScoreMatchingTrainer(model, lr=1e-3, n_projections=4)
losses = trainer.train(data, n_epochs=2000, batch_size=256)

# Visualize learned score field
def plot_scores(model, data, title="Learned Score Field"):
    x = torch.linspace(-4, 4, 20)
    y = torch.linspace(-4, 4, 20)
    X, Y = torch.meshgrid(x, y, indexing='xy')
    grid = torch.stack([X.flatten(), Y.flatten()], dim=1)
    
    with torch.no_grad():
        scores = model(grid)
    
    plt.figure(figsize=(8, 8))
    plt.scatter(data[:, 0], data[:, 1], alpha=0.3, s=1, label='Data')
    plt.quiver(X.numpy(), Y.numpy(), 
               scores[:, 0].reshape(20, 20).numpy(),
               scores[:, 1].reshape(20, 20).numpy(),
               alpha=0.7)
    plt.title(title)
    plt.axis('equal')
    plt.legend()

plot_scores(model, data)
```

---

## 8. Summary

| Aspect | Description |
|--------|-------------|
| **Objective** | $\mathcal{L}_{\text{SSM}} = \mathbb{E}_{\mathbf{x}, \mathbf{v}}[\frac{1}{2}(\mathbf{v}^\top \mathbf{s}_\theta)^2 + \mathbf{v}^\top \nabla_{\mathbf{x}} \mathbf{s}_\theta \, \mathbf{v}]$ |
| **Key trick** | Hutchinson's trace estimator: $\text{tr}(\mathbf{A}) = \mathbb{E}[\mathbf{v}^\top \mathbf{A} \mathbf{v}]$ |
| **Computation** | VJP gives $\mathbf{v}^\top \nabla \mathbf{s} \, \mathbf{v}$ in one backward pass |
| **Complexity** | $O(M)$ backward passes instead of $O(D)$ |
| **Bias** | Unbiased (unlike DSM) |
| **Best for** | When noise perturbation is problematic |

!!! tip "Key Takeaways"
    1. **SSM uses random projections** to estimate the Jacobian trace efficiently
    2. **Hutchinson's estimator** converts $O(D)$ to $O(M)$ backward passes
    3. **VJP trick** computes $\mathbf{v}^\top \nabla \mathbf{s} \, \mathbf{v}$ in one backward pass
    4. **Rademacher vectors** typically give lower variance than Gaussian
    5. **Use DSM for most cases**; SSM when noise perturbation is unacceptable

---

## Exercises

1. **Projection comparison**: Compare Rademacher vs Gaussian projections on a 2D Gaussian mixture. Measure variance of the loss estimate.

2. **Number of projections**: Plot final loss vs $M \in \{1, 2, 4, 8, 16\}$. What's the sweet spot for compute vs accuracy?

3. **SSM vs DSM**: Train both methods on the same 2D data. Compare:
   - Training curves
   - Final score field quality
   - Training time per epoch

4. **Antithetic sampling**: Implement and compare variance with/without antithetic sampling.

5. **High-dimensional scaling**: Test SSM on data with $D \in \{10, 50, 100, 500\}$. At what dimension does it become impractical?

---

## References

1. Song, Y., et al. (2019). "Sliced Score Matching: A Scalable Approach to Density and Score Estimation." *UAI*.
2. Hutchinson, M. F. (1989). "A Stochastic Estimator of the Trace of the Influence Matrix for Laplacian Smoothing Splines." *Communications in Statistics*.
3. Song, Y., & Ermon, S. (2019). "Generative Modeling by Estimating Gradients of the Data Distribution." *NeurIPS*.
