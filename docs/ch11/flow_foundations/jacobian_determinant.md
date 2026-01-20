# Jacobian Determinant

## Introduction

The **Jacobian determinant** is the key computational challenge in normalizing flows. Computing a general $d \times d$ determinant costs $O(d^3)$, which is prohibitive for high-dimensional data. This document explains the Jacobian, why it matters, and how flow architectures achieve efficient computation.

## The Jacobian Matrix

### Definition

For a function $f: \mathbb{R}^d \to \mathbb{R}^d$, the Jacobian matrix at point $z$ is:

$$J_f(z) = \frac{\partial f}{\partial z} = \begin{pmatrix}
\frac{\partial f_1}{\partial z_1} & \frac{\partial f_1}{\partial z_2} & \cdots & \frac{\partial f_1}{\partial z_d} \\
\frac{\partial f_2}{\partial z_1} & \frac{\partial f_2}{\partial z_2} & \cdots & \frac{\partial f_2}{\partial z_d} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial f_d}{\partial z_1} & \frac{\partial f_d}{\partial z_2} & \cdots & \frac{\partial f_d}{\partial z_d}
\end{pmatrix}$$

Element $(i, j)$ is $\frac{\partial f_i}{\partial z_j}$: how the $i$-th output changes with the $j$-th input.

### Example: 2D Linear Transformation

$$f(z) = \begin{pmatrix} 2 & 1 \\ 0 & 3 \end{pmatrix} \begin{pmatrix} z_1 \\ z_2 \end{pmatrix} = \begin{pmatrix} 2z_1 + z_2 \\ 3z_2 \end{pmatrix}$$

Jacobian:
$$J_f = \begin{pmatrix} 2 & 1 \\ 0 & 3 \end{pmatrix}$$

Determinant: $\det(J_f) = 2 \cdot 3 - 1 \cdot 0 = 6$

This means the transformation expands area by factor 6.

## Geometric Interpretation

### Determinant as Volume Scaling

The absolute value of the Jacobian determinant measures how much a transformation scales infinitesimal volume:

$$\text{Volume}(f(R)) = |\det J_f| \cdot \text{Volume}(R)$$

for a small region $R$ around point $z$.

### Sign of Determinant

- $\det J > 0$: Orientation preserving
- $\det J < 0$: Orientation reversing (reflection)
- $\det J = 0$: Transformation is singular (not invertible)

For flows, we need $\det J \neq 0$ everywhere to ensure invertibility.

## Computational Complexity

### The Challenge

For a general $d \times d$ matrix:
- **Storage**: $O(d^2)$ elements
- **Determinant**: $O(d^3)$ via LU decomposition

For images (e.g., 64×64×3 = 12,288 dimensions):
- Jacobian: 150 million elements
- Determinant: $10^{12}$ operations per sample

This is **completely infeasible** for direct computation.

### Solution: Structured Jacobians

Flow architectures are designed to have Jacobians with special structure:

| Structure | Determinant Cost | Example |
|-----------|-----------------|---------|
| Diagonal | $O(d)$ | Element-wise transforms |
| Triangular | $O(d)$ | Autoregressive, coupling |
| Block diagonal | $O(\sum d_i^3)$ | Multi-scale |
| Low-rank update | $O(d \cdot r^2)$ | Planar, Sylvester |

## Triangular Jacobians

### Key Property

For a triangular matrix, the determinant is the **product of diagonal elements**:

$$\det \begin{pmatrix}
a_{11} & 0 & \cdots & 0 \\
a_{21} & a_{22} & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
a_{d1} & a_{d2} & \cdots & a_{dd}
\end{pmatrix} = \prod_{i=1}^{d} a_{ii}$$

In log-space:
$$\log |\det J| = \sum_{i=1}^{d} \log |a_{ii}|$$

This is $O(d)$ instead of $O(d^3)$!

### Achieving Triangular Structure

**Autoregressive transformations**: Output $i$ depends only on inputs $1, \ldots, i$

$$x_i = f_i(z_1, z_2, \ldots, z_i)$$

Jacobian:
$$J = \begin{pmatrix}
\frac{\partial x_1}{\partial z_1} & 0 & \cdots & 0 \\
\frac{\partial x_2}{\partial z_1} & \frac{\partial x_2}{\partial z_2} & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial x_d}{\partial z_1} & \frac{\partial x_d}{\partial z_2} & \cdots & \frac{\partial x_d}{\partial z_d}
\end{pmatrix}$$

Lower triangular! Determinant = $\prod_i \frac{\partial x_i}{\partial z_i}$

**Coupling layers**: Split input, transform one part based on the other

$$y_A = z_A, \quad y_B = g(z_B; z_A)$$

Jacobian has block structure leading to efficient determinant.

## Diagonal Jacobians

### Element-wise Transformations

When $f$ applies the same scalar function to each dimension independently:

$$f(z)_i = h(z_i)$$

Jacobian is diagonal:
$$J = \text{diag}(h'(z_1), h'(z_2), \ldots, h'(z_d))$$

Determinant:
$$\det J = \prod_{i=1}^{d} h'(z_i)$$

$$\log |\det J| = \sum_{i=1}^{d} \log |h'(z_i)|$$

### Common Element-wise Flows

**Affine**:
$$f(z) = \alpha \odot z + \beta$$
$$\log |\det J| = \sum_i \log |\alpha_i|$$

**Leaky ReLU**:
$$f(z)_i = \max(\alpha z_i, z_i)$$
$$\log |\det J| = \sum_{i: z_i < 0} \log |\alpha|$$

**Sigmoid/Logit**:
$$f(z) = \sigma(z) = \frac{1}{1 + e^{-z}}$$
$$\log |\det J| = \sum_i \log[\sigma(z_i)(1 - \sigma(z_i))]$$

## Matrix Determinant Lemma

### For Low-Rank Updates

If $A$ is invertible and $U, V$ are $d \times r$ matrices:

$$\det(A + UV^T) = \det(A) \det(I_r + V^T A^{-1} U)$$

This reduces a $d \times d$ determinant to an $r \times r$ determinant when $r \ll d$.

### Application: Planar Flows

Planar flow: $f(z) = z + u \cdot h(w^T z + b)$

This is a rank-1 update to identity. Using the matrix determinant lemma:

$$\det J_f = 1 + u^T \frac{\partial h}{\partial z} = 1 + h'(w^T z + b) \cdot u^T w$$

Cost: $O(d)$ instead of $O(d^3)$!

## Implementation Patterns

### Pattern 1: Explicit Diagonal

```python
class DiagonalFlow(nn.Module):
    """Flow with diagonal Jacobian."""
    
    def __init__(self, dim):
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(dim))
        self.shift = nn.Parameter(torch.zeros(dim))
    
    def forward(self, z):
        x = z * torch.exp(self.log_scale) + self.shift
        log_det = self.log_scale.sum()  # Same for all samples
        return x, log_det.expand(z.shape[0])
    
    def inverse(self, x):
        z = (x - self.shift) * torch.exp(-self.log_scale)
        log_det = -self.log_scale.sum()
        return z, log_det.expand(x.shape[0])
```

### Pattern 2: Triangular via Autoregressive

```python
class AutoregressiveFlow(nn.Module):
    """Flow with triangular Jacobian via autoregressive structure."""
    
    def __init__(self, dim, hidden_dim=64):
        super().__init__()
        self.dim = dim
        
        # MADE-style network that outputs scale and shift
        self.net = MADENetwork(dim, hidden_dim)
    
    def forward(self, z):
        # Get autoregressive parameters
        log_scale, shift = self.net(z)  # Both (batch, dim)
        
        # Transform
        x = z * torch.exp(log_scale) + shift
        
        # Log det is sum of log scales (triangular Jacobian)
        log_det = log_scale.sum(dim=-1)
        
        return x, log_det
```

### Pattern 3: Block Structure

```python
class CouplingFlow(nn.Module):
    """Flow with block-triangular Jacobian."""
    
    def __init__(self, dim, hidden_dim=64):
        super().__init__()
        self.dim = dim
        self.split = dim // 2
        
        # Network to compute scale and shift for second half
        self.net = nn.Sequential(
            nn.Linear(self.split, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * (dim - self.split))
        )
    
    def forward(self, z):
        z1, z2 = z[:, :self.split], z[:, self.split:]
        
        # Compute transformation parameters from z1
        params = self.net(z1)
        log_scale, shift = params.chunk(2, dim=-1)
        log_scale = torch.tanh(log_scale)  # Bound for stability
        
        # Transform z2
        x1 = z1  # Identity
        x2 = z2 * torch.exp(log_scale) + shift
        
        x = torch.cat([x1, x2], dim=-1)
        
        # Log det: only from transformed part
        log_det = log_scale.sum(dim=-1)
        
        return x, log_det
```

## Computing Log-Det in Practice

### Using PyTorch's slogdet

```python
def compute_log_det_general(jacobian):
    """Compute log|det(J)| for general matrix."""
    # slogdet returns (sign, log_abs_det)
    sign, log_abs_det = torch.slogdet(jacobian)
    return log_abs_det
```

### Numerical Stability

```python
def stable_log_det_diagonal(diagonal):
    """Numerically stable log-det for diagonal matrix."""
    # Avoid log(0) by clamping
    return torch.log(torch.abs(diagonal) + 1e-8).sum(dim=-1)


def stable_log_det_triangular(triangular):
    """Log-det for triangular matrix."""
    diagonal = torch.diagonal(triangular, dim1=-2, dim2=-1)
    return torch.log(torch.abs(diagonal) + 1e-8).sum(dim=-1)
```

## Verifying Jacobian Computations

### Numerical Gradient Check

```python
def numerical_log_det(flow, z, eps=1e-5):
    """Compute log|det J| numerically for verification."""
    batch_size, dim = z.shape
    
    # Compute full Jacobian numerically
    jacobian = torch.zeros(batch_size, dim, dim)
    
    for i in range(dim):
        z_plus = z.clone()
        z_minus = z.clone()
        z_plus[:, i] += eps
        z_minus[:, i] -= eps
        
        x_plus, _ = flow.forward(z_plus)
        x_minus, _ = flow.forward(z_minus)
        
        jacobian[:, :, i] = (x_plus - x_minus) / (2 * eps)
    
    # Compute log det
    _, log_det = torch.slogdet(jacobian)
    return log_det


def verify_jacobian(flow, z, tol=1e-3):
    """Verify flow's log_det against numerical computation."""
    _, log_det_flow = flow.forward(z)
    log_det_numerical = numerical_log_det(flow, z)
    
    error = (log_det_flow - log_det_numerical).abs().max().item()
    
    if error > tol:
        print(f"WARNING: Jacobian error {error:.2e} exceeds tolerance {tol}")
    else:
        print(f"✓ Jacobian verified (error: {error:.2e})")
    
    return error
```

## Summary

| Jacobian Type | Determinant Formula | Cost | Flow Type |
|---------------|--------------------| -----|-----------|
| General | LU decomposition | $O(d^3)$ | Infeasible |
| Diagonal | $\prod_i J_{ii}$ | $O(d)$ | Element-wise |
| Triangular | $\prod_i J_{ii}$ | $O(d)$ | Autoregressive |
| Block triangular | $\prod_k \det(J_k)$ | $O(d)$ | Coupling |
| Rank-$r$ update | Matrix det lemma | $O(dr^2)$ | Planar, Sylvester |

Key takeaway: **Flow architecture design is fundamentally about achieving efficient Jacobian determinant computation** while maintaining expressiveness.

## References

1. Papamakarios, G., et al. (2021). Normalizing Flows for Probabilistic Modeling and Inference. *JMLR*.
2. Dinh, L., et al. (2015). NICE: Non-linear Independent Components Estimation. *ICLR Workshop*.
3. Kingma, D. P., et al. (2016). Improved Variational Inference with Inverse Autoregressive Flow. *NeurIPS*.
