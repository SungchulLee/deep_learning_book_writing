# Linear Flows

## Introduction

Linear flows are the simplest normalizing flows, applying linear (affine) transformations to the input. While limited in expressiveness alone, they form essential building blocks in more complex architectures and provide intuition for understanding flows.

## Affine Transformations

### Definition

An affine transformation in $\mathbb{R}^d$:

$$f(z) = Az + b$$

where:
- $A \in \mathbb{R}^{d \times d}$ is an invertible matrix
- $b \in \mathbb{R}^d$ is a bias vector

### Inverse and Jacobian

**Inverse**:
$$f^{-1}(x) = A^{-1}(x - b)$$

**Jacobian**:
$$J_f = A$$

**Log-determinant**:
$$\log |\det J_f| = \log |\det A|$$

### Implementation

```python
import torch
import torch.nn as nn
import numpy as np

class AffineFlow(nn.Module):
    """General affine transformation flow."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        # Initialize A as identity + small noise
        self.A = nn.Parameter(torch.eye(dim) + 0.01 * torch.randn(dim, dim))
        self.b = nn.Parameter(torch.zeros(dim))
    
    def forward(self, z: torch.Tensor):
        """z -> x = Az + b"""
        x = z @ self.A.T + self.b
        log_det = torch.slogdet(self.A)[1]
        return x, log_det.expand(z.shape[0])
    
    def inverse(self, x: torch.Tensor):
        """x -> z = A^{-1}(x - b)"""
        A_inv = torch.inverse(self.A)
        z = (x - self.b) @ A_inv.T
        log_det = -torch.slogdet(self.A)[1]
        return z, log_det.expand(x.shape[0])
```

### Computational Cost

- Forward/Inverse: $O(d^2)$ for matrix-vector multiplication
- Log-determinant: $O(d^3)$ for general matrix

This $O(d^3)$ cost motivates structured linear transformations.

## Diagonal Flows

### Definition

Restrict $A$ to be diagonal:

$$f(z) = \text{diag}(\alpha) \cdot z + b = \alpha \odot z + b$$

where $\alpha \in \mathbb{R}^d$ with $\alpha_i \neq 0$.

### Properties

- **Parameters**: $2d$ instead of $d^2 + d$
- **Log-det**: $\sum_i \log |\alpha_i|$ — only $O(d)$
- **Inverse**: $z = (x - b) \oslash \alpha$

### Implementation

```python
class DiagonalAffineFlow(nn.Module):
    """Diagonal affine transformation (element-wise scaling and shift)."""
    
    def __init__(self, dim: int):
        super().__init__()
        # Use log-scale for positivity and stability
        self.log_scale = nn.Parameter(torch.zeros(dim))
        self.shift = nn.Parameter(torch.zeros(dim))
    
    def forward(self, z: torch.Tensor):
        scale = torch.exp(self.log_scale)
        x = z * scale + self.shift
        log_det = self.log_scale.sum()
        return x, log_det.expand(z.shape[0])
    
    def inverse(self, x: torch.Tensor):
        scale = torch.exp(self.log_scale)
        z = (x - self.shift) / scale
        log_det = -self.log_scale.sum()
        return z, log_det.expand(x.shape[0])
```

## Triangular Flows

### Lower Triangular Matrix

$$A = \begin{pmatrix}
a_{11} & 0 & \cdots & 0 \\
a_{21} & a_{22} & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
a_{d1} & a_{d2} & \cdots & a_{dd}
\end{pmatrix}$$

**Log-det**: $\sum_i \log |a_{ii}|$ — only $O(d)$!

### Implementation

```python
class LowerTriangularFlow(nn.Module):
    """Lower triangular affine transformation."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        # Parameterize lower triangular entries
        # Number of entries: d + d(d-1)/2 = d(d+1)/2
        n_entries = dim * (dim + 1) // 2
        self.entries = nn.Parameter(torch.randn(n_entries) * 0.01)
        self.bias = nn.Parameter(torch.zeros(dim))
        
        # Create mask for lower triangular
        self.register_buffer('tril_indices', 
                            torch.tril_indices(dim, dim))
    
    def _get_matrix(self):
        """Construct lower triangular matrix from parameters."""
        L = torch.zeros(self.dim, self.dim, device=self.entries.device)
        L[self.tril_indices[0], self.tril_indices[1]] = self.entries
        
        # Ensure positive diagonal for invertibility
        diag_idx = torch.arange(self.dim)
        L[diag_idx, diag_idx] = torch.exp(L[diag_idx, diag_idx])
        
        return L
    
    def forward(self, z: torch.Tensor):
        L = self._get_matrix()
        x = z @ L.T + self.bias
        
        # Log det = sum of log diagonal
        log_det = torch.diagonal(L).log().sum()
        return x, log_det.expand(z.shape[0])
    
    def inverse(self, x: torch.Tensor):
        L = self._get_matrix()
        # Solve L^T z = (x - b)
        z = torch.linalg.solve_triangular(
            L.T, (x - self.bias).T, upper=True
        ).T
        
        log_det = -torch.diagonal(L).log().sum()
        return z, log_det.expand(x.shape[0])
```

## LU Decomposition Flow

### Idea

Any invertible matrix can be decomposed as $A = PLU$ where:
- $P$ is a permutation matrix
- $L$ is lower triangular with ones on diagonal
- $U$ is upper triangular

**Log-det**: $\log|\det A| = \sum_i \log |U_{ii}|$

### Implementation

```python
class LUFlow(nn.Module):
    """Affine flow using LU decomposition for efficient log-det."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        # Initialize with random orthogonal matrix
        W_init = torch.linalg.qr(torch.randn(dim, dim))[0]
        
        # LU decomposition of initialization
        P, L, U = torch.linalg.lu(W_init)
        
        # Store permutation (fixed)
        self.register_buffer('P', P)
        
        # Parameterize L (lower triangular, ones on diagonal)
        # Store strictly lower triangular part
        self.L_entries = nn.Parameter(torch.tril(L, -1))
        
        # Parameterize U (upper triangular)
        # Store log of diagonal separately for positivity
        self.U_diag_log = nn.Parameter(torch.diagonal(U).abs().log())
        self.U_upper = nn.Parameter(torch.triu(U, 1))
        
        self.bias = nn.Parameter(torch.zeros(dim))
    
    def _get_W(self):
        """Reconstruct W = PLU."""
        # L: lower triangular with ones on diagonal
        L = self.L_entries + torch.eye(self.dim, device=self.L_entries.device)
        
        # U: upper triangular
        U = self.U_upper + torch.diag(torch.exp(self.U_diag_log))
        
        return self.P @ L @ U
    
    def forward(self, z: torch.Tensor):
        W = self._get_W()
        x = z @ W.T + self.bias
        
        # Log det from U diagonal
        log_det = self.U_diag_log.sum()
        return x, log_det.expand(z.shape[0])
    
    def inverse(self, x: torch.Tensor):
        W = self._get_W()
        z = (x - self.bias) @ torch.inverse(W).T
        
        log_det = -self.U_diag_log.sum()
        return z, log_det.expand(x.shape[0])
```

## 1×1 Convolution (Glow)

### For Image Data

A 1×1 convolution with $c$ channels is equivalent to applying the same $c \times c$ linear transformation at every spatial location:

$$y_{:,i,j} = W \cdot x_{:,i,j}$$

**Log-det** (for $h \times w$ spatial dimensions):
$$\log |\det J| = h \cdot w \cdot \log |\det W|$$

### Implementation

```python
class Invertible1x1Conv(nn.Module):
    """Invertible 1x1 convolution for image flows (Glow)."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        
        # Initialize with random orthogonal matrix
        W_init = torch.linalg.qr(torch.randn(channels, channels))[0]
        
        # LU decomposition
        P, L, U = torch.linalg.lu(W_init)
        
        self.register_buffer('P', P)
        self.L_entries = nn.Parameter(torch.tril(L, -1))
        self.U_diag_log = nn.Parameter(torch.diagonal(U).abs().log())
        self.U_upper = nn.Parameter(torch.triu(U, 1))
    
    def _get_W(self):
        L = self.L_entries + torch.eye(self.channels, device=self.L_entries.device)
        U = self.U_upper + torch.diag(torch.exp(self.U_diag_log))
        return self.P @ L @ U
    
    def forward(self, z: torch.Tensor):
        """z: (batch, channels, height, width)"""
        batch, c, h, w = z.shape
        
        W = self._get_W()
        
        # Apply as 1x1 conv
        x = torch.nn.functional.conv2d(z, W.view(c, c, 1, 1))
        
        # Log det: multiply by number of spatial locations
        log_det = h * w * self.U_diag_log.sum()
        
        return x, log_det.expand(batch)
    
    def inverse(self, x: torch.Tensor):
        batch, c, h, w = x.shape
        
        W = self._get_W()
        W_inv = torch.inverse(W)
        
        z = torch.nn.functional.conv2d(x, W_inv.view(c, c, 1, 1))
        
        log_det = -h * w * self.U_diag_log.sum()
        
        return z, log_det.expand(batch)
```

## Orthogonal Flows

### Definition

Orthogonal transformations: $A^T A = I$

**Key property**: $|\det A| = 1$, so $\log |\det A| = 0$

### Householder Reflections

Any orthogonal matrix can be written as product of Householder reflections:

$$H_v = I - 2\frac{vv^T}{\|v\|^2}$$

```python
class HouseholderFlow(nn.Module):
    """Orthogonal flow using Householder reflections."""
    
    def __init__(self, dim: int, n_reflections: int = None):
        super().__init__()
        self.dim = dim
        self.n_reflections = n_reflections or dim
        
        # Householder vectors
        self.vs = nn.Parameter(torch.randn(self.n_reflections, dim))
    
    def _householder_matrix(self, v):
        """Compute Householder reflection matrix."""
        v_norm = v / (v.norm() + 1e-8)
        return torch.eye(self.dim, device=v.device) - 2 * torch.outer(v_norm, v_norm)
    
    def _get_orthogonal(self):
        """Construct orthogonal matrix from Householder reflections."""
        Q = torch.eye(self.dim, device=self.vs.device)
        for i in range(self.n_reflections):
            H = self._householder_matrix(self.vs[i])
            Q = Q @ H
        return Q
    
    def forward(self, z: torch.Tensor):
        Q = self._get_orthogonal()
        x = z @ Q.T
        # Orthogonal: log det = 0
        log_det = torch.zeros(z.shape[0], device=z.device)
        return x, log_det
    
    def inverse(self, x: torch.Tensor):
        Q = self._get_orthogonal()
        z = x @ Q  # Q^{-1} = Q^T
        log_det = torch.zeros(x.shape[0], device=x.device)
        return z, log_det
```

## Activation Normalization (ActNorm)

### Motivation

Batch normalization is not easily invertible due to running statistics. ActNorm is a learnable diagonal affine transformation with data-dependent initialization.

### Implementation

```python
class ActNorm(nn.Module):
    """
    Activation Normalization (Glow).
    
    Data-dependent initialization: first batch normalizes to zero mean, unit variance.
    Then parameters are learned.
    """
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        self.log_scale = nn.Parameter(torch.zeros(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        
        self.initialized = False
    
    def initialize(self, x: torch.Tensor):
        """Initialize to normalize first batch."""
        with torch.no_grad():
            # Flatten spatial dimensions if present
            if x.dim() == 4:  # (batch, channels, h, w)
                x_flat = x.permute(0, 2, 3, 1).reshape(-1, self.dim)
            else:
                x_flat = x
            
            mean = x_flat.mean(dim=0)
            std = x_flat.std(dim=0) + 1e-6
            
            self.bias.data = -mean
            self.log_scale.data = -torch.log(std)
        
        self.initialized = True
    
    def forward(self, z: torch.Tensor):
        if not self.initialized:
            self.initialize(z)
        
        if z.dim() == 4:  # Image: (batch, channels, h, w)
            scale = torch.exp(self.log_scale).view(1, -1, 1, 1)
            bias = self.bias.view(1, -1, 1, 1)
            x = (z + bias) * scale
            
            # Log det: multiply by spatial dimensions
            h, w = z.shape[2], z.shape[3]
            log_det = h * w * self.log_scale.sum()
        else:
            scale = torch.exp(self.log_scale)
            x = (z + self.bias) * scale
            log_det = self.log_scale.sum()
        
        return x, log_det.expand(z.shape[0])
    
    def inverse(self, x: torch.Tensor):
        if x.dim() == 4:
            scale = torch.exp(self.log_scale).view(1, -1, 1, 1)
            bias = self.bias.view(1, -1, 1, 1)
            z = x / scale - bias
            
            h, w = x.shape[2], x.shape[3]
            log_det = -h * w * self.log_scale.sum()
        else:
            scale = torch.exp(self.log_scale)
            z = x / scale - self.bias
            log_det = -self.log_scale.sum()
        
        return z, log_det.expand(x.shape[0])
```

## Summary

| Linear Flow | Parameters | Log-det Cost | Use Case |
|-------------|------------|--------------|----------|
| General Affine | $d^2 + d$ | $O(d^3)$ | Small $d$ only |
| Diagonal | $2d$ | $O(d)$ | Scaling/shifting |
| Triangular | $d(d+1)/2 + d$ | $O(d)$ | Moderate expressiveness |
| LU | $d^2 + d$ | $O(d)$ | Full expressiveness, efficient |
| 1×1 Conv | $c^2$ | $O(c)$ per pixel | Images (Glow) |
| Orthogonal | $kd$ | $O(1)$ | Volume-preserving |
| ActNorm | $2d$ | $O(d)$ | Normalization layer |

Linear flows alone are limited but essential as:
- Building blocks in coupling layers
- Channel mixing (1×1 conv)
- Normalization (ActNorm)
- Permutations

## References

1. Kingma, D. P., & Dhariwal, P. (2018). Glow: Generative Flow with Invertible 1×1 Convolutions. *NeurIPS*.
2. Tomczak, J. M., & Welling, M. (2016). Improving Variational Auto-Encoders using Householder Flow. *arXiv*.
3. Papamakarios, G., et al. (2021). Normalizing Flows for Probabilistic Modeling and Inference. *JMLR*.
