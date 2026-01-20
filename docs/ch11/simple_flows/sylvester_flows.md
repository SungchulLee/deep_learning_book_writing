# Sylvester Flows

## Introduction

**Sylvester flows** (van den Berg et al., 2018) generalize planar flows from rank-1 to rank-$M$ updates, significantly increasing expressiveness while maintaining efficient log-determinant computation. They are named after the Sylvester determinant identity used in their derivation.

## Motivation

### Limitations of Planar Flows

Planar flow: $f(z) = z + u h(w^T z + b)$
- Rank-1 update to identity
- Limited expressiveness
- Requires many layers

### Sylvester's Insight

Instead of a single vector $u$, use $M$ vectors:
$$f(z) = z + A h(B^T z + b)$$

where:
- $A \in \mathbb{R}^{d \times M}$ — "output" directions
- $B \in \mathbb{R}^{d \times M}$ — "input" directions  
- $h$ — element-wise nonlinearity
- $b \in \mathbb{R}^M$ — bias

This is a **rank-$M$ update** to the identity.

## Sylvester Determinant Identity

### The Identity

For matrices $A \in \mathbb{R}^{d \times M}$ and $B \in \mathbb{R}^{d \times M}$:

$$\det(I_d + AB^T) = \det(I_M + B^T A)$$

This reduces a $d \times d$ determinant to an $M \times M$ determinant!

### Application to Flows

For $f(z) = z + A h(B^T z + b)$:

$$\det \frac{\partial f}{\partial z} = \det(I_d + A \cdot \text{diag}(h') \cdot B^T)$$

Using Sylvester's identity:
$$= \det(I_M + \text{diag}(h') \cdot B^T A)$$

**Cost**: $O(M^3 + dM)$ instead of $O(d^3)$

When $M \ll d$, this is a massive speedup.

## Mathematical Details

### Jacobian

$$J_f = \frac{\partial f}{\partial z} = I + A \cdot \text{diag}(h'(B^T z + b)) \cdot B^T$$

where $h' = \frac{\partial h}{\partial (B^T z + b)}$ is the derivative of the activation.

### Log-Determinant

$$\log |\det J_f| = \log |\det(I_M + \text{diag}(h') \cdot B^T A)|$$

This $M \times M$ determinant can be computed in $O(M^3)$.

### Invertibility Constraint

For invertibility, we need $\det J_f > 0$ everywhere.

**Sufficient condition**: If $B^T A$ is positive definite and $h' > 0$, the flow is invertible.

Common approach: Use orthogonal or triangular parameterization.

## Orthogonal Sylvester Flow

### Parameterization

Set $A = QR$ and $B = QS$ where:
- $Q \in \mathbb{R}^{d \times M}$ has orthonormal columns ($Q^T Q = I_M$)
- $R, S \in \mathbb{R}^{M \times M}$ are diagonal matrices with positive entries

Then: $B^T A = S Q^T Q R = SR$ is diagonal positive definite.

### Implementation

```python
import torch
import torch.nn as nn
import numpy as np

class OrthogonalSylvesterFlow(nn.Module):
    """
    Orthogonal Sylvester Flow.
    
    f(z) = z + Q R h(S Q^T z + b)
    
    where Q has orthonormal columns, R and S are diagonal positive.
    """
    
    def __init__(self, dim: int, M: int):
        """
        Args:
            dim: Input dimensionality
            M: Rank of the update (M << dim for efficiency)
        """
        super().__init__()
        self.dim = dim
        self.M = M
        
        # Orthonormal Q: parameterized via Householder reflections
        self.v = nn.Parameter(torch.randn(M, dim) * 0.01)
        
        # Diagonal R and S (use log for positivity)
        self.log_R = nn.Parameter(torch.zeros(M))
        self.log_S = nn.Parameter(torch.zeros(M))
        
        # Bias
        self.b = nn.Parameter(torch.zeros(M))
    
    def _get_Q(self):
        """Compute orthonormal Q from Householder reflections."""
        # Gram-Schmidt on v
        Q = torch.zeros(self.dim, self.M, device=self.v.device)
        
        for i in range(self.M):
            v = self.v[i]
            for j in range(i):
                v = v - torch.dot(v, Q[:, j]) * Q[:, j]
            Q[:, i] = v / (torch.norm(v) + 1e-8)
        
        return Q
    
    def forward(self, z: torch.Tensor):
        """
        Forward transformation.
        
        Args:
            z: Input (batch, dim)
        
        Returns:
            x: Transformed (batch, dim)
            log_det: Log determinant (batch,)
        """
        Q = self._get_Q()  # (dim, M)
        R = torch.exp(self.log_R)  # (M,)
        S = torch.exp(self.log_S)  # (M,)
        
        # Compute S Q^T z + b
        Qz = z @ Q  # (batch, M)
        pre_activation = S * Qz + self.b  # (batch, M)
        
        # Activation and derivative
        h = torch.tanh(pre_activation)  # (batch, M)
        h_prime = 1 - h ** 2  # (batch, M)
        
        # Transform: z + Q R h
        x = z + (h * R) @ Q.T  # (batch, dim)
        
        # Log determinant
        # det(I_M + diag(h') S R) = prod(1 + h'_i S_i R_i)
        diag_terms = 1 + h_prime * S * R  # (batch, M)
        log_det = torch.log(torch.abs(diag_terms) + 1e-8).sum(dim=1)  # (batch,)
        
        return x, log_det
    
    def inverse(self, x: torch.Tensor, n_iter: int = 100, tol: float = 1e-6):
        """
        Inverse transformation (iterative).
        
        Args:
            x: Transformed tensor
            n_iter: Max iterations
            tol: Convergence tolerance
        
        Returns:
            z: Original tensor
            log_det: Log determinant
        """
        Q = self._get_Q()
        R = torch.exp(self.log_R)
        S = torch.exp(self.log_S)
        
        # Fixed point iteration
        z = x.clone()
        
        for _ in range(n_iter):
            Qz = z @ Q
            pre_activation = S * Qz + self.b
            h = torch.tanh(pre_activation)
            
            z_new = x - (h * R) @ Q.T
            
            if (z_new - z).abs().max() < tol:
                break
            z = z_new
        
        # Compute log det
        _, log_det_forward = self.forward(z)
        log_det = -log_det_forward
        
        return z, log_det
```

## Householder Sylvester Flow

### Using Householder Reflections

Householder reflections provide another way to construct orthogonal matrices:

$$H_v = I - 2\frac{vv^T}{\|v\|^2}$$

Stack $K$ reflections to get richer orthogonal matrices.

### Implementation

```python
class HouseholderSylvesterFlow(nn.Module):
    """
    Sylvester flow using Householder reflections for orthogonality.
    """
    
    def __init__(self, dim: int, M: int, n_householder: int = None):
        """
        Args:
            dim: Input dimensionality
            M: Rank of update
            n_householder: Number of Householder reflections (default: M)
        """
        super().__init__()
        self.dim = dim
        self.M = M
        self.n_householder = n_householder or M
        
        # Householder vectors
        self.vs = nn.Parameter(torch.randn(self.n_householder, dim))
        
        # Diagonal scaling
        self.log_R = nn.Parameter(torch.zeros(M))
        self.log_S = nn.Parameter(torch.zeros(M))
        self.b = nn.Parameter(torch.zeros(M))
    
    def _householder_matrix(self, v):
        """Single Householder reflection."""
        v_norm = v / (v.norm() + 1e-8)
        return torch.eye(self.dim, device=v.device) - 2 * torch.outer(v_norm, v_norm)
    
    def _get_Q(self):
        """Build orthogonal matrix from Householder reflections."""
        # Start with identity, apply reflections
        Q = torch.eye(self.dim, device=self.vs.device)
        for i in range(self.n_householder):
            H = self._householder_matrix(self.vs[i])
            Q = Q @ H
        
        # Take first M columns
        return Q[:, :self.M]
    
    def forward(self, z: torch.Tensor):
        Q = self._get_Q()
        R = torch.exp(self.log_R)
        S = torch.exp(self.log_S)
        
        Qz = z @ Q
        pre_activation = S * Qz + self.b
        h = torch.tanh(pre_activation)
        h_prime = 1 - h ** 2
        
        x = z + (h * R) @ Q.T
        
        diag_terms = 1 + h_prime * S * R
        log_det = torch.log(torch.abs(diag_terms) + 1e-8).sum(dim=1)
        
        return x, log_det
    
    def inverse(self, x, n_iter=100, tol=1e-6):
        Q = self._get_Q()
        R = torch.exp(self.log_R)
        S = torch.exp(self.log_S)
        
        z = x.clone()
        for _ in range(n_iter):
            Qz = z @ Q
            h = torch.tanh(S * Qz + self.b)
            z_new = x - (h * R) @ Q.T
            
            if (z_new - z).abs().max() < tol:
                break
            z = z_new
        
        _, log_det_forward = self.forward(z)
        return z, -log_det_forward
```

## Triangular Sylvester Flow

### Alternative Parameterization

Instead of orthogonal $Q$, use triangular matrices:

$$f(z) = z + R_1 h(R_2^T z + b)$$

where $R_1, R_2 \in \mathbb{R}^{d \times M}$ are restricted to have specific structure.

This can sometimes be easier to optimize.

## Comparison with Planar Flows

### Expressiveness vs. Cost

| Aspect | Planar ($M=1$) | Sylvester ($M$) |
|--------|----------------|-----------------|
| Rank of update | 1 | $M$ |
| Parameters | $2d + 1$ | $O(dM + M^2)$ |
| Log-det cost | $O(d)$ | $O(M^3 + dM)$ |
| Expressiveness | Low | Medium-High |
| Layers needed | Many | Fewer |

### Practical Guidance

- $M = 1$: Equivalent to planar flow
- $M \approx d/4$: Good balance for moderate $d$
- $M = d$: Full expressiveness but $O(d^3)$ cost

## Training Example

```python
def train_sylvester_flow(target_samples, M=4, n_flows=8, n_epochs=1000, lr=1e-3):
    """Train Sylvester flows on target distribution."""
    dim = target_samples.shape[1]
    
    # Build model
    flows = nn.ModuleList([OrthogonalSylvesterFlow(dim, M) for _ in range(n_flows)])
    base_dist = torch.distributions.MultivariateNormal(
        torch.zeros(dim), torch.eye(dim)
    )
    
    optimizer = torch.optim.Adam(
        [p for flow in flows for p in flow.parameters()], 
        lr=lr
    )
    
    losses = []
    
    for epoch in range(n_epochs):
        idx = torch.randint(0, len(target_samples), (256,))
        x = target_samples[idx]
        
        # Inverse pass
        log_det_total = torch.zeros(x.shape[0])
        z = x
        for flow in reversed(flows):
            z, log_det = flow.inverse(z)
            log_det_total += log_det
        
        # Log probability
        log_pz = base_dist.log_prob(z)
        log_px = log_pz + log_det_total
        loss = -log_px.mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    return flows, losses
```

## Advantages and Limitations

### Advantages

1. **Higher expressiveness** than planar/radial with moderate $M$
2. **Efficient** when $M \ll d$
3. **Smooth parameterization** with orthogonal constraints
4. **Theoretically grounded** via Sylvester identity

### Limitations

1. **No analytical inverse**: Iterative inversion required
2. **Orthogonality constraints**: Can be tricky to maintain
3. **Hyperparameter $M$**: Needs tuning
4. **Superseded**: Coupling flows often preferred in practice

## Summary

Sylvester flows:
- **Generalize** planar flows from rank-1 to rank-$M$
- **Key identity**: $\det(I + AB^T) = \det(I + B^T A)$
- **Parameterization**: Orthogonal or Householder for stability
- **Trade-off**: Expressiveness ($M$) vs. computation ($M^3$)
- **Use case**: When planar flows need more power but coupling is overkill

## References

1. van den Berg, R., et al. (2018). Sylvester Normalizing Flows for Variational Inference. *UAI*.
2. Rezende, D. J., & Mohamed, S. (2015). Variational Inference with Normalizing Flows. *ICML*.
3. Tomczak, J. M., & Welling, M. (2016). Improving Variational Auto-Encoders using Householder Flow. *arXiv*.
