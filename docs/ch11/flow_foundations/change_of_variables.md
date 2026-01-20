# Change of Variables Formula

## Introduction

The **change of variables formula** is the mathematical foundation of normalizing flows. It tells us how probability density transforms when we apply an invertible function to a random variable. Understanding this formula deeply is essential for working with flows.

## Intuition: Conservation of Probability

### The Core Idea

Probability must be conserved under transformation. If we transform random variable $Z$ to $X = f(Z)$:

- The total probability mass remains 1
- Probability "stretches" or "compresses" based on how $f$ distorts space

### Visual Intuition (1D)

Consider $Z \sim \text{Uniform}(0, 1)$ and $X = f(Z) = Z^2$:

```
Z:  [====|====|====|====]  Uniform on [0,1]
     0   0.25 0.5  0.75  1
     
X:  [==|=|=|========]      Compressed near 0, stretched near 1
    0  0.06 0.25   1
```

The transformation $f(z) = z^2$ compresses probability near 0 and stretches it near 1. The density must adjust to keep total probability = 1.

## Mathematical Derivation

### 1D Case

Let $Z$ have density $p_Z(z)$ and $X = f(Z)$ where $f$ is invertible and differentiable.

**Step 1**: Probability conservation
$$P(X \leq x) = P(f(Z) \leq x) = P(Z \leq f^{-1}(x))$$

**Step 2**: Differentiate both sides
$$p_X(x) = p_Z(f^{-1}(x)) \cdot \left| \frac{d f^{-1}(x)}{dx} \right|$$

**Step 3**: Using inverse function theorem, $\frac{d f^{-1}}{dx} = \frac{1}{f'(f^{-1}(x))}$

$$\boxed{p_X(x) = p_Z(f^{-1}(x)) \cdot \left| \frac{1}{f'(f^{-1}(x))} \right|}$$

### Why the Absolute Value?

The derivative can be negative (decreasing function), but density must be positive. The absolute value ensures $p_X(x) > 0$.

### Example: Exponential Transformation

Let $Z \sim \mathcal{N}(0, 1)$ and $X = e^Z$ (log-normal transformation).

- $f(z) = e^z$, so $f^{-1}(x) = \ln(x)$
- $f'(z) = e^z$, so $|f'(f^{-1}(x))| = |e^{\ln x}| = x$

$$p_X(x) = p_Z(\ln x) \cdot \frac{1}{x} = \frac{1}{\sqrt{2\pi}} e^{-(\ln x)^2/2} \cdot \frac{1}{x}$$

This is the log-normal density!

## Multivariate Case

### The General Formula

For $Z \in \mathbb{R}^d$ with density $p_Z(z)$ and $X = f(Z)$ where $f: \mathbb{R}^d \to \mathbb{R}^d$ is a diffeomorphism (smooth invertible map):

$$\boxed{p_X(x) = p_Z(f^{-1}(x)) \cdot \left| \det \frac{\partial f^{-1}(x)}{\partial x} \right|}$$

Or equivalently, using the Jacobian of $f$ at $z = f^{-1}(x)$:

$$p_X(x) = p_Z(z) \cdot \left| \det \frac{\partial f(z)}{\partial z} \right|^{-1}$$

### The Jacobian Matrix

The Jacobian of $f: \mathbb{R}^d \to \mathbb{R}^d$ is:

$$J_f = \frac{\partial f}{\partial z} = \begin{pmatrix}
\frac{\partial f_1}{\partial z_1} & \cdots & \frac{\partial f_1}{\partial z_d} \\
\vdots & \ddots & \vdots \\
\frac{\partial f_d}{\partial z_1} & \cdots & \frac{\partial f_d}{\partial z_d}
\end{pmatrix}$$

The **determinant** of this matrix measures the local volume change.

### Interpretation of Jacobian Determinant

$$|\det J_f(z)| = \text{factor by which } f \text{ scales volume near } z$$

- $|\det J| > 1$: Volume expansion
- $|\det J| < 1$: Volume contraction  
- $|\det J| = 1$: Volume preserving

## Log-Space Formulation

For numerical stability, we work in log-space:

$$\log p_X(x) = \log p_Z(f^{-1}(x)) + \log \left| \det \frac{\partial f^{-1}}{\partial x} \right|$$

Or using the forward Jacobian:

$$\boxed{\log p_X(x) = \log p_Z(z) - \log \left| \det \frac{\partial f}{\partial z} \right|}$$

where $z = f^{-1}(x)$.

## Composing Transformations

### Chain of Flows

For composed transformation $f = f_K \circ f_{K-1} \circ \cdots \circ f_1$:

$$z_0 \xrightarrow{f_1} z_1 \xrightarrow{f_2} z_2 \xrightarrow{} \cdots \xrightarrow{f_K} z_K = x$$

### Chain Rule for Jacobians

$$\det J_f = \det J_{f_K} \cdot \det J_{f_{K-1}} \cdot \ldots \cdot \det J_{f_1}$$

In log-space:

$$\log |\det J_f| = \sum_{k=1}^{K} \log |\det J_{f_k}|$$

### Log-Likelihood of Data

$$\log p_X(x) = \log p_Z(z_0) + \sum_{k=1}^{K} \log \left| \det \frac{\partial f_k}{\partial z_{k-1}} \right|$$

where $z_0 = f_1^{-1}(f_2^{-1}(\cdots f_K^{-1}(x)))$.

## Implementation

### Basic Flow Class

```python
import torch
import torch.nn as nn
import numpy as np

class Flow(nn.Module):
    """Base class for normalizing flow transformations."""
    
    def forward(self, z):
        """
        Forward transformation: z -> x
        
        Returns:
            x: Transformed samples
            log_det: Log absolute determinant of Jacobian
        """
        raise NotImplementedError
    
    def inverse(self, x):
        """
        Inverse transformation: x -> z
        
        Returns:
            z: Latent samples
            log_det: Log absolute determinant of Jacobian
        """
        raise NotImplementedError


class FlowSequence(nn.Module):
    """Compose multiple flows."""
    
    def __init__(self, flows, base_distribution):
        super().__init__()
        self.flows = nn.ModuleList(flows)
        self.base_dist = base_distribution
    
    def forward(self, z):
        """Transform z -> x through all flows."""
        log_det_total = 0
        
        for flow in self.flows:
            z, log_det = flow.forward(z)
            log_det_total += log_det
        
        return z, log_det_total
    
    def inverse(self, x):
        """Transform x -> z through all flows (reversed)."""
        log_det_total = 0
        
        for flow in reversed(self.flows):
            x, log_det = flow.inverse(x)
            log_det_total += log_det
        
        return x, log_det_total
    
    def log_prob(self, x):
        """Compute log p(x) using change of variables."""
        # Inverse: x -> z
        z, log_det = self.inverse(x)
        
        # Base distribution log probability
        log_pz = self.base_dist.log_prob(z)
        
        # Change of variables formula
        log_px = log_pz + log_det
        
        return log_px
    
    def sample(self, n_samples, device='cpu'):
        """Sample x ~ p(x)."""
        # Sample from base
        z = self.base_dist.sample(n_samples, device)
        
        # Transform to data space
        x, _ = self.forward(z)
        
        return x
```

### Gaussian Base Distribution

```python
class GaussianBase:
    """Standard Gaussian base distribution."""
    
    def __init__(self, dim):
        self.dim = dim
    
    def sample(self, n_samples, device='cpu'):
        return torch.randn(n_samples, self.dim, device=device)
    
    def log_prob(self, z):
        """Log probability of standard Gaussian."""
        # log p(z) = -0.5 * (z^2 + log(2π)) summed over dimensions
        return -0.5 * (z ** 2 + np.log(2 * np.pi)).sum(dim=-1)
```

## Verification: Checking Implementations

### Numerical Jacobian Check

```python
def numerical_jacobian(f, z, eps=1e-5):
    """Compute Jacobian numerically for verification."""
    d = z.shape[-1]
    jac = torch.zeros(d, d)
    
    for i in range(d):
        z_plus = z.clone()
        z_minus = z.clone()
        z_plus[0, i] += eps
        z_minus[0, i] -= eps
        
        f_plus, _ = f.forward(z_plus)
        f_minus, _ = f.forward(z_minus)
        
        jac[:, i] = (f_plus - f_minus).squeeze() / (2 * eps)
    
    return jac


def verify_log_det(flow, z, tol=1e-4):
    """Verify that flow's log_det matches numerical computation."""
    # Flow's log det
    _, log_det_flow = flow.forward(z)
    
    # Numerical log det
    jac = numerical_jacobian(flow, z)
    log_det_numerical = torch.log(torch.abs(torch.det(jac)))
    
    error = (log_det_flow - log_det_numerical).abs().item()
    
    print(f"Flow log det: {log_det_flow.item():.6f}")
    print(f"Numerical log det: {log_det_numerical.item():.6f}")
    print(f"Error: {error:.2e}")
    
    assert error < tol, f"Log det error {error} exceeds tolerance {tol}"
    print("✓ Log det verification passed")
```

### Invertibility Check

```python
def verify_invertibility(flow, z, tol=1e-5):
    """Verify that inverse(forward(z)) = z."""
    x, log_det_fwd = flow.forward(z)
    z_reconstructed, log_det_inv = flow.inverse(x)
    
    reconstruction_error = (z - z_reconstructed).abs().max().item()
    log_det_error = (log_det_fwd + log_det_inv).abs().max().item()
    
    print(f"Reconstruction error: {reconstruction_error:.2e}")
    print(f"Log det sum (should be 0): {log_det_error:.2e}")
    
    assert reconstruction_error < tol, "Invertibility check failed"
    assert log_det_error < tol, "Log det consistency check failed"
    print("✓ Invertibility verification passed")
```

## Common Pitfalls

### 1. Sign Errors in Log-Det

The Jacobian determinant can be negative (for reflections), but we need the absolute value:

```python
# WRONG
log_det = torch.log(det)  # Can be NaN if det < 0

# CORRECT
log_det = torch.log(torch.abs(det))
# Or use log of absolute value directly
log_det = torch.slogdet(jacobian)[1]  # Returns log|det|
```

### 2. Forgetting Log-Det Term

```python
# WRONG - missing Jacobian correction
log_prob = base_dist.log_prob(z)

# CORRECT
log_prob = base_dist.log_prob(z) + log_det
```

### 3. Wrong Sign Convention

Be consistent about whether `log_det` is for forward or inverse:

```python
# If log_det is |det(df/dz)|:
log_px = log_pz - log_det  # Note: MINUS

# If log_det is |det(df^{-1}/dx)|:
log_px = log_pz + log_det  # Note: PLUS
```

## Summary

The change of variables formula:

$$p_X(x) = p_Z(f^{-1}(x)) \cdot \left| \det J_{f^{-1}}(x) \right|$$

Key points:
1. **Probability conservation** requires density adjustment
2. **Jacobian determinant** measures volume change
3. **Log-space** formulation for numerical stability
4. **Composition** multiplies determinants (adds in log-space)
5. **Always verify** implementations numerically

## References

1. Papamakarios, G., et al. (2021). Normalizing Flows for Probabilistic Modeling and Inference. *JMLR*, Section 2.
2. Rudin, W. (1976). Principles of Mathematical Analysis. McGraw-Hill. (Change of variables theorem)
