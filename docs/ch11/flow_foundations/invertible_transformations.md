# Invertible Transformations

## Introduction

Invertibility is the defining constraint of normalizing flows. A transformation must be **bijective** (one-to-one and onto) to enable both sampling (forward pass) and density evaluation (inverse pass). This document explores what invertibility means, why it matters, and how to design invertible neural networks.

## Mathematical Definition

### Bijection

A function $f: \mathbb{R}^d \to \mathbb{R}^d$ is a **bijection** (invertible) if:

1. **Injective (one-to-one)**: $f(z_1) = f(z_2) \implies z_1 = z_2$
2. **Surjective (onto)**: For every $x$, there exists $z$ such that $f(z) = x$

Equivalently, there exists $f^{-1}$ such that:
- $f^{-1}(f(z)) = z$ for all $z$
- $f(f^{-1}(x)) = x$ for all $x$

### Diffeomorphism

For normalizing flows, we need **diffeomorphisms**: smooth bijections with smooth inverses.

- Both $f$ and $f^{-1}$ are continuously differentiable
- The Jacobian $\frac{\partial f}{\partial z}$ is non-singular everywhere

## Why Invertibility Matters

### For Density Evaluation (Training)

To compute $\log p(x)$, we need to transform data back to latent space:

$$\log p(x) = \log p_Z(f^{-1}(x)) + \log \left| \det J_{f^{-1}} \right|$$

Without $f^{-1}$, we cannot evaluate likelihood → cannot train with MLE.

### For Sampling (Generation)

To generate new data, we transform latent samples:

$$z \sim p_Z(z), \quad x = f(z)$$

Without $f$, we cannot generate samples.

### Bijective Latent Space

Every data point $x$ maps to a unique latent $z$:
- **Deterministic encoding**: No stochasticity (unlike VAE)
- **Perfect reconstruction**: $f(f^{-1}(x)) = x$ exactly
- **Meaningful interpolation**: Latent space is structured

## Building Invertible Networks

### Design Principles

Standard neural networks are **not** invertible:
- ReLU is not invertible ($\text{ReLU}(1) = \text{ReLU}(2) = 1$ for clamped inputs)
- Pooling loses information
- Fully-connected layers with $\text{out} < \text{in}$ lose dimensions

**Key insight**: Design each layer to be invertible, then compose them.

### Invertible Building Blocks

#### 1. Affine Transformations

$$f(z) = Az + b, \quad A \in \mathbb{R}^{d \times d}, \det(A) \neq 0$$

Inverse:
$$f^{-1}(x) = A^{-1}(x - b)$$

**Implementation**:
```python
class AffineFlow(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.W = nn.Parameter(torch.eye(dim))
        self.b = nn.Parameter(torch.zeros(dim))
    
    def forward(self, z):
        x = z @ self.W.T + self.b
        log_det = torch.slogdet(self.W)[1]
        return x, log_det.expand(z.shape[0])
    
    def inverse(self, x):
        W_inv = torch.inverse(self.W)
        z = (x - self.b) @ W_inv.T
        log_det = -torch.slogdet(self.W)[1]
        return z, log_det.expand(x.shape[0])
```

#### 2. Element-wise Invertible Functions

Apply an invertible scalar function to each dimension:

| Function | Formula | Inverse |
|----------|---------|---------|
| Affine | $\alpha z + \beta$ | $(x - \beta) / \alpha$ |
| Exp/Log | $e^z$ | $\log(x)$ |
| Sigmoid/Logit | $\sigma(z)$ | $\log(x/(1-x))$ |
| Leaky ReLU | $\max(\alpha z, z)$ | $\max(x/\alpha, x)$ |
| Softplus | $\log(1 + e^z)$ | $\log(e^x - 1)$ |

```python
class LeakyReLUFlow(nn.Module):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.alpha = negative_slope
    
    def forward(self, z):
        x = torch.where(z >= 0, z, self.alpha * z)
        log_det = torch.where(z >= 0, 
                              torch.zeros_like(z), 
                              torch.full_like(z, np.log(self.alpha)))
        return x, log_det.sum(dim=-1)
    
    def inverse(self, x):
        z = torch.where(x >= 0, x, x / self.alpha)
        log_det = torch.where(x >= 0,
                              torch.zeros_like(x),
                              torch.full_like(x, -np.log(self.alpha)))
        return z, log_det.sum(dim=-1)
```

#### 3. Permutations

Reorder dimensions:

$$f(z) = Pz, \quad P \text{ is a permutation matrix}$$

Inverse: $f^{-1}(x) = P^T x$ (transpose = inverse for permutation matrices)

Log-det: $\log|\det P| = 0$ (permutations preserve volume)

```python
class Permutation(nn.Module):
    def __init__(self, dim, mode='reverse'):
        super().__init__()
        if mode == 'reverse':
            perm = torch.arange(dim - 1, -1, -1)
        elif mode == 'random':
            perm = torch.randperm(dim)
        else:
            perm = torch.arange(dim)
        
        self.register_buffer('perm', perm)
        self.register_buffer('inv_perm', torch.argsort(perm))
    
    def forward(self, z):
        return z[:, self.perm], torch.zeros(z.shape[0], device=z.device)
    
    def inverse(self, x):
        return x[:, self.inv_perm], torch.zeros(x.shape[0], device=x.device)
```

#### 4. Coupling Layers

Split dimensions and transform one part based on the other:

$$z = [z_A, z_B]$$
$$x_A = z_A$$
$$x_B = g(z_B; \theta(z_A))$$

Where $g$ is invertible in $z_B$ for any $\theta$.

**Affine coupling**:
$$x_B = z_B \odot \exp(s(z_A)) + t(z_A)$$

Inverse:
$$z_B = (x_B - t(x_A)) \odot \exp(-s(x_A))$$

```python
class AffineCoupling(nn.Module):
    def __init__(self, dim, hidden_dim=64):
        super().__init__()
        self.dim = dim
        self.split = dim // 2
        
        # s and t networks (can be arbitrary)
        self.net = nn.Sequential(
            nn.Linear(self.split, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * (dim - self.split))
        )
    
    def forward(self, z):
        z_a, z_b = z[:, :self.split], z[:, self.split:]
        
        params = self.net(z_a)
        log_s, t = params.chunk(2, dim=-1)
        log_s = torch.tanh(log_s) * 2  # Bound for stability
        
        x_a = z_a
        x_b = z_b * torch.exp(log_s) + t
        
        x = torch.cat([x_a, x_b], dim=-1)
        log_det = log_s.sum(dim=-1)
        
        return x, log_det
    
    def inverse(self, x):
        x_a, x_b = x[:, :self.split], x[:, self.split:]
        
        params = self.net(x_a)  # Same network, x_a = z_a
        log_s, t = params.chunk(2, dim=-1)
        log_s = torch.tanh(log_s) * 2
        
        z_a = x_a
        z_b = (x_b - t) * torch.exp(-log_s)
        
        z = torch.cat([z_a, z_b], dim=-1)
        log_det = -log_s.sum(dim=-1)
        
        return z, log_det
```

## Non-Invertible Operations to Avoid

### Operations That Destroy Information

| Operation | Why Not Invertible | Alternative |
|-----------|-------------------|-------------|
| Max Pooling | Multiple inputs → same output | Invertible downsampling |
| ReLU | Negative values → 0 | Leaky ReLU |
| Dropout | Random zeroing | Deterministic regularization |
| Batch Norm (naive) | Running stats | ActNorm, Invertible BN |

### Dimensionality Reduction

Standard networks often reduce dimensions:
```
Input (784) → Hidden (256) → Hidden (64) → Output (10)
```

Flows must preserve dimensions:
```
Input (784) → Layer (784) → Layer (784) → Output (784)
```

**Solution**: Multi-scale architecture (factor out dimensions, don't reduce)

## Verifying Invertibility

### Numerical Test

```python
def verify_invertibility(flow, z, tol=1e-5):
    """Verify f^{-1}(f(z)) = z."""
    x, _ = flow.forward(z)
    z_reconstructed, _ = flow.inverse(x)
    
    error = (z - z_reconstructed).abs().max().item()
    
    if error > tol:
        raise AssertionError(f"Invertibility error: {error:.2e}")
    
    print(f"✓ Invertibility verified (max error: {error:.2e})")
    return error


def verify_both_directions(flow, z, x, tol=1e-5):
    """Verify both f^{-1}(f(z)) = z and f(f^{-1}(x)) = x."""
    # Forward then inverse
    x_from_z, _ = flow.forward(z)
    z_back, _ = flow.inverse(x_from_z)
    error1 = (z - z_back).abs().max().item()
    
    # Inverse then forward
    z_from_x, _ = flow.inverse(x)
    x_back, _ = flow.forward(z_from_x)
    error2 = (x - x_back).abs().max().item()
    
    print(f"Forward→Inverse error: {error1:.2e}")
    print(f"Inverse→Forward error: {error2:.2e}")
    
    assert error1 < tol and error2 < tol, "Invertibility check failed"
    return max(error1, error2)
```

### Jacobian Singularity Check

```python
def check_jacobian_nonsingular(flow, z, eps=1e-6):
    """Check that Jacobian is non-singular."""
    batch_size, dim = z.shape
    
    # Compute numerical Jacobian
    jac = torch.zeros(batch_size, dim, dim)
    for i in range(dim):
        z_plus = z.clone()
        z_minus = z.clone()
        z_plus[:, i] += eps
        z_minus[:, i] -= eps
        
        x_plus, _ = flow.forward(z_plus)
        x_minus, _ = flow.forward(z_minus)
        
        jac[:, :, i] = (x_plus - x_minus) / (2 * eps)
    
    # Check determinant is not close to zero
    det = torch.det(jac)
    min_abs_det = det.abs().min().item()
    
    if min_abs_det < eps:
        print(f"WARNING: Near-singular Jacobian (min |det| = {min_abs_det:.2e})")
    else:
        print(f"✓ Jacobian non-singular (min |det| = {min_abs_det:.2e})")
    
    return min_abs_det
```

## Practical Considerations

### Numerical Precision

Even mathematically invertible functions can have numerical issues:

```python
# Potential numerical issues
x = torch.exp(z)  # Overflow for large z
z_back = torch.log(x)  # Underflow for small x

# More stable
x = torch.exp(torch.clamp(z, -20, 20))
z_back = torch.log(torch.clamp(x, 1e-8, None))
```

### Conditioning of Transformations

Well-conditioned transformations have bounded Jacobian eigenvalues:

```python
def check_conditioning(flow, z):
    """Check condition number of Jacobian."""
    # Compute Jacobian
    jac = compute_jacobian(flow, z)
    
    # Condition number = max eigenvalue / min eigenvalue
    eigenvalues = torch.linalg.eigvalsh(jac @ jac.T)
    condition_number = eigenvalues.max() / eigenvalues.min()
    
    print(f"Condition number: {condition_number.item():.2f}")
    
    if condition_number > 1000:
        print("WARNING: Poorly conditioned transformation")
```

### Initialization

Initialize flows close to identity to start with stable transformations:

```python
def init_near_identity(flow):
    """Initialize flow parameters for near-identity transformation."""
    for name, param in flow.named_parameters():
        if 'weight' in name:
            # Small random weights
            nn.init.normal_(param, 0, 0.01)
        elif 'bias' in name:
            nn.init.zeros_(param)
        elif 'log_scale' in name or 'log_s' in name:
            # log(1) = 0 → scale = 1
            nn.init.zeros_(param)
```

## Summary

Invertibility requirements for normalizing flows:

1. **Mathematical**: Bijective mapping with smooth inverse
2. **Computational**: Efficient forward AND inverse pass
3. **Numerical**: Stable computation in finite precision
4. **Jacobian**: Non-singular everywhere

Building blocks for invertible networks:
- Affine transformations
- Element-wise invertible functions
- Permutations
- Coupling layers
- Autoregressive structures

Always verify invertibility numerically during development!

## References

1. Dinh, L., et al. (2015). NICE: Non-linear Independent Components Estimation. *ICLR Workshop*.
2. Dinh, L., et al. (2017). Density Estimation Using Real-NVP. *ICLR*.
3. Papamakarios, G., et al. (2021). Normalizing Flows for Probabilistic Modeling and Inference. *JMLR*.
