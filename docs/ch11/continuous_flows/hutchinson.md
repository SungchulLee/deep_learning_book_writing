# Hutchinson's Trace Estimator

## Introduction

Hutchinson's trace estimator is a stochastic method for efficiently computing the trace of a matrix using random vector probing. In the context of continuous normalizing flows, it reduces the cost of computing log-determinants from O(D²) to O(D), making CNFs practical for high-dimensional data.

## The Trace Computation Problem

### Why Trace Matters for CNFs

The instantaneous change of variables for CNFs requires:

$$\frac{\partial \log p(\mathbf{z}(t))}{\partial t} = -\text{tr}\left(\frac{\partial f}{\partial \mathbf{z}}\right)$$

### Exact Computation is Expensive

For a function $f: \mathbb{R}^D \to \mathbb{R}^D$, the trace of its Jacobian is:

$$\text{tr}(J) = \sum_{i=1}^{D} \frac{\partial f_i}{\partial z_i}$$

Computing each diagonal element requires one backward pass:
- D backward passes needed
- Each pass is O(D)
- **Total: O(D²)**

For D = 784 (MNIST), this is ~600,000 operations per sample!

## Hutchinson's Estimator

### The Key Identity

For any matrix $A \in \mathbb{R}^{D \times D}$ and random vector $\boldsymbol{\epsilon}$ with:
- $\mathbb{E}[\boldsymbol{\epsilon}] = \mathbf{0}$
- $\mathbb{E}[\boldsymbol{\epsilon}\boldsymbol{\epsilon}^T] = I$

The following holds:

$$\text{tr}(A) = \mathbb{E}_{\boldsymbol{\epsilon}}\left[\boldsymbol{\epsilon}^T A \boldsymbol{\epsilon}\right]$$

### Proof

$$\mathbb{E}[\boldsymbol{\epsilon}^T A \boldsymbol{\epsilon}] = \mathbb{E}\left[\sum_{i=1}^{D} \sum_{j=1}^{D} \epsilon_i A_{ij} \epsilon_j\right]$$

$$= \sum_{i,j} A_{ij} \mathbb{E}[\epsilon_i \epsilon_j]$$

Since $\mathbb{E}[\epsilon_i \epsilon_j] = \delta_{ij}$ (Kronecker delta):

$$= \sum_{i} A_{ii} = \text{tr}(A)$$

### Monte Carlo Approximation

In practice, we use K samples:

$$\text{tr}(A) \approx \frac{1}{K} \sum_{k=1}^{K} \boldsymbol{\epsilon}_k^T A \boldsymbol{\epsilon}_k$$

## Noise Distributions

### Rademacher Distribution

$$\epsilon_i \in \{-1, +1\} \text{ with probability } \frac{1}{2} \text{ each}$$

Properties:
- $\mathbb{E}[\epsilon_i] = 0$
- $\mathbb{E}[\epsilon_i^2] = 1$
- $\mathbb{E}[\epsilon_i \epsilon_j] = 0$ for $i \neq j$

```python
def sample_rademacher(shape, device='cpu'):
    """Sample Rademacher random variables."""
    return torch.randint(0, 2, shape, device=device).float() * 2 - 1
```

### Gaussian Distribution

$$\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, I)$$

Properties:
- $\mathbb{E}[\epsilon_i] = 0$
- $\mathbb{E}[\epsilon_i^2] = 1$
- $\mathbb{E}[\epsilon_i \epsilon_j] = 0$ for $i \neq j$

```python
def sample_gaussian(shape, device='cpu'):
    """Sample standard Gaussian random variables."""
    return torch.randn(shape, device=device)
```

### Comparison

| Distribution | Variance | Computation | Recommendation |
|--------------|----------|-------------|----------------|
| Rademacher | Lower for sparse J | Simpler | Default choice |
| Gaussian | Higher | Simpler | Large D |

## Implementation

### Basic Hutchinson Estimator

```python
import torch

def hutchinson_trace(A, n_samples=1, noise_type='rademacher'):
    """
    Estimate trace of matrix A using Hutchinson's method.
    
    Args:
        A: Matrix (D, D)
        n_samples: Number of noise samples
        noise_type: 'rademacher' or 'gaussian'
    
    Returns:
        Estimated trace (scalar)
    """
    D = A.shape[0]
    
    estimates = []
    for _ in range(n_samples):
        if noise_type == 'rademacher':
            epsilon = torch.randint(0, 2, (D,), device=A.device).float() * 2 - 1
        else:
            epsilon = torch.randn(D, device=A.device)
        
        # ε^T A ε
        estimate = torch.dot(epsilon, A @ epsilon)
        estimates.append(estimate)
    
    return torch.stack(estimates).mean()
```

### Efficient Implementation with Autograd

For neural networks, we don't form J explicitly. Instead, use vector-Jacobian products:

```python
def hutchinson_trace_estimator(f, z, n_samples=1):
    """
    Estimate tr(df/dz) using Hutchinson's method with autograd.
    
    Args:
        f: Function output, shape (batch, D)
        z: Function input, shape (batch, D), requires_grad=True
        n_samples: Number of noise samples
    
    Returns:
        Trace estimates, shape (batch,)
    """
    batch_size, D = z.shape
    
    estimates = []
    for _ in range(n_samples):
        # Sample noise
        epsilon = torch.randint(0, 2, (batch_size, D), 
                               device=z.device).float() * 2 - 1
        
        # Compute ε^T J ε using vector-Jacobian product
        # First: compute ε^T f = (f * ε).sum()
        # Then: gradient gives ε^T J
        epsilon_f = (f * epsilon).sum()
        
        # ε^T J via backward pass
        epsilon_J = torch.autograd.grad(
            epsilon_f, z, 
            create_graph=True,  # For training
            retain_graph=True
        )[0]
        
        # ε^T J ε
        estimate = (epsilon_J * epsilon).sum(dim=-1)
        estimates.append(estimate)
    
    return torch.stack(estimates).mean(dim=0)
```

### Integration in CNF

```python
class CNFWithHutchinson(nn.Module):
    """CNF using Hutchinson trace estimator."""
    
    def __init__(self, dim, hidden_dims, n_trace_samples=1):
        super().__init__()
        self.dim = dim
        self.n_trace_samples = n_trace_samples
        self.dynamics = self._build_dynamics(dim, hidden_dims)
    
    def _build_dynamics(self, dim, hidden_dims):
        layers = []
        prev = dim + 1  # +1 for time
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.Tanh()])
            prev = h
        layers.append(nn.Linear(prev, dim))
        return nn.Sequential(*layers)
    
    def compute_dynamics_and_trace(self, t, z):
        """Compute f(z,t) and tr(df/dz) together."""
        batch_size = z.shape[0]
        
        with torch.enable_grad():
            z = z.requires_grad_(True)
            
            # Compute f
            t_vec = t.expand(batch_size, 1)
            f = self.dynamics(torch.cat([z, t_vec], dim=-1))
            
            # Hutchinson trace estimate
            trace = torch.zeros(batch_size, device=z.device)
            
            for _ in range(self.n_trace_samples):
                epsilon = torch.randint(
                    0, 2, (batch_size, self.dim), device=z.device
                ).float() * 2 - 1
                
                epsilon_f = (f * epsilon).sum()
                epsilon_J = torch.autograd.grad(
                    epsilon_f, z, 
                    create_graph=self.training,
                    retain_graph=True
                )[0]
                
                trace += (epsilon_J * epsilon).sum(dim=-1)
            
            trace = trace / self.n_trace_samples
        
        return f, trace
```

## Variance Analysis

### Variance of the Estimator

For a single sample:

$$\text{Var}[\boldsymbol{\epsilon}^T A \boldsymbol{\epsilon}] = 2 \|A\|_F^2 - 2 \sum_i A_{ii}^2$$

For Rademacher noise, the variance is:

$$\text{Var}_{\text{Rad}} = 2 \sum_{i \neq j} A_{ij}^2$$

### Variance Reduction

**1. Multiple samples**:
$$\text{Var}\left[\frac{1}{K}\sum_k \boldsymbol{\epsilon}_k^T A \boldsymbol{\epsilon}_k\right] = \frac{1}{K} \text{Var}[\boldsymbol{\epsilon}^T A \boldsymbol{\epsilon}]$$

**2. Control variates**:
```python
def hutchinson_with_control_variate(f, z, baseline_trace=None):
    """Variance reduction using control variates."""
    epsilon = sample_rademacher(z.shape, z.device)
    
    # Standard estimate
    epsilon_f = (f * epsilon).sum()
    epsilon_J = torch.autograd.grad(epsilon_f, z, create_graph=True)[0]
    estimate = (epsilon_J * epsilon).sum(dim=-1)
    
    if baseline_trace is not None:
        # Subtract baseline and add its known trace
        estimate = estimate - baseline_trace + baseline_trace.mean()
    
    return estimate
```

**3. Antithetic sampling**:
```python
def hutchinson_antithetic(f, z):
    """Antithetic sampling for variance reduction."""
    epsilon = sample_rademacher(z.shape, z.device)
    
    # Estimate with +ε
    estimate_pos = compute_trace_estimate(f, z, epsilon)
    
    # Estimate with -ε (antithetic)
    estimate_neg = compute_trace_estimate(f, z, -epsilon)
    
    # Average (reduces variance for symmetric distributions)
    return (estimate_pos + estimate_neg) / 2
```

## Computational Complexity

### Cost Breakdown

| Operation | Exact Trace | Hutchinson (K samples) |
|-----------|-------------|------------------------|
| Forward pass | O(D) | O(D) |
| Backward passes | D × O(D) = O(D²) | K × O(D) |
| **Total** | **O(D²)** | **O(KD)** |

### Practical Speedup

For D = 784 (MNIST) with K = 1:
- Exact: ~600,000 operations
- Hutchinson: ~800 operations
- **Speedup: ~750×**

## Bias-Variance Trade-off

### Unbiasedness

Hutchinson's estimator is **unbiased**:

$$\mathbb{E}[\hat{\text{tr}}] = \text{tr}(A)$$

This holds for any number of samples K ≥ 1.

### Trade-off in Practice

| K (samples) | Bias | Variance | Cost |
|-------------|------|----------|------|
| 1 | 0 | High | O(D) |
| 5 | 0 | Medium | O(5D) |
| 10 | 0 | Low | O(10D) |
| D | 0 | ~0 | O(D²) |

**Recommendation**: K = 1 during training, K = 5-10 for evaluation.

## Application to Different Matrices

### Jacobians of Neural Networks

The primary use case:

```python
# For f: R^D -> R^D (neural network)
# Want: tr(df/dz)

z.requires_grad_(True)
f = network(z)
trace = hutchinson_trace_estimator(f, z)
```

### Hessians

Can also estimate trace of Hessian:

```python
def hutchinson_hessian_trace(loss, params):
    """Estimate tr(H) for Hessian H = d²L/dθ²."""
    # First gradient
    grad = torch.autograd.grad(loss, params, create_graph=True)[0]
    
    # Hutchinson on the Hessian
    epsilon = sample_rademacher(params.shape, params.device)
    grad_epsilon = (grad * epsilon).sum()
    hessian_epsilon = torch.autograd.grad(grad_epsilon, params)[0]
    
    return (hessian_epsilon * epsilon).sum()
```

## Summary

Hutchinson's trace estimator provides:

1. **Unbiased estimation**: $\mathbb{E}[\boldsymbol{\epsilon}^T A \boldsymbol{\epsilon}] = \text{tr}(A)$
2. **O(D) complexity**: Single backward pass per sample
3. **Easy implementation**: Uses standard autograd
4. **Flexible noise**: Rademacher or Gaussian
5. **Variance control**: Multiple samples, antithetic, control variates

This estimator is the key enabler for continuous normalizing flows, reducing computational cost from O(D²) to O(D) while maintaining unbiased gradients for training.

## References

1. Hutchinson, M. F. (1989). A Stochastic Estimator of the Trace of the Influence Matrix for Laplacian Smoothing Splines. *Communications in Statistics - Simulation and Computation*.
2. Grathwohl, W., et al. (2019). FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models. *ICLR*.
3. Avron, H., & Toledo, S. (2011). Randomized Algorithms for Estimating the Trace of an Implicit Symmetric Positive Semi-definite Matrix. *Journal of the ACM*.
4. Adams, R. P., et al. (2018). Estimating the Spectral Density of Large Implicit Matrices. *arXiv*.
