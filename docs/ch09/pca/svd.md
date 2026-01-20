# SVD for PCA

Efficient PCA computation via Singular Value Decomposition.

---

## SVD Definition

For matrix $X \in \mathbb{R}^{n \times d}$:

$$X = U \Sigma V^T$$

where:
- $U \in \mathbb{R}^{n \times n}$: Left singular vectors (orthonormal)
- $\Sigma \in \mathbb{R}^{n \times d}$: Singular values (diagonal)
- $V \in \mathbb{R}^{d \times d}$: Right singular vectors (orthonormal)

---

## Connection to PCA

### Key Relationship

$$X^T X = V \Sigma^T U^T U \Sigma V^T = V \Sigma^2 V^T$$

Comparing with eigendecomposition $C = V \Lambda V^T$:

$$\lambda_i = \frac{\sigma_i^2}{n-1}$$

**Principal components are right singular vectors!**

---

## SVD-based PCA

```python
import numpy as np

def pca_svd(X, n_components):
    """
    PCA using SVD (more numerically stable).
    
    Args:
        X: Data [n_samples, n_features]
        n_components: Number of components
    
    Returns:
        components: Principal components [n_features, n_components]
        explained_variance: Variance per component
        projected: Projected data [n_samples, n_components]
    """
    # Center data
    mean = X.mean(axis=0)
    X_centered = X - mean
    
    # SVD (full_matrices=False for efficiency)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    
    # Principal components are rows of Vt (columns of V)
    components = Vt[:n_components].T
    
    # Eigenvalues from singular values
    n = X.shape[0]
    explained_variance = (S[:n_components] ** 2) / (n - 1)
    
    # Projected data: X @ V = U @ S
    projected = U[:, :n_components] * S[:n_components]
    
    return components, explained_variance, projected
```

---

## Truncated SVD

For large matrices, compute only top-k singular values:

```python
from scipy.sparse.linalg import svds

def pca_truncated(X, n_components):
    """PCA using truncated SVD (efficient for large data)."""
    X_centered = X - X.mean(axis=0)
    
    # Compute only top-k singular values
    U, S, Vt = svds(X_centered, k=n_components)
    
    # Sort by singular value (svds returns in ascending order)
    idx = np.argsort(S)[::-1]
    
    return Vt[idx].T, S[idx]**2 / (X.shape[0]-1)
```

---

## Complexity Comparison

| Method | Complexity | When to Use |
|--------|------------|-------------|
| **Eigendecomposition** | $O(nd^2 + d^3)$ | $d \ll n$, dense |
| **Full SVD** | $O(\min(nd^2, n^2d))$ | General case |
| **Truncated SVD** | $O(ndk)$ | Large data, few components |
| **Randomized SVD** | $O(nd\log k)$ | Very large data |

---

## PyTorch Implementation

```python
import torch

def pca_torch(X, n_components):
    """PCA using PyTorch SVD."""
    X_centered = X - X.mean(dim=0)
    
    # torch.linalg.svd
    U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
    
    components = Vh[:n_components].T
    variance = (S[:n_components] ** 2) / (X.shape[0] - 1)
    projected = U[:, :n_components] * S[:n_components]
    
    return components, variance, projected
```

---

## Summary

| Aspect | Eigendecomposition | SVD |
|--------|-------------------|-----|
| **Compute** | Covariance matrix first | Directly on data |
| **Stability** | Can be unstable | More stable |
| **Efficiency** | Good for $d \ll n$ | Better for $d \approx n$ or $d > n$ |
| **Truncation** | Must compute all eigenvalues | Can compute only top-k |
