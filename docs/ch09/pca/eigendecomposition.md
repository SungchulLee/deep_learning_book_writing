# Eigendecomposition for PCA

Computing principal components via covariance matrix eigenvectors.

---

## Covariance Matrix

### Definition

For centered data $X \in \mathbb{R}^{n \times d}$:

$$C = \frac{1}{n-1} X^T X$$

### Properties

- Symmetric: $C = C^T$
- Positive semi-definite: $v^T C v \geq 0$
- Eigenvalues are non-negative

---

## Eigendecomposition

### The Eigenproblem

$$C v_i = \lambda_i v_i$$

where:
- $\lambda_i$: Eigenvalue (variance in direction $v_i$)
- $v_i$: Eigenvector (principal direction)

### Matrix Form

$$C = V \Lambda V^T$$

where:
- $V = [v_1, ..., v_d]$: Orthonormal eigenvector matrix
- $\Lambda = \text{diag}(\lambda_1, ..., \lambda_d)$: Eigenvalue matrix

---

## PCA via Eigendecomposition

```python
import numpy as np

def pca_eigen(X, n_components):
    """
    PCA using eigendecomposition of covariance matrix.
    
    Args:
        X: Data [n_samples, n_features]
        n_components: Number of components to keep
    
    Returns:
        components: Principal components [n_features, n_components]
        explained_variance: Variance for each component
        projected: Projected data [n_samples, n_components]
    """
    # Center data
    mean = X.mean(axis=0)
    X_centered = X - mean
    
    # Covariance matrix
    n = X.shape[0]
    cov = X_centered.T @ X_centered / (n - 1)
    
    # Eigendecomposition (eigh for symmetric matrices)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Sort by eigenvalue (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Select top-k components
    components = eigenvectors[:, :n_components]
    explained_variance = eigenvalues[:n_components]
    
    # Project data
    projected = X_centered @ components
    
    return components, explained_variance, projected
```

---

## Numerical Considerations

### Use `eigh` for Symmetric Matrices

```python
# ✓ Correct: eigh for symmetric matrices
eigenvalues, eigenvectors = np.linalg.eigh(cov)

# ✗ Less stable: general eig
eigenvalues, eigenvectors = np.linalg.eig(cov)
```

### Handle Near-Zero Eigenvalues

```python
# Regularization for numerical stability
cov_reg = cov + 1e-10 * np.eye(cov.shape[0])
```

---

## Complexity Analysis

| Step | Complexity |
|------|------------|
| Covariance computation | $O(nd^2)$ |
| Eigendecomposition | $O(d^3)$ |
| Projection | $O(ndk)$ |
| **Total** | $O(nd^2 + d^3)$ |

For high-dimensional data ($d > n$), SVD is more efficient.

---

## Summary

- Covariance eigenvectors give principal directions
- Eigenvalues give variance explained
- Use `eigh` for numerical stability
- Consider SVD for high-dimensional data
