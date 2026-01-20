# Reconstruction Error

Quantifying information loss in dimensionality reduction.

---

## Reconstruction in PCA

### Projection and Reconstruction

1. **Project:** $z = W^T x$ (encode)
2. **Reconstruct:** $\hat{x} = W z = W W^T x$ (decode)

where $W \in \mathbb{R}^{d \times k}$ contains top-k principal components.

### Reconstruction Error

$$\mathcal{E} = \|x - \hat{x}\|^2 = \|x - WW^T x\|^2$$

---

## Optimal Reconstruction

### Theorem

PCA minimizes reconstruction error among all linear projections:

$$W^* = \arg\min_W \sum_{i=1}^n \|x_i - WW^T x_i\|^2 \quad \text{s.t.} \quad W^T W = I$$

### Closed-Form Error

Total reconstruction error using top-k components:

$$\mathcal{E}_k = \sum_{i=k+1}^{d} \lambda_i$$

**Error equals sum of discarded eigenvalues!**

---

## Implementation

```python
import numpy as np

def compute_reconstruction_error(X, W):
    """
    Compute reconstruction error.
    
    Args:
        X: Original data [n_samples, n_features]
        W: Principal components [n_features, n_components]
    
    Returns:
        mse: Mean squared error per sample
        total_error: Total reconstruction error
    """
    X_centered = X - X.mean(axis=0)
    
    # Project and reconstruct
    Z = X_centered @ W  # [n, k]
    X_reconstructed = Z @ W.T  # [n, d]
    
    # Error
    error = X_centered - X_reconstructed
    mse = (error ** 2).mean(axis=1)
    total_error = (error ** 2).sum()
    
    return mse, total_error


def reconstruction_error_from_eigenvalues(eigenvalues, k):
    """
    Theoretical reconstruction error from eigenvalues.
    
    Args:
        eigenvalues: All eigenvalues (sorted descending)
        k: Number of components kept
    
    Returns:
        error: Sum of discarded eigenvalues
        explained: Fraction of variance explained
    """
    total_variance = eigenvalues.sum()
    kept_variance = eigenvalues[:k].sum()
    discarded_variance = eigenvalues[k:].sum()
    
    return discarded_variance, kept_variance / total_variance
```

---

## Choosing Number of Components

### By Explained Variance

Keep components until explained variance exceeds threshold:

```python
def choose_n_components(eigenvalues, threshold=0.95):
    """Find minimum k for desired explained variance."""
    total = eigenvalues.sum()
    cumsum = np.cumsum(eigenvalues)
    k = np.searchsorted(cumsum / total, threshold) + 1
    return k
```

### By Reconstruction Error

Set maximum acceptable error:

```python
def choose_by_error(eigenvalues, max_error):
    """Find minimum k for acceptable reconstruction error."""
    cumsum_discarded = eigenvalues.sum() - np.cumsum(eigenvalues)
    k = np.searchsorted(-cumsum_discarded, -max_error) + 1
    return k
```

### Scree Plot

```python
def plot_scree(eigenvalues):
    """Plot eigenvalues and cumulative variance."""
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Eigenvalue plot
    ax1.plot(eigenvalues, 'o-')
    ax1.set_xlabel('Component')
    ax1.set_ylabel('Eigenvalue')
    ax1.set_title('Scree Plot')
    
    # Cumulative variance
    cumsum = np.cumsum(eigenvalues) / eigenvalues.sum()
    ax2.plot(cumsum, 'o-')
    ax2.axhline(0.95, color='r', linestyle='--', label='95%')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Explained Variance')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
```

---

## Summary

| Metric | Formula | Use |
|--------|---------|-----|
| **Total error** | $\sum_{i=k+1}^d \lambda_i$ | Absolute error |
| **Explained variance** | $\frac{\sum_{i=1}^k \lambda_i}{\sum_{i=1}^d \lambda_i}$ | Relative quality |
| **Per-sample MSE** | $\frac{1}{n}\sum_j \|x_j - \hat{x}_j\|^2$ | Average error |

---

## Key Insight

PCA provides the **optimal linear reconstruction** — any other k-dimensional linear subspace will have higher reconstruction error. This optimality motivates the use of PCA as a baseline for comparison with nonlinear methods like autoencoders.
