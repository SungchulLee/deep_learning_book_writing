# Variance Maximization

PCA finds directions that maximize variance in projected data.

---

## The Variance Perspective

### Problem Statement

Given data $X \in \mathbb{R}^{n \times d}$, find direction $w \in \mathbb{R}^d$ such that projecting data onto $w$ maximizes variance.

### Mathematical Formulation

Variance of projected data:

$$\text{Var}(Xw) = w^T C w$$

where $C = \frac{1}{n}X^TX$ is the covariance matrix (assuming centered data).

### Optimization Problem

$$\max_w w^T C w \quad \text{subject to} \quad \|w\| = 1$$

---

## Solution via Lagrange Multipliers

### Lagrangian

$$\mathcal{L}(w, \lambda) = w^T C w - \lambda(w^T w - 1)$$

### Stationary Condition

$$\frac{\partial \mathcal{L}}{\partial w} = 2Cw - 2\lambda w = 0$$

$$Cw = \lambda w$$

**Result:** Optimal $w$ is an eigenvector of $C$.

### Which Eigenvector?

$$\text{Var} = w^T C w = w^T \lambda w = \lambda$$

**Maximum variance** corresponds to **largest eigenvalue**.

---

## Multiple Components

### Sequential Maximization

1. **First PC:** Eigenvector with largest eigenvalue $\lambda_1$
2. **Second PC:** Maximize variance orthogonal to first PC
3. **k-th PC:** Eigenvector with k-th largest eigenvalue

### All Components at Once

$$\max_W \text{trace}(W^T C W) \quad \text{s.t.} \quad W^T W = I$$

Solution: $W = [w_1, w_2, ..., w_k]$ where $w_i$ are top-k eigenvectors.

---

## Implementation

```python
def pca_variance_maximization(X, k):
    """PCA by variance maximization."""
    # Center
    X_centered = X - X.mean(axis=0)
    
    # Covariance
    C = X_centered.T @ X_centered / (X.shape[0] - 1)
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    
    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    
    # Top-k components
    W = eigenvectors[:, idx[:k]]
    explained_variance = eigenvalues[idx[:k]]
    
    return W, explained_variance
```

---

## Explained Variance Ratio

$$\text{EVR}_k = \frac{\lambda_k}{\sum_{i=1}^d \lambda_i}$$

Total variance explained by first $k$ components:

$$\text{Cumulative EVR} = \frac{\sum_{i=1}^k \lambda_i}{\sum_{i=1}^d \lambda_i}$$

---

## Summary

| Concept | Key Point |
|---------|-----------|
| **Objective** | Maximize variance of projections |
| **Solution** | Eigenvectors of covariance matrix |
| **Ranking** | By eigenvalue (variance explained) |
| **Orthogonality** | Principal components are orthogonal |
