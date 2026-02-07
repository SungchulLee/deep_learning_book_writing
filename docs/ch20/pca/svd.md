# Singular Value Decomposition for PCA

Efficient and numerically stable PCA computation via SVD.

---

## Overview

Singular Value Decomposition (SVD) provides an alternative — and often superior — route to computing PCA. Rather than forming the covariance matrix and eigendecomposing it, SVD factorizes the data matrix directly. This avoids squaring condition numbers (improving numerical stability) and enables efficient truncated computation for high-dimensional data.

---

## SVD Definition

Every real matrix $\mathbf{X} \in \mathbb{R}^{n \times d}$ admits a factorization:

$$\mathbf{X} = \mathbf{U}\mathbf{S}\mathbf{V}^T$$

where:

- $\mathbf{U} \in \mathbb{R}^{n \times n}$: **Left singular vectors** (orthonormal: $\mathbf{U}^T\mathbf{U} = \mathbf{I}$)
- $\mathbf{S} \in \mathbb{R}^{n \times d}$: **Singular values** on the diagonal, $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_l \geq 0$ where $l = \min(n, d)$
- $\mathbf{V} \in \mathbb{R}^{d \times d}$: **Right singular vectors** (orthonormal: $\mathbf{V}^T\mathbf{V} = \mathbf{I}$)

### Outer Product Form

The SVD can be written as a sum of rank-1 matrices:

$$\mathbf{X} = \sum_{i=1}^{l} \sigma_i \, \mathbf{u}_i \mathbf{v}_i^T$$

Each term $\sigma_i \mathbf{u}_i \mathbf{v}_i^T$ captures one "mode" of the data, with singular value $\sigma_i$ indicating its importance.

### Compact (Economy) SVD

When $n > d$, the full $\mathbf{U}$ has $n - d$ columns that multiply zero singular values. The compact SVD drops these:

$$\mathbf{X} = \mathbf{U}_l \mathbf{S}_l \mathbf{V}^T$$

where $\mathbf{U}_l \in \mathbb{R}^{n \times l}$ and $\mathbf{S}_l \in \mathbb{R}^{l \times l}$. This is what `np.linalg.svd(X, full_matrices=False)` computes.

---

## Understanding the SVD Components

### Right Singular Vectors: Basis for Row Space

The columns of $\mathbf{V}$ (equivalently, rows of $\mathbf{V}^T$) form an orthonormal basis for the row space of $\mathbf{X}$:

$$\mathbf{v}_i^T \mathbf{v}_j = \delta_{ij}, \qquad \operatorname{span}\{\mathbf{v}_1, \ldots, \mathbf{v}_r\} = \operatorname{row}(\mathbf{X})$$

where $r = \operatorname{rank}(\mathbf{X})$.

### Left Singular Vectors: Basis for Column Space

The columns of $\mathbf{U}$ form an orthonormal basis for the column space of $\mathbf{X}$. Given the relationship $\mathbf{X}\mathbf{v}_i = \sigma_i \mathbf{u}_i$, each left singular vector is the (normalized) image of the corresponding right singular vector under $\mathbf{X}$.

### Singular Values: Scaling Factors

The singular value $\sigma_i$ measures how much $\mathbf{X}$ "stretches" along the $i$-th direction. Larger singular values correspond to directions of greater data variation.

---

## Computing the SVD

### Connection to Eigendecompositions

The SVD is intimately related to the eigendecompositions of $\mathbf{X}^T\mathbf{X}$ and $\mathbf{X}\mathbf{X}^T$:

**From $\mathbf{X}^T\mathbf{X}$ (right singular vectors):**

$$\mathbf{X}^T\mathbf{X} = \mathbf{V}\mathbf{S}^T\mathbf{U}^T\mathbf{U}\mathbf{S}\mathbf{V}^T = \mathbf{V}\mathbf{S}^2\mathbf{V}^T$$

The eigenvalues of $\mathbf{X}^T\mathbf{X}$ are $\sigma_i^2$, and its eigenvectors are the $\mathbf{v}_i$.

**From $\mathbf{X}\mathbf{X}^T$ (left singular vectors):**

$$\mathbf{X}\mathbf{X}^T = \mathbf{U}\mathbf{S}\mathbf{V}^T\mathbf{V}\mathbf{S}^T\mathbf{U}^T = \mathbf{U}\mathbf{S}^2\mathbf{U}^T$$

The eigenvalues of $\mathbf{X}\mathbf{X}^T$ are also $\sigma_i^2$, with eigenvectors $\mathbf{u}_i$.

### Recommended Computation

**Option A: Start from $\mathbf{V}$ (when $d \leq n$).**

1. Eigendecompose $\mathbf{X}^T\mathbf{X}$ to get $\sigma_i^2$ and $\mathbf{v}_i$
2. Compute $\sigma_i = \sqrt{\sigma_i^2}$
3. Recover $\mathbf{U}$ via $\mathbf{U} = \mathbf{X}\mathbf{V}\mathbf{S}^{-1}$

**Option B: Start from $\mathbf{U}$ (when $n < d$).**

1. Eigendecompose $\mathbf{X}\mathbf{X}^T$ to get $\sigma_i^2$ and $\mathbf{u}_i$
2. Compute $\sigma_i = \sqrt{\sigma_i^2}$
3. Recover $\mathbf{V}^T$ via $\mathbf{V}^T = \mathbf{S}^{-1}\mathbf{U}^T\mathbf{X}$

!!! warning "Avoid the Naïve Approach"
    Computing eigenvectors of $\mathbf{X}^T\mathbf{X}$ and $\mathbf{X}\mathbf{X}^T$ separately and trying to pair them is error-prone. The recommended approach computes one set of vectors and derives the other, ensuring correct pairing.

In practice, numerical libraries (LAPACK) compute SVD directly using algorithms like Golub–Kahan bidiagonalization followed by QR iteration, avoiding explicit formation of $\mathbf{X}^T\mathbf{X}$ entirely.

---

## Connection to PCA

### Key Relationship

For centered data $\mathbf{X}$, the covariance matrix eigendecomposition and SVD are related by:

$$\boldsymbol{\Sigma} = \frac{1}{n-1}\mathbf{X}^T\mathbf{X} = \frac{1}{n-1}\mathbf{V}\mathbf{S}^2\mathbf{V}^T = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^T$$

Therefore:

$$\lambda_i = \frac{\sigma_i^2}{n - 1}$$

**The right singular vectors of $\mathbf{X}$ are the principal components**, and eigenvalues relate to singular values by a factor of $n - 1$.

### PCA Outputs from SVD

| PCA quantity | From SVD |
|-------------|----------|
| Principal directions (loadings) | Columns of $\mathbf{V}$ (rows of $\mathbf{V}^T$) |
| Eigenvalues (variances) | $\sigma_i^2 / (n-1)$ |
| Scores (projections) | $\mathbf{U}_k \mathbf{S}_k$ (first $k$ columns of $\mathbf{U}$, scaled) |
| Reconstruction | $\mathbf{U}_k \mathbf{S}_k \mathbf{V}_k^T$ |

---

## Low-Rank Approximation

### The Eckart–Young–Mirsky Theorem

The best rank-$k$ approximation to $\mathbf{X}$ in the Frobenius norm is obtained by truncating the SVD to its top $k$ terms:

$$\mathbf{X}_k = \sum_{i=1}^{k} \sigma_i \, \mathbf{u}_i \mathbf{v}_i^T = \mathbf{U}_k \mathbf{S}_k \mathbf{V}_k^T$$

No other rank-$k$ matrix achieves smaller error:

$$\mathbf{X}_k = \arg\min_{\operatorname{rank}(\mathbf{A}) \leq k} \|\mathbf{X} - \mathbf{A}\|_F$$

### Approximation Error

$$\|\mathbf{X} - \mathbf{X}_k\|_F^2 = \sum_{i=k+1}^{l} \sigma_i^2$$

### Equivalence to PCA Reconstruction

For centered data, the truncated SVD approximation is exactly the PCA reconstruction:

$$\hat{\mathbf{X}} = \mathbf{X}\mathbf{V}_k\mathbf{V}_k^T = \mathbf{U}_k\mathbf{S}_k\mathbf{V}_k^T = \mathbf{X}_k$$

This connects the algebraic optimality of low-rank SVD approximation to the statistical optimality of PCA.

---

## SVD-Based PCA Implementation

```python
import numpy as np

def pca_svd(X, n_components):
    """
    PCA via SVD — more numerically stable than eigendecomposition.

    Args:
        X: Data matrix [n_samples, n_features]
        n_components: Number of principal components

    Returns:
        components: Principal directions [n_features, n_components]
        explained_variance: Variance per component [n_components]
        scores: Projected data [n_samples, n_components]
    """
    # Center data
    mean = X.mean(axis=0)
    X_centered = X - mean

    # Compact SVD
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # Principal components: first k rows of Vt (columns of V)
    components = Vt[:n_components].T

    # Eigenvalues from singular values
    n = X.shape[0]
    explained_variance = (S[:n_components] ** 2) / (n - 1)

    # Scores: U_k * S_k (more efficient than X @ V_k)
    scores = U[:, :n_components] * S[:n_components]

    return components, explained_variance, scores
```

---

## Truncated SVD for Large Data

For very large matrices where only the top $k$ components are needed, **truncated SVD** avoids computing the full decomposition:

```python
from scipy.sparse.linalg import svds

def pca_truncated(X, n_components):
    """PCA using truncated SVD — efficient for large, sparse data.

    Only computes the top-k singular triplets, avoiding O(min(n,d)^3) cost.
    """
    X_centered = X - X.mean(axis=0)

    # Compute only top-k singular values/vectors
    U, S, Vt = svds(X_centered, k=n_components)

    # svds returns singular values in ascending order
    idx = np.argsort(S)[::-1]
    U, S, Vt = U[:, idx], S[idx], Vt[idx]

    components = Vt.T
    explained_variance = S ** 2 / (X.shape[0] - 1)
    scores = U * S

    return components, explained_variance, scores
```

### Randomized SVD

For even larger matrices, randomized algorithms approximate the top-$k$ SVD in $O(nd \log k)$ time:

```python
from sklearn.utils.extmath import randomized_svd

def pca_randomized(X, n_components, random_state=42):
    """PCA using randomized SVD — fast approximate computation."""
    X_centered = X - X.mean(axis=0)

    U, S, Vt = randomized_svd(X_centered, n_components=n_components,
                               random_state=random_state)

    components = Vt.T
    explained_variance = S ** 2 / (X.shape[0] - 1)
    scores = U * S

    return components, explained_variance, scores
```

The randomized approach works by projecting onto a random low-dimensional subspace, then computing an exact SVD of the projected matrix. With high probability, the approximation error is within a small factor of optimal.

---

## PyTorch Implementation

```python
import torch

def pca_svd_torch(X, n_components):
    """PCA via SVD using PyTorch (GPU-compatible)."""
    X_centered = X - X.mean(dim=0)

    # torch.linalg.svd with full_matrices=False for efficiency
    U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)

    components = Vh[:n_components].T
    explained_variance = (S[:n_components] ** 2) / (X.shape[0] - 1)
    scores = U[:, :n_components] * S[:n_components]

    return components, explained_variance, scores
```

---

## Complexity Comparison

| Method | Complexity | Memory | When to Use |
|--------|------------|--------|-------------|
| **Eigendecomposition** | $O(nd^2 + d^3)$ | $O(d^2)$ | $d \ll n$, dense data |
| **Full SVD** | $O(\min(nd^2, n^2d))$ | $O(nd)$ | General case, moderate $n, d$ |
| **Truncated SVD** | $O(ndk)$ per iteration | $O(nk + dk)$ | Large data, few components |
| **Randomized SVD** | $O(nd\log k)$ | $O(nk + dk)$ | Very large data |

### Decision Guide

The choice depends on the relationship between $n$ (samples), $d$ (features), and $k$ (components):

- **$d \ll n$, moderate $d$:** Eigendecomposition is simple and fast; the $d \times d$ covariance matrix fits in memory.
- **$d \approx n$:** Full SVD avoids forming $\boldsymbol{\Sigma}$ and is more stable.
- **$d \gg n$:** SVD on $\mathbf{X}$ directly is essential. Alternatively, eigendecompose the $n \times n$ Gram matrix.
- **$n$ and $d$ both very large, $k$ small:** Truncated or randomized SVD computes only the needed components.

---

## Numerical Stability: SVD vs. Eigendecomposition

### Why SVD is More Stable

Eigendecomposing $\boldsymbol{\Sigma} = \frac{1}{n-1}\mathbf{X}^T\mathbf{X}$ effectively **squares** the singular values: $\lambda_i = \sigma_i^2 / (n-1)$. This squaring amplifies the condition number:

$$\kappa(\boldsymbol{\Sigma}) = \frac{\lambda_1}{\lambda_d} = \left(\frac{\sigma_1}{\sigma_d}\right)^2 = \kappa(\mathbf{X})^2$$

If $\mathbf{X}$ has condition number $10^4$, then $\boldsymbol{\Sigma}$ has condition number $10^8$ — dangerously close to the limits of double-precision arithmetic ($\sim 10^{16}$). SVD applied directly to $\mathbf{X}$ avoids this squaring.

### Practical Impact

```python
# Ill-conditioned data: singular values span many orders of magnitude
X = np.random.randn(1000, 50)
X[:, -1] *= 1e-8  # One feature with tiny variance

# Eigendecomposition may give inaccurate small eigenvalues
cov = X.T @ X / (X.shape[0] - 1)
eigvals_eig = np.linalg.eigh(cov)[0]

# SVD gives more accurate small singular values
_, S, _ = np.linalg.svd(X - X.mean(axis=0), full_matrices=False)
eigvals_svd = S ** 2 / (X.shape[0] - 1)

# Compare: SVD result is typically more accurate for small eigenvalues
```

---

## Verification: SVD vs. Eigendecomposition

```python
import numpy as np

np.random.seed(42)
X = np.random.randn(200, 10)
X_centered = X - X.mean(axis=0)
k = 3

# Method 1: Eigendecomposition
cov = X_centered.T @ X_centered / (X.shape[0] - 1)
eigvals, eigvecs = np.linalg.eigh(cov)
idx = np.argsort(eigvals)[::-1]
W_eig = eigvecs[:, idx[:k]]
scores_eig = X_centered @ W_eig

# Method 2: SVD
U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
W_svd = Vt[:k].T
scores_svd = U[:, :k] * S[:k]

# Compare reconstructions (should be nearly identical)
recon_eig = scores_eig @ W_eig.T
recon_svd = scores_svd @ W_svd.T
print(f"Max reconstruction difference: {np.abs(recon_eig - recon_svd).max():.2e}")
# Typically ~1e-14 (machine precision)
```

---

## Summary

SVD is the preferred computational backend for PCA in production implementations (including scikit-learn's `PCA`). It provides identical results to eigendecomposition but with superior numerical properties and greater flexibility for large-scale data.

| Aspect | Eigendecomposition | SVD |
|--------|-------------------|-----|
| **Operates on** | Covariance matrix $\boldsymbol{\Sigma}$ | Data matrix $\mathbf{X}$ |
| **Stability** | Squares condition number | Preserves condition number |
| **Efficiency ($d \gg n$)** | $O(d^3)$ — impractical | $O(n^2 d)$ or truncated |
| **Truncation** | Must compute all eigenvalues | Can compute only top-$k$ |
| **Low-rank approx.** | Via reconstruction | Direct from truncated SVD |
| **Software** | `np.linalg.eigh` | `np.linalg.svd`, `scipy.sparse.linalg.svds` |
