# Eigendecomposition for PCA

Computing principal components via the spectral decomposition of the covariance matrix.

---

## Overview

The most direct way to compute PCA is through the **eigendecomposition** of the sample covariance matrix. Since the covariance matrix is real symmetric and positive semi-definite, its eigenvalues are non-negative and its eigenvectors form an orthonormal basis — properties that give PCA its clean geometric interpretation.

---

## The Covariance Matrix

### Definition

For centered data $\mathbf{X} \in \mathbb{R}^{n \times d}$ (each row is an observation with $\bar{\mathbf{x}} = \mathbf{0}$), the sample covariance matrix is:

$$\boldsymbol{\Sigma} = \frac{1}{n-1}\mathbf{X}^T\mathbf{X} \in \mathbb{R}^{d \times d}$$

Entry $(j, l)$ of $\boldsymbol{\Sigma}$ is the sample covariance between features $j$ and $l$:

$$\Sigma_{jl} = \frac{1}{n-1}\sum_{i=1}^n x_j^{(i)} x_l^{(i)}$$

Diagonal entries $\Sigma_{jj}$ are feature variances; off-diagonal entries measure pairwise linear dependence.

### Properties

$\boldsymbol{\Sigma}$ has three key properties that make PCA tractable:

**Symmetry.** $\boldsymbol{\Sigma} = \boldsymbol{\Sigma}^T$ since $(\mathbf{X}^T\mathbf{X})^T = \mathbf{X}^T\mathbf{X}$.

**Positive semi-definiteness.** For any $\mathbf{v} \in \mathbb{R}^d$:

$$\mathbf{v}^T \boldsymbol{\Sigma} \mathbf{v} = \frac{1}{n-1}\mathbf{v}^T \mathbf{X}^T \mathbf{X} \mathbf{v} = \frac{1}{n-1}\|\mathbf{X}\mathbf{v}\|^2 \geq 0$$

This ensures all eigenvalues are non-negative, which is essential since eigenvalues represent variances.

**Rank.** $\operatorname{rank}(\boldsymbol{\Sigma}) \leq \min(n-1, d)$. When $n < d$ (more features than observations), $\boldsymbol{\Sigma}$ is rank-deficient with at least $d - n + 1$ zero eigenvalues.

---

## Spectral Decomposition

### The Eigenvalue Problem

The eigendecomposition of $\boldsymbol{\Sigma}$ finds scalars $\lambda$ and non-zero vectors $\mathbf{v}$ satisfying:

$$\boldsymbol{\Sigma}\mathbf{v}_i = \lambda_i \mathbf{v}_i, \quad i = 1, \ldots, d$$

Because $\boldsymbol{\Sigma}$ is symmetric, the **spectral theorem** guarantees:

1. All eigenvalues $\lambda_i$ are **real** (and non-negative by PSD)
2. Eigenvectors corresponding to distinct eigenvalues are **orthogonal**
3. There exists an orthonormal eigenbasis for $\mathbb{R}^d$

### Matrix Form

$$\boldsymbol{\Sigma} = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^T$$

where:

- $\mathbf{V} = [\mathbf{v}_1, \ldots, \mathbf{v}_d] \in \mathbb{R}^{d \times d}$ is orthogonal ($\mathbf{V}^T\mathbf{V} = \mathbf{V}\mathbf{V}^T = \mathbf{I}$)
- $\boldsymbol{\Lambda} = \operatorname{diag}(\lambda_1, \ldots, \lambda_d)$ with $\lambda_1 \geq \cdots \geq \lambda_d \geq 0$

### Interpretation for PCA

Each eigenpair $(\lambda_i, \mathbf{v}_i)$ has a direct statistical meaning:

- $\mathbf{v}_i$ defines the $i$-th **principal direction** — a unit vector in feature space
- $\lambda_i$ is the **variance** of the data projected onto $\mathbf{v}_i$: $\operatorname{Var}(\mathbf{X}\mathbf{v}_i) = \lambda_i$
- The eigenvectors form an orthonormal coordinate system aligned with the data's principal axes

---

## PCA via Eigendecomposition: Algorithm

```python
import numpy as np

def pca_eigen(X, n_components):
    """
    PCA using eigendecomposition of the covariance matrix.

    Args:
        X: Data matrix [n_samples, n_features]
        n_components: Number of principal components to retain

    Returns:
        components: Principal directions [n_features, n_components]
        explained_variance: Variance per component [n_components]
        scores: Projected data [n_samples, n_components]
    """
    # Step 1: Center the data
    mean = X.mean(axis=0)
    X_centered = X - mean

    # Step 2: Compute covariance matrix
    n = X.shape[0]
    cov = X_centered.T @ X_centered / (n - 1)

    # Step 3: Eigendecomposition (eigh for symmetric matrices)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Step 4: Sort by eigenvalue (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Step 5: Select top-k components
    components = eigenvectors[:, :n_components]
    explained_variance = eigenvalues[:n_components]

    # Step 6: Project data onto principal subspace
    scores = X_centered @ components

    return components, explained_variance, scores
```

### Step-by-Step Walkthrough

**Step 1: Centering.** Subtract the column means so that $\frac{1}{n}\sum_i \mathbf{x}^{(i)} = \mathbf{0}$. This is required because the covariance formula assumes zero mean.

**Step 2: Covariance.** Form $\boldsymbol{\Sigma} = \frac{1}{n-1}\mathbf{X}_c^T\mathbf{X}_c$. This is an $O(nd^2)$ matrix multiplication producing a $d \times d$ symmetric matrix.

**Step 3: Eigendecomposition.** Use `eigh` (specialized for symmetric matrices) rather than `eig` (for general matrices). `eigh` is more numerically stable and returns real eigenvalues guaranteed to be sorted.

**Step 4: Sorting.** `np.linalg.eigh` returns eigenvalues in ascending order. Reverse to get descending order (largest variance first).

**Step 5: Truncation.** Keep only the top $k$ eigenvectors and eigenvalues.

**Step 6: Projection.** Multiply centered data by the loading matrix to obtain scores.

---

## Numerical Considerations

### Always Use `eigh` for Symmetric Matrices

```python
# Correct: eigh exploits symmetry for stability and speed
eigenvalues, eigenvectors = np.linalg.eigh(cov)

# Avoid: eig may return complex values due to numerical noise
eigenvalues, eigenvectors = np.linalg.eig(cov)
```

`np.linalg.eigh` uses the symmetric QR algorithm (or divide-and-conquer), which guarantees real eigenvalues and runs roughly $3\times$ faster than `eig` for symmetric inputs. The general `eig` can return small imaginary components due to floating-point asymmetry in $\boldsymbol{\Sigma}$.

### Handling Near-Zero Eigenvalues

When eigenvalues are very small (e.g., $10^{-15}$), they may be slightly negative due to floating-point arithmetic. Two remedies:

**Clipping:** Set negative eigenvalues to zero.

```python
eigenvalues = np.maximum(eigenvalues, 0)
```

**Regularization:** Add a small ridge to the diagonal before decomposition.

```python
cov_reg = cov + epsilon * np.eye(d)  # epsilon ≈ 1e-10
```

Regularization also helps when $\boldsymbol{\Sigma}$ is singular (rank-deficient), which occurs whenever $n \leq d$.

### Ensuring Consistent Sign Convention

Eigenvectors are defined only up to sign ($\mathbf{v}$ and $-\mathbf{v}$ are both valid). For reproducibility, enforce a convention such as making the largest-magnitude element of each eigenvector positive:

```python
def fix_sign(eigenvectors):
    """Enforce consistent eigenvector sign convention."""
    max_abs_idx = np.argmax(np.abs(eigenvectors), axis=0)
    signs = np.sign(eigenvectors[max_abs_idx, range(eigenvectors.shape[1])])
    return eigenvectors * signs
```

---

## PyTorch Implementation

```python
import torch

def pca_eigen_torch(X, n_components):
    """PCA via eigendecomposition using PyTorch."""
    X_centered = X - X.mean(dim=0)
    n = X.shape[0]
    cov = X_centered.T @ X_centered / (n - 1)

    # torch.linalg.eigh returns eigenvalues in ascending order
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)

    # Select top-k (last k columns for ascending order)
    components = eigenvectors[:, -n_components:].flip(dims=[1])
    explained_variance = eigenvalues[-n_components:].flip(dims=[0])

    scores = X_centered @ components
    return components, explained_variance, scores
```

---

## Computational Complexity

| Step | Operation | Complexity |
|------|-----------|------------|
| Centering | Column means + subtraction | $O(nd)$ |
| Covariance matrix | $\mathbf{X}^T\mathbf{X}$ | $O(nd^2)$ |
| Eigendecomposition | Symmetric eigenproblem | $O(d^3)$ |
| Projection | $\mathbf{X}\mathbf{W}$ | $O(ndk)$ |
| **Total** | | $O(nd^2 + d^3)$ |

The eigendecomposition approach is efficient when $d \ll n$ (few features, many observations). The $d^3$ eigendecomposition cost dominates when $d$ is large.

### When Eigendecomposition Becomes Impractical

For high-dimensional data ($d > n$, common in genomics or NLP), forming the $d \times d$ covariance matrix is prohibitively expensive. In this regime, **SVD** applied directly to $\mathbf{X}$ is preferred (see the SVD section), or the **dual trick** can be used: eigendecompose the smaller $n \times n$ Gram matrix $\mathbf{X}\mathbf{X}^T$ and recover the principal components via $\mathbf{v}_i = \frac{1}{\sigma_i}\mathbf{X}^T\mathbf{u}_i$.

---

## Verification: Eigendecomposition vs. sklearn

```python
from sklearn.decomposition import PCA
import numpy as np

# Generate test data
np.random.seed(42)
X = np.random.randn(500, 10)

# Our implementation
W, var, scores = pca_eigen(X, n_components=3)

# sklearn
pca = PCA(n_components=3)
scores_sk = pca.fit_transform(X)

# Compare explained variance (should match)
print("Our variance: ", var)
print("sklearn var:  ", pca.explained_variance_)

# Compare reconstruction error
X_recon_ours = scores @ W.T + X.mean(axis=0)
X_recon_sk = pca.inverse_transform(scores_sk)
print("Max recon diff:", np.abs(X_recon_ours - X_recon_sk).max())
```

---

## Summary

Eigendecomposition is the most conceptually transparent route to PCA: diagonalize the covariance matrix and read off the principal directions and variances. Its main limitations are the $O(d^3)$ cost and numerical sensitivity when $d$ is large or $\boldsymbol{\Sigma}$ is ill-conditioned. For these cases, SVD provides a more robust and often more efficient alternative.

| Aspect | Detail |
|--------|--------|
| **Input** | Covariance matrix $\boldsymbol{\Sigma} \in \mathbb{R}^{d \times d}$ |
| **Output** | Eigenvectors (principal directions) + eigenvalues (variances) |
| **Numerical method** | Use `eigh` for symmetric matrices |
| **Complexity** | $O(nd^2 + d^3)$ |
| **Best regime** | $d \ll n$ (low-dimensional, many samples) |
| **Limitation** | Expensive for $d > n$; prefer SVD in that regime |
