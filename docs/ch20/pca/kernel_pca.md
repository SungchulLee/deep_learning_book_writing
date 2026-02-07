# Kernel PCA

Nonlinear dimensionality reduction via the kernel trick.

---

## Overview

Classical PCA is limited to finding linear subspaces. **Kernel PCA** extends PCA to capture nonlinear structure by implicitly mapping data into a high-dimensional (possibly infinite-dimensional) feature space and performing linear PCA there. The **kernel trick** makes this computationally feasible: we never explicitly compute the feature-space coordinates, only inner products between mapped data points.

Kernel PCA bridges the gap between linear PCA and fully nonlinear methods like autoencoders, providing nonlinear dimensionality reduction with a closed-form solution.

---

## Motivation

### The Limitation of Linearity

PCA finds the subspace that maximizes variance, but this subspace is constrained to be **flat** (a hyperplane). For data lying on a curved manifold — such as the Swiss roll or concentric circles — the best flat projection loses essential structure.

### The Feature Map Idea

Instead of working in the original space $\mathbb{R}^d$, map data to a higher-dimensional feature space $\mathcal{F}$ via $\phi: \mathbb{R}^d \to \mathcal{F}$, then perform linear PCA in $\mathcal{F}$:

$$\mathbf{x} \in \mathbb{R}^d \xrightarrow{\phi} \phi(\mathbf{x}) \in \mathcal{F} \xrightarrow{\text{PCA}} \text{principal components in } \mathcal{F}$$

A linear subspace in a high-dimensional feature space corresponds to a **nonlinear** manifold in the original space. The challenge is that $\mathcal{F}$ may be extremely high-dimensional or even infinite-dimensional, making explicit computation impossible.

---

## The Kernel Trick

### Kernel Function

A **kernel function** $\kappa: \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}$ computes the inner product in feature space without explicitly computing the feature map:

$$\kappa(\mathbf{x}, \mathbf{x}') = \langle \phi(\mathbf{x}), \phi(\mathbf{x}') \rangle_{\mathcal{F}}$$

### Common Kernels

| Kernel | Formula | Feature Space |
|--------|---------|---------------|
| **Linear** | $\kappa(\mathbf{x}, \mathbf{x}') = \mathbf{x}^T\mathbf{x}'$ | $\mathbb{R}^d$ (identity map) |
| **Polynomial** | $\kappa(\mathbf{x}, \mathbf{x}') = (\mathbf{x}^T\mathbf{x}' + c)^p$ | $\mathbb{R}^{\binom{d+p}{p}}$ (finite) |
| **RBF (Gaussian)** | $\kappa(\mathbf{x}, \mathbf{x}') = \exp\!\left(-\frac{\|\mathbf{x} - \mathbf{x}'\|^2}{2\gamma^2}\right)$ | $\ell^2$ (infinite-dimensional) |
| **Sigmoid** | $\kappa(\mathbf{x}, \mathbf{x}') = \tanh(\alpha \, \mathbf{x}^T\mathbf{x}' + c)$ | (not always valid PD kernel) |

The **RBF kernel** is particularly powerful because its implicit feature space is infinite-dimensional, allowing it to approximate any continuous nonlinear structure.

### Mercer's Condition

A function $\kappa$ is a valid kernel if and only if it is **positive semi-definite**: for any set of points $\{\mathbf{x}^{(1)}, \ldots, \mathbf{x}^{(n)}\}$, the Gram matrix $K_{ij} = \kappa(\mathbf{x}^{(i)}, \mathbf{x}^{(j)})$ is positive semi-definite. This ensures the existence of a feature map $\phi$ such that $\kappa$ computes inner products in $\mathcal{F}$.

---

## Derivation

### PCA in Feature Space

Let $\boldsymbol{\Phi} = [\phi(\mathbf{x}^{(1)}), \ldots, \phi(\mathbf{x}^{(n)})]^T$ denote the $n \times D$ matrix of feature-space representations (where $D = \dim(\mathcal{F})$ may be infinite). Assuming centered features ($\frac{1}{n}\sum_i \phi(\mathbf{x}^{(i)}) = \mathbf{0}$), the covariance in feature space is:

$$\mathbf{C}_\phi = \frac{1}{n}\boldsymbol{\Phi}^T\boldsymbol{\Phi}$$

PCA seeks eigenvectors $\mathbf{v}$ satisfying $\mathbf{C}_\phi \mathbf{v} = \lambda \mathbf{v}$.

### The Representer Theorem

Since $\mathbf{C}_\phi \mathbf{v} = \frac{1}{n}\boldsymbol{\Phi}^T(\boldsymbol{\Phi}\mathbf{v}) = \lambda\mathbf{v}$, any eigenvector with $\lambda > 0$ must lie in the span of the mapped data:

$$\mathbf{v} = \sum_{i=1}^n \alpha_i \, \phi(\mathbf{x}^{(i)}) = \boldsymbol{\Phi}^T\boldsymbol{\alpha}$$

for some coefficient vector $\boldsymbol{\alpha} \in \mathbb{R}^n$.

### Kernelized Eigenproblem

Substituting $\mathbf{v} = \boldsymbol{\Phi}^T\boldsymbol{\alpha}$ into $\mathbf{C}_\phi\mathbf{v} = \lambda\mathbf{v}$ and left-multiplying by $\boldsymbol{\Phi}$:

$$\frac{1}{n}\boldsymbol{\Phi}\boldsymbol{\Phi}^T\boldsymbol{\Phi}\boldsymbol{\Phi}^T\boldsymbol{\alpha} = \lambda\boldsymbol{\Phi}\boldsymbol{\Phi}^T\boldsymbol{\alpha}$$

Defining the **kernel (Gram) matrix** $\mathbf{K} = \boldsymbol{\Phi}\boldsymbol{\Phi}^T$ where $K_{ij} = \kappa(\mathbf{x}^{(i)}, \mathbf{x}^{(j)})$:

$$\frac{1}{n}\mathbf{K}^2\boldsymbol{\alpha} = \lambda\mathbf{K}\boldsymbol{\alpha}$$

For non-degenerate $\mathbf{K}$, this simplifies to:

$$\mathbf{K}\boldsymbol{\alpha} = n\lambda\boldsymbol{\alpha}$$

This is an **$n \times n$ eigenvalue problem** — its size depends on the number of samples $n$, not the (possibly infinite) dimension of $\mathcal{F}$.

### Normalization

The feature-space eigenvectors must satisfy $\|\mathbf{v}\| = 1$:

$$\|\mathbf{v}\|^2 = \boldsymbol{\alpha}^T\mathbf{K}\boldsymbol{\alpha} = n\lambda \|\boldsymbol{\alpha}\|^2 = 1$$

Therefore normalize: $\boldsymbol{\alpha} \leftarrow \boldsymbol{\alpha} / \sqrt{n\lambda}$.

### Projection

The projection of a point $\mathbf{x}$ onto the $j$-th kernel principal component:

$$z_j = \langle \mathbf{v}_j, \phi(\mathbf{x}) \rangle = \sum_{i=1}^n \alpha_{ji} \, \kappa(\mathbf{x}^{(i)}, \mathbf{x}) = \boldsymbol{\alpha}_j^T \mathbf{k}_\mathbf{x}$$

where $\mathbf{k}_\mathbf{x} = [\kappa(\mathbf{x}^{(1)}, \mathbf{x}), \ldots, \kappa(\mathbf{x}^{(n)}, \mathbf{x})]^T$.

---

## Centering in Feature Space

PCA requires centered data. Since we cannot explicitly center in $\mathcal{F}$, we center the kernel matrix:

$$\tilde{\mathbf{K}} = \mathbf{H}\mathbf{K}\mathbf{H}$$

where $\mathbf{H} = \mathbf{I}_n - \frac{1}{n}\mathbf{1}\mathbf{1}^T$ is the centering matrix. Expanding:

$$\tilde{\mathbf{K}} = \mathbf{K} - \frac{1}{n}\mathbf{1}\mathbf{1}^T\mathbf{K} - \frac{1}{n}\mathbf{K}\mathbf{1}\mathbf{1}^T + \frac{1}{n^2}\mathbf{1}\mathbf{1}^T\mathbf{K}\mathbf{1}\mathbf{1}^T$$

For a new test point, the centered kernel vector is:

$$\tilde{k}_i = \kappa(\mathbf{x}^{(i)}, \mathbf{x}_*) - \frac{1}{n}\sum_{j=1}^n \kappa(\mathbf{x}^{(j)}, \mathbf{x}_*) - \frac{1}{n}\sum_{j=1}^n \kappa(\mathbf{x}^{(i)}, \mathbf{x}^{(j)}) + \frac{1}{n^2}\sum_{j,l} \kappa(\mathbf{x}^{(j)}, \mathbf{x}^{(l)})$$

---

## Implementation

```python
import numpy as np
from scipy.spatial.distance import cdist

class KernelPCA:
    """Kernel PCA for nonlinear dimensionality reduction.

    Performs PCA in an implicit high-dimensional feature space
    defined by the chosen kernel function.

    Args:
        n_components: Number of principal components
        kernel: 'rbf', 'poly', or 'linear'
        gamma: RBF kernel bandwidth
        degree: Polynomial kernel degree
        coef0: Polynomial kernel offset
    """

    def __init__(self, n_components, kernel='rbf', gamma=1.0,
                 degree=3, coef0=1.0):
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0

    def _compute_kernel(self, X, Y=None):
        """Compute kernel matrix K[i,j] = kappa(X[i], Y[j])."""
        if Y is None:
            Y = X
        if self.kernel == 'rbf':
            dists = cdist(X, Y, metric='sqeuclidean')
            return np.exp(-dists / (2 * self.gamma ** 2))
        elif self.kernel == 'poly':
            return (X @ Y.T + self.coef0) ** self.degree
        elif self.kernel == 'linear':
            return X @ Y.T
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def _center_kernel(self, K):
        """Center kernel matrix: K_centered = H K H."""
        n = K.shape[0]
        one_n = np.ones((n, n)) / n
        return K - one_n @ K - K @ one_n + one_n @ K @ one_n

    def fit(self, X):
        """Fit Kernel PCA on training data.

        Computes the kernel matrix, centers it, and finds
        the top eigenvectors (dual coefficients alpha).
        """
        self.X_train = X.copy()
        n = X.shape[0]

        # Compute and center kernel matrix
        K = self._compute_kernel(X)
        K_centered = self._center_kernel(K)

        # Eigendecomposition of centered kernel matrix
        eigenvalues, eigenvectors = np.linalg.eigh(K_centered)

        # Sort descending, select top components
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx[:self.n_components]]
        eigenvectors = eigenvectors[:, idx[:self.n_components]]

        # Normalize: alpha_j / sqrt(n * lambda_j)
        self.alphas = eigenvectors / np.sqrt(
            np.maximum(eigenvalues, 1e-10)
        )
        self.eigenvalues = eigenvalues / n

        # Cache statistics for test-time centering
        self.K_train = K
        self.K_train_col_mean = K.mean(axis=0)
        self.K_train_mean = K.mean()

        return self

    def transform(self, X):
        """Project new data onto kernel principal components.

        Computes kernels between test and training points,
        centers appropriately, and projects.
        """
        K_test = self._compute_kernel(X, self.X_train)

        # Center test kernel matrix
        K_test_centered = (
            K_test
            - K_test.mean(axis=1, keepdims=True)
            - self.K_train_col_mean[np.newaxis, :]
            + self.K_train_mean
        )

        return K_test_centered @ self.alphas

    def fit_transform(self, X):
        self.fit(X)
        K_centered = self._center_kernel(self.K_train)
        return K_centered @ self.alphas
```

---

## Example: Concentric Circles

A classic example where linear PCA fails but Kernel PCA succeeds:

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles
from sklearn.decomposition import PCA, KernelPCA

# Generate concentric circles
X, y = make_circles(n_samples=500, factor=0.3, noise=0.05,
                     random_state=42)

# Linear PCA (cannot separate circles)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Kernel PCA with RBF kernel (separates circles)
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=10)
X_kpca = kpca.fit_transform(X)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=20)
axes[0].set_title('Original Data')

axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', s=20)
axes[1].set_title('Linear PCA')

axes[2].scatter(X_kpca[:, 0], X_kpca[:, 1], c=y, cmap='coolwarm', s=20)
axes[2].set_title('Kernel PCA (RBF)')

plt.tight_layout()
plt.show()
```

In this example, linear PCA projects both circles onto overlapping ranges, while the RBF kernel PCA maps them to separable regions.

---

## Hyperparameter Selection

### Kernel Bandwidth ($\gamma$ for RBF)

The RBF bandwidth $\gamma$ controls the "locality" of the kernel:

- **Small $\gamma$** (wide kernel): Emphasizes global structure, approaches linear PCA
- **Large $\gamma$** (narrow kernel): Emphasizes local structure, risks overfitting

A practical heuristic is to set $\gamma$ based on the median pairwise distance:

```python
from scipy.spatial.distance import pdist

def median_heuristic(X):
    """Set RBF gamma to median pairwise distance."""
    dists = pdist(X, metric='euclidean')
    return np.median(dists)
```

### Grid Search with Reconstruction Error

Since Kernel PCA lacks a direct reconstruction in the original space, one approach is to use the kernel alignment or a downstream task metric:

```python
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# Use classification accuracy as a proxy for good projections
pipe = Pipeline([
    ('kpca', KernelPCA(kernel='rbf')),
    ('clf', SVC())
])

param_grid = {
    'kpca__n_components': [2, 5, 10, 20],
    'kpca__gamma': [0.01, 0.1, 1.0, 10.0],
}

search = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy')
search.fit(X_train, y_train)
print(f"Best params: {search.best_params_}")
```

---

## The Pre-Image Problem

### Challenge

Kernel PCA maps data to a representation $\mathbf{z}$ in a space defined by the kernel eigenvectors. However, there is no direct inverse mapping back to the original space $\mathbb{R}^d$. Given a point $\mathbf{z}$ in the kernel PC space, finding the corresponding $\mathbf{x} \in \mathbb{R}^d$ is known as the **pre-image problem**.

### Approximate Solution

Mika et al. (1999) proposed an iterative fixed-point method. The idea is to find $\mathbf{x}$ that minimizes $\|\phi(\mathbf{x}) - \hat{\phi}\|^2$ where $\hat{\phi}$ is the reconstruction in feature space:

```python
def approximate_preimage(kpca, z, X_train, gamma, n_iter=100, lr=0.1):
    """Approximate pre-image via gradient descent (RBF kernel)."""
    # Initialize with weighted average of training points
    K_weights = np.exp(-0.5 * np.sum(z ** 2))
    x = X_train.mean(axis=0).copy()

    for _ in range(n_iter):
        k = np.exp(-np.sum((X_train - x) ** 2, axis=1)
                    / (2 * gamma ** 2))
        grad = np.sum(k[:, None] * (X_train - x), axis=0)
        x += lr * grad / (k.sum() + 1e-10)

    return x
```

This is an inherent limitation compared to autoencoders, which learn an explicit decoder.

---

## Comparison with Other Methods

| Method | Linearity | Reconstruction | Scalability | Hyperparameters |
|--------|-----------|---------------|-------------|-----------------|
| **PCA** | Linear | Exact | $O(nd^2)$ | $k$ only |
| **Kernel PCA** | Nonlinear | Approximate (pre-image) | $O(n^3)$ | $k$, kernel params |
| **Autoencoder** | Nonlinear | Exact (decoder) | $O(ndk)$ per epoch | Architecture, training |
| **t-SNE** | Nonlinear | None | $O(n^2)$ or $O(n\log n)$ | Perplexity |
| **UMAP** | Nonlinear | Approximate | $O(n^{1.14})$ | $k$, neighbors |

### When to Use Kernel PCA

Kernel PCA is most useful when nonlinear structure needs to be captured but the dataset is small enough that the $O(n^3)$ cost is acceptable (typically $n < 10{,}000$). For larger datasets, randomized approximations or neural network methods (autoencoders) are more practical.

---

## Quantitative Finance Application

In finance, nonlinear factor structures arise naturally. Interest rate term structures, for example, are dominated by three linear factors (level, slope, curvature) that PCA captures well. However, credit spreads and volatility surfaces exhibit nonlinear dependencies that Kernel PCA can model:

```python
# Volatility surface analysis
# X: [n_days, n_strikes * n_maturities] flattened vol surface
# Linear PCA captures level/skew/term structure modes
# Kernel PCA can capture smile dynamics, regime changes

from sklearn.decomposition import KernelPCA

# RBF kernel captures nonlinear regime structure
kpca = KernelPCA(n_components=5, kernel='rbf', gamma=0.1)
vol_factors = kpca.fit_transform(vol_surfaces)

# Factor 1: overall vol level (similar to PCA)
# Factors 2-5: nonlinear modes capturing smile deformation,
#              term structure twists, and regime transitions
```

---

## Summary

| Aspect | Detail |
|--------|--------|
| **Idea** | PCA in implicit high-dimensional feature space via kernel trick |
| **Eigenproblem** | $\mathbf{K}\boldsymbol{\alpha} = n\lambda\boldsymbol{\alpha}$ ($n \times n$, not $D \times D$) |
| **Projection** | $z_j = \sum_i \alpha_{ji}\kappa(\mathbf{x}^{(i)}, \mathbf{x})$ |
| **Centering** | Center the kernel matrix: $\tilde{\mathbf{K}} = \mathbf{H}\mathbf{K}\mathbf{H}$ |
| **Complexity** | $O(n^3)$ for eigendecomposition of $\mathbf{K}$ |
| **Limitation** | No direct inverse mapping (pre-image problem); $O(n^3)$ scaling |
| **Key advantage** | Nonlinear dimensionality reduction with closed-form solution |
| **vs. Autoencoders** | Simpler (no training), but less scalable and no explicit decoder |
