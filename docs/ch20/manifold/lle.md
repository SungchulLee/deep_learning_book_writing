# Locally Linear Embedding

Preserving local reconstruction weights for nonlinear dimensionality reduction.

---

## Overview

**Locally Linear Embedding (LLE)** assumes that each data point can be approximated as a **linear combination of its neighbors**. It finds low-dimensional coordinates that preserve these local linear relationships — without requiring distance computations between all pairs of points.

---

## Core Intuition

On a smooth manifold, any sufficiently small neighborhood looks approximately **flat** (like a tangent plane). LLE exploits this:

1. Describe each point as a weighted sum of its neighbors (local linearity)
2. Find low-dimensional coordinates that preserve those same weights

The key insight is that the **reconstruction weights** encode the local geometry of the manifold and are invariant to rotation, scaling, and translation.

---

## Algorithm

### Step 1: Find Neighbors

For each point $\mathbf{x}_i$, identify its $k$ nearest neighbors $\mathcal{N}(i)$.

### Step 2: Compute Reconstruction Weights

For each point, find weights $w_{ij}$ that best reconstruct $\mathbf{x}_i$ from its neighbors:

$$\min_{w_{ij}} \left\|\mathbf{x}_i - \sum_{j \in \mathcal{N}(i)} w_{ij} \mathbf{x}_j\right\|^2 \quad \text{s.t.} \quad \sum_{j \in \mathcal{N}(i)} w_{ij} = 1$$

The constraint $\sum_j w_{ij} = 1$ ensures translation invariance.

### Step 3: Find Embedding

Find low-dimensional coordinates $\mathbf{Y}$ that minimize:

$$\Phi(\mathbf{Y}) = \sum_{i=1}^n \left\|\mathbf{y}_i - \sum_{j \in \mathcal{N}(i)} w_{ij} \mathbf{y}_j\right\|^2$$

using the **fixed** weights from Step 2, subject to:

- $\sum_i \mathbf{y}_i = \mathbf{0}$ (centering)
- $\frac{1}{n}\mathbf{Y}^T\mathbf{Y} = \mathbf{I}$ (unit covariance)

---

## Solving for Weights

### Local Gram Matrix

For point $i$ with neighbors $\mathcal{N}(i) = \{j_1, \ldots, j_k\}$, define the local covariance:

$$G_{jl}^{(i)} = (\mathbf{x}_i - \mathbf{x}_j)^T (\mathbf{x}_i - \mathbf{x}_l) \quad j, l \in \mathcal{N}(i)$$

### Closed-Form Solution

$$w_{ij} = \frac{\sum_l (G^{(i)})^{-1}_{jl}}{\sum_{j'l'} (G^{(i)})^{-1}_{j'l'}}$$

```python
import numpy as np

def compute_weights(X, neighbors, reg=1e-3):
    """
    Compute LLE reconstruction weights.
    
    Args:
        X: Data matrix [n, d]
        neighbors: Neighbor indices [n, k]
        reg: Regularization for singular local Gram matrices
    
    Returns:
        W: Weight matrix [n, n] (sparse)
    """
    n, k = X.shape[0], neighbors.shape[1]
    W = np.zeros((n, n))
    
    for i in range(n):
        # Neighbor differences
        Z = X[neighbors[i]] - X[i]    # [k, d]
        
        # Local Gram matrix
        G = Z @ Z.T                    # [k, k]
        
        # Regularize
        G += reg * np.eye(k) * np.trace(G)
        
        # Solve for weights (sum-to-one constraint)
        w = np.linalg.solve(G, np.ones(k))
        w /= w.sum()
        
        W[i, neighbors[i]] = w
    
    return W
```

---

## Solving for Embedding

### Matrix Formulation

Define the sparse weight matrix $\mathbf{W}$ (with $w_{ij} = 0$ for non-neighbors). The embedding cost is:

$$\Phi(\mathbf{Y}) = \|\mathbf{Y} - \mathbf{W}\mathbf{Y}\|_F^2 = \text{tr}(\mathbf{Y}^T \mathbf{M} \mathbf{Y})$$

where $\mathbf{M} = (\mathbf{I} - \mathbf{W})^T (\mathbf{I} - \mathbf{W})$.

### Eigenvalue Problem

Minimizing $\text{tr}(\mathbf{Y}^T \mathbf{M} \mathbf{Y})$ subject to $\mathbf{Y}^T\mathbf{Y} = n\mathbf{I}$ and $\mathbf{Y}^T\mathbf{1} = \mathbf{0}$ leads to:

$$\mathbf{M} \mathbf{y} = \lambda \mathbf{y}$$

The embedding coordinates are the **bottom** $d+1$ eigenvectors of $\mathbf{M}$ (excluding the constant eigenvector with $\lambda = 0$).

```python
def lle_embedding(W, n_components=2):
    """
    Compute LLE embedding from weight matrix.
    
    Args:
        W: Weight matrix [n, n]
        n_components: Embedding dimensionality
    
    Returns:
        Y: Embedded coordinates [n, n_components]
    """
    n = W.shape[0]
    
    # Cost matrix M = (I - W)^T (I - W)
    I_W = np.eye(n) - W
    M = I_W.T @ I_W
    
    # Symmetrize (numerical stability)
    M = (M + M.T) / 2
    
    # Smallest eigenvectors (skip first: constant vector)
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    
    # Bottom eigenvectors (eigenvalues already ascending from eigh)
    # Skip index 0 (near-zero eigenvalue, constant eigenvector)
    Y = eigenvectors[:, 1:n_components + 1]
    
    return Y
```

---

## Full Implementation

```python
import numpy as np
from sklearn.neighbors import NearestNeighbors

def lle(X, n_components=2, n_neighbors=10, reg=1e-3):
    """
    Locally Linear Embedding from scratch.
    
    Args:
        X: Data matrix [n_samples, n_features]
        n_components: Embedding dimensionality
        n_neighbors: Number of neighbors per point
        reg: Regularization strength
    
    Returns:
        Y: Embedded coordinates [n, n_components]
        reconstruction_error: Final embedding cost
    """
    n = X.shape[0]
    
    # Step 1: Find neighbors
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(X)
    _, indices = nn.kneighbors(X)
    
    # Step 2: Compute weights
    W = compute_weights(X, indices, reg=reg)
    
    # Step 3: Embedding
    Y = lle_embedding(W, n_components)
    
    # Reconstruction error
    I_W = np.eye(n) - W
    error = np.trace(Y.T @ I_W.T @ I_W @ Y)
    
    return Y, error
```

---

## Swiss Roll Example

```python
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll

X, color = make_swiss_roll(n_samples=1500, noise=0.3, random_state=42)

Y_lle, error = lle(X, n_components=2, n_neighbors=12)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = fig.add_subplot(121, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap='Spectral', s=5)
ax.set_title('Swiss Roll (3D)')

axes[1].scatter(Y_lle[:, 0], Y_lle[:, 1], c=color, cmap='Spectral', s=5)
axes[1].set_title(f'LLE (2D), error={error:.4f}')

plt.tight_layout()
plt.show()
```

---

## scikit-learn Interface

```python
from sklearn.manifold import LocallyLinearEmbedding

# Standard LLE
lle_sk = LocallyLinearEmbedding(
    n_components=2,
    n_neighbors=10,
    method='standard',
    random_state=42
)
Y = lle_sk.fit_transform(X)
print(f"Reconstruction error: {lle_sk.reconstruction_error_:.6f}")

# Modified LLE (uses multiple weight vectors)
lle_mod = LocallyLinearEmbedding(
    n_components=2,
    n_neighbors=10,
    method='modified',
    random_state=42
)
Y_mod = lle_mod.fit_transform(X)

# Hessian LLE (uses local Hessian estimates)
lle_hess = LocallyLinearEmbedding(
    n_components=2,
    n_neighbors=10,
    method='hessian',
    random_state=42
)
Y_hess = lle_hess.fit_transform(X)
```

### LLE Variants

| Variant | `method=` | Key Idea |
|---------|-----------|----------|
| **Standard** | `'standard'` | Reconstruction weights |
| **Modified** | `'modified'` | Multiple weight vectors per neighborhood |
| **Hessian** | `'hessian'` | Local Hessian estimator (more robust) |
| **LTSA** | `'ltsa'` | Local tangent space alignment |

---

## Choosing n_neighbors

| $k$ too small | $k$ too large |
|---------------|---------------|
| Insufficient local geometry | Violates local linearity |
| Degenerate weight solutions | Captures non-local structure |
| Fragmented embedding | Over-smoothed embedding |

### Constraint

LLE requires $k < d$ (number of neighbors less than ambient dimension) for the local Gram matrix to be well-conditioned. In practice, $k$ is often chosen as $d + 1$ to $2d$.

For the modified LLE variant, $k > n\_components$ is required, and $k > n\_components \cdot (n\_components + 3) / 2$ is recommended.

---

## Application: Factor Model Residual Structure

LLE can reveal nonlinear structure in factor model residuals that linear methods miss:

```python
import numpy as np
import matplotlib.pyplot as plt

def residual_lle(returns, n_factors=3, n_components=2, 
                 n_neighbors=15):
    """
    Apply LLE to factor model residuals to find
    nonlinear structure missed by PCA factors.
    
    Args:
        returns: Return matrix [n_periods, n_assets]
        n_factors: Number of PCA factors to remove
        n_components: LLE embedding dimension
        n_neighbors: LLE neighborhood size
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import LocallyLinearEmbedding
    
    # Extract PCA factors
    pca = PCA(n_components=n_factors)
    factors = pca.fit_transform(returns)
    reconstructed = pca.inverse_transform(factors)
    residuals = returns - reconstructed
    
    # LLE on residual covariance structure
    # Treat each asset as a point in "residual behavior space"
    # Transpose: each asset is a sample, each date is a feature
    lle = LocallyLinearEmbedding(
        n_components=n_components,
        n_neighbors=n_neighbors,
        method='standard'
    )
    Y = lle.fit_transform(residuals.T)  # [n_assets, 2]
    
    return Y


# Example
np.random.seed(42)
n_periods, n_assets = 500, 50
market = np.random.randn(n_periods)
returns = np.outer(market, np.random.randn(n_assets))
returns += 0.3 * np.random.randn(n_periods, n_assets)

Y = residual_lle(returns)
plt.scatter(Y[:, 0], Y[:, 1], s=30, alpha=0.7)
plt.xlabel('LLE 1')
plt.ylabel('LLE 2')
plt.title('Asset Residual Structure (LLE)')
plt.tight_layout()
plt.show()
```

---

## LLE vs Isomap

| Aspect | Isomap | LLE |
|--------|--------|-----|
| **Preserves** | Global geodesic distances | Local linear reconstruction |
| **Approach** | Distance → MDS | Weights → eigen-problem |
| **Global structure** | Good | Often distorted |
| **Local structure** | Good | Excellent |
| **Non-convex manifolds** | Fails | Handles better |
| **Complexity** | $O(n^3)$ | $O(n^2 k^3)$ |

---

## Limitations

| Limitation | Consequence |
|------------|-------------|
| **No out-of-sample** | Cannot embed new points (standard LLE) |
| **Sensitive to $k$** | Wrong $k$ breaks local linearity assumption |
| **Uneven sampling** | Sparse regions get poor weight estimates |
| **Global distortion** | Only local relationships preserved |
| **Regularization** | Singular local Gram matrices require regularization |

---

## Complexity Analysis

| Step | Complexity |
|------|------------|
| k-NN search | $O(n^2 d)$ or $O(nd \log n)$ |
| Weight computation | $O(nk^3)$ |
| Eigendecomposition of $\mathbf{M}$ | $O(n^3)$ or $O(n^2 k)$ sparse |
| **Total** | $O(n^2 d + n^2 k)$ |

---

## Summary

| Concept | Key Point |
|---------|-----------|
| **Core idea** | Preserve local linear reconstruction weights |
| **Step 1** | Find $k$ nearest neighbors |
| **Step 2** | Solve constrained least squares for weights |
| **Step 3** | Eigen-problem on $\mathbf{M} = (\mathbf{I}-\mathbf{W})^T(\mathbf{I}-\mathbf{W})$ |
| **Strength** | Simple, local geometry preserved |
| **Weakness** | No global structure, sensitive to $k$ |
| **Variants** | Modified LLE, Hessian LLE, LTSA |

---

## What's Next

LLE and Isomap are deterministic spectral methods. **t-SNE** introduces a fundamentally different, probabilistic approach: modeling pairwise similarities as probability distributions and minimizing KL divergence between high-dimensional and low-dimensional distributions.
