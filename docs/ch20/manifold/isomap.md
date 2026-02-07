# Isomap

Geodesic distance preservation for nonlinear manifold unfolding.

---

## Overview

**Isomap** (Isometric Feature Mapping) extends classical MDS by replacing Euclidean distances with **geodesic distances** — shortest paths along the data manifold. This simple modification enables recovery of the intrinsic low-dimensional structure of curved manifolds like the Swiss roll.

---

## Motivation

Classical MDS preserves Euclidean (straight-line) distances. On a curved manifold, the Euclidean distance between two points can be much shorter than the **geodesic distance** (distance along the surface):

```
Euclidean:  A -------- B     (short, cuts through manifold)
Geodesic:   A ~~~~~~~~~~ B   (long, follows the surface)
```

On a Swiss roll, nearby points in Euclidean space may be far apart on the manifold. Isomap respects this by computing distances **along** the manifold.

---

## Algorithm

Isomap consists of three steps:

### Step 1: Construct Neighborhood Graph

Build a graph $G$ where each point connects to its $k$ nearest neighbors (or all neighbors within radius $\epsilon$):

$$G_{ij} = \begin{cases} \|\mathbf{x}_i - \mathbf{x}_j\| & \text{if } j \in \mathcal{N}_k(i) \\ \infty & \text{otherwise} \end{cases}$$

### Step 2: Compute Geodesic Distances

Approximate geodesic distances using **shortest paths** in $G$ via Dijkstra's or Floyd-Warshall algorithm:

$$d_G(i, j) = \text{shortest path from } i \text{ to } j \text{ in } G$$

### Step 3: Apply Classical MDS

Run classical MDS on the geodesic distance matrix $\mathbf{D}_G$:

$$\mathbf{B} = -\frac{1}{2}\mathbf{H}\mathbf{D}_G^{(2)}\mathbf{H} \quad \Longrightarrow \quad \mathbf{Y} = \mathbf{Q}_k \mathbf{\Lambda}_k^{1/2}$$

---

## Mathematical Foundation

### Geodesic Distances on Manifolds

For a smooth manifold $\mathcal{M}$, the geodesic distance is:

$$d_{\mathcal{M}}(\mathbf{x}_i, \mathbf{x}_j) = \inf_{\gamma} \int_0^1 \left\|\frac{d\gamma(t)}{dt}\right\| dt$$

where $\gamma: [0,1] \to \mathcal{M}$ is a path with $\gamma(0) = \mathbf{x}_i$, $\gamma(1) = \mathbf{x}_j$.

### Graph Approximation

As $n \to \infty$ and neighborhood radius $\to 0$ appropriately, shortest-path distances in the neighborhood graph converge to true geodesic distances:

$$d_G(i, j) \to d_{\mathcal{M}}(\mathbf{x}_i, \mathbf{x}_j)$$

### Isometry Guarantee

If the manifold is **isometric** to a convex subset of $\mathbb{R}^d$ (i.e., geodesic distances equal Euclidean distances in the unfolded space), then Isomap recovers the true low-dimensional coordinates.

---

## Implementation

```python
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import shortest_path
from sklearn.neighbors import NearestNeighbors

def isomap(X, n_components=2, n_neighbors=10):
    """
    Isomap: geodesic-distance MDS.
    
    Args:
        X: Data matrix [n_samples, n_features]
        n_components: Embedding dimensionality
        n_neighbors: Number of nearest neighbors for graph
    
    Returns:
        Y: Embedded coordinates [n, n_components]
        eigenvalues: Top eigenvalues from MDS
    """
    n = X.shape[0]
    
    # Step 1: k-NN graph
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(X)
    distances, indices = nn.kneighbors(X)
    
    # Build sparse adjacency (symmetric)
    graph = np.full((n, n), np.inf)
    np.fill_diagonal(graph, 0)
    for i in range(n):
        for j_idx, j in enumerate(indices[i]):
            d = distances[i, j_idx]
            graph[i, j] = min(graph[i, j], d)
            graph[j, i] = min(graph[j, i], d)
    
    # Step 2: Shortest paths (geodesic approximation)
    D_geo = shortest_path(graph, method='D')
    
    # Check connectivity
    if np.any(np.isinf(D_geo)):
        raise ValueError(
            "Graph is disconnected. Increase n_neighbors."
        )
    
    # Step 3: Classical MDS on geodesic distances
    D_sq = D_geo ** 2
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ D_sq @ H
    
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    k = n_components
    Lambda_k = np.diag(np.sqrt(np.maximum(eigenvalues[:k], 0)))
    Y = eigenvectors[:, :k] @ Lambda_k
    
    return Y, eigenvalues[:k]
```

---

## Swiss Roll Example

```python
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import PCA

# Generate Swiss roll
X, color = make_swiss_roll(n_samples=1500, noise=0.3, random_state=42)

# PCA (fails — cannot unroll)
Y_pca = PCA(n_components=2).fit_transform(X)

# Isomap (succeeds — unfolds the manifold)
Y_iso, eigenvalues = isomap(X, n_components=2, n_neighbors=12)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Original 3D
ax = fig.add_subplot(131, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap='Spectral', s=5)
ax.set_title('Swiss Roll (3D)')

# PCA
axes[1].scatter(Y_pca[:, 0], Y_pca[:, 1], c=color, cmap='Spectral', s=5)
axes[1].set_title('PCA (2D) — Overlapping')

# Isomap
axes[2].scatter(Y_iso[:, 0], Y_iso[:, 1], c=color, cmap='Spectral', s=5)
axes[2].set_title('Isomap (2D) — Unfolded')

plt.tight_layout()
plt.show()
```

---

## Choosing n_neighbors

The neighborhood size $k$ controls the trade-off between local fidelity and global connectivity:

| $k$ too small | $k$ too large |
|---------------|---------------|
| Disconnected graph | Short-circuits through manifold |
| Missing geodesic paths | Loses nonlinear structure |
| Noisy distance estimates | Approaches Euclidean MDS |

### Practical Heuristic

```python
def residual_variance(X, n_neighbors_range, n_components=2):
    """
    Select n_neighbors by residual variance curve.
    
    The residual variance measures how well geodesic distances
    are preserved in the embedding (1 - R^2).
    """
    from sklearn.manifold import Isomap as SkIsomap
    
    residuals = []
    for k in n_neighbors_range:
        iso = SkIsomap(n_components=n_components, n_neighbors=k)
        iso.fit(X)
        # Reconstruction error from sklearn
        residuals.append(iso.reconstruction_error())
    
    return residuals
```

---

## scikit-learn Interface

```python
from sklearn.manifold import Isomap

# Basic usage
iso = Isomap(n_components=2, n_neighbors=10)
Y = iso.fit_transform(X)

# Access geodesic distance matrix
D_geo = iso.dist_matrix_

# Reconstruction error
print(f"Reconstruction error: {iso.reconstruction_error():.4f}")

# Transform new points (out-of-sample via kernel trick)
Y_new = iso.transform(X_new)
```

### Key Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `n_components` | 2 | Embedding dimensionality |
| `n_neighbors` | 5 | Neighbors for graph construction |
| `metric` | 'minkowski' | Distance metric for neighbors |
| `path_method` | 'auto' | Shortest path algorithm |

---

## Application: Yield Curve Manifold

Government yield curves (rates across maturities) lie on a low-dimensional manifold driven by level, slope, and curvature factors:

```python
import numpy as np
import matplotlib.pyplot as plt

def yield_curve_isomap(yields, maturities):
    """
    Embed yield curve time series via Isomap.
    
    Args:
        yields: Yield matrix [n_dates, n_maturities]
        maturities: List of maturity labels
    """
    from sklearn.manifold import Isomap
    
    iso = Isomap(n_components=3, n_neighbors=15)
    Y = iso.fit_transform(yields)
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color by time
    colors = np.arange(len(yields))
    scatter = ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2],
                         c=colors, cmap='viridis', s=10)
    
    ax.set_xlabel('Isomap 1 (≈ Level)')
    ax.set_ylabel('Isomap 2 (≈ Slope)')
    ax.set_zlabel('Isomap 3 (≈ Curvature)')
    ax.set_title('Yield Curve Manifold via Isomap')
    plt.colorbar(scatter, label='Time Index', shrink=0.6)
    plt.tight_layout()
    plt.show()


# Synthetic yield curve data (Nelson-Siegel factors)
np.random.seed(42)
n_dates = 500
maturities = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])

# Latent factors: level, slope, curvature (random walk)
level = np.cumsum(0.01 * np.random.randn(n_dates)) + 3.0
slope = np.cumsum(0.005 * np.random.randn(n_dates)) - 0.5
curv  = np.cumsum(0.003 * np.random.randn(n_dates)) + 0.3

# Nelson-Siegel model
tau = 1.5
yields = np.zeros((n_dates, len(maturities)))
for i, m in enumerate(maturities):
    factor1 = 1.0
    factor2 = (1 - np.exp(-m / tau)) / (m / tau)
    factor3 = factor2 - np.exp(-m / tau)
    yields[:, i] = (level * factor1 + slope * factor2 + 
                    curv * factor3 + 0.05 * np.random.randn(n_dates))

yield_curve_isomap(yields, maturities)
```

---

## When Isomap Fails

### Non-Convex Manifolds

Isomap assumes the manifold is **isometric to a convex region** of $\mathbb{R}^d$. It fails on:

- Manifolds with holes (shortest paths detour around holes)
- Non-convex manifolds where geodesics are ambiguous

### Noisy Data

Noise can create spurious short-circuit edges in the neighborhood graph, collapsing geodesic distances.

### Uneven Sampling

Sparse regions create long graph edges that poorly approximate geodesics.

---

## Complexity Analysis

| Step | Complexity |
|------|------------|
| k-NN graph | $O(n^2 d)$ or $O(nd \log n)$ with KD-tree |
| Shortest paths (Dijkstra) | $O(n^2 \log n)$ |
| Classical MDS | $O(n^3)$ |
| **Total** | $O(n^3)$ |

For large $n$, the $O(n^3)$ MDS step dominates. **Landmark Isomap** selects $m \ll n$ landmark points to reduce complexity to $O(nm^2)$.

---

## Summary

| Concept | Key Point |
|---------|-----------|
| **Core idea** | Replace Euclidean distances with geodesic distances |
| **Step 1** | Build k-NN neighborhood graph |
| **Step 2** | Shortest paths approximate geodesic distances |
| **Step 3** | Classical MDS on geodesic distance matrix |
| **Strength** | Unfolds curved manifolds (Swiss roll) |
| **Weakness** | Fails on non-convex manifolds, sensitive to $k$ |
| **Complexity** | $O(n^3)$ |

---

## What's Next

Isomap preserves global geodesic distances. **Locally Linear Embedding (LLE)** takes a fundamentally different approach: instead of preserving distances, it preserves **local linear reconstruction weights**, making it more robust to manifold curvature.
