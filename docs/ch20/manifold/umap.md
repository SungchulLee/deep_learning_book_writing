# UMAP

Uniform Manifold Approximation and Projection for scalable dimensionality reduction.

---

## Overview

**UMAP** (Uniform Manifold Approximation and Projection) combines ideas from topological data analysis and stochastic gradient descent to produce embeddings that preserve both local and global structure. It is faster than t-SNE, supports out-of-sample mapping, and provides a more principled mathematical framework.

---

## Mathematical Foundation

UMAP is grounded in two theoretical pillars:

### 1. Riemannian Geometry / Fuzzy Topology

UMAP assumes data is uniformly distributed on a Riemannian manifold. Under this assumption, the local distance metric varies across the manifold, and UMAP estimates a **fuzzy simplicial set** (a weighted graph) that captures the manifold's topology.

### 2. Cross-Entropy Optimization

Rather than KL divergence (as in t-SNE), UMAP minimizes a **fuzzy set cross-entropy** between high-dimensional and low-dimensional representations:

$$C = \sum_{i \neq j} \left[ p_{ij} \log\frac{p_{ij}}{q_{ij}} + (1 - p_{ij}) \log\frac{1 - p_{ij}}{1 - q_{ij}} \right]$$

The second term (absent in t-SNE) is a **repulsive force** that explicitly penalizes dissimilar points being placed too close together.

---

## Algorithm

### Step 1: Construct Fuzzy Simplicial Set (High-D Graph)

For each point $\mathbf{x}_i$, find its $k$ nearest neighbors and define local connectivity:

$$p_{j|i} = \exp\left(-\frac{d(\mathbf{x}_i, \mathbf{x}_j) - \rho_i}{\sigma_i}\right)$$

where:

- $\rho_i = \min_{j \in \mathcal{N}(i)} d(\mathbf{x}_i, \mathbf{x}_j)$ — distance to nearest neighbor
- $\sigma_i$ — local bandwidth (found via binary search to match $\log_2(k)$ effective neighbors)

Symmetrize via fuzzy union:

$$p_{ij} = p_{j|i} + p_{i|j} - p_{j|i} \cdot p_{i|j}$$

### Step 2: Initialize Low-Dimensional Layout

Typically use spectral embedding (eigenvectors of the graph Laplacian) or random initialization.

### Step 3: Optimize Low-Dimensional Layout

Define low-dimensional affinities using a smooth approximation:

$$q_{ij} = \left(1 + a \|\mathbf{y}_i - \mathbf{y}_j\|^{2b}\right)^{-1}$$

where $a$ and $b$ are derived from `min_dist` parameter (default: $a \approx 1.93$, $b \approx 0.79$ for `min_dist=0.1`).

Minimize the cross-entropy via stochastic gradient descent with **negative sampling** for efficiency.

---

## UMAP vs t-SNE: Key Differences

| Aspect | t-SNE | UMAP |
|--------|-------|------|
| **Loss** | $\text{KL}(P \| Q)$ | Fuzzy cross-entropy |
| **Low-D kernel** | Fixed Cauchy: $(1+d^2)^{-1}$ | Parametric: $(1 + a \cdot d^{2b})^{-1}$ |
| **Repulsion** | Implicit (normalization) | Explicit (CE second term) |
| **Normalization** | Global (sum to 1) | Local (no global normalization) |
| **Optimization** | Full gradient or Barnes-Hut | SGD with negative sampling |
| **Global structure** | Poor | Better preserved |
| **Speed** | $O(n \log n)$ per step (BH) | $O(n \cdot k)$ per step |
| **Out-of-sample** | Not supported | Supported via `transform()` |

---

## Implementation: Simplified UMAP

```python
import numpy as np
from sklearn.neighbors import NearestNeighbors

def compute_fuzzy_simplicial_set(X, n_neighbors=15):
    """
    Build UMAP's fuzzy simplicial set (high-D graph).
    
    Args:
        X: Data matrix [n, d]
        n_neighbors: Number of neighbors
    
    Returns:
        P: Symmetric fuzzy membership matrix [n, n]
    """
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(X)
    distances, indices = nn.kneighbors(X)
    
    n = X.shape[0]
    
    # rho_i: distance to nearest neighbor
    rho = distances[:, 1]  # Skip self (index 0)
    
    # sigma_i: binary search to match target = log2(n_neighbors)
    target = np.log2(n_neighbors)
    sigma = np.ones(n)
    
    for i in range(n):
        lo, hi = 1e-10, 1e4
        dists_i = distances[i, 1:]  # Skip self
        
        for _ in range(64):
            mid = (lo + hi) / 2
            vals = np.exp(-(np.maximum(dists_i - rho[i], 0)) / mid)
            if vals.sum() > target:
                hi = mid
            else:
                lo = mid
            if abs(vals.sum() - target) < 1e-5:
                break
        sigma[i] = mid
    
    # Asymmetric affinities
    P_asym = np.zeros((n, n))
    for i in range(n):
        for j_idx, j in enumerate(indices[i, 1:]):
            d = max(distances[i, j_idx + 1] - rho[i], 0)
            P_asym[i, j] = np.exp(-d / sigma[i])
    
    # Fuzzy union (symmetrize)
    P = P_asym + P_asym.T - P_asym * P_asym.T
    
    return P


def umap_embed(P, n_components=2, n_epochs=200, lr=1.0,
               min_dist=0.1, seed=42):
    """
    UMAP embedding via SGD on cross-entropy.
    
    Args:
        P: Fuzzy simplicial set [n, n]
        n_components: Embedding dimension
        n_epochs: Optimization epochs
        lr: Learning rate
        min_dist: Minimum embedding distance (controls tightness)
    
    Returns:
        Y: Embedding [n, n_components]
    """
    np.random.seed(seed)
    n = P.shape[0]
    
    # Compute a, b from min_dist (approximate)
    # These control the low-D kernel shape
    from scipy.optimize import curve_fit
    
    def kernel(d, a, b):
        return 1.0 / (1.0 + a * d ** (2 * b))
    
    x_fit = np.linspace(0, 3, 300)
    y_fit = np.where(x_fit <= min_dist, 1.0, 
                     np.exp(-(x_fit - min_dist)))
    (a, b), _ = curve_fit(kernel, x_fit, y_fit)
    
    # Initialize (spectral or random)
    Y = np.random.randn(n, n_components) * 0.01
    
    # Get edges with positive weight
    rows, cols = np.nonzero(P > 0)
    weights = P[rows, cols]
    
    n_edges = len(rows)
    
    for epoch in range(n_epochs):
        # Decaying learning rate
        alpha = lr * (1.0 - epoch / n_epochs)
        
        for edge_idx in range(n_edges):
            i, j = rows[edge_idx], cols[edge_idx]
            w = weights[edge_idx]
            
            diff = Y[i] - Y[j]
            dist_sq = (diff ** 2).sum()
            
            # Attractive force (positive edges)
            grad_coeff = (-2.0 * a * b * dist_sq ** (b - 1)) / \
                         (1.0 + a * dist_sq ** b)
            grad = w * grad_coeff * diff
            
            Y[i] += alpha * grad
            Y[j] -= alpha * grad
            
            # Negative sampling (repulsive force)
            for _ in range(5):
                k = np.random.randint(n)
                if k == i:
                    continue
                
                diff_neg = Y[i] - Y[k]
                dist_sq_neg = (diff_neg ** 2).sum() + 1e-6
                
                grad_rep = (2.0 * b) / \
                           ((1e-3 + dist_sq_neg) * 
                            (1.0 + a * dist_sq_neg ** b))
                
                Y[i] += alpha * grad_rep * diff_neg
    
    return Y


def simple_umap(X, n_components=2, n_neighbors=15, 
                min_dist=0.1, n_epochs=200, seed=42):
    """
    Simplified UMAP pipeline.
    
    Args:
        X: Data matrix [n, d]
        n_components: Embedding dimension
        n_neighbors: Number of neighbors
        min_dist: Minimum distance in embedding
        n_epochs: Optimization epochs
    
    Returns:
        Y: Embedding [n, n_components]
    """
    P = compute_fuzzy_simplicial_set(X, n_neighbors)
    Y = umap_embed(P, n_components, n_epochs, 
                   min_dist=min_dist, seed=seed)
    return Y
```

---

## Key Hyperparameters

### n_neighbors

Controls the balance between local and global structure:

| n_neighbors | Effect |
|-------------|--------|
| Small (5–10) | Fine local detail, may fragment global structure |
| Medium (15–30) | Balanced (default: 15) |
| Large (50–200) | More global structure, less local detail |

### min_dist

Controls how tightly points cluster in the embedding:

| min_dist | Effect |
|----------|--------|
| 0.0 | Points can overlap → dense clusters |
| 0.1 | Slight separation (default) |
| 0.5–1.0 | Spread out → focus on global topology |

### n_components

UMAP works well for $d > 2$ (unlike t-SNE which is typically limited to 2–3):

```python
# 2D for visualization
Y_2d = umap.UMAP(n_components=2).fit_transform(X)

# Higher dimensions for downstream ML
Y_50d = umap.UMAP(n_components=50).fit_transform(X)
```

---

## umap-learn Interface

```python
import umap

# Basic usage
reducer = umap.UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    metric='euclidean',
    random_state=42
)
Y = reducer.fit_transform(X)

# Out-of-sample (key advantage over t-SNE)
Y_new = reducer.transform(X_new)

# Custom metrics
reducer_corr = umap.UMAP(
    metric='correlation',
    n_neighbors=20
)
Y_corr = reducer_corr.fit_transform(X)

# Supervised UMAP (use labels to guide embedding)
reducer_sup = umap.UMAP()
Y_sup = reducer_sup.fit_transform(X, y=labels)
```

### Precomputed Distances

```python
from scipy.spatial.distance import pdist, squareform

D = squareform(pdist(X, metric='cosine'))

reducer = umap.UMAP(metric='precomputed', n_neighbors=15)
Y = reducer.fit_transform(D)
```

---

## Application: Portfolio Regime Clustering

```python
import numpy as np
import matplotlib.pyplot as plt
import umap

def portfolio_regime_umap(returns, window=60, step=10,
                          n_neighbors=15, min_dist=0.1):
    """
    Cluster market regimes using UMAP on rolling features.
    
    Args:
        returns: Return matrix [n_periods, n_assets]
        window: Rolling window
        step: Step size
        n_neighbors: UMAP neighbors
        min_dist: UMAP minimum distance
    """
    features = []
    timestamps = []
    
    for t in range(0, returns.shape[0] - window, step):
        chunk = returns[t:t+window]
        means = chunk.mean(axis=0)
        vols = chunk.std(axis=0)
        corr = np.corrcoef(chunk.T)
        upper_corr = corr[np.triu_indices_from(corr, k=1)]
        
        features.append(np.concatenate([means, vols, upper_corr]))
        timestamps.append(t + window)
    
    features = np.array(features)
    timestamps = np.array(timestamps)
    
    # UMAP embedding
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42
    )
    Y = reducer.fit_transform(features)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(Y[:, 0], Y[:, 1], c=timestamps,
                         cmap='viridis', s=20, alpha=0.7)
    ax.plot(Y[:, 0], Y[:, 1], 'k-', alpha=0.1, linewidth=0.5)
    
    plt.colorbar(scatter, label='Time Index')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title('Market Regime Clusters (UMAP)')
    plt.tight_layout()
    plt.show()
    
    return reducer  # Can transform new windows


# Example with regime shifts
np.random.seed(42)
n_assets = 10
returns = np.vstack([
    np.random.randn(400, n_assets) * 0.01,             # Low vol
    np.random.randn(200, n_assets) * 0.03 + 0.001,     # High vol
    np.random.randn(400, n_assets) * 0.01 - 0.001,     # Low vol
])

reducer = portfolio_regime_umap(returns)
```

---

## Application: Asset Embedding for Clustering

UMAP embeddings can serve as features for downstream clustering:

```python
import numpy as np
import umap
from sklearn.cluster import HDBSCAN

def asset_clustering(returns, n_neighbors=10, min_dist=0.0):
    """
    Cluster assets using UMAP + HDBSCAN.
    
    Args:
        returns: Return matrix [n_periods, n_assets]
    
    Returns:
        labels: Cluster labels per asset
        Y: UMAP embedding [n_assets, 2]
    """
    # Correlation distance
    corr = np.corrcoef(returns.T)
    D = np.sqrt(2 * (1 - corr))
    
    # UMAP on precomputed distance
    reducer = umap.UMAP(
        n_components=2,
        metric='precomputed',
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42
    )
    Y = reducer.fit_transform(D)
    
    # HDBSCAN clustering on embedding
    clusterer = HDBSCAN(min_cluster_size=3)
    labels = clusterer.fit_predict(Y)
    
    return labels, Y
```

---

## Parametric UMAP

Standard UMAP is non-parametric (no explicit mapping function). **Parametric UMAP** trains a neural network to learn $f: \mathbb{R}^D \to \mathbb{R}^d$:

```python
# Requires: pip install parametric-umap
from umap.parametric_umap import ParametricUMAP
import tensorflow as tf

encoder = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(2)
])

pumap = ParametricUMAP(
    encoder=encoder,
    n_components=2,
    n_neighbors=15,
    min_dist=0.1
)
Y = pumap.fit_transform(X)

# True out-of-sample mapping
Y_new = pumap.transform(X_new)
```

---

## Complexity Analysis

| Step | Complexity |
|------|------------|
| k-NN graph | $O(n^{1.14} d)$ approximate (NN-descent) |
| Fuzzy simplicial set | $O(nk)$ |
| SGD optimization ($E$ epochs) | $O(nkE)$ |
| **Total** | $O(n^{1.14} d + nkE)$ |

UMAP scales much better than t-SNE for large $n$ due to approximate NN search and SGD with negative sampling.

---

## Limitations

| Limitation | Detail |
|------------|--------|
| **Stochastic** | Different seeds give different layouts |
| **Hyperparameter-sensitive** | `n_neighbors`, `min_dist` significantly affect results |
| **Inter-cluster distances** | More meaningful than t-SNE but still approximate |
| **Theory-practice gap** | Topological justification is debated |
| **Density distortion** | Relative densities may not be preserved |

---

## Method Comparison: Swiss Roll

```python
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, Isomap, LocallyLinearEmbedding, TSNE
import umap

X, color = make_swiss_roll(n_samples=1500, noise=0.3, random_state=42)

methods = {
    'PCA': PCA(n_components=2).fit_transform(X),
    'MDS': MDS(n_components=2, random_state=42, normalized_stress='auto').fit_transform(X),
    'Isomap': Isomap(n_components=2, n_neighbors=12).fit_transform(X),
    'LLE': LocallyLinearEmbedding(n_components=2, n_neighbors=12).fit_transform(X),
    't-SNE': TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X),
    'UMAP': umap.UMAP(n_components=2, n_neighbors=15, random_state=42).fit_transform(X),
}

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for ax, (name, Y) in zip(axes.flat, methods.items()):
    ax.scatter(Y[:, 0], Y[:, 1], c=color, cmap='Spectral', s=5)
    ax.set_title(name)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()
```

---

## Summary

| Concept | Key Point |
|---------|-----------|
| **Framework** | Fuzzy simplicial sets (topological) |
| **High-D graph** | Local connectivity with adaptive bandwidth |
| **Low-D kernel** | $(1 + a \cdot d^{2b})^{-1}$, controlled by `min_dist` |
| **Loss** | Fuzzy cross-entropy (attractive + repulsive) |
| **Optimization** | SGD with negative sampling |
| **Key advantages** | Fast, out-of-sample, better global structure than t-SNE |
| **Key parameters** | `n_neighbors` (local/global), `min_dist` (tightness) |
| **Complexity** | $O(n^{1.14} d + nkE)$ — near-linear in $n$ |
