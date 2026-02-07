# Multidimensional Scaling

Embedding high-dimensional data in low dimensions by preserving pairwise distances.

---

## Overview

**Multidimensional Scaling (MDS)** finds a low-dimensional embedding where pairwise distances between points approximate those in the original space. Unlike PCA, which preserves global variance, MDS directly optimizes distance preservation — making it the natural bridge from linear methods to manifold learning.

---

## Problem Formulation

### Input

A distance (or dissimilarity) matrix $\mathbf{D} \in \mathbb{R}^{n \times n}$ where $d_{ij}$ is the distance between points $i$ and $j$:

$$d_{ij} = \|\mathbf{x}_i - \mathbf{x}_j\|$$

### Goal

Find low-dimensional coordinates $\mathbf{Y} = [\mathbf{y}_1, \ldots, \mathbf{y}_n]^T \in \mathbb{R}^{n \times k}$ such that:

$$\|\mathbf{y}_i - \mathbf{y}_j\| \approx d_{ij} \quad \forall\; i, j$$

---

## Classical MDS

Classical MDS provides an **analytical solution** when distances are Euclidean, converting the distance matrix into an inner product matrix via **double centering**.

### Double Centering

Given the squared distance matrix $\mathbf{D}^{(2)}$ with entries $d_{ij}^2$:

$$\mathbf{B} = -\frac{1}{2} \mathbf{H} \mathbf{D}^{(2)} \mathbf{H}$$

where $\mathbf{H} = \mathbf{I} - \frac{1}{n}\mathbf{1}\mathbf{1}^T$ is the centering matrix.

### Why Double Centering Works

For Euclidean distances from centered data $\mathbf{X}$:

$$d_{ij}^2 = \|\mathbf{x}_i - \mathbf{x}_j\|^2 = \mathbf{x}_i^T\mathbf{x}_i - 2\mathbf{x}_i^T\mathbf{x}_j + \mathbf{x}_j^T\mathbf{x}_j$$

Double centering removes the diagonal terms, recovering the Gram matrix:

$$b_{ij} = \mathbf{x}_i^T \mathbf{x}_j \quad \Longrightarrow \quad \mathbf{B} = \mathbf{X}\mathbf{X}^T$$

### Eigendecomposition

$$\mathbf{B} = \mathbf{Q} \mathbf{\Lambda} \mathbf{Q}^T$$

The $k$-dimensional embedding is:

$$\mathbf{Y} = \mathbf{Q}_k \mathbf{\Lambda}_k^{1/2}$$

where $\mathbf{Q}_k$ and $\mathbf{\Lambda}_k$ contain the top-$k$ eigenvectors and eigenvalues.

---

## Algorithm: Classical MDS

1. Compute squared distance matrix $\mathbf{D}^{(2)}$
2. Double center: $\mathbf{B} = -\frac{1}{2}\mathbf{H}\mathbf{D}^{(2)}\mathbf{H}$
3. Eigendecompose $\mathbf{B} = \mathbf{Q}\mathbf{\Lambda}\mathbf{Q}^T$
4. Select top-$k$ positive eigenvalues and eigenvectors
5. Embed: $\mathbf{Y} = \mathbf{Q}_k \mathbf{\Lambda}_k^{1/2}$

---

## Implementation

```python
import numpy as np

def classical_mds(D, n_components=2):
    """
    Classical (metric) MDS via double centering.
    
    Args:
        D: Distance matrix [n, n] (symmetric, zero diagonal)
        n_components: Embedding dimensionality
    
    Returns:
        Y: Embedded coordinates [n, n_components]
        eigenvalues: Top eigenvalues
    """
    n = D.shape[0]
    
    # Squared distance matrix
    D_sq = D ** 2
    
    # Centering matrix
    H = np.eye(n) - np.ones((n, n)) / n
    
    # Double centering -> Gram matrix
    B = -0.5 * H @ D_sq @ H
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    
    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Top-k positive eigenvalues
    k = n_components
    Lambda_k = np.diag(np.sqrt(np.maximum(eigenvalues[:k], 0)))
    Q_k = eigenvectors[:, :k]
    
    Y = Q_k @ Lambda_k
    
    return Y, eigenvalues[:k]
```

---

## Equivalence to PCA

When the distance matrix is Euclidean from centered data, classical MDS produces the **same embedding as PCA** (up to rotation/reflection).

### Proof

For centered data $\mathbf{X}$ with SVD $\mathbf{X} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^T$:

- PCA scores: $\mathbf{Z} = \mathbf{U}_k\mathbf{\Sigma}_k$
- Gram matrix: $\mathbf{B} = \mathbf{X}\mathbf{X}^T = \mathbf{U}\mathbf{\Sigma}^2\mathbf{U}^T$
- MDS embedding: $\mathbf{Y} = \mathbf{U}_k\mathbf{\Sigma}_k$

The embeddings are identical.

### Verification

```python
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform

np.random.seed(42)
X = np.random.randn(100, 10)
X_centered = X - X.mean(axis=0)

# PCA
Y_pca = PCA(n_components=2).fit_transform(X_centered)

# Classical MDS
D = squareform(pdist(X_centered))
Y_mds, _ = classical_mds(D, n_components=2)

# Align signs
for i in range(2):
    if np.corrcoef(Y_pca[:, i], Y_mds[:, i])[0, 1] < 0:
        Y_mds[:, i] *= -1

print(f"Max difference: {np.abs(Y_pca - Y_mds).max():.2e}")
# Near machine epsilon
```

---

## Metric MDS (Stress Minimization)

When distances are not perfectly Euclidean, classical MDS may produce negative eigenvalues. **Metric MDS** minimizes a stress function directly.

### Stress Function

Kruskal's raw stress:

$$\text{Stress}(\mathbf{Y}) = \sqrt{\frac{\sum_{i < j} (d_{ij} - \|\mathbf{y}_i - \mathbf{y}_j\|)^2}{\sum_{i < j} d_{ij}^2}}$$

### SMACOF Algorithm

Scaling by MAjorizing a COmplicated Function:

1. Initialize $\mathbf{Y}^{(0)}$ (e.g., from classical MDS)
2. At each iteration $t$, compute the Guttman transform:
   $$\mathbf{Y}^{(t+1)} = \frac{1}{n} \mathbf{Z}(\mathbf{Y}^{(t)}) \mathbf{Y}^{(t)}$$
   where $\mathbf{Z}$ is derived from the ratio $d_{ij} / \|\mathbf{y}_i^{(t)} - \mathbf{y}_j^{(t)}\|$
3. Repeat until stress converges

### Stress Interpretation

| Stress | Quality |
|--------|---------|
| < 0.05 | Excellent |
| 0.05–0.10 | Good |
| 0.10–0.20 | Fair |
| > 0.20 | Poor |

---

## Metric MDS with PyTorch

```python
import torch

def metric_mds_torch(D, n_components=2, n_iter=300, lr=0.01, seed=42):
    """
    Metric MDS via gradient descent on stress.
    
    Args:
        D: Distance matrix [n, n] (numpy array)
        n_components: Embedding dimensionality
        n_iter: Optimization iterations
        lr: Learning rate
    
    Returns:
        Y: Embedded coordinates [n, n_components]
        stress_history: Stress per iteration
    """
    torch.manual_seed(seed)
    n = D.shape[0]
    D_tensor = torch.tensor(D, dtype=torch.float32)
    
    # Upper triangle mask
    mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    
    # Initialize from classical MDS
    Y_init, _ = classical_mds(D, n_components)
    Y = torch.tensor(Y_init, dtype=torch.float32, requires_grad=True)
    
    optimizer = torch.optim.Adam([Y], lr=lr)
    stress_history = []
    
    for _ in range(n_iter):
        optimizer.zero_grad()
        
        # Pairwise distances in embedding
        diff = Y.unsqueeze(0) - Y.unsqueeze(1)       # [n, n, k]
        dist_embed = torch.sqrt((diff ** 2).sum(-1) + 1e-12)
        
        # Stress
        stress = ((D_tensor[mask] - dist_embed[mask]) ** 2).sum()
        stress.backward()
        optimizer.step()
        
        stress_history.append(stress.item())
    
    return Y.detach().numpy(), stress_history
```

---

## Non-Metric MDS

Non-metric MDS preserves only the **rank order** of distances, not magnitudes.

### Objective

Find $\mathbf{Y}$ such that:

$$d_{ij} < d_{kl} \implies \|\mathbf{y}_i - \mathbf{y}_j\| < \|\mathbf{y}_k - \mathbf{y}_l\|$$

### Stress with Monotone Regression

$$\text{Stress}_{\text{NM}} = \sqrt{\frac{\sum_{i < j} (\hat{d}_{ij} - f(d_{ij}))^2}{\sum_{i < j} \hat{d}_{ij}^2}}$$

where $f$ is a monotone function fitted via isotonic regression.

### When to Use

- Dissimilarities are ordinal (rankings, not exact distances)
- Original metric is unknown or unreliable
- Perceptual similarity data

---

## scikit-learn Interface

```python
from sklearn.manifold import MDS
from scipy.spatial.distance import pdist, squareform

# --- Metric MDS from features ---
mds_metric = MDS(
    n_components=2,
    metric=True,
    dissimilarity='euclidean',
    random_state=42,
    normalized_stress='auto'
)
Y_metric = mds_metric.fit_transform(X)
print(f"Stress: {mds_metric.stress_:.4f}")

# --- From precomputed distance matrix ---
D = squareform(pdist(X))
mds_pre = MDS(
    n_components=2,
    metric=True,
    dissimilarity='precomputed',
    random_state=42,
    normalized_stress='auto'
)
Y_pre = mds_pre.fit_transform(D)

# --- Non-metric MDS ---
mds_nm = MDS(
    n_components=2,
    metric=False,
    dissimilarity='precomputed',
    random_state=42,
    normalized_stress='auto'
)
Y_nm = mds_nm.fit_transform(D)
```

---

## Evaluating Quality: Shepard Diagram

A Shepard diagram plots original distances vs embedded distances — perfect embedding lies on the diagonal:

```python
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

def shepard_diagram(D_original, Y_embedded):
    """
    Assess MDS quality with a Shepard diagram.
    
    Args:
        D_original: Original distance matrix [n, n]
        Y_embedded: Embedded coordinates [n, k]
    """
    D_embed = squareform(pdist(Y_embedded))
    mask = np.triu_indices_from(D_original, k=1)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(D_original[mask], D_embed[mask], alpha=0.3, s=10)
    
    lims = [0, max(D_original[mask].max(), D_embed[mask].max())]
    ax.plot(lims, lims, 'r--', alpha=0.5, label='Perfect preservation')
    
    ax.set_xlabel('Original Distance')
    ax.set_ylabel('Embedded Distance')
    ax.set_title('Shepard Diagram')
    ax.legend()
    plt.tight_layout()
    plt.show()
```

---

## Application: Correlation-Distance Asset Map

MDS is natural for quantitative finance where assets are compared via correlation-based distances.

### Correlation Distance

Given return correlation $\rho_{ij}$:

$$d_{ij} = \sqrt{2(1 - \rho_{ij})}$$

This is a proper metric: perfectly correlated assets have distance 0, uncorrelated $\sqrt{2}$, negatively correlated 2.

```python
import numpy as np
import matplotlib.pyplot as plt

def asset_mds_map(returns, asset_names):
    """
    Visualize asset relationships via MDS on correlation distances.
    
    Args:
        returns: Return matrix [n_periods, n_assets]
        asset_names: List of asset name strings
    """
    corr = np.corrcoef(returns.T)
    D = np.sqrt(2 * (1 - corr))
    
    Y, eigenvalues = classical_mds(D, n_components=2)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(Y[:, 0], Y[:, 1], s=100, alpha=0.7)
    
    for i, name in enumerate(asset_names):
        ax.annotate(name, (Y[i, 0], Y[i, 1]),
                    fontsize=9, ha='center', va='bottom',
                    xytext=(0, 8), textcoords='offset points')
    
    ax.set_xlabel('MDS Dimension 1')
    ax.set_ylabel('MDS Dimension 2')
    ax.set_title('Asset Structure via Correlation-Distance MDS')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()


# Example with synthetic sector returns
np.random.seed(42)
n_periods = 500
market = np.random.randn(n_periods)
tech   = 0.8 * market + 0.2 * np.random.randn(n_periods)
semis  = 0.7 * market + 0.5 * tech + 0.2 * np.random.randn(n_periods)
banks  = 0.6 * market + 0.3 * np.random.randn(n_periods)
energy = 0.3 * market + 0.5 * np.random.randn(n_periods)
gold   = -0.1 * market + 0.6 * np.random.randn(n_periods)
bonds  = -0.3 * market + 0.4 * np.random.randn(n_periods)

returns = np.column_stack([market, tech, semis, banks, energy, gold, bonds])
names = ['Market', 'Tech', 'Semis', 'Banks', 'Energy', 'Gold', 'Bonds']

asset_mds_map(returns, names)
```

---

## Comparison of MDS Variants

| Variant | Preserves | Method | Complexity |
|---------|-----------|--------|------------|
| **Classical** | Euclidean distances | Eigendecomposition | $O(n^3)$ |
| **Metric** | Distance magnitudes | SMACOF (iterative) | $O(n^2 T)$ |
| **Non-metric** | Distance rank order | SMACOF + isotonic regression | $O(n^2 T)$ |

---

## Complexity Analysis

| Step | Classical MDS | Metric MDS |
|------|--------------|------------|
| Distance matrix | $O(n^2 d)$ | $O(n^2 d)$ |
| Double centering | $O(n^2)$ | — |
| Eigendecomposition | $O(n^3)$ | — |
| SMACOF iterations | — | $O(n^2 T)$ |
| **Total** | $O(n^2 d + n^3)$ | $O(n^2 d + n^2 T)$ |

---

## Limitations

| Limitation | Consequence | Addressed By |
|------------|-------------|--------------|
| **$O(n^2)$ memory** | Full distance matrix required | Landmark MDS |
| **No out-of-sample** | Cannot project new points | Parametric methods |
| **Global focus** | May distort local neighborhoods | t-SNE, UMAP |
| **Euclidean assumption** | Negative eigenvalues for non-Euclidean data | Metric MDS |

---

## Summary

| Concept | Key Point |
|---------|-----------|
| **Objective** | Preserve pairwise distances in low dimensions |
| **Classical MDS** | Analytical via double centering + eigendecomposition |
| **Metric MDS** | Iterative stress minimization (SMACOF) |
| **Non-metric MDS** | Preserves rank order of distances only |
| **PCA equivalence** | Classical MDS on Euclidean distances = PCA |
| **Finance use** | Correlation-distance maps for portfolio structure |

---

## What's Next

MDS preserves global distance structure but cannot "unfold" nonlinear manifolds. **Isomap** extends classical MDS by replacing Euclidean distances with geodesic distances computed along the data manifold.
