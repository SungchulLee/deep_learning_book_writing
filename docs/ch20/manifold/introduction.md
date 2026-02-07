# Manifold Learning

Nonlinear dimensionality reduction that recovers intrinsic low-dimensional structure.

---

## Overview

Real-world high-dimensional data often lies on or near a **low-dimensional manifold** — a curved surface embedded in ambient space. **Manifold learning** methods discover this intrinsic structure by preserving geometric relationships (distances, neighborhoods, or probability distributions) rather than assuming a linear subspace as PCA does.

---

## The Manifold Hypothesis

Many high-dimensional datasets have far fewer **intrinsic degrees of freedom** than their ambient dimensionality suggests.

**Examples:**

- A 28×28 grayscale digit image lives in $\mathbb{R}^{784}$, but plausible handwritten digits occupy a much lower-dimensional manifold
- Daily returns on 500 stocks live in $\mathbb{R}^{500}$, but a handful of latent factors (market, sector, momentum) drive most variation
- A robot arm with 3 joints traces a 3-dimensional manifold in its high-dimensional sensor space

### Formal Statement

Data $\{\mathbf{x}_1, \ldots, \mathbf{x}_n\} \subset \mathbb{R}^D$ lies on or near a smooth manifold $\mathcal{M}$ of intrinsic dimension $d \ll D$. The goal is to find a mapping $f: \mathcal{M} \to \mathbb{R}^d$ that preserves the manifold's geometry.

---

## Learning Objectives

- Understand MDS as distance-preserving embedding and its equivalence to PCA for Euclidean data
- Derive Isomap as geodesic-distance MDS and recognize when it outperforms linear methods
- Implement LLE and understand its local linearity assumption
- Master t-SNE's probabilistic framework, perplexity tuning, and common pitfalls
- Apply UMAP for scalable, structure-preserving embeddings
- Choose the right method for a given dataset and application in quantitative finance

---

## Why PCA Fails on Manifolds

PCA finds the best **linear** subspace, but manifolds are generally **nonlinear**:

```
Swiss Roll (3D):           PCA (2D projection):

   ╭──────╮               Points far apart on the
  ╭──────╮│               manifold overlap when
 ╭──────╮│╯               projected linearly
 │●●●●●●│╯               
 ╰──────╯                 ●●●●●●  (overlapping!)
```

PCA projects onto the plane of maximum variance, collapsing the rolled structure. Manifold learning methods "unroll" data by respecting intrinsic geometry.

---

## Key Equations

| Method | Objective |
|--------|-----------|
| **MDS** | $\min_{\mathbf{Y}} \sum_{i<j}(d_{ij} - \|\mathbf{y}_i - \mathbf{y}_j\|)^2$ |
| **Isomap** | Classical MDS on geodesic distances $d_G(i,j)$ |
| **LLE** | $\min_{\mathbf{Y}} \sum_i \|\mathbf{y}_i - \sum_j w_{ij}\mathbf{y}_j\|^2$ |
| **t-SNE** | $\min_{\mathbf{Y}} \text{KL}(P \| Q)$ where $q_{ij} \propto (1 + \|\mathbf{y}_i - \mathbf{y}_j\|^2)^{-1}$ |
| **UMAP** | $\min_{\mathbf{Y}} \text{CE}(P, Q)$ with fuzzy simplicial set construction |

---

## Taxonomy

### Global vs Local Methods

| Type | Methods | Preserves | Strength |
|------|---------|-----------|----------|
| **Global** | MDS, Isomap | All pairwise distances | Faithful large-scale geometry |
| **Local** | LLE, t-SNE, UMAP | Neighborhood relationships | Cluster separation, local detail |

### Parametric vs Non-Parametric

| Type | Property | Consequence |
|------|----------|-------------|
| **Non-parametric** | No explicit mapping function | Cannot embed new points directly |
| **Parametric** | Learns $f: \mathbb{R}^D \to \mathbb{R}^d$ | Out-of-sample extension possible |

All classical manifold learning methods (MDS, Isomap, LLE, t-SNE) are non-parametric. UMAP offers a parametric variant via neural networks.

---

## Section Contents

1. **MDS** — Distance-preserving embeddings via double centering or stress minimization
2. **Isomap** — Geodesic distances on the data graph + classical MDS
3. **LLE** — Locally linear reconstruction weights preserved in embedding
4. **t-SNE** — Probabilistic neighbor embedding with heavy-tailed distributions
5. **UMAP** — Topological data analysis for fast, scalable embeddings

---

## Quick Comparison

```python
from sklearn.manifold import MDS, Isomap, LocallyLinearEmbedding, TSNE
from sklearn.datasets import make_swiss_roll
import umap

X, color = make_swiss_roll(n_samples=1000, noise=0.1)

# Each method: fit_transform(X) -> Y [n, 2]
Y_mds    = MDS(n_components=2).fit_transform(X)
Y_iso    = Isomap(n_components=2, n_neighbors=10).fit_transform(X)
Y_lle    = LocallyLinearEmbedding(n_components=2, n_neighbors=10).fit_transform(X)
Y_tsne   = TSNE(n_components=2, perplexity=30).fit_transform(X)
Y_umap   = umap.UMAP(n_components=2, n_neighbors=15).fit_transform(X)
```

---

## Summary

| Consideration | Recommendation |
|---------------|----------------|
| **Linear data, interpretability** | PCA |
| **Distance/dissimilarity input** | MDS |
| **Curved manifold, global structure** | Isomap |
| **Small data, local geometry** | LLE |
| **Visualization of clusters** | t-SNE |
| **Large-scale, general purpose** | UMAP |

---

## What's Next

The following sections develop each method in full mathematical detail with from-scratch implementations, scikit-learn usage, and applications to quantitative finance.
