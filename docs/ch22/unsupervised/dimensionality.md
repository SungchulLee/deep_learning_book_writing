# Dimensionality Reduction

Dimensionality reduction transforms high-dimensional data into lower-dimensional representations while preserving important structure. It's essential for visualization, noise reduction, feature extraction, and computational efficiency.

---

## Principal Component Analysis (PCA)

### 1. Basic Usage

```python
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# Load and scale data
X, y = load_iris(return_X_y=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"Original shape: {X_scaled.shape}")
print(f"Reduced shape: {X_pca.shape}")
```

### 2. Explained Variance

```python
# Full PCA to see all components
pca_full = PCA()
pca_full.fit(X_scaled)

# Explained variance ratio
print(f"Explained variance ratio: {pca_full.explained_variance_ratio_}")
print(f"Cumulative variance: {np.cumsum(pca_full.explained_variance_ratio_)}")

# Plot explained variance
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].bar(range(1, len(pca_full.explained_variance_ratio_)+1),
            pca_full.explained_variance_ratio_)
axes[0].set_xlabel('Principal Component')
axes[0].set_ylabel('Variance Ratio')
axes[0].set_title('Explained Variance per Component')

axes[1].plot(range(1, len(pca_full.explained_variance_ratio_)+1),
             np.cumsum(pca_full.explained_variance_ratio_), 'bo-')
axes[1].axhline(y=0.95, color='r', linestyle='--', label='95% variance')
axes[1].set_xlabel('Number of Components')
axes[1].set_ylabel('Cumulative Variance Ratio')
axes[1].legend()
plt.tight_layout()
plt.show()
```

### 3. Selecting Number of Components

```python
# Method 1: Specify variance to retain
pca_95 = PCA(n_components=0.95)  # Retain 95% variance
X_pca_95 = pca_95.fit_transform(X_scaled)
print(f"Components for 95% variance: {pca_95.n_components_}")

# Method 2: Specify exact number
pca_2 = PCA(n_components=2)

# Method 3: Use 'mle' (Minka's MLE)
pca_mle = PCA(n_components='mle')
```

### 4. Components (Loadings)

```python
# Principal components (loadings matrix)
print(f"Components shape: {pca.components_.shape}")

# Feature contributions to each PC
feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

for i, component in enumerate(pca.components_):
    print(f"\nPC{i+1} loadings:")
    for name, loading in zip(feature_names, component):
        print(f"  {name}: {loading:.3f}")
```

### 5. Reconstruction

```python
# Transform and inverse transform
X_reconstructed = pca.inverse_transform(X_pca)

# Reconstruction error
reconstruction_error = np.mean((X_scaled - X_reconstructed) ** 2)
print(f"Mean reconstruction error: {reconstruction_error:.4f}")
```

---

## Kernel PCA

### 1. Non-linear Dimensionality Reduction

```python
from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_moons

# Non-linearly separable data
X_moons, y_moons = make_moons(n_samples=200, noise=0.05, random_state=42)

# Standard PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_moons)

# Kernel PCA with RBF kernel
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
X_kpca = kpca.fit_transform(X_moons)
```

### 2. Different Kernels

```python
kernels = ['linear', 'poly', 'rbf', 'sigmoid', 'cosine']

for kernel in kernels:
    kpca = KernelPCA(n_components=2, kernel=kernel, gamma=10)
    X_kpca = kpca.fit_transform(X_moons)
    print(f"{kernel}: done")
```

---

## t-SNE

### 1. Basic Usage

```python
from sklearn.manifold import TSNE

# t-SNE: t-distributed Stochastic Neighbor Embedding
# Best for visualization (2D/3D only)
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.7)
plt.title('t-SNE: Iris Dataset')
plt.show()
```

### 2. Perplexity Parameter

```python
# Perplexity: roughly the number of nearest neighbors
# Typical range: 5-50

perplexities = [5, 15, 30, 50]

for perp in perplexities:
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)
    print(f"Perplexity {perp}: done")
```

### 3. Important Considerations

```python
# t-SNE limitations:
# 1. Non-deterministic (use random_state)
# 2. Cannot transform new data (no transform method)
# 3. Distances in t-SNE space are not meaningful
# 4. Cluster sizes in t-SNE don't reflect true sizes
# 5. Should PCA first for high-dimensional data

# PCA preprocessing for high-dimensional data
pca_50 = PCA(n_components=50)
X_pca_50 = pca_50.fit_transform(X_scaled)

tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_pca_50)
```

---

## UMAP

```python
# UMAP: Uniform Manifold Approximation and Projection
# Better than t-SNE for: speed, preserving global structure, transforming new data

try:
    from umap import UMAP
    
    umap = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    X_umap = umap.fit_transform(X_scaled)
    
    # Unlike t-SNE, UMAP can transform new data
    X_new_umap = umap.transform(X_scaled[:10])
except ImportError:
    print("UMAP not installed. Install with: pip install umap-learn")
```

---

## Linear Discriminant Analysis (LDA)

### 1. Supervised Dimensionality Reduction

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# LDA uses class labels for dimensionality reduction
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X_scaled, y)  # Note: requires y!

# Maximum components = min(n_classes - 1, n_features)
n_classes = len(np.unique(y))
max_components = n_classes - 1
print(f"Max LDA components: {max_components}")
```

### 2. PCA vs LDA

```python
# PCA: unsupervised, maximizes variance
# LDA: supervised, maximizes class separation
```

---

## Other Methods

### 1. Factor Analysis

```python
from sklearn.decomposition import FactorAnalysis

fa = FactorAnalysis(n_components=2, random_state=42)
X_fa = fa.fit_transform(X_scaled)

# Noise variance per feature
print(f"Noise variance: {fa.noise_variance_}")
```

### 2. Independent Component Analysis (ICA)

```python
from sklearn.decomposition import FastICA

# ICA finds statistically independent components
ica = FastICA(n_components=2, random_state=42)
X_ica = ica.fit_transform(X_scaled)
```

### 3. Non-negative Matrix Factorization (NMF)

```python
from sklearn.decomposition import NMF

# NMF requires non-negative data
X_positive = X_scaled - X_scaled.min() + 0.01

nmf = NMF(n_components=2, random_state=42)
X_nmf = nmf.fit_transform(X_positive)
```

### 4. Truncated SVD (for Sparse Data)

```python
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import random as sparse_random

# TruncatedSVD works with sparse matrices
X_sparse = sparse_random(1000, 100, density=0.1, random_state=42)

svd = TruncatedSVD(n_components=10, random_state=42)
X_svd = svd.fit_transform(X_sparse)
```

---

## Comparison of Methods

| Method | Linear | Preserves | Scales | New Data | Use Case |
|--------|--------|-----------|--------|----------|----------|
| PCA | Yes | Global variance | Well | Yes | General |
| Kernel PCA | No | Non-linear | Medium | Yes | Non-linear |
| t-SNE | No | Local structure | Poorly | No | Visualization |
| UMAP | No | Local + global | Well | Yes | Visualization |
| LDA | Yes | Class separation | Well | Yes | Classification |
| ICA | Yes | Independence | Medium | Yes | Source separation |
| NMF | Yes | Parts | Medium | Yes | Interpretable |

---

## PyTorch Comparison

### 1. PCA in PyTorch

```python
import torch

def pca_pytorch(X, n_components):
    """PCA implementation in PyTorch"""
    X = torch.FloatTensor(X)
    
    # Center the data
    X_centered = X - X.mean(dim=0)
    
    # Compute covariance matrix
    cov = X_centered.T @ X_centered / (X.shape[0] - 1)
    
    # Eigendecomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    
    # Sort by eigenvalue (descending)
    idx = eigenvalues.argsort(descending=True)
    eigenvectors = eigenvectors[:, idx]
    
    # Select top components
    components = eigenvectors[:, :n_components]
    
    # Project data
    X_pca = X_centered @ components
    
    return X_pca.numpy(), components.numpy()
```

### 2. Autoencoder for Non-linear Dimensionality Reduction

```python
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )
    
    def encode(self, x):
        return self.encoder(x)
    
    def forward(self, x):
        z = self.encode(x)
        return self.decoder(z)
```

---

## Summary

**Choosing a method:**

1. **PCA**: Default choice for linear dimensionality reduction
2. **t-SNE/UMAP**: Best for 2D/3D visualization
3. **LDA**: When you have class labels and want supervised reduction
4. **Kernel PCA**: For non-linear relationships
5. **ICA**: When you need independent components
6. **Autoencoders**: For complex non-linear reduction

**Key considerations:**
- **Always scale** data before PCA/ICA
- **PCA first** for high-dimensional data before t-SNE
- **Choose n_components** based on explained variance
- **t-SNE distances are not meaningful** - only use for visualization
- **UMAP is faster** than t-SNE and preserves more global structure
