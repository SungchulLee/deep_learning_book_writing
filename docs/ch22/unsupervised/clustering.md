# Clustering

Clustering groups similar data points together without labeled examples. It's a fundamental unsupervised learning technique used for customer segmentation, anomaly detection, data exploration, and feature engineering.

---

## K-Means Clustering

### 1. Basic Usage

```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)

# Scale features (important for distance-based methods)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit K-Means
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
y_pred = kmeans.fit_predict(X_scaled)

# Cluster centers
print(f"Cluster centers shape: {kmeans.cluster_centers_.shape}")
print(f"Inertia (within-cluster sum of squares): {kmeans.inertia_:.2f}")
```

### 2. Visualization

```python
# Plot clusters
plt.figure(figsize=(10, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_pred, cmap='viridis', alpha=0.6)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            c='red', marker='X', s=200, edgecolors='black', linewidths=2,
            label='Centroids')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-Means Clustering')
plt.legend()
plt.show()
```

### 3. Elbow Method

```python
# Find optimal number of clusters
inertias = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
plt.show()
```

### 4. Silhouette Score

```python
from sklearn.metrics import silhouette_score, silhouette_samples

# Silhouette score ranges from -1 to 1
# Higher is better (tight clusters, well separated)
silhouette_scores = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)
    print(f"k={k}: Silhouette Score = {score:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), silhouette_scores, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs Number of Clusters')
plt.grid(True)
plt.show()
```

### 5. K-Means++ Initialization

```python
# K-Means++ is the default (smarter centroid initialization)
kmeans_plus = KMeans(n_clusters=4, init='k-means++', random_state=42)

# Random initialization (may require more iterations)
kmeans_random = KMeans(n_clusters=4, init='random', random_state=42)

# Custom initialization
initial_centers = X_scaled[:4]  # First 4 points
kmeans_custom = KMeans(n_clusters=4, init=initial_centers, n_init=1)
```

### 6. Mini-Batch K-Means (Large Datasets)

```python
from sklearn.cluster import MiniBatchKMeans

# Much faster for large datasets
mbkmeans = MiniBatchKMeans(n_clusters=4, batch_size=100, random_state=42)
y_pred_mb = mbkmeans.fit_predict(X_scaled)

print(f"Standard K-Means inertia: {kmeans.inertia_:.2f}")
print(f"Mini-Batch K-Means inertia: {mbkmeans.inertia_:.2f}")
```

---

## DBSCAN

### 1. Basic Usage

```python
from sklearn.cluster import DBSCAN

# DBSCAN: Density-Based Spatial Clustering
# Automatically determines number of clusters
# Can find arbitrary shaped clusters
# Identifies outliers (noise points)

dbscan = DBSCAN(eps=0.5, min_samples=5)
y_pred = dbscan.fit_predict(X_scaled)

# Labels: -1 means noise (outlier)
n_clusters = len(set(y_pred)) - (1 if -1 in y_pred else 0)
n_noise = list(y_pred).count(-1)

print(f"Number of clusters: {n_clusters}")
print(f"Number of noise points: {n_noise}")
```

### 2. Core vs Border vs Noise Points

```python
# Core samples (have at least min_samples neighbors within eps)
core_samples_mask = np.zeros_like(y_pred, dtype=bool)
core_samples_mask[dbscan.core_sample_indices_] = True

# Visualize
plt.figure(figsize=(10, 6))

# Core points
plt.scatter(X_scaled[core_samples_mask, 0], X_scaled[core_samples_mask, 1],
            c=y_pred[core_samples_mask], cmap='viridis', marker='o', s=60,
            label='Core')

# Border points
border_mask = ~core_samples_mask & (y_pred != -1)
plt.scatter(X_scaled[border_mask, 0], X_scaled[border_mask, 1],
            c=y_pred[border_mask], cmap='viridis', marker='s', s=40,
            label='Border')

# Noise points
plt.scatter(X_scaled[y_pred == -1, 0], X_scaled[y_pred == -1, 1],
            c='red', marker='x', s=50, label='Noise')

plt.title('DBSCAN Clustering')
plt.legend()
plt.show()
```

### 3. Tuning eps and min_samples

```python
from sklearn.neighbors import NearestNeighbors

# Estimate eps using k-distance graph
k = 5
nbrs = NearestNeighbors(n_neighbors=k).fit(X_scaled)
distances, indices = nbrs.kneighbors(X_scaled)

# Sort and plot k-th nearest neighbor distance
k_distances = np.sort(distances[:, k-1])

plt.figure(figsize=(10, 6))
plt.plot(k_distances)
plt.xlabel('Points sorted by distance')
plt.ylabel(f'{k}-th nearest neighbor distance')
plt.title('K-Distance Graph (Elbow = good eps)')
plt.grid(True)
plt.show()
```

### 4. DBSCAN vs K-Means

```python
from sklearn.datasets import make_moons

# Generate non-convex data
X_moons, y_moons = make_moons(n_samples=200, noise=0.05, random_state=42)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# K-Means (fails on non-convex shapes)
kmeans = KMeans(n_clusters=2, random_state=42)
y_kmeans = kmeans.fit_predict(X_moons)
axes[0].scatter(X_moons[:, 0], X_moons[:, 1], c=y_kmeans, cmap='viridis')
axes[0].set_title('K-Means (fails)')

# DBSCAN (handles non-convex shapes)
dbscan = DBSCAN(eps=0.2, min_samples=5)
y_dbscan = dbscan.fit_predict(X_moons)
axes[1].scatter(X_moons[:, 0], X_moons[:, 1], c=y_dbscan, cmap='viridis')
axes[1].set_title('DBSCAN (succeeds)')

plt.tight_layout()
plt.show()
```

---

## Hierarchical Clustering

### 1. Agglomerative Clustering

```python
from sklearn.cluster import AgglomerativeClustering

# Bottom-up hierarchical clustering
agg = AgglomerativeClustering(n_clusters=4, linkage='ward')
y_pred = agg.fit_predict(X_scaled)

print(f"Cluster labels: {np.unique(y_pred)}")
```

### 2. Linkage Methods

```python
linkages = ['ward', 'complete', 'average', 'single']

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for ax, linkage in zip(axes, linkages):
    agg = AgglomerativeClustering(n_clusters=4, linkage=linkage)
    y_pred = agg.fit_predict(X_scaled)
    ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_pred, cmap='viridis')
    ax.set_title(f'Linkage: {linkage}')

plt.tight_layout()
plt.show()
```

### 3. Dendrogram

```python
from scipy.cluster.hierarchy import dendrogram, linkage

# Compute linkage matrix
Z = linkage(X_scaled, method='ward')

# Plot dendrogram
plt.figure(figsize=(12, 6))
dendrogram(Z, truncate_mode='lastp', p=30)
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.title('Hierarchical Clustering Dendrogram')
plt.show()
```

### 4. Cutting the Dendrogram

```python
from scipy.cluster.hierarchy import fcluster

# Cut at a specific number of clusters
labels_4 = fcluster(Z, t=4, criterion='maxclust')

# Cut at a specific distance
labels_dist = fcluster(Z, t=5.0, criterion='distance')

print(f"Clusters (n=4): {np.unique(labels_4)}")
print(f"Clusters (distance=5.0): {np.unique(labels_dist)}")
```

---

## Gaussian Mixture Models

### 1. Basic Usage

```python
from sklearn.mixture import GaussianMixture

# Soft clustering with probability assignments
gmm = GaussianMixture(n_components=4, random_state=42)
y_pred = gmm.fit_predict(X_scaled)

# Probability of belonging to each cluster
proba = gmm.predict_proba(X_scaled)
print(f"Probability shape: {proba.shape}")
print(f"First sample probabilities: {proba[0]}")
```

### 2. Covariance Types

```python
covariance_types = ['full', 'tied', 'diag', 'spherical']

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for ax, cov_type in zip(axes, covariance_types):
    gmm = GaussianMixture(n_components=4, covariance_type=cov_type, random_state=42)
    y_pred = gmm.fit_predict(X_scaled)
    ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_pred, cmap='viridis')
    ax.set_title(f'Covariance: {cov_type}')

plt.tight_layout()
plt.show()
```

### 3. Model Selection with BIC/AIC

```python
# Use BIC (Bayesian Information Criterion) to select number of components
n_components_range = range(1, 10)
bics = []
aics = []

for n in n_components_range:
    gmm = GaussianMixture(n_components=n, random_state=42)
    gmm.fit(X_scaled)
    bics.append(gmm.bic(X_scaled))
    aics.append(gmm.aic(X_scaled))

plt.figure(figsize=(10, 6))
plt.plot(n_components_range, bics, 'b-', label='BIC')
plt.plot(n_components_range, aics, 'r-', label='AIC')
plt.xlabel('Number of Components')
plt.ylabel('Information Criterion')
plt.title('GMM Model Selection')
plt.legend()
plt.grid(True)
plt.show()

print(f"Optimal components (BIC): {n_components_range[np.argmin(bics)]}")
```

### 4. GMM for Anomaly Detection

```python
# Fit GMM and compute log-likelihood
gmm = GaussianMixture(n_components=4, random_state=42)
gmm.fit(X_scaled)

# Score samples (log probability density)
scores = gmm.score_samples(X_scaled)

# Low scores indicate outliers
threshold = np.percentile(scores, 5)  # Bottom 5%
outliers = scores < threshold

plt.figure(figsize=(10, 6))
plt.scatter(X_scaled[~outliers, 0], X_scaled[~outliers, 1], 
            c='blue', alpha=0.6, label='Normal')
plt.scatter(X_scaled[outliers, 0], X_scaled[outliers, 1], 
            c='red', marker='x', s=100, label='Anomaly')
plt.title('GMM Anomaly Detection')
plt.legend()
plt.show()
```

---

## Spectral Clustering

### 1. Basic Usage

```python
from sklearn.cluster import SpectralClustering

# Good for non-convex clusters
# Uses graph Laplacian
spectral = SpectralClustering(n_clusters=4, affinity='rbf', random_state=42)
y_pred = spectral.fit_predict(X_scaled)

plt.figure(figsize=(10, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_pred, cmap='viridis')
plt.title('Spectral Clustering')
plt.show()
```

### 2. Different Affinities

```python
# Spectral clustering on moons data
X_moons_scaled = StandardScaler().fit_transform(X_moons)

affinities = ['rbf', 'nearest_neighbors']

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, affinity in zip(axes, affinities):
    if affinity == 'nearest_neighbors':
        spectral = SpectralClustering(n_clusters=2, affinity=affinity, 
                                       n_neighbors=10, random_state=42)
    else:
        spectral = SpectralClustering(n_clusters=2, affinity=affinity, 
                                       random_state=42)
    y_pred = spectral.fit_predict(X_moons_scaled)
    ax.scatter(X_moons_scaled[:, 0], X_moons_scaled[:, 1], c=y_pred, cmap='viridis')
    ax.set_title(f'Affinity: {affinity}')

plt.tight_layout()
plt.show()
```

---

## HDBSCAN (Advanced)

### 1. Basic Usage

```python
# Note: HDBSCAN needs to be installed separately
# pip install hdbscan

try:
    from sklearn.cluster import HDBSCAN
    
    # Hierarchical DBSCAN - no need to specify eps
    hdbscan = HDBSCAN(min_cluster_size=5, min_samples=5)
    y_pred = hdbscan.fit_predict(X_scaled)
    
    n_clusters = len(set(y_pred)) - (1 if -1 in y_pred else 0)
    print(f"Number of clusters: {n_clusters}")
    
except ImportError:
    print("HDBSCAN not available in this sklearn version")
```

---

## Clustering Evaluation

### 1. External Metrics (When Labels Available)

```python
from sklearn.metrics import (
    adjusted_rand_score, normalized_mutual_info_score,
    homogeneity_score, completeness_score, v_measure_score
)

# When true labels are available
kmeans = KMeans(n_clusters=4, random_state=42)
y_pred = kmeans.fit_predict(X_scaled)

print(f"Adjusted Rand Index: {adjusted_rand_score(y_true, y_pred):.4f}")
print(f"Normalized Mutual Info: {normalized_mutual_info_score(y_true, y_pred):.4f}")
print(f"Homogeneity: {homogeneity_score(y_true, y_pred):.4f}")
print(f"Completeness: {completeness_score(y_true, y_pred):.4f}")
print(f"V-Measure: {v_measure_score(y_true, y_pred):.4f}")
```

### 2. Internal Metrics (No Labels)

```python
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score
)

# When true labels are NOT available
print(f"Silhouette Score: {silhouette_score(X_scaled, y_pred):.4f}")
print(f"Calinski-Harabasz Index: {calinski_harabasz_score(X_scaled, y_pred):.4f}")
print(f"Davies-Bouldin Index: {davies_bouldin_score(X_scaled, y_pred):.4f}")
```

---

## Comparison of Methods

| Method | Cluster Shape | # Clusters | Outliers | Scalability |
|--------|---------------|------------|----------|-------------|
| K-Means | Spherical | Must specify | No | Good |
| DBSCAN | Arbitrary | Auto | Yes | Medium |
| Hierarchical | Any | Must specify | No | Poor |
| GMM | Elliptical | Must specify | Soft | Good |
| Spectral | Arbitrary | Must specify | No | Poor |
| HDBSCAN | Arbitrary | Auto | Yes | Medium |

---

## PyTorch Comparison

### 1. K-Means in PyTorch

```python
import torch

def kmeans_pytorch(X, n_clusters, n_iters=100):
    """K-Means implementation in PyTorch"""
    X = torch.FloatTensor(X)
    n_samples = X.shape[0]
    
    # Initialize centroids randomly
    indices = torch.randperm(n_samples)[:n_clusters]
    centroids = X[indices].clone()
    
    for _ in range(n_iters):
        # Compute distances to centroids
        distances = torch.cdist(X, centroids)
        
        # Assign to nearest centroid
        labels = distances.argmin(dim=1)
        
        # Update centroids
        new_centroids = torch.stack([
            X[labels == k].mean(dim=0) if (labels == k).sum() > 0 
            else centroids[k]
            for k in range(n_clusters)
        ])
        
        # Check convergence
        if torch.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    
    return labels.numpy(), centroids.numpy()

# Usage
labels, centers = kmeans_pytorch(X_scaled, n_clusters=4)
```

### 2. Deep Clustering (DEC Concept)

```python
import torch.nn as nn

class DeepClustering(nn.Module):
    """Deep Embedded Clustering concept"""
    def __init__(self, input_dim, latent_dim, n_clusters):
        super().__init__()
        # Autoencoder for feature learning
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
        # Cluster centers in latent space
        self.cluster_centers = nn.Parameter(torch.randn(n_clusters, latent_dim))
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z
    
    def get_cluster_assignments(self, x):
        z = self.encode(x)
        distances = torch.cdist(z, self.cluster_centers)
        return distances.argmin(dim=1)
```

---

## Summary

**Choosing a clustering algorithm:**

1. **Start with K-Means** for simple, spherical clusters
2. **Use DBSCAN/HDBSCAN** for arbitrary shapes and outlier detection
3. **Try GMM** for soft assignments and probability estimates
4. **Consider Spectral Clustering** for complex manifolds
5. **Use Hierarchical** when you need a dendrogram

**Key preprocessing:**
- **Always scale** your features for distance-based methods
- Consider **dimensionality reduction** (PCA) for high-dimensional data
- Handle **outliers** before clustering (or use DBSCAN)

**Evaluation:**
- Use **silhouette score** for internal validation
- Use **ARI/NMI** when ground truth is available
- Visualize clusters to sanity check results
