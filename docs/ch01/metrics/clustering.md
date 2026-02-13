# Clustering Metrics

Evaluating clusters is fundamentally harder than evaluating supervised models because there are no ground-truth labels in most practical settings. Metrics fall into two categories: **internal** (structure-based, no labels needed) and **external** (compare to known labels when available).

## Internal Metrics (No Ground Truth)

### Silhouette Score

Measures how similar each point is to its own cluster versus the nearest other cluster. For point $i$:

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

where $a(i)$ is the mean distance to points in the same cluster and $b(i)$ is the mean distance to points in the nearest other cluster.

- Range: $[-1, 1]$. Higher is better; near 0 means overlapping clusters; negative means misassignment.

```python
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import numpy as np

X, y_true = make_blobs(n_samples=500, centers=4, cluster_std=0.8, random_state=42)
X_scaled = StandardScaler().fit_transform(X)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

score = silhouette_score(X_scaled, labels)
print(f"Silhouette Score: {score:.4f}")

# Per-sample silhouette values
sample_scores = silhouette_samples(X_scaled, labels)
print(f"Min: {sample_scores.min():.3f}, Max: {sample_scores.max():.3f}")
```

### Choosing $k$ with Silhouette

```python
k_range = range(2, 10)
silhouette_scores = []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

best_k = k_range[np.argmax(silhouette_scores)]
print(f"Best k = {best_k}, silhouette = {max(silhouette_scores):.4f}")
```

### Calinski–Harabasz Index

Ratio of between-cluster dispersion to within-cluster dispersion (higher is better):

$$\text{CH} = \frac{\text{tr}(B_K)}{\text{tr}(W_K)} \cdot \frac{n - K}{K - 1}$$

where $B_K$ and $W_K$ are the between- and within-cluster scatter matrices.

```python
from sklearn.metrics import calinski_harabasz_score

ch_score = calinski_harabasz_score(X_scaled, labels)
print(f"Calinski-Harabasz: {ch_score:.2f}")
```

### Davies–Bouldin Index

Average similarity between each cluster and its most similar cluster (lower is better):

$$\text{DB} = \frac{1}{K}\sum_{i=1}^{K} \max_{j \neq i} \frac{s_i + s_j}{d(c_i, c_j)}$$

where $s_i$ is the average distance within cluster $i$ and $d(c_i, c_j)$ is the distance between centroids.

```python
from sklearn.metrics import davies_bouldin_score

db_score = davies_bouldin_score(X_scaled, labels)
print(f"Davies-Bouldin: {db_score:.4f}")
```

### Elbow Method (Inertia)

Not a metric per se, but the within-cluster sum of squares (WCSS) is commonly plotted:

```python
inertias = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

# Look for "elbow" in the plot
import matplotlib.pyplot as plt
plt.plot(range(1, 11), inertias, 'bo-')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.title('Elbow Method')
```

## External Metrics (Ground Truth Available)

When true labels exist (e.g., for validating a clustering algorithm on labelled data):

### Adjusted Rand Index (ARI)

Measures agreement between predicted and true labels, adjusted for chance:

$$\text{ARI} = \frac{\text{RI} - \mathbb{E}[\text{RI}]}{\max(\text{RI}) - \mathbb{E}[\text{RI}]}$$

- Range: $[-1, 1]$. ARI $= 1$ means perfect agreement; ARI $\approx 0$ means random.

```python
from sklearn.metrics import adjusted_rand_score

ari = adjusted_rand_score(y_true, labels)
print(f"Adjusted Rand Index: {ari:.4f}")
```

### Normalised Mutual Information (NMI)

```python
from sklearn.metrics import normalized_mutual_info_score

nmi = normalized_mutual_info_score(y_true, labels)
print(f"NMI: {nmi:.4f}")
```

### Adjusted Mutual Information (AMI)

```python
from sklearn.metrics import adjusted_mutual_info_score

ami = adjusted_mutual_info_score(y_true, labels)
print(f"AMI: {ami:.4f}")
```

### Homogeneity, Completeness, V-Measure

```python
from sklearn.metrics import homogeneity_completeness_v_measure

h, c, v = homogeneity_completeness_v_measure(y_true, labels)
print(f"Homogeneity: {h:.4f}")    # each cluster contains only one class
print(f"Completeness: {c:.4f}")   # all members of a class are in one cluster
print(f"V-Measure: {v:.4f}")      # harmonic mean of h and c
```

### Fowlkes–Mallows Index

```python
from sklearn.metrics import fowlkes_mallows_score

fm = fowlkes_mallows_score(y_true, labels)
print(f"Fowlkes-Mallows: {fm:.4f}")
```

## Comprehensive Comparison

```python
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

algorithms = {
    'KMeans': KMeans(n_clusters=4, random_state=42, n_init=10),
    'DBSCAN': DBSCAN(eps=0.5, min_samples=5),
    'Agglom': AgglomerativeClustering(n_clusters=4),
}

for name, algo in algorithms.items():
    pred = algo.fit_predict(X_scaled)
    n_clusters = len(set(pred)) - (1 if -1 in pred else 0)
    
    if n_clusters > 1:
        sil = silhouette_score(X_scaled, pred)
        ch = calinski_harabasz_score(X_scaled, pred)
        db = davies_bouldin_score(X_scaled, pred)
        ari = adjusted_rand_score(y_true, pred)
        print(f"{name:10s}: k={n_clusters}, Sil={sil:.3f}, CH={ch:.0f}, "
              f"DB={db:.3f}, ARI={ari:.3f}")
```

## Quantitative Finance: Regime Clustering Evaluation

When clustering market regimes (bull, bear, volatile, quiet), internal metrics guide the choice of $k$ since true regime labels are unavailable:

```python
# Features: rolling volatility, returns, correlation, spread
# Evaluate regime clusters using multiple internal metrics

results = []
for k in range(2, 7):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    regime_labels = km.fit_predict(market_features_scaled)
    
    results.append({
        'k': k,
        'silhouette': silhouette_score(market_features_scaled, regime_labels),
        'calinski_harabasz': calinski_harabasz_score(market_features_scaled, regime_labels),
        'davies_bouldin': davies_bouldin_score(market_features_scaled, regime_labels),
        'inertia': km.inertia_,
    })

import pandas as pd
print(pd.DataFrame(results).round(3))
```

## Summary

| Metric | Labels Needed | Range | Better | Use For |
|--------|:---:|-------|--------|---------|
| Silhouette | No | $[-1, 1]$ | Higher | Cluster compactness + separation |
| Calinski–Harabasz | No | $[0, \infty)$ | Higher | Compact, well-separated clusters |
| Davies–Bouldin | No | $[0, \infty)$ | Lower | Cluster similarity |
| Inertia (elbow) | No | $[0, \infty)$ | Lower | Quick $k$ selection |
| Adjusted Rand | Yes | $[-1, 1]$ | Higher | Agreement with ground truth |
| NMI / AMI | Yes | $[0, 1]$ | Higher | Information-theoretic agreement |
| V-Measure | Yes | $[0, 1]$ | Higher | Homogeneity + completeness |
