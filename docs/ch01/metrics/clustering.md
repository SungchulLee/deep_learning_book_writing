# Clustering Metrics

Evaluating clusters is harder than evaluating supervised models because ground-truth labels are often unavailable. This page covers the key internal and external metrics used to assess clustering quality.

## Definition

Clustering metrics quantify how well a partition of data points into groups captures meaningful structure. **Internal metrics** use only the data and cluster assignments; **external metrics** compare assignments against known labels. The silhouette score is the most widely used internal metric:

$$
s(i) = \frac{b(i) - a(i)}{\max(a(i),\; b(i))}
$$

where $a(i)$ is the mean intra-cluster distance and $b(i)$ is the mean nearest-cluster distance for point $i$. Values range from $-1$ (misassigned) to $+1$ (well-clustered).

## Explanation

**Internal metrics** (no labels required):

- **Silhouette score**: Balances cohesion ($a$) and separation ($b$). Average over all points gives the overall score. Higher is better.
- **Calinski-Harabasz index**: Ratio of between-cluster to within-cluster variance, scaled by degrees of freedom. Higher means more compact and separated clusters.
- **Davies-Bouldin index**: Average worst-case cluster similarity. Lower is better.

**External metrics** (labels required):

- **Adjusted Rand Index (ARI)**: Measures pairwise agreement between predicted and true labels, corrected for chance. ARI $= 1$ is perfect; ARI $\approx 0$ is random.
- **Normalized Mutual Information (NMI)**: Information-theoretic measure of shared information between clusterings, normalized to $[0, 1]$.

Choosing $k$: compute silhouette scores for a range of $k$ values and select the $k$ that maximizes it.

## Examples

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                             davies_bouldin_score, adjusted_rand_score)

X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=42)
X_scaled = StandardScaler().fit_transform(X)

# Fit KMeans and compute all metrics
km = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = km.fit_predict(X_scaled)

print(f"Silhouette:        {silhouette_score(X_scaled, labels):.4f}")
print(f"Calinski-Harabasz: {calinski_harabasz_score(X_scaled, labels):.1f}")
print(f"Davies-Bouldin:    {davies_bouldin_score(X_scaled, labels):.4f}")
print(f"Adjusted Rand:     {adjusted_rand_score(y_true, labels):.4f}")

# Select k using silhouette score
for k in range(2, 8):
    lab = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_scaled)
    print(f"  k={k}: silhouette={silhouette_score(X_scaled, lab):.4f}")
```
