# K-Nearest Neighbors

K-Nearest Neighbors (KNN) is a non-parametric method that predicts based on the closest training examples. It serves as an important baseline and illustrates the bias-variance tradeoff that motivates more sophisticated models in deep learning.

## Definition

For a query point $\mathbf{x}$, KNN finds the $k$ nearest training points $\mathcal{N}_k(\mathbf{x})$ under a chosen distance metric. For classification, it predicts by majority vote:

$$
\hat{y} = \arg\max_{c} \sum_{i \in \mathcal{N}_k(\mathbf{x})} \mathbf{1}[y_i = c]
$$

For regression, it predicts the (weighted) mean of neighbor targets:

$$
\hat{y} = \frac{\sum_{i \in \mathcal{N}_k(\mathbf{x})} w_i \, y_i}{\sum_{i \in \mathcal{N}_k(\mathbf{x})} w_i}
$$

## Explanation

KNN has no training phase -- it stores all data and computes distances at prediction time. This makes it a **lazy learner**. Key considerations:

- **Feature scaling is mandatory**: Distance metrics are sensitive to feature magnitudes. Always standardize before applying KNN.
- **Choosing $k$**: Small $k$ gives low bias but high variance (sensitive to noise). Large $k$ gives high bias but low variance (over-smoothed boundaries). Select $k$ via cross-validation.
- **Curse of dimensionality**: In high dimensions, distances between points become nearly uniform, making nearest-neighbor queries uninformative. This is a key motivation for learning low-dimensional representations with neural networks.
- **Computational cost**: Brute-force search costs $O(nd)$ per query. Tree structures (KD-tree, Ball-tree) reduce this to $O(d \log n)$ in low dimensions but degrade in high dimensions.

## Examples

```python
import torch
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

# Generate data and scale
X, y = make_classification(n_samples=500, n_features=10, n_informative=5,
                           n_classes=3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Select k via cross-validation
best_k, best_score = 1, 0.0
for k in range(1, 21):
    score = cross_val_score(KNeighborsClassifier(n_neighbors=k),
                            X_train_s, y_train, cv=5).mean()
    if score > best_score:
        best_k, best_score = k, score
print(f"Best k={best_k}, CV accuracy={best_score:.4f}")

knn = KNeighborsClassifier(n_neighbors=best_k, weights="distance")
knn.fit(X_train_s, y_train)
print(f"Test accuracy: {knn.score(X_test_s, y_test):.4f}")

# KNN in PyTorch (manual implementation for learned embeddings)
X_tr = torch.tensor(X_train_s, dtype=torch.float32)
X_te = torch.tensor(X_test_s, dtype=torch.float32)
y_tr = torch.tensor(y_train)

dists = torch.cdist(X_te[:5], X_tr)  # pairwise distances
_, idx = dists.topk(best_k, largest=False)
neighbor_labels = y_tr[idx]
print(f"Neighbor labels for first 5 queries:\n{neighbor_labels}")
```
