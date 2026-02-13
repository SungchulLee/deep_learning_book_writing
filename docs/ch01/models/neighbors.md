# Neighbors

$k$-Nearest Neighbors ($k$-NN) is a non-parametric method that predicts based on the $k$ closest training examples. It makes no assumptions about the data distribution, naturally handles multi-class problems, and serves as a strong baseline for both classification and regression on small-to-medium datasets.

## KNeighborsClassifier

### Basic Usage

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

X, y = make_classification(n_samples=1000, n_features=10, n_informative=5,
                           n_classes=3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling is critical for distance-based methods
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

print(f"Accuracy: {knn.score(X_test_scaled, y_test):.4f}")
```

### Prediction Mechanism

For a query point $\mathbf{x}$, find the $k$ nearest training points $\mathcal{N}_k(\mathbf{x})$ and predict by majority vote:

$$\hat{y} = \arg\max_{c} \sum_{i \in \mathcal{N}_k(\mathbf{x})} \mathbf{1}[y_i = c]$$

```python
# Probabilities: fraction of neighbors in each class
y_proba = knn.predict_proba(X_test_scaled[:5])
print(y_proba.round(3))
```

### Distance Weighting

Weight closer neighbours more heavily:

```python
knn_weighted = KNeighborsClassifier(n_neighbors=5, weights='distance')
knn_weighted.fit(X_train_scaled, y_train)

# Weighted vote: w_i = 1/d(x, x_i)
print(f"Uniform: {knn.score(X_test_scaled, y_test):.4f}")
print(f"Weighted: {knn_weighted.score(X_test_scaled, y_test):.4f}")
```

### Choosing $k$

```python
from sklearn.model_selection import cross_val_score

k_range = range(1, 31)
cv_scores = []

for k in k_range:
    knn_k = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn_k, X_train_scaled, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

best_k = k_range[np.argmax(cv_scores)]
print(f"Best k = {best_k}, CV accuracy = {max(cv_scores):.4f}")
```

**Trade-off**: small $k$ → low bias, high variance (sensitive to noise); large $k$ → high bias, low variance (over-smoothed boundaries).

## KNeighborsRegressor

```python
from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=500, n_features=5, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

knn_reg = KNeighborsRegressor(n_neighbors=5, weights='distance')
knn_reg.fit(X_train_s, y_train)

print(f"R²: {knn_reg.score(X_test_s, y_test):.4f}")
```

Prediction is the (weighted) mean of the $k$ nearest targets:

$$\hat{y} = \frac{\sum_{i \in \mathcal{N}_k(\mathbf{x})} w_i \, y_i}{\sum_{i \in \mathcal{N}_k(\mathbf{x})} w_i}$$

## Distance Metrics

```python
# Minkowski distance (generalises Euclidean and Manhattan)
# p=2 → Euclidean (default), p=1 → Manhattan
knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)

# Explicit metrics
knn_manhattan = KNeighborsClassifier(metric='manhattan')
knn_chebyshev = KNeighborsClassifier(metric='chebyshev')   # L∞

# Custom metric
from sklearn.metrics import DistanceMetric
knn_mahal = KNeighborsClassifier(metric='mahalanobis',
                                  metric_params={'V': np.cov(X_train_s.T)})
```

| Metric | Formula | Best For |
|--------|---------|----------|
| Euclidean ($L_2$) | $\sqrt{\sum (x_i - y_i)^2}$ | Default, isotropic data |
| Manhattan ($L_1$) | $\sum |x_i - y_i|$ | High-dimensional, sparse |
| Chebyshev ($L_\infty$) | $\max |x_i - y_i|$ | Grid-like structure |
| Mahalanobis | Accounts for covariance | Correlated features |

## Efficient Search: BallTree and KDTree

For large datasets, brute-force $O(n \cdot d)$ per query is too slow. Tree structures enable $O(\log n)$ lookups:

```python
# Auto-select algorithm
knn = KNeighborsClassifier(n_neighbors=5, algorithm='auto')
# 'auto' chooses among: 'ball_tree', 'kd_tree', 'brute'

# Explicit
knn_bt = KNeighborsClassifier(algorithm='ball_tree', leaf_size=30)
knn_kd = KNeighborsClassifier(algorithm='kd_tree', leaf_size=30)
```

| Algorithm | Time Complexity | Best For |
|-----------|----------------|----------|
| `brute` | $O(n \cdot d)$ per query | Small $n$ or high $d$ |
| `kd_tree` | $O(d \cdot \log n)$ average | Low $d$ ($< 20$) |
| `ball_tree` | $O(d \cdot \log n)$ | Any metric, moderate $d$ |

**Curse of dimensionality**: tree-based speedups degrade when $d$ is large because the fraction of the space explored grows exponentially. For $d > 20$, brute force is often competitive.

### Direct BallTree Usage

```python
from sklearn.neighbors import BallTree

tree = BallTree(X_train_s, leaf_size=40, metric='euclidean')

# Query: distances and indices of 5 nearest neighbors
distances, indices = tree.query(X_test_s[:5], k=5)
print(f"Nearest distances: {distances[0].round(3)}")
print(f"Nearest indices: {indices[0]}")
```

## Radius Neighbors

Query all points within a fixed radius instead of a fixed $k$:

```python
from sklearn.neighbors import RadiusNeighborsClassifier

rnn = RadiusNeighborsClassifier(radius=1.0, weights='distance',
                                 outlier_label='most_frequent')
rnn.fit(X_train_s, y_train)
```

Useful when the density of points varies significantly across the feature space.

## Pipeline Example

```python
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier()),
])

from sklearn.model_selection import GridSearchCV

param_grid = {
    'knn__n_neighbors': [3, 5, 7, 11, 15],
    'knn__weights': ['uniform', 'distance'],
    'knn__metric': ['euclidean', 'manhattan'],
}

grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)
print(f"Best: {grid.best_params_}, Accuracy: {grid.best_score_:.4f}")
```

## Quantitative Finance: KNN for Missing Data and Similarity

### KNN Imputation

$k$-NN is used by `KNNImputer` to fill missing values based on feature similarity across assets:

```python
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5, weights='distance')
X_imputed = imputer.fit_transform(X_with_missing)
```

### Peer Group Identification

In equity research, KNN identifies "peer" companies based on fundamental characteristics (market cap, P/E, sector exposure) for relative valuation:

```python
# Find 10 most similar companies to a target
tree = BallTree(fundamentals_scaled)
distances, peer_indices = tree.query(target_company.reshape(1, -1), k=10)
peer_companies = company_names[peer_indices[0]]
```

## Summary

| Aspect | Detail |
|--------|--------|
| **Type** | Non-parametric, instance-based |
| **Training** | $O(1)$ — just stores data |
| **Prediction** | $O(n \cdot d)$ brute; $O(d \log n)$ with trees |
| **Scaling** | Required (distance-based) |
| **Key hyperparameter** | $k$ (number of neighbours) |
| **Strengths** | No training, naturally non-linear, multi-class |
| **Weaknesses** | Slow prediction, curse of dimensionality, memory-intensive |
