# Transformers

Transformers create new feature representations from raw data. This section covers polynomial expansion, discretisation, power transforms, dimensionality reduction (PCA, t-SNE, UMAP), and temporal/text feature engineering.

## Polynomial Features

Polynomial expansion captures non-linear relationships and feature interactions without changing the model class:

```python
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

X = np.array([[1, 2], [3, 4], [5, 6]])

# Degree 2: adds x₁², x₂², x₁·x₂, plus bias
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

print(f"Original: {X.shape} → Polynomial: {X_poly.shape}")
print(f"Features: {poly.get_feature_names_out()}")
# ['1', 'x0', 'x1', 'x0^2', 'x0 x1', 'x1^2']
```

### Interaction Only

```python
# Only cross-terms, no powers (x₁·x₂ but not x₁²)
poly_inter = PolynomialFeatures(degree=2, interaction_only=True)
X_inter = poly_inter.fit_transform(X)
print(f"Features: {poly_inter.get_feature_names_out()}")
# ['1', 'x0', 'x1', 'x0 x1']
```

### Without Bias

```python
poly_no_bias = PolynomialFeatures(degree=2, include_bias=False)
X_nb = poly_no_bias.fit_transform(X)
# Omits the constant '1' column
```

**Caution**: Feature count grows as $\binom{d + p}{p}$ where $d$ is the original dimensionality and $p$ is the degree. For $d = 100, p = 2$: 5,151 features. Combine with feature selection or regularisation.

## Binning (Discretisation)

Convert continuous features to categorical bins:

```python
from sklearn.preprocessing import KBinsDiscretizer

X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])

# Uniform-width bins
binner = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
X_binned = binner.fit_transform(X)
print(f"Bin edges: {binner.bin_edges_[0].round(2)}")
```

### Strategies

```python
for strategy in ['uniform', 'quantile', 'kmeans']:
    binner = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy=strategy)
    X_binned = binner.fit_transform(X)
    print(f"{strategy:10s}: edges = {binner.bin_edges_[0].round(2)}")
```

| Strategy | Bin Width | Use When |
|----------|-----------|----------|
| `uniform` | Equal width | Uniformly distributed data |
| `quantile` | Equal frequency | Skewed distributions |
| `kmeans` | Cluster-based | Natural groupings |

### Encoding Options

```python
# 'ordinal': integer labels (0, 1, 2)
# 'onehot': sparse one-hot matrix
# 'onehot-dense': dense one-hot matrix
```

## Power Transforms

Make distributions more Gaussian to improve linear model performance:

### PowerTransformer

```python
from sklearn.preprocessing import PowerTransformer

# Right-skewed data (common in finance: prices, volumes)
X_skewed = np.random.exponential(scale=2, size=(1000, 2))

# Yeo-Johnson: handles negative values
pt_yj = PowerTransformer(method='yeo-johnson', standardize=True)
X_yj = pt_yj.fit_transform(X_skewed)
print(f"Optimal lambdas: {pt_yj.lambdas_}")

# Box-Cox: requires strictly positive data
pt_bc = PowerTransformer(method='box-cox', standardize=True)
X_bc = pt_bc.fit_transform(X_skewed)  # X must be > 0
```

The Box-Cox transform is $x' = (x^\lambda - 1)/\lambda$ for $\lambda \neq 0$ and $\log(x)$ for $\lambda = 0$, with $\lambda$ chosen by maximum likelihood.

### QuantileTransformer

```python
from sklearn.preprocessing import QuantileTransformer

# Map to uniform [0, 1] via empirical CDF
qt_uniform = QuantileTransformer(output_distribution='uniform', n_quantiles=1000)
X_uniform = qt_uniform.fit_transform(X_skewed)

# Map to standard normal
qt_normal = QuantileTransformer(output_distribution='normal', n_quantiles=1000)
X_normal = qt_normal.fit_transform(X_skewed)
```

Non-parametric and robust to outliers, but can distort distances between points.

## FunctionTransformer

Wrap any callable as a scikit-learn transformer:

```python
from sklearn.preprocessing import FunctionTransformer

log_transformer = FunctionTransformer(
    func=np.log1p,
    inverse_func=np.expm1,
    validate=True
)

X_log = log_transformer.fit_transform(X_skewed)
X_back = log_transformer.inverse_transform(X_log)
```

Useful for injecting custom one-liners into a pipeline without writing a full class.

## Dimensionality Reduction

### Principal Component Analysis (PCA)

PCA finds the orthogonal directions of maximum variance:

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
X_scaled = StandardScaler().fit_transform(X)

# Reduce to 2 components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total explained: {pca.explained_variance_ratio_.sum():.2%}")
```

#### Choosing the Number of Components

```python
# Keep 95% of variance
pca_auto = PCA(n_components=0.95)
X_auto = pca_auto.fit_transform(X_scaled)
print(f"Components needed for 95%: {pca_auto.n_components_}")

# Scree plot
import matplotlib.pyplot as plt
pca_full = PCA().fit(X_scaled)
plt.plot(np.cumsum(pca_full.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
plt.legend()
```

#### Loadings (Component Weights)

```python
# Each row of components_ is a principal direction
loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i+1}' for i in range(pca.n_components_)],
    index=feature_names
)
print(loadings.round(3))
```

### Incremental PCA

For datasets too large to fit in memory:

```python
from sklearn.decomposition import IncrementalPCA

ipca = IncrementalPCA(n_components=10, batch_size=200)
for batch in np.array_split(X_scaled, 10):
    ipca.partial_fit(batch)
X_reduced = ipca.transform(X_scaled)
```

### t-SNE

Non-linear embedding for **visualisation** (2D or 3D only):

```python
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.6)
plt.title('t-SNE Embedding')
```

**Key properties**: preserves local structure but not global distances; perplexity controls the effective number of neighbours; non-deterministic; cannot transform new data (no `transform` method — only `fit_transform`).

### UMAP

Uniform Manifold Approximation and Projection — faster than t-SNE, preserves more global structure, and supports `transform` on new data:

```python
# pip install umap-learn
from umap import UMAP

reducer = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
X_umap = reducer.fit_transform(X_scaled)

# Can transform new data
X_new_umap = reducer.transform(X_new_scaled)
```

### Truncated SVD

For sparse data (text, one-hot encoded) where PCA's centering would destroy sparsity:

```python
from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=50, random_state=42)
X_svd = svd.fit_transform(X_sparse)  # works with scipy.sparse
```

## Date/Time Feature Engineering

Temporal features are critical for financial time series:

```python
import pandas as pd

df = pd.DataFrame({
    'timestamp': pd.date_range('2024-01-01', periods=1000, freq='h')
})

# Calendar components
df['year'] = df['timestamp'].dt.year
df['month'] = df['timestamp'].dt.month
df['day'] = df['timestamp'].dt.day
df['hour'] = df['timestamp'].dt.hour
df['dayofweek'] = df['timestamp'].dt.dayofweek
df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)

# Cyclical encoding (prevents discontinuity at midnight, month boundaries)
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
```

## Rolling Window Features

```python
ts = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=252),
    'price': np.cumsum(np.random.randn(252)) + 100
})

# Rolling statistics
ts['sma_20'] = ts['price'].rolling(20).mean()
ts['std_20'] = ts['price'].rolling(20).std()
ts['returns'] = ts['price'].pct_change()

# Lag features
ts['lag_1'] = ts['returns'].shift(1)
ts['lag_5'] = ts['returns'].shift(5)

# Expanding statistics
ts['expanding_mean'] = ts['price'].expanding().mean()
```

## Text Features

```python
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

corpus = ['earnings beat expectations', 'revenue missed forecast']

# TF-IDF (most common for ML)
tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
X_tfidf = tfidf.fit_transform(corpus)

# Simple counts
count = CountVectorizer(max_features=1000)
X_count = count.fit_transform(corpus)

# Basic text statistics as features
df_text = pd.DataFrame({'text': corpus})
df_text['n_chars'] = df_text['text'].str.len()
df_text['n_words'] = df_text['text'].str.split().str.len()
```

## Custom Feature Transformer

```python
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Add log-transformed copies of positive features."""
    
    def __init__(self, add_log=True):
        self.add_log = add_log
    
    def fit(self, X, y=None):
        X = np.asarray(X)
        if self.add_log:
            self.positive_cols_ = [i for i in range(X.shape[1]) if (X[:, i] > 0).all()]
        self.n_features_in_ = X.shape[1]
        return self
    
    def transform(self, X):
        X = np.asarray(X)
        if self.add_log and self.positive_cols_:
            log_features = np.log1p(X[:, self.positive_cols_])
            return np.hstack([X, log_features])
        return X
```

## Summary

| Transformer | Input → Output | Use Case |
|-------------|----------------|----------|
| `PolynomialFeatures` | $d \to \binom{d+p}{p}$ | Non-linear relationships |
| `KBinsDiscretizer` | Continuous → categorical | Non-monotonic effects |
| `PowerTransformer` | Skewed → Gaussian | Improve linear models |
| `QuantileTransformer` | Any → uniform/normal | Robust to outliers |
| `FunctionTransformer` | Custom callable | Quick one-liners |
| `PCA` | $d \to k$ (linear) | Reduce multicollinearity |
| `TSNE` / `UMAP` | $d \to 2\text{–}3$ (non-linear) | Visualisation |
| `TruncatedSVD` | Sparse $d \to k$ | Text features |
| `TfidfVectorizer` | Text → sparse matrix | NLP pipelines |
