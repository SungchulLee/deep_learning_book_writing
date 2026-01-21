# Feature Engineering

Feature engineering transforms raw data into features that better represent the underlying patterns, improving model performance. It's often the most impactful part of the ML pipeline.

---

## Polynomial Features

### 1. Basic Usage

```python
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

X = np.array([[1, 2], [3, 4], [5, 6]])

# Degree 2 polynomial features
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

print(f"Original shape: {X.shape}")
print(f"Polynomial shape: {X_poly.shape}")
print(f"Feature names: {poly.get_feature_names_out()}")
```

### 2. Interaction Only

```python
# Only interaction terms (no x², x³, etc.)
poly_inter = PolynomialFeatures(degree=2, interaction_only=True)
X_inter = poly_inter.fit_transform(X)

print(f"Features: {poly_inter.get_feature_names_out()}")
```

---

## Binning (Discretization)

### 1. KBinsDiscretizer

```python
from sklearn.preprocessing import KBinsDiscretizer

X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])

# Uniform bins
binner = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
X_binned = binner.fit_transform(X)

print(f"Bin edges: {binner.bin_edges_}")
```

### 2. Different Strategies

```python
# uniform: Equal width bins
# quantile: Equal frequency bins  
# kmeans: Bins based on k-means clustering

for strategy in ['uniform', 'quantile', 'kmeans']:
    binner = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy=strategy)
    X_binned = binner.fit_transform(X)
    print(f"{strategy}: {binner.bin_edges_[0].round(2)}")
```

---

## Log and Power Transforms

### 1. PowerTransformer

```python
from sklearn.preprocessing import PowerTransformer

# Generate skewed data
X_skewed = np.random.exponential(scale=2, size=(100, 2))

# Yeo-Johnson (handles negative values)
pt = PowerTransformer(method='yeo-johnson')
X_transformed = pt.fit_transform(X_skewed)
```

### 2. QuantileTransformer

```python
from sklearn.preprocessing import QuantileTransformer

# Transform to normal distribution
qt = QuantileTransformer(output_distribution='normal')
X_normal = qt.fit_transform(X_skewed)
```

---

## Date/Time Features

```python
import pandas as pd

df = pd.DataFrame({
    'timestamp': pd.date_range('2024-01-01', periods=100, freq='H')
})

# Extract components
df['year'] = df['timestamp'].dt.year
df['month'] = df['timestamp'].dt.month
df['day'] = df['timestamp'].dt.day
df['hour'] = df['timestamp'].dt.hour
df['dayofweek'] = df['timestamp'].dt.dayofweek
df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)

# Cyclical encoding
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
```

---

## Aggregation Features

```python
df = pd.DataFrame({
    'user_id': [1, 1, 1, 2, 2, 3],
    'purchase_amount': [100, 150, 200, 50, 75, 300]
})

# User-level aggregations
user_stats = df.groupby('user_id')['purchase_amount'].agg([
    'mean', 'std', 'min', 'max', 'count'
]).add_prefix('user_')

# Merge back
df = df.merge(user_stats, left_on='user_id', right_index=True)
```

---

## Rolling Window Features

```python
# Time series data
ts_df = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=10),
    'value': [10, 12, 15, 14, 18, 20, 19, 22, 25, 24]
})

# Rolling statistics
ts_df['rolling_mean_3'] = ts_df['value'].rolling(window=3).mean()
ts_df['rolling_std_3'] = ts_df['value'].rolling(window=3).std()

# Lag features
ts_df['lag_1'] = ts_df['value'].shift(1)
ts_df['lag_2'] = ts_df['value'].shift(2)
```

---

## Text Features

```python
from sklearn.feature_extraction.text import TfidfVectorizer

texts = ['machine learning is great', 'deep learning is powerful']

# TF-IDF features
tfidf = TfidfVectorizer(max_features=10)
X_tfidf = tfidf.fit_transform(texts)

# Basic text statistics
import pandas as pd
df = pd.DataFrame({'text': texts})
df['char_count'] = df['text'].str.len()
df['word_count'] = df['text'].str.split().str.len()
```

---

## Custom Feature Transformer

```python
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, add_log=True):
        self.add_log = add_log
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        if self.add_log:
            for col in range(X.shape[1]):
                if (X[:, col] > 0).all():
                    X = np.column_stack([X, np.log1p(X[:, col])])
        return X

# Use in pipeline
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

pipe = make_pipeline(
    FeatureEngineer(add_log=True),
    StandardScaler(),
    LogisticRegression()
)
```

---

## Feature Selection After Engineering

```python
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

# Generate many features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Select important features
selector = SelectFromModel(
    RandomForestClassifier(n_estimators=100, random_state=42),
    threshold='median'
)
X_selected = selector.fit_transform(X_poly, y)

print(f"Before: {X_poly.shape[1]}, After: {X_selected.shape[1]}")
```

---

## Summary

| Technique | When to Use |
|-----------|------------|
| Polynomial | Capture non-linear relationships |
| Binning | Convert continuous to categorical |
| Log/Power | Fix skewed distributions |
| Interactions | Capture feature combinations |
| Aggregations | Group-level statistics |
| Time features | Temporal patterns |

**Best practices:**
- Start simple, add complexity as needed
- Use domain knowledge
- Validate on held-out data
- Watch for data leakage
- Use pipelines for reproducibility
