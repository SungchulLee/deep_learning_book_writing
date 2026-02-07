# Imputers

Missing data requires careful handling through deletion, imputation, or indicator features, with strategies depending on missingness mechanism (MCAR, MAR, MNAR) and data type.

---

## Detection

### 1. Check Missing
```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'A': [1, 2, np.nan, 4],
    'B': [5, np.nan, np.nan, 8]
})

print(df.isnull().sum())  # Count per column
print(df.isnull().sum() / len(df))  # Proportion
```

### 2. Visualization
```python
import missingno as msno
msno.matrix(df)  # Visual pattern
msno.heatmap(df)  # Correlation of missingness
```

---

## SimpleImputer

### 1. Mean Imputation
```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
```

### 2. Median (Robust)
```python
imputer = SimpleImputer(strategy='median')
# Better for skewed data or outliers
```

### 3. Most Frequent
```python
imputer = SimpleImputer(strategy='most_frequent')
# For categorical data
```

### 4. Constant
```python
imputer = SimpleImputer(strategy='constant', fill_value=0)
# Fill with specific value
```

---

## KNN Imputer

### 1. Use Neighbors
```python
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)
X_imputed = imputer.fit_transform(X)
# Weighted average of k nearest neighbors
```

### 2. Distance Metric
```python
imputer = KNNImputer(n_neighbors=5, weights='distance')
# Closer neighbors have more weight
```

---

## Iterative Imputer

### 1. MICE Algorithm
```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imputer = IterativeImputer(max_iter=10, random_state=0)
X_imputed = imputer.fit_transform(X)
# Models each feature with missing values as function of others
```

### 2. Estimator
```python
from sklearn.ensemble import RandomForestRegressor

imputer = IterativeImputer(
    estimator=RandomForestRegressor(n_estimators=10),
    max_iter=10
)
# Use RF instead of default BayesianRidge
```

---

## Missing Indicator

### 1. Add Indicator Column
```python
from sklearn.impute import MissingIndicator

indicator = MissingIndicator()
X_indicator = indicator.fit_transform(X)
# Binary columns: 1 if was missing, 0 otherwise
```

### 2. Combined Strategy
```python
from sklearn.pipeline import Pipeline, FeatureUnion

imputer = Pipeline([
    ('features', FeatureUnion([
        ('imputer', SimpleImputer(strategy='mean')),
        ('indicator', Pipeline([
            ('indicator', MissingIndicator()),
            ('select', 'passthrough')
        ]))
    ]))
])
# Both imputed values and missing indicators
```

---

## Dropping

### 1. Drop Rows
```python
# Remove rows with any missing
df_dropped = df.dropna()

# Remove rows with all missing
df_dropped = df.dropna(how='all')

# Remove rows with >50% missing
df_dropped = df.dropna(thresh=len(df.columns)*0.5)
```

### 2. Drop Columns
```python
# Remove columns with >50% missing
threshold = 0.5
df_dropped = df.loc[:, df.isnull().mean() < threshold]
```

---

## Summary

| Method | When to Use | Pros | Cons |
|--------|-------------|------|------|
| **Mean/Median** | MCAR, numerical | Fast, simple | Reduces variance |
| **Most Frequent** | Categorical | Simple | Ignores other features |
| **KNN** | MAR, structured | Uses feature similarity | Slow, sensitive to scale |
| **Iterative** | MAR, complex patterns | Sophisticated | Slow, risk of overfitting |
| **Drop** | High missingness | Clean | Loses data |
| **Indicator** | Informative missingness | Preserves info | Extra features |

**Key insight:** No single best method; choose based on amount (<5% can drop, >30% problematic), mechanism (MCAR/MAR/MNAR), and whether missingness is informative.
