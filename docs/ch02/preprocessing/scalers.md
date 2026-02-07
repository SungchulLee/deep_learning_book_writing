# Scalers

Feature scaling transforms numerical features to similar scales, preventing features with large ranges from dominating distance-based algorithms and improving gradient descent convergence in optimization.

---

## StandardScaler

### 1. Z-Score Normalization

**Standardize to mean=0, std=1:**

```python
from sklearn.preprocessing import StandardScaler
import numpy as np

# Original data
X = np.array([[1, 2000], [2, 3000], [3, 4000]])

# Fit and transform
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(X_scaled)
# [[-1.22, -1.22],
#  [ 0.  ,  0.  ],
#  [ 1.22,  1.22]]

# Check: mean ≈ 0, std ≈ 1
print(X_scaled.mean(axis=0))  # [0, 0]
print(X_scaled.std(axis=0))   # [1, 1]
```

### 2. Formula

**z = (x - μ) / σ**

```python
# Manual calculation
mean = X.mean(axis=0)
std = X.std(axis=0)
X_manual = (X - mean) / std

# Verify
print(np.allclose(X_scaled, X_manual))  # True
```

### 3. Inverse Transform

```python
# Transform back to original scale
X_original = scaler.inverse_transform(X_scaled)
print(np.allclose(X, X_original))  # True
```

### 4. New Data

```python
# Apply to new data
X_new = np.array([[1.5, 2500]])
X_new_scaled = scaler.transform(X_new)
print(X_new_scaled)  # Uses training mean/std
```

### 5. Fit vs Fit-Transform

```python
# Training data
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)

# Or combined
X_train_scaled = scaler.fit_transform(X_train)

# Test data (use training statistics!)
X_test_scaled = scaler.transform(X_test)
```

### 6. When to Use

- Linear models (regression, SVM)
- Neural networks
- PCA
- K-means clustering
- **Not needed:** Tree-based models (decision trees, random forests)

### 7. With Missing Values

```python
# StandardScaler ignores NaN by default
X_with_nan = np.array([[1, np.nan], [2, 3], [3, 4]])
scaler = StandardScaler()
# Will raise error - handle NaN first with imputation
```

---

## MinMaxScaler

### 1. Range Normalization

**Scale to [0, 1]:**

```python
from sklearn.preprocessing import MinMaxScaler

X = np.array([[1, 2000], [2, 3000], [3, 4000]])

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

print(X_scaled)
# [[0. , 0. ],
#  [0.5, 0.5],
#  [1. , 1. ]]

# Check range
print(X_scaled.min(axis=0))  # [0, 0]
print(X_scaled.max(axis=0))  # [1, 1]
```

### 2. Formula

**x' = (x - min) / (max - min)**

```python
# Manual
X_min = X.min(axis=0)
X_max = X.max(axis=0)
X_manual = (X - X_min) / (X_max - X_min)

print(np.allclose(X_scaled, X_manual))  # True
```

### 3. Custom Range

```python
# Scale to [a, b]
scaler = MinMaxScaler(feature_range=(-1, 1))
X_scaled = scaler.fit_transform(X)

print(X_scaled.min(axis=0))  # [-1, -1]
print(X_scaled.max(axis=0))  # [1, 1]
```

### 4. Outlier Sensitivity

```python
# MinMaxScaler sensitive to outliers
X_with_outlier = np.array([[1], [2], [3], [100]])

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_with_outlier)

print(X_scaled[:3])  # Very small values [0.0, 0.01, 0.02]
# All compressed near 0 due to outlier at 100
```

### 5. When to Use

- Neural networks (especially output layer)
- Image pixel values
- When bounded output is needed
- **Avoid with outliers** (use RobustScaler instead)

### 6. Clipping

```python
# Clip values to range
scaler = MinMaxScaler(clip=True)
X_train_scaled = scaler.fit_transform(X_train)

# Test values outside training range will be clipped
X_test = np.array([[5, 5000]])  # Beyond training range
X_test_scaled = scaler.transform(X_test)
# Clipped to [0, 1] range
```

### 7. Sparse Data

```python
# Preserves sparsity
from scipy.sparse import csr_matrix
X_sparse = csr_matrix([[0, 1, 0], [0, 0, 3], [4, 0, 0]])

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_sparse)
# Output remains sparse
```

---

## RobustScaler

### 1. Robust to Outliers

**Uses median and IQR:**

```python
from sklearn.preprocessing import RobustScaler

X = np.array([[1], [2], [3], [100]])  # Outlier at 100

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

print(X_scaled)
# [[-0.5 ],
#  [ 0.  ],
#  [ 0.5 ],
#  [49.  ]]  # Outlier less influential
```

### 2. Formula

**x' = (x - median) / IQR**

```python
# IQR = Q3 - Q1
median = np.median(X)
q1 = np.percentile(X, 25)
q3 = np.percentile(X, 75)
iqr = q3 - q1

X_manual = (X - median) / iqr
```

### 3. Comparison

```python
X_with_outliers = np.array([[1], [2], [3], [100]])

# StandardScaler
std_scaler = StandardScaler()
X_std = std_scaler.fit_transform(X_with_outliers)
print("StandardScaler:", X_std.T)
# Very affected by outlier

# RobustScaler
robust_scaler = RobustScaler()
X_robust = robust_scaler.fit_transform(X_with_outliers)
print("RobustScaler:", X_robust.T)
# Less affected
```

### 4. Quantile Range

```python
# Use different quantile range
scaler = RobustScaler(quantile_range=(10, 90))
# Uses 10th and 90th percentiles instead of 25th/75th
```

### 5. Centering

```python
# Option to not center
scaler = RobustScaler(with_centering=False)
# Only scales by IQR, doesn't subtract median
```

### 6. When to Use

- Data with outliers
- Robust statistics needed
- Real-world messy data
- Alternative to StandardScaler when outliers present

### 7. Unit Variance

```python
# Option for unit variance
scaler = RobustScaler(unit_variance=True)
# Scale to unit variance instead of IQR
```

---

## MaxAbsScaler

### 1. Scale by Maximum

**Scale to [-1, 1] by dividing by max absolute value:**

```python
from sklearn.preprocessing import MaxAbsScaler

X = np.array([[1, 2000], [2, 3000], [3, 4000]])

scaler = MaxAbsScaler()
X_scaled = scaler.fit_transform(X)

print(X_scaled)
# [[0.33, 0.5 ],
#  [0.67, 0.75],
#  [1.  , 1.  ]]

# Max abs value in each column = 1
print(np.abs(X_scaled).max(axis=0))  # [1, 1]
```

### 2. Formula

**x' = x / max(|x|)**

```python
max_abs = np.abs(X).max(axis=0)
X_manual = X / max_abs
```

### 3. Preserves Sparsity

```python
# Doesn't center, preserves sparsity and sign
from scipy.sparse import csr_matrix
X_sparse = csr_matrix([[0, 1, 0], [0, 0, 3], [4, 0, 0]])

scaler = MaxAbsScaler()
X_scaled = scaler.fit_transform(X_sparse)
# Sparsity preserved (no centering)
```

### 4. When to Use

- Sparse data (text, one-hot encoded)
- Data already centered at zero
- Need to preserve sparsity
- Neural networks with sparse inputs

### 5. Signed Data

```python
# Works with negative values
X_signed = np.array([[-10, 2], [5, -4], [3, 8]])
scaler = MaxAbsScaler()
X_scaled = scaler.fit_transform(X_signed)

print(X_scaled)
# [[-1.  ,  0.25],
#  [ 0.5 , -0.5 ],
#  [ 0.3 ,  1.  ]]
```

### 6. No Shift

```python
# Zeros remain zeros
X = np.array([[0, 1, 0], [0, 0, 3]])
X_scaled = scaler.fit_transform(X)
print(X_scaled)
# Zeros unchanged
```

### 7. Single Sample

```python
# Works with single sample
X_single = np.array([[5, 10]])
scaler = MaxAbsScaler()
X_scaled = scaler.fit_transform(X_single)
```

---

## Normalizer

### 1. Unit Norm

**Scale samples (rows) to unit norm:**

```python
from sklearn.preprocessing import Normalizer

X = np.array([[3, 4], [1, 2]])

normalizer = Normalizer(norm='l2')
X_normalized = normalizer.fit_transform(X)

print(X_normalized)
# [[0.6, 0.8],    # sqrt(3²+4²) = 5
#  [0.447, 0.894]] # sqrt(1²+2²) = sqrt(5)

# Check norms
norms = np.linalg.norm(X_normalized, axis=1)
print(norms)  # [1.0, 1.0]
```

### 2. L1 Norm

```python
normalizer = Normalizer(norm='l1')
X_normalized = normalizer.fit_transform(X)

# Sum of absolute values = 1
print(X_normalized.sum(axis=1))  # [1.0, 1.0]
```

### 3. L2 Norm (Default)

```python
# L2 = Euclidean norm
# For row x: x / ||x||₂
```

### 4. Max Norm

```python
normalizer = Normalizer(norm='max')
X_normalized = normalizer.fit_transform(X)

# Max absolute value in row = 1
print(np.abs(X_normalized).max(axis=1))  # [1.0, 1.0]
```

### 5. When to Use

- Text classification (TF-IDF vectors)
- When direction matters more than magnitude
- Cosine similarity
- Neural network inputs (per-sample)

### 6. Per-Sample Scaling

```python
# Normalizer scales ROWS, not columns
# Each sample independently
# No fit needed (stateless)

# Can use transform directly
X_normalized = Normalizer().transform(X)
```

### 7. Text Example

```python
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = ['This is document one', 'This is document two']
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(corpus)

# Already L2 normalized by default in TfidfVectorizer
```

---

## PowerTransformer

### 1. Box-Cox Transform

**Make data more Gaussian:**

```python
from sklearn.preprocessing import PowerTransformer

# Right-skewed data
X = np.array([[1], [10], [100], [1000]])

transformer = PowerTransformer(method='box-cox')
X_transformed = transformer.fit_transform(X)

print("Original skewness:", np.mean(((X - X.mean()) / X.std())**3))
print("Transformed skewness:", np.mean(((X_transformed - X_transformed.mean()) / X_transformed.std())**3))
# Reduced skewness
```

### 2. Yeo-Johnson Transform

```python
# Works with negative values (Box-Cox requires x > 0)
X_with_negatives = np.array([[-1], [0], [1], [10]])

transformer = PowerTransformer(method='yeo-johnson')
X_transformed = transformer.fit_transform(X_with_negatives)
```

### 3. Standardization

```python
# Automatically standardizes after transform
transformer = PowerTransformer(standardize=True)  # Default
X_transformed = transformer.fit_transform(X)

# Mean ≈ 0, Var ≈ 1
print(X_transformed.mean())  # ~0
print(X_transformed.std())   # ~1
```

### 4. When to Use

- Right-skewed data (income, prices)
- Want approximately normal distribution
- Improve linear model performance
- Before parametric tests

### 5. Lambda Parameter

```python
# Box-Cox: x' = (x^λ - 1) / λ if λ ≠ 0, else log(x)
# Optimal λ found via MLE

transformer.fit(X)
print(transformer.lambdas_)  # Optimal λ for each feature
```

### 6. Comparison

```python
import matplotlib.pyplot as plt

# Original
plt.subplot(1, 2, 1)
plt.hist(X, bins=20)
plt.title('Original')

# Transformed
plt.subplot(1, 2, 2)
plt.hist(X_transformed, bins=20)
plt.title('Box-Cox Transformed')
plt.show()
```

### 7. Inverse Transform

```python
X_back = transformer.inverse_transform(X_transformed)
print(np.allclose(X, X_back))  # True
```

---

## QuantileTransformer

### 1. Uniform Distribution

**Transform to uniform [0, 1]:**

```python
from sklearn.preprocessing import QuantileTransformer

X = np.array([[1], [10], [100], [1000]])

transformer = QuantileTransformer(output_distribution='uniform')
X_transformed = transformer.fit_transform(X)

print(X_transformed)
# [[0.  ],
#  [0.33],
#  [0.67],
#  [1.  ]]  # Uniform spacing
```

### 2. Normal Distribution

```python
# Transform to standard normal
transformer = QuantileTransformer(output_distribution='normal')
X_transformed = transformer.fit_transform(X)

# Should be approximately N(0, 1)
from scipy import stats
print(stats.normaltest(X_transformed))  # Test normality
```

### 3. Robust to Outliers

```python
X_with_outliers = np.array([[1], [2], [3], [1000]])

# QuantileTransformer
qt = QuantileTransformer()
X_qt = qt.fit_transform(X_with_outliers)

# StandardScaler
ss = StandardScaler()
X_ss = ss.fit_transform(X_with_outliers)

print("QuantileTransformer:", X_qt.T)
print("StandardScaler:", X_ss.T)
# QuantileTransformer less affected by outlier
```

### 4. Quantiles

```python
# Number of quantiles
transformer = QuantileTransformer(n_quantiles=1000)
# More quantiles = smoother transformation
```

### 5. When to Use

- Non-normal distributions
- Many outliers
- Want specific target distribution (uniform/normal)
- Improve model performance with skewed data

### 6. Non-Parametric

```python
# No assumptions about data distribution
# Uses empirical CDF
# Can handle any distribution shape
```

### 7. Subsample

```python
# For large datasets
transformer = QuantileTransformer(
    n_quantiles=1000,
    subsample=10000  # Use 10k samples to estimate quantiles
)
```

---

## Practical Examples

### 1. Preprocessing Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
```

### 2. Column-Specific Scaling

```python
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer([
    ('std', StandardScaler(), [0, 1]),      # Numerical
    ('robust', RobustScaler(), [2, 3]),     # With outliers
    ('minmax', MinMaxScaler(), [4])         # Bounded
])

X_transformed = ct.fit_transform(X)
```

### 3. Comparing Scalers

```python
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge

X, y = make_regression(n_features=10, noise=10, random_state=42)

scalers = {
    'None': None,
    'Standard': StandardScaler(),
    'MinMax': MinMaxScaler(),
    'Robust': RobustScaler()
}

for name, scaler in scalers.items():
    if scaler:
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X
    
    model = Ridge()
    model.fit(X_scaled, y)
    print(f"{name}: R² = {model.score(X_scaled, y):.3f}")
```

### 4. Neural Network Input

```python
# Scale inputs to [0, 1] for neural networks
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Feed to neural network
model.fit(X_train_scaled, y_train)
```

### 5. Time Series

```python
# Scale time series data
# Important: fit only on training data!

scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)  # Use train statistics
```

### 6. Inverse Transform

```python
# Make predictions, then inverse transform
y_pred_scaled = model.predict(X_test_scaled)
y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
# Back to original scale
```

### 7. Saving Scaler

```python
import joblib

# Save fitted scaler
joblib.dump(scaler, 'scaler.pkl')

# Load later
scaler_loaded = joblib.load('scaler.pkl')
X_new_scaled = scaler_loaded.transform(X_new)
```

---

## Summary

| Scaler | Range | Use Case | Outliers |
|--------|-------|----------|----------|
| **StandardScaler** | No specific | General purpose, linear models | Sensitive |
| **MinMaxScaler** | [0, 1] | Neural networks, bounded output | Very sensitive |
| **RobustScaler** | No specific | Data with outliers | Robust |
| **MaxAbsScaler** | [-1, 1] | Sparse data | Somewhat robust |
| **Normalizer** | Unit norm per sample | Text, cosine similarity | N/A (per sample) |
| **PowerTransformer** | ~N(0,1) | Skewed → Gaussian | Moderate |
| **QuantileTransformer** | User choice | Arbitrary → Uniform/Normal | Very robust |

**Key insights:**
- **Always fit only on training data**, then transform both train and test
- **StandardScaler** is default for most ML algorithms
- **RobustScaler** when outliers present
- **MinMaxScaler** for neural networks
- **QuantileTransformer** for heavily skewed data
- **Tree-based models don't need scaling**
