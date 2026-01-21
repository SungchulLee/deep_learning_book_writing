# Train-Test Split

Train-test split divides data into training (model fitting) and test (evaluation) sets to assess generalization performance on unseen data and detect overfitting.

---

## Basic Split

### 1. Random Split
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# 80% train, 20% test
```

### 2. Test Size
```python
# Proportion (0.0 to 1.0)
train_test_split(X, y, test_size=0.3)  # 30% test

# Absolute number
train_test_split(X, y, test_size=100)  # 100 samples for test
```

### 3. Train Size
```python
# Specify train size instead
train_test_split(X, y, train_size=0.8)  # 80% train
```

---

## Stratification

### 1. Stratified Split
```python
# Preserve class proportions
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Check proportions
print(np.bincount(y_train) / len(y_train))
print(np.bincount(y_test) / len(y_test))
# Same distribution
```

### 2. Why Stratify
```python
# Imbalanced dataset: 90% class 0, 10% class 1
# Without stratify: test might have 0% class 1 by chance
# With stratify: test has ~10% class 1
```

### 3. Continuous Target
```python
# For regression, bin target first
y_binned = pd.qcut(y, q=4, labels=False)  # 4 quantiles
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y_binned
)
```

---

## Shuffle

### 1. Shuffle Data
```python
# Default: shuffle=True
train_test_split(X, y, shuffle=True)  # Random permutation

# No shuffle (preserves order)
train_test_split(X, y, shuffle=False)  # Temporal order preserved
```

### 2. Time Series
```python
# Don't shuffle time series!
# Use temporal split instead
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
```

---

## Multiple Splits

### 1. Train-Val-Test
```python
# First split: train+val vs test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Second split: train vs val
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42
)
# 60% train, 20% val, 20% test
```

### 2. Typical Ratios
```python
# Small data (<1k): 60/20/20
# Medium data (1k-100k): 70/15/15
# Large data (>100k): 80/10/10 or 90/5/5
```

---

## Random State

### 1. Reproducibility
```python
# Same split every time
train_test_split(X, y, random_state=42)

# Different split each run
train_test_split(X, y, random_state=None)
```

### 2. Multiple Experiments
```python
# Use different random states for robustness
for seed in [42, 123, 999]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=seed
    )
    # Train and evaluate
```

---

## Practical Examples

### 1. With Pandas
```python
from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(df, test_size=0.2)

X_train = df_train.drop('target', axis=1)
y_train = df_train['target']
X_test = df_test.drop('target', axis=1)
y_test = df_test['target']
```

### 2. Pipeline
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(X, y)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])

pipeline.fit(X_train, y_train)
score = pipeline.score(X_test, y_test)
```

### 3. Check Leak
```python
# Ensure no data leakage
# Fit preprocessors ONLY on training data

scaler = StandardScaler()
scaler.fit(X_train)  # Fit on train only

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use train statistics
```

---

## Summary

**Key principles:**
- **Test set:** Never used for training, hyperparameter tuning, or feature selection
- **Stratify:** For classification with imbalanced classes
- **Don't shuffle:** Time series or ordered data
- **Random state:** For reproducibility
- **Typical split:** 80/20 or 70/30
- **Validation set:** Use cross-validation or separate val set for hyperparameter tuning

**Common mistake:** Fitting preprocessors on entire dataset before split (data leakage)
