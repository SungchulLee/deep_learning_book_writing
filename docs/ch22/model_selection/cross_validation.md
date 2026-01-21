# Cross-Validation

Cross-validation evaluates model performance on multiple train-test splits, providing robust performance estimates and reducing variance from single split randomness.

---

## K-Fold CV

### 1. Basic K-Fold
```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
scores = cross_val_score(model, X, y, cv=5)

print(f"Scores: {scores}")
print(f"Mean: {scores.mean():.3f} (+/- {scores.std():.3f})")
# 5-fold: train on 80%, test on 20%, repeat 5 times
```

### 2. Custom Folds
```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kf)
```

### 3. Manual Loop
```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=5)
scores = []

for train_idx, test_idx in kf.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    scores.append(score)

print(f"Mean: {np.mean(scores):.3f}")
```

---

## Stratified K-Fold

### 1. Preserve Class Distribution
```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf)
# Each fold has same class proportions
```

### 2. Why Stratify
```python
# Imbalanced data: 95% class 0, 5% class 1
# Regular K-Fold: some folds might have 0% class 1
# Stratified: each fold has ~5% class 1
```

---

## Leave-One-Out

### 1. LOO CV
```python
from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()
scores = cross_val_score(model, X, y, cv=loo)
# n_splits = n_samples (very expensive!)
print(f"Mean: {scores.mean():.3f}")
```

### 2. When to Use
```python
# Very small datasets (<100 samples)
# Computationally expensive: n iterations
# Low bias, high variance in estimate
```

---

## Time Series CV

### 1. TimeSeriesSplit
```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    # Train on past, test on future
```

### 2. Expanding Window
```python
# Fold 1: train [0:100], test [100:120]
# Fold 2: train [0:120], test [120:140]
# Fold 3: train [0:140], test [140:160]
# Training set grows each fold
```

### 3. Never Shuffle
```python
# DON'T shuffle time series!
# Would leak future information into past
```

---

## Metrics

### 1. Scoring Parameter
```python
# Classification
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
scores = cross_val_score(model, X, y, cv=5, scoring='f1')
scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')

# Regression
scores = cross_val_score(model, X, y, cv=5, scoring='r2')
scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
```

### 2. Multiple Metrics
```python
from sklearn.model_selection import cross_validate

scoring = ['accuracy', 'precision', 'recall', 'f1']
results = cross_validate(model, X, y, cv=5, scoring=scoring)

print(results['test_accuracy'].mean())
print(results['test_f1'].mean())
```

---

## Cross Val Predict

### 1. Get Predictions
```python
from sklearn.model_selection import cross_val_predict

y_pred = cross_val_predict(model, X, y, cv=5)
# Predictions for each sample (when in test fold)

# Evaluate
from sklearn.metrics import accuracy_score
print(accuracy_score(y, y_pred))
```

### 2. Predict Proba
```python
y_pred_proba = cross_val_predict(
    model, X, y, cv=5, method='predict_proba'
)
# Probability predictions
```

---

## Practical Examples

### 1. Compare Models
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

models = {
    'Logistic': LogisticRegression(),
    'RF': RandomForestClassifier(),
    'SVM': SVC()
}

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5)
    print(f"{name}: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

### 2. With Pipeline
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])

scores = cross_val_score(pipeline, X, y, cv=5)
# Scaler fit on each training fold separately (no leakage!)
```

### 3. Group K-Fold
```python
from sklearn.model_selection import GroupKFold

# Data has groups (e.g., patients, users)
# Ensure same group not in train and test

groups = [0, 0, 1, 1, 2, 2, 3, 3]  # Group labels
gkf = GroupKFold(n_splits=4)

for train_idx, test_idx in gkf.split(X, y, groups):
    # Groups in train never in test
    pass
```

---

## Summary

| Method | Use Case | n_splits | Characteristics |
|--------|----------|----------|-----------------|
| **KFold** | General | 5-10 | Standard, fast |
| **StratifiedKFold** | Classification | 5-10 | Preserves class distribution |
| **TimeSeriesSplit** | Time series | 5+ | Respects temporal order |
| **LeaveOneOut** | Small data | n | Expensive, high variance |
| **GroupKFold** | Grouped data | 5-10 | Prevents group leakage |

**Key insights:**
- **K=5 or 10** is standard (bias-variance tradeoff)
- **Stratify** for classification with imbalanced classes
- **Never shuffle** time series
- **More folds** = less bias, more variance, more computation
- **Use with pipelines** to prevent data leakage
