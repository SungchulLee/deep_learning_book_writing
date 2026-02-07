# Cross-Validation

Cross-validation is a resampling technique that provides robust estimates of model generalisation performance. By rotating the held-out set across multiple folds, it reduces variance compared to a single train-test split and enables principled hyperparameter selection.

## Train-Test Split (Prerequisite)

Before discussing cross-validation, recall the basic split:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### Stratification

Preserve class proportions (essential for imbalanced data):

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

### Train-Validation-Test Split

```python
# 60% train, 20% validation, 20% test
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
```

### Time Series: No Shuffling

```python
# Preserve temporal order
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
```

**Key rule**: the test set is never used for training, preprocessing fitting, or hyperparameter tuning. Fit all transformers on training data only, then apply to test. A single split has high variance in the performance estimate, which motivates cross-validation.

## Why Cross-Validation?

A single train-validation split has two problems:

1. **High variance**: performance estimates depend heavily on which examples land in the validation set
2. **Data inefficiency**: 20–30% of data is withheld from training

Cross-validation addresses both by using every example for both training and validation (in different folds).

## K-Fold Cross-Validation

### Procedure

Partition the dataset into $K$ disjoint folds $\mathcal{F}_1, \ldots, \mathcal{F}_K$ of size $\approx n/K$. For each fold $k$:

1. Train on $\mathcal{D} \setminus \mathcal{F}_k$
2. Evaluate on $\mathcal{F}_k$

### Estimator

$$\hat{R}_{\text{CV}} = \frac{1}{K} \sum_{k=1}^{K} \mathcal{L}\bigl(\hat{f}^{(-k)}, \mathcal{F}_k\bigr)$$

where $\hat{f}^{(-k)}$ is the model trained without fold $k$.

### Standard Error

$$\text{SE}(\hat{R}_{\text{CV}}) \approx \sqrt{\frac{1}{K(K-1)} \sum_{k=1}^{K} (\mathcal{L}_k - \hat{R}_{\text{CV}})^2}$$

This approximation treats fold errors as independent, which underestimates true variance due to overlapping training sets.

### Choosing K

| K | Bias | Variance | Computation | Use case |
|---|------|----------|-------------|----------|
| 5 | Moderate | Low | Moderate | Large datasets (default) |
| 10 | Low | Moderate | Moderate | Common default |
| n (LOOCV) | Very low | High | Expensive | Small datasets |

## Variants

### Stratified K-Fold

Preserves class distribution in each fold. Essential for imbalanced classification—standard K-fold may produce folds with few or no minority-class examples.

### Leave-One-Out (LOOCV)

The special case $K = n$:

$$\hat{R}_{\text{LOO}} = \frac{1}{n} \sum_{i=1}^{n} \mathcal{L}\bigl(\hat{f}^{(-i)}, (x_i, y_i)\bigr)$$

Nearly unbiased but high variance and expensive ($n$ model fits).

For linear regression with squared error, LOOCV has a closed form using the hat matrix $H = X(X^\top X)^{-1}X^\top$:

$$\hat{R}_{\text{LOO}} = \frac{1}{n} \sum_{i=1}^{n} \left(\frac{y_i - \hat{y}_i}{1 - h_{ii}}\right)^2$$

### Repeated K-Fold

Run K-fold $R$ times with different random partitions and average. Reduces variance from a particular partition at cost of $R \times K$ fits.

### Time Series Split

Standard K-fold violates temporal ordering. **Forward chaining** respects causality:

- Fold 1: train on $[0, T_0)$, validate on $[T_0, T_1)$
- Fold 2: train on $[0, T_1)$, validate on $[T_1, T_2)$
- ...

Never shuffle time series data.

### Group K-Fold

When data has natural groups (patients, users, sessions), ensure the same group never appears in both train and validation. Prevents leakage from correlated observations.

## Scikit-learn Implementation

### Basic K-Fold

```python
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

# Simple API
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

# Custom folds
kf = KFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kf)
```

### Stratified K-Fold

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf)
# Each fold preserves class proportions
```

### Time Series Split

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, val_idx in tscv.split(X):
    print(f"Train: [0..{train_idx[-1]}], Val: [{val_idx[0]}..{val_idx[-1]}]")
```

### Group K-Fold

```python
from sklearn.model_selection import GroupKFold

groups = [0, 0, 1, 1, 2, 2, 3, 3]  # e.g., patient IDs
gkf = GroupKFold(n_splits=4)

for train_idx, val_idx in gkf.split(X, y, groups):
    # Same group never in both train and val
    pass
```

### Multiple Metrics

```python
from sklearn.model_selection import cross_validate

results = cross_validate(
    model, X, y, cv=5,
    scoring=['accuracy', 'precision', 'recall', 'f1'],
    return_train_score=True,
)
print(f"Test F1: {results['test_f1'].mean():.3f}")
print(f"Train F1: {results['train_f1'].mean():.3f}")
# Large gap indicates overfitting
```

### Cross-Validated Predictions

```python
from sklearn.model_selection import cross_val_predict

y_pred = cross_val_predict(model, X, y, cv=5)
# Each sample's prediction from when it was in the validation fold

y_proba = cross_val_predict(model, X, y, cv=5, method='predict_proba')
```

### With Pipelines (Preventing Leakage)

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression()),
])

scores = cross_val_score(pipeline, X, y, cv=5)
# Scaler fits only on training fold—no leakage
```

## Hyperparameter Tuning

### GridSearchCV

```python
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

param_grid = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]}

grid = GridSearchCV(Ridge(), param_grid, cv=5,
                    scoring='neg_mean_squared_error',
                    return_train_score=True)
grid.fit(X, y)

print(f"Best alpha: {grid.best_params_['alpha']}")
print(f"Best CV MSE: {-grid.best_score_:.4f}")
```

### Nested Cross-Validation

When both model selection and performance estimation are needed, use nested CV to avoid optimistic bias:

- **Outer loop** ($K_{\text{outer}}$ folds): estimates generalisation
- **Inner loop** ($K_{\text{inner}}$ folds): selects hyperparameters

```python
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold

outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)

grid = GridSearchCV(Ridge(), {'alpha': [0.01, 0.1, 1.0, 10.0]},
                    cv=inner_cv, scoring='neg_mean_squared_error')

nested_scores = cross_val_score(grid, X, y, cv=outer_cv,
                                 scoring='neg_mean_squared_error')
print(f"Nested CV MSE: {-nested_scores.mean():.4f} ± {nested_scores.std():.4f}")
```

## PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import KFold


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


def pytorch_kfold_cv(X, y, n_splits=5, epochs=100, lr=0.01):
    """K-Fold CV for a PyTorch model."""
    dataset = TensorDataset(X, y)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_losses = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=32, shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=64)

        model = MLP(X.shape[1])
        opt = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        # Train
        model.train()
        for _ in range(epochs):
            for xb, yb in train_loader:
                opt.zero_grad()
                criterion(model(xb), yb).backward()
                opt.step()

        # Evaluate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                val_loss += criterion(model(xb), yb).item() * xb.size(0)
        val_loss /= len(val_idx)
        fold_losses.append(val_loss)
        print(f"Fold {fold + 1}: Val MSE = {val_loss:.4f}")

    mean = sum(fold_losses) / len(fold_losses)
    std = (sum((l - mean)**2 for l in fold_losses) / len(fold_losses))**0.5
    print(f"\nCV MSE: {mean:.4f} ± {std:.4f}")
    return fold_losses
```

## Cross-Validation for Overfitting Detection

CV serves a critical diagnostic role:

1. **Detects overfitting**: large gap between train and CV scores signals need for regularisation
2. **Tunes regularisation**: select $\lambda$, dropout rate, etc. by minimising CV error
3. **Prevents selection bias**: evaluating many configurations on a single validation set overfits to that set

Compare `mean_train_score` and `mean_test_score` in `GridSearchCV.cv_results_` to diagnose whether regularisation is too weak (large gap) or too strong (both scores high).

## Practical Guidelines

| Situation | Recommendation |
|-----------|----------------|
| Default | 5-fold or 10-fold with shuffle |
| Classification | Stratified K-fold |
| Small data ($n < 100$) | LOOCV or repeated K-fold |
| Time series | TimeSeriesSplit (no shuffle) |
| Grouped data | GroupKFold |
| Expensive models | 3-fold or 5-fold |
| Final performance estimate | Nested CV |

**Always use pipelines** when preprocessing depends on training data (scaling, imputation) to prevent leakage.

## Summary

| Splitter | Use case | Key property |
|----------|----------|--------------|
| `KFold` | General | Standard |
| `StratifiedKFold` | Classification | Preserves class distribution |
| `TimeSeriesSplit` | Time series | Respects temporal order |
| `LeaveOneOut` | Small data | Nearly unbiased |
| `GroupKFold` | Grouped data | Prevents group leakage |
| `RepeatedKFold` | Reduce variance | Multiple random partitions |

## References

1. Stone, M. (1974). "Cross-Validatory Choice and Assessment of Statistical Predictions." *JRSS-B*.
2. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer. Section 7.10.
3. Arlot, S., & Celisse, A. (2010). "A Survey of Cross-Validation Procedures for Model Selection." *Statistics Surveys*.
4. Varma, S., & Simon, R. (2006). "Bias in Error Estimation When Using Cross-Validation for Model Selection." *BMC Bioinformatics*.
