# Cross-Validation

## Overview

Cross-validation provides a more robust estimate of generalization performance than a single train-test split by systematically rotating which data serves as validation.

## K-Fold Cross-Validation

The dataset is partitioned into $k$ equal folds. The model is trained $k$ times, each time using $k-1$ folds for training and the remaining fold for validation:

```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)

fold_scores = []
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model = create_model()
    train(model, X_train, y_train)
    score = evaluate(model, X_val, y_val)
    fold_scores.append(score)

mean_score = np.mean(fold_scores)
std_score = np.std(fold_scores)
```

## Stratified K-Fold

For classification, stratified k-fold ensures each fold maintains the class distribution of the original dataset:

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in skf.split(X, y):
    # Each fold has approximately the same class proportions
    pass
```

## Time Series Cross-Validation

Standard k-fold violates temporal ordering. For time series, use **walk-forward** (expanding window) or **sliding window** validation:

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, val_idx in tscv.split(X):
    # Training set always precedes validation set chronologically
    # Each successive split uses more training data
    pass
```

## Purged Cross-Validation for Finance

Financial data requires additional care to prevent information leakage from overlapping samples:

```python
class PurgedKFold:
    """K-fold with purging and embargo for financial time series."""
    def __init__(self, n_splits=5, purge_days=5, embargo_days=5):
        self.n_splits = n_splits
        self.purge_days = purge_days
        self.embargo_days = embargo_days

    def split(self, X, timestamps):
        fold_size = len(X) // self.n_splits
        for i in range(self.n_splits):
            val_start = i * fold_size
            val_end = min((i + 1) * fold_size, len(X))

            # Purge: remove training samples too close to validation
            train_mask = np.ones(len(X), dtype=bool)
            train_mask[val_start:val_end] = False

            # Remove samples within purge_days of validation boundaries
            purge_start = max(0, val_start - self.purge_days)
            purge_end = min(len(X), val_end + self.embargo_days)
            train_mask[purge_start:purge_end] = False

            train_idx = np.where(train_mask)[0]
            val_idx = np.arange(val_start, val_end)
            yield train_idx, val_idx
```

**Purging** removes training samples whose labels overlap with the validation period. **Embargo** adds a buffer after the validation set to prevent look-ahead leakage from features computed over rolling windows.

## Choosing K

- $k = 5$ or $k = 10$: Standard choices balancing bias and variance of the estimate.
- $k = N$ (leave-one-out): Nearly unbiased but high variance and computationally expensive.
- For deep learning, $k = 5$ is typical given the computational cost of training.

## Key Takeaways

- Cross-validation provides a more reliable generalization estimate than a single split.
- Use stratified k-fold for classification; time series split for temporal data.
- Financial applications require purged cross-validation to prevent information leakage.
