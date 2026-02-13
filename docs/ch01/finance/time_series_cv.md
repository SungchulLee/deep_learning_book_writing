# Time Series CV

Standard cross-validation shuffles data randomly, violating the temporal ordering that is fundamental to financial prediction. Time series cross-validation respects causality: the model is always trained on past data and evaluated on future data, mirroring actual deployment.

## Why Standard K-Fold Fails for Finance

```python
from sklearn.model_selection import KFold
import numpy as np

dates = pd.date_range('2020-01-01', periods=1000, freq='B')
X = np.random.randn(1000, 10)
y = np.random.randn(1000)

# WRONG: Future data leaks into training
kf = KFold(n_splits=5, shuffle=True)
for train_idx, test_idx in kf.split(X):
    train_dates = dates[train_idx]
    test_dates = dates[test_idx]
    # test_dates contain dates BEFORE some train_dates — look-ahead bias!
```

## Walk-Forward Validation

### Basic TimeSeriesSplit

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    print(f"Fold {fold+1}: "
          f"Train [{train_idx[0]}..{train_idx[-1]}], "
          f"Test [{test_idx[0]}..{test_idx[-1]}]")

# Fold 1: Train [0..166],   Test [167..332]
# Fold 2: Train [0..332],   Test [333..499]
# Fold 3: Train [0..499],   Test [500..665]
# Fold 4: Train [0..665],   Test [666..832]
# Fold 5: Train [0..832],   Test [833..999]
```

The training window **expands** with each fold. The test window always follows the training window temporally.

### With Gap (Embargo)

```python
# gap parameter adds buffer between train and test
tscv = TimeSeriesSplit(n_splits=5, gap=20)
# 20-day gap prevents label leakage from overlapping return horizons

for train_idx, test_idx in tscv.split(X):
    print(f"Train end: {train_idx[-1]}, Test start: {test_idx[0]}, "
          f"Gap: {test_idx[0] - train_idx[-1] - 1}")
```

### Fixed-Size Rolling Window

```python
tscv = TimeSeriesSplit(n_splits=5, max_train_size=500)
# Training window capped at 500 observations — rolls forward

for train_idx, test_idx in tscv.split(X):
    print(f"Train size: {len(train_idx)}, Test size: {len(test_idx)}")
```

## Using with Pipelines

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', Ridge(alpha=1.0)),
])

tscv = TimeSeriesSplit(n_splits=5, gap=5)
scores = cross_val_score(
    pipe, X, y, cv=tscv, scoring='neg_mean_squared_error'
)
print(f"Walk-forward MSE: {-scores.mean():.6f} ± {scores.std():.6f}")
```

The scaler re-fits on each training window, ensuring no future data leaks into the standardisation.

## Purging and Embargo

In finance, labels often depend on overlapping future windows (e.g., 5-day forward returns). Without purging, training data near the test boundary contains information about test-period outcomes.

### Purged K-Fold

```python
def purged_kfold(n_samples, n_splits, embargo_pct=0.01):
    """Generate purged walk-forward train/test indices.
    
    Removes training samples whose labels overlap with the test period
    and adds an embargo gap after each test set.
    """
    test_size = n_samples // n_splits
    embargo = int(n_samples * embargo_pct)
    
    for i in range(n_splits):
        test_start = i * test_size
        test_end = min((i + 1) * test_size, n_samples)
        
        # Training: everything before test, minus embargo
        train_end = max(0, test_start - embargo)
        train_idx = np.arange(0, train_end)
        
        # Can optionally add data after test + embargo
        post_embargo_start = min(test_end + embargo, n_samples)
        if post_embargo_start < n_samples:
            train_idx = np.concatenate([
                train_idx,
                np.arange(post_embargo_start, n_samples)
            ])
        
        test_idx = np.arange(test_start, test_end)
        
        if len(train_idx) > 0 and len(test_idx) > 0:
            yield train_idx.astype(int), test_idx.astype(int)
```

### Combinatorial Purged Cross-Validation (CPCV)

Lopez de Prado (2018) proposes CPCV, which generates multiple test paths by selecting $k$ out of $N$ groups as test, with purging applied:

```python
from itertools import combinations

def cpcv_splits(n_samples, n_groups=6, n_test_groups=2, embargo_pct=0.01):
    """Combinatorial Purged Cross-Validation.
    
    Generates C(n_groups, n_test_groups) train/test splits.
    """
    group_size = n_samples // n_groups
    embargo = int(n_samples * embargo_pct)
    
    for test_groups in combinations(range(n_groups), n_test_groups):
        test_idx = []
        for g in test_groups:
            start = g * group_size
            end = min((g + 1) * group_size, n_samples)
            test_idx.extend(range(start, end))
        test_idx = np.array(test_idx)
        
        # Purge: remove training samples near test boundaries
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[test_idx] = False
        
        for g in test_groups:
            purge_start = max(0, g * group_size - embargo)
            purge_end = min(n_samples, (g + 1) * group_size + embargo)
            train_mask[purge_start:purge_end] = False
        
        train_idx = np.where(train_mask)[0]
        if len(train_idx) > 0:
            yield train_idx, test_idx
```

## Custom CV Splitter for Scikit-learn

Implement the splitter interface to use with `cross_val_score`:

```python
from sklearn.model_selection import BaseCrossValidator

class WalkForwardCV(BaseCrossValidator):
    """Walk-forward with expanding window and embargo."""
    
    def __init__(self, n_splits=5, min_train_size=100, embargo=10):
        self.n_splits = n_splits
        self.min_train_size = min_train_size
        self.embargo = embargo
    
    def split(self, X, y=None, groups=None):
        n = len(X)
        test_size = (n - self.min_train_size) // self.n_splits
        
        for i in range(self.n_splits):
            test_start = self.min_train_size + i * test_size
            test_end = min(test_start + test_size, n)
            train_end = test_start - self.embargo
            
            if train_end > 0:
                train_idx = np.arange(0, train_end)
                test_idx = np.arange(test_start, test_end)
                yield train_idx, test_idx
    
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

# Use like any sklearn splitter
wfcv = WalkForwardCV(n_splits=5, min_train_size=200, embargo=20)
scores = cross_val_score(pipe, X, y, cv=wfcv, scoring='neg_mean_squared_error')
```

## Walk-Forward with GridSearch

```python
from sklearn.model_selection import GridSearchCV

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', Ridge()),
])

param_grid = {'model__alpha': np.logspace(-3, 3, 20)}

grid = GridSearchCV(
    pipe, param_grid,
    cv=TimeSeriesSplit(n_splits=5, gap=5),
    scoring='neg_mean_squared_error',
)
grid.fit(X, y)
print(f"Best alpha: {grid.best_params_['model__alpha']:.4f}")
```

## Performance Evaluation

```python
from sklearn.model_selection import cross_validate

results = cross_validate(
    pipe, X, y,
    cv=TimeSeriesSplit(n_splits=5),
    scoring={
        'mse': 'neg_mean_squared_error',
        'mae': 'neg_mean_absolute_error',
        'r2': 'r2',
    },
    return_train_score=True,
)

import pandas as pd
summary = pd.DataFrame({
    'Train MSE': -results['train_mse'],
    'Test MSE': -results['test_mse'],
    'Train R²': results['train_r2'],
    'Test R²': results['test_r2'],
})
print(summary.round(6))
print(f"\nMean Test MSE: {-results['test_mse'].mean():.6f}")
print(f"Train-Test Gap: {(-results['train_mse'] + results['test_mse']).mean():.6f}")
```

## Summary

| Splitter | Window | Use Case |
|----------|--------|----------|
| `TimeSeriesSplit` | Expanding | Standard walk-forward |
| `TimeSeriesSplit(max_train_size=N)` | Rolling | Regime-adaptive models |
| `TimeSeriesSplit(gap=G)` | Expanding + embargo | Overlapping return labels |
| Custom `WalkForwardCV` | Configurable | Purging + embargo |
| CPCV | Combinatorial | Multiple backtest paths |

**Key rules for financial CV**:
1. **Never shuffle** time series data
2. **Always purge** when labels use overlapping future windows
3. **Add embargo** to prevent information leakage near boundaries
4. **Re-fit preprocessing** on each training window (use pipelines)
5. **Report standard errors** — a single walk-forward path has high variance

## References

1. Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. Chapters 7, 12.
2. Bailey, D. H., et al. (2014). "The Deflated Sharpe Ratio." *Journal of Portfolio Management*.
