# Custom Scorers

Scikit-learn's built-in metrics cover standard cases, but many real-world problems—especially in finance—require custom loss functions. `make_scorer` bridges any Python function into the `scoring` parameter of cross-validation and grid search.

## `make_scorer` Basics

```python
from sklearn.metrics import make_scorer
import numpy as np

# Custom metric function: mean absolute percentage error
def mean_absolute_percentage_error(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# Wrap as scorer (greater_is_better=False because lower MAPE is better)
mape_scorer = make_scorer(mean_absolute_percentage_error, greater_is_better=False)
```

### Using in Cross-Validation

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge

scores = cross_val_score(Ridge(), X, y, cv=5, scoring=mape_scorer)
print(f"MAPE: {-scores.mean():.2f}% ± {scores.std():.2f}%")
# Negative because sklearn maximises by convention
```

### Using in GridSearchCV

```python
from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(
    Ridge(),
    {'alpha': [0.01, 0.1, 1.0, 10.0]},
    cv=5,
    scoring=mape_scorer,
)
grid.fit(X_train, y_train)
print(f"Best alpha: {grid.best_params_['alpha']}, MAPE: {-grid.best_score_:.2f}%")
```

## Asymmetric Cost Functions

In many applications, false positives and false negatives have different costs:

```python
def asymmetric_cost(y_true, y_pred, fp_cost=1.0, fn_cost=10.0):
    """Weighted misclassification cost.
    
    False negatives (missing a default) are 10x more costly than false positives.
    """
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return fp * fp_cost + fn * fn_cost

cost_scorer = make_scorer(asymmetric_cost, greater_is_better=False)
```

### With Probability Threshold

```python
def threshold_cost(y_true, y_proba, threshold=0.3, fp_cost=1, fn_cost=10):
    """Apply custom threshold to probabilities, then compute cost."""
    y_pred = (y_proba[:, 1] >= threshold).astype(int)
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return fp * fp_cost + fn * fn_cost

# needs_proba=True to receive predict_proba output
threshold_scorer = make_scorer(
    threshold_cost, greater_is_better=False, needs_proba=True
)
```

## `needs_proba` and `needs_threshold`

| Parameter | Input Received | Use When |
|-----------|---------------|----------|
| Default | `y_pred` from `predict()` | Hard predictions |
| `needs_proba=True` | `y_proba` from `predict_proba()` | Probability-based metrics |
| `needs_threshold=True` | `y_score` from `decision_function()` | Score-based metrics |

```python
# Brier score (needs probabilities)
def brier_score(y_true, y_proba):
    return np.mean((y_proba[:, 1] - y_true) ** 2)

brier_scorer = make_scorer(brier_score, greater_is_better=False, needs_proba=True)
```

## Passing Extra Parameters

```python
def weighted_mse(y_true, y_pred, sample_weight=None):
    errors = (y_true - y_pred) ** 2
    if sample_weight is not None:
        return np.average(errors, weights=sample_weight)
    return np.mean(errors)

# Pass extra kwargs to the scorer
wmse_scorer = make_scorer(
    weighted_mse,
    greater_is_better=False,
    sample_weight=weights_array,
)
```

## Multi-Metric Scoring

Use multiple scorers simultaneously in `GridSearchCV`:

```python
from sklearn.metrics import make_scorer, accuracy_score, f1_score

scoring = {
    'accuracy': 'accuracy',
    'f1': 'f1_weighted',
    'custom_cost': cost_scorer,
}

grid = GridSearchCV(
    model, param_grid, cv=5,
    scoring=scoring,
    refit='custom_cost',  # which metric to optimise
)
grid.fit(X_train, y_train)

# Results for all metrics
print(grid.cv_results_['mean_test_accuracy'])
print(grid.cv_results_['mean_test_f1'])
print(grid.cv_results_['mean_test_custom_cost'])
```

## Built-in Scorer Strings

For convenience, sklearn accepts string names for common metrics:

| String | Metric | Task |
|--------|--------|------|
| `'accuracy'` | Accuracy | Classification |
| `'f1'` | F1 (binary) | Classification |
| `'f1_weighted'` | Weighted F1 | Multi-class |
| `'roc_auc'` | ROC-AUC | Binary classification |
| `'roc_auc_ovr'` | ROC-AUC (one-vs-rest) | Multi-class |
| `'neg_mean_squared_error'` | $-$MSE | Regression |
| `'neg_mean_absolute_error'` | $-$MAE | Regression |
| `'r2'` | $R^2$ | Regression |

The `neg_` prefix exists because sklearn always maximises the score.

## Quantitative Finance: Custom Scorers

### Sharpe-Like Scorer for Return Prediction

```python
def information_ratio(y_true, y_pred):
    """IC-based scorer: rank correlation between predicted and actual returns."""
    from scipy.stats import spearmanr
    corr, _ = spearmanr(y_pred, y_true)
    return corr

ic_scorer = make_scorer(information_ratio, greater_is_better=True)
```

### Profit-Based Scorer

```python
def trading_pnl(y_true, y_pred):
    """PnL from going long when predicted return > 0, short otherwise."""
    positions = np.sign(y_pred)
    pnl = positions * y_true
    sharpe = pnl.mean() / (pnl.std() + 1e-8) * np.sqrt(252)
    return sharpe

pnl_scorer = make_scorer(trading_pnl, greater_is_better=True)

# Use in model selection
grid = GridSearchCV(
    model, param_grid, cv=TimeSeriesSplit(5), scoring=pnl_scorer
)
```

### Regulatory-Aware Credit Scorer

```python
def gini_coefficient(y_true, y_proba):
    """Gini coefficient (2 * AUC - 1), standard in credit risk."""
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_true, y_proba[:, 1])
    return 2 * auc - 1

gini_scorer = make_scorer(gini_coefficient, greater_is_better=True, needs_proba=True)
```

## Summary

| Step | How |
|------|-----|
| Define metric function | `def metric(y_true, y_pred): ...` |
| Wrap with `make_scorer` | `scorer = make_scorer(metric, greater_is_better=...)` |
| Use in CV / Grid Search | `cross_val_score(model, X, y, scoring=scorer)` |
| For probabilities | `make_scorer(..., needs_proba=True)` |
| Multiple metrics | `scoring={'name': scorer, ...}` |
