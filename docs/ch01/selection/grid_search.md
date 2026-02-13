# Grid Search

`GridSearchCV` performs exhaustive search over a specified parameter grid, evaluating every combination via cross-validation to find the optimal hyperparameter configuration.

## Basic GridSearchCV

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,           # parallel
    return_train_score=True,
)

grid.fit(X_train, y_train)
print(f"Best params: {grid.best_params_}")
print(f"Best CV score: {grid.best_score_:.4f}")
print(f"Test score: {grid.best_estimator_.score(X_test, y_test):.4f}")
```

## Total Combinations

The number of fits is $|\text{grid}| \times K$:

$$\text{Total fits} = \prod_{p} |V_p| \times K$$

For the grid above: $3 \times 3 \times 3 \times 5 = 135$ fits. This grows combinatorially, so grid search is practical only for small parameter spaces.

## Analysing Results

```python
import pandas as pd

results = pd.DataFrame(grid.cv_results_)
cols = ['params', 'mean_test_score', 'std_test_score', 'rank_test_score',
        'mean_train_score', 'mean_fit_time']
print(results[cols].sort_values('rank_test_score').head(10))

# Detect overfitting: large gap between train and test scores
results['gap'] = results['mean_train_score'] - results['mean_test_score']
```

## Multi-Metric Evaluation

```python
grid_multi = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring=['accuracy', 'f1_weighted', 'roc_auc_ovr'],
    refit='f1_weighted',   # which metric to use for best_estimator_
    return_train_score=True,
)

grid_multi.fit(X_train, y_train)

# Access results for each metric
print(f"Best F1: {grid_multi.cv_results_['mean_test_f1_weighted'].max():.4f}")
print(f"Best AUC: {grid_multi.cv_results_['mean_test_roc_auc_ovr'].max():.4f}")
```

## Pipeline Grid Search

Grid search naturally extends to pipelines using the `step__param` syntax:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC()),
])

param_grid = {
    'svm__C': [0.1, 1, 10],
    'svm__kernel': ['linear', 'rbf'],
    'svm__gamma': ['scale', 'auto'],
}

grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)
print(f"Best: {grid.best_params_}")
```

## Refit and the Best Estimator

By default, `refit=True` refits the best model on the entire training set after finding the best parameters:

```python
# grid.best_estimator_ is fitted on all of X_train
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
```

Set `refit=False` when you only need the CV results (e.g., for comparison) without the final refit.

## Custom CV Splitters

```python
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit

# Stratified for classification
grid = GridSearchCV(model, param_grid, cv=StratifiedKFold(5, shuffle=True, random_state=42))

# Time series
grid = GridSearchCV(model, param_grid, cv=TimeSeriesSplit(n_splits=5))
```

## When Grid Search Is Too Expensive

Grid search scales poorly with the number of parameters. Consider switching to:

- **[Randomized Search](random_search.md)** — samples a fixed budget of combinations
- **[Bayesian Optimization](bayesian.md)** — uses a surrogate model to guide search

**Rule of thumb**: use grid search when the total grid size is $< 100$ combinations. Beyond that, randomised search is more efficient.

## Quantitative Finance: Model Selection with Walk-Forward CV

```python
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge

param_grid = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}

grid = GridSearchCV(
    Ridge(),
    param_grid,
    cv=TimeSeriesSplit(n_splits=5),
    scoring='neg_mean_squared_error',
    return_train_score=True,
)
grid.fit(X_train, y_train)

print(f"Best alpha: {grid.best_params_['alpha']}")
print(f"Best walk-forward MSE: {-grid.best_score_:.6f}")
```

## Summary

| Aspect | Detail |
|--------|--------|
| **Search strategy** | Exhaustive (all combinations) |
| **Cost** | $\prod |V_p| \times K$ fits |
| **Strengths** | Guaranteed to find best in grid, reproducible |
| **Weaknesses** | Combinatorial explosion, ignores parameter interactions |
| **When to use** | Small grids ($< 100$ combos), discrete parameters |
