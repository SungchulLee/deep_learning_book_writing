# Feature Selection

Feature selection reduces dimensionality by retaining only the most informative features, improving model interpretability, reducing overfitting, and speeding up training. Methods fall into three categories: filter, wrapper, and embedded.

## Filter Methods

Filter methods score each feature independently using a statistical test, then select the top $k$ or those exceeding a threshold. They are fast and model-agnostic.

### SelectKBest

```python
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.datasets import make_classification
import numpy as np

X, y = make_classification(n_samples=500, n_features=20, n_informative=5, random_state=42)

# ANOVA F-test (assumes linear relationship)
selector_f = SelectKBest(score_func=f_classif, k=10)
X_selected = selector_f.fit_transform(X, y)

print(f"Original: {X.shape[1]} features → Selected: {X_selected.shape[1]}")
print(f"F-scores: {selector_f.scores_[:5].round(2)}")
print(f"p-values: {selector_f.pvalues_[:5].round(4)}")
print(f"Selected mask: {selector_f.get_support()}")
```

### Scoring Functions

| Function | Task | Captures |
|----------|------|----------|
| `f_classif` | Classification | Linear dependence (ANOVA F) |
| `f_regression` | Regression | Linear dependence (F-test) |
| `mutual_info_classif` | Classification | Any dependence (non-parametric) |
| `mutual_info_regression` | Regression | Any dependence (non-parametric) |
| `chi2` | Classification | Non-negative features only |

```python
# Mutual information captures non-linear relationships
selector_mi = SelectKBest(score_func=mutual_info_classif, k=10)
X_selected_mi = selector_mi.fit_transform(X, y)

# Compare rankings
import pandas as pd
comparison = pd.DataFrame({
    'F-score': selector_f.scores_,
    'MI-score': selector_mi.scores_,
}).round(3)
```

### SelectPercentile

```python
from sklearn.feature_selection import SelectPercentile

# Keep top 25% of features
selector = SelectPercentile(score_func=f_classif, percentile=25)
X_selected = selector.fit_transform(X, y)
```

### VarianceThreshold

Remove low-variance features (unsupervised — does not use `y`):

```python
from sklearn.feature_selection import VarianceThreshold

# Remove features with variance < 0.01
selector = VarianceThreshold(threshold=0.01)
X_filtered = selector.fit_transform(X)
```

## Wrapper Methods

Wrapper methods evaluate feature subsets by training a model, making them more expensive but better at capturing feature interactions.

### Recursive Feature Elimination (RFE)

Iteratively removes the least important feature based on model coefficients or feature importances:

```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

estimator = LogisticRegression(max_iter=1000)
rfe = RFE(estimator, n_features_to_select=10, step=1)
X_rfe = rfe.fit_transform(X, y)

print(f"Selected features: {np.where(rfe.support_)[0]}")
print(f"Feature ranking: {rfe.ranking_}")
```

### RFECV — RFE with Cross-Validation

Automatically selects the optimal number of features:

```python
from sklearn.feature_selection import RFECV

rfecv = RFECV(
    estimator=LogisticRegression(max_iter=1000),
    step=1,
    cv=5,
    scoring='accuracy',
    min_features_to_select=2
)
rfecv.fit(X, y)

print(f"Optimal features: {rfecv.n_features_}")
print(f"CV scores per n_features: {rfecv.cv_results_['mean_test_score']}")
```

### Sequential Feature Selection

Forward or backward selection:

```python
from sklearn.feature_selection import SequentialFeatureSelector

# Forward selection
sfs_forward = SequentialFeatureSelector(
    LogisticRegression(max_iter=1000),
    n_features_to_select=10,
    direction='forward',
    cv=5
)
X_forward = sfs_forward.fit_transform(X, y)

# Backward elimination
sfs_backward = SequentialFeatureSelector(
    LogisticRegression(max_iter=1000),
    n_features_to_select=10,
    direction='backward',
    cv=5
)
X_backward = sfs_backward.fit_transform(X, y)
```

## Embedded Methods

Embedded methods perform selection as part of the model training process itself.

### SelectFromModel

Selects features based on importance weights from a fitted model:

```python
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

# Tree-based feature importance
rf = RandomForestClassifier(n_estimators=100, random_state=42)
selector = SelectFromModel(rf, threshold='median')
X_selected = selector.fit_transform(X, y)

print(f"Selected {X_selected.shape[1]} of {X.shape[1]} features")
```

### L1 Regularisation (Sparsity)

Lasso and L1-penalised logistic regression drive irrelevant coefficients to exactly zero:

```python
from sklearn.linear_model import Lasso, LogisticRegression

# Lasso for regression
lasso = Lasso(alpha=0.01)
selector = SelectFromModel(lasso)
X_selected = selector.fit_transform(X, y)

# L1 logistic regression for classification
l1_lr = LogisticRegression(penalty='l1', solver='saga', C=0.1, max_iter=1000)
selector = SelectFromModel(l1_lr)
X_selected = selector.fit_transform(X, y)
```

### Tree-Based Importance

```python
from sklearn.ensemble import GradientBoostingClassifier

gbm = GradientBoostingClassifier(n_estimators=100)
gbm.fit(X, y)

# Impurity-based importance
importances = gbm.feature_importances_
indices = np.argsort(importances)[::-1]

print("Top 10 features:")
for i in range(10):
    print(f"  Feature {indices[i]}: {importances[indices[i]]:.4f}")
```

## Dimensionality Reduction as Feature Extraction

Unlike selection (keeping original features), extraction creates new features. PCA is the most common unsupervised approach:

```python
from sklearn.decomposition import PCA

# Keep 95% of variance
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X)
print(f"Reduced: {X.shape[1]} → {X_pca.shape[1]} components")
print(f"Explained variance: {pca.explained_variance_ratio_.cumsum()[-1]:.2%}")
```

See [Transformers](transformers.md) for more on PCA, t-SNE, UMAP, and other dimensionality reduction techniques.

## Pipeline Integration

Feature selection should always be inside a pipeline to prevent leakage:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', SelectKBest(f_classif, k=10)),
    ('classifier', LogisticRegression(max_iter=1000)),
])

from sklearn.model_selection import cross_val_score
scores = cross_val_score(pipe, X, y, cv=5, scoring='accuracy')
```

### Grid Search Over Selection Parameters

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'selector__k': [5, 10, 15, 20],
    'classifier__C': [0.1, 1.0, 10.0],
}

grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy')
grid.fit(X, y)
print(f"Best k={grid.best_params_['selector__k']}, C={grid.best_params_['classifier__C']}")
```

## Quantitative Finance: Factor Selection

In factor investing, feature selection identifies which candidate factors have genuine predictive power for cross-sectional returns:

```python
# Candidate factors: value, momentum, size, quality, volatility, ...
# Target: forward 1-month returns

from sklearn.feature_selection import mutual_info_regression

mi_scores = mutual_info_regression(factor_matrix, forward_returns)
factor_ranking = pd.Series(mi_scores, index=factor_names).sort_values(ascending=False)

# Select factors with MI significantly above zero
# Use RFECV with a Ridge model for the final factor set
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', RFECV(Ridge(alpha=1.0), cv=TimeSeriesSplit(5))),
    ('model', Ridge(alpha=1.0)),
])
```

## Summary

| Method | Type | Speed | Captures Interactions | Use When |
|--------|------|-------|----------------------|----------|
| `SelectKBest` | Filter | Fast | No | Quick screening, many features |
| `VarianceThreshold` | Filter | Very fast | No | Remove near-constant features |
| `RFE` / `RFECV` | Wrapper | Slow | Via model | Need optimal subset size |
| `SelectFromModel` | Embedded | Moderate | Via model | Tree or L1 importance available |
| `SequentialFeatureSelector` | Wrapper | Slow | Via model | Small feature sets |
| PCA | Extraction | Moderate | Yes (linear) | Reduce multicollinearity |
