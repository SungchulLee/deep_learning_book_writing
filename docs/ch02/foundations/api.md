# API Overview

Scikit-learn's power comes from a **uniform interface**: every algorithm—whether a scaler, a random forest, or a clustering method—exposes the same small set of methods. Internalising these conventions lets you swap components freely and build complex workflows from interchangeable parts.

## The Three Core Methods

Every scikit-learn object implements a subset of three methods:

$$\text{Estimator} \xrightarrow{\texttt{fit}(X, y)} \text{Fitted Estimator} \xrightarrow{\texttt{predict}(X) \;\text{or}\; \texttt{transform}(X)} \text{Output}$$

### `fit(X, y=None)` — Learn from Data

All estimators implement `fit`. It learns parameters from the training data and stores them as attributes with a trailing underscore:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)

# Learned parameters: mean_ and scale_
print(scaler.mean_)    # per-feature means
print(scaler.scale_)   # per-feature standard deviations
```

The trailing underscore convention (`mean_`, `coef_`, `classes_`) distinguishes **learned attributes** from **constructor parameters** (`with_mean`, `C`, `n_estimators`).

### `transform(X)` — Apply a Learned Transformation

Transformers (scalers, encoders, decomposition methods) implement `transform` to map data using the parameters learned during `fit`:

```python
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)   # uses training statistics
```

The convenience method `fit_transform(X, y)` combines both steps and may be optimised internally (e.g., PCA computes the SVD once rather than fitting then transforming separately):

```python
X_train_scaled = scaler.fit_transform(X_train)  # fit + transform in one call
X_test_scaled = scaler.transform(X_test)          # transform only
```

### `predict(X)` — Generate Predictions

Predictors (classifiers, regressors, clusterers) implement `predict`. Many also provide `predict_proba` for probability estimates and `decision_function` for raw scores:

```python
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)       # shape (n_samples, n_classes)
y_score = clf.decision_function(X_test)   # raw log-odds for binary
```

## Parameter Conventions

### Constructor Parameters vs. Learned Attributes

```python
from sklearn.ensemble import RandomForestClassifier

# Constructor parameters — set before fitting
rf = RandomForestClassifier(
    n_estimators=100,      # hyperparameter
    max_depth=10,          # hyperparameter
    random_state=42        # reproducibility
)

rf.fit(X_train, y_train)

# Learned attributes — available after fitting
print(rf.feature_importances_)   # trailing underscore
print(rf.n_classes_)
print(rf.classes_)
```

### `get_params` / `set_params`

Every estimator supports introspection and modification of its constructor parameters:

```python
params = rf.get_params()
# {'n_estimators': 100, 'max_depth': 10, 'random_state': 42, ...}

rf.set_params(n_estimators=200)
# Returns self, enabling chaining
```

This interface is what enables `GridSearchCV` and `Pipeline` to manipulate nested parameters using the `step__param` syntax:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression())
])

pipe.set_params(clf__C=0.1)
pipe.get_params()['clf__C']  # 0.1
```

### Cloning

`clone` creates a new estimator with the same constructor parameters but **without** fitted state:

```python
from sklearn.base import clone

rf_fresh = clone(rf)
# Same hyperparameters, no fitted attributes
```

This is used internally by cross-validation to ensure each fold trains a fresh model.

## Estimator Tags and Type Checking

Scikit-learn uses duck typing augmented with mixin classes to classify estimators:

```python
from sklearn.base import is_classifier, is_regressor
from sklearn.utils.estimator_checks import check_estimator

print(is_classifier(LogisticRegression()))  # True
print(is_regressor(LogisticRegression()))   # False

# Full API compliance check (useful for custom estimators)
check_estimator(LogisticRegression())
```

| Role | Required Methods | Mixin |
|------|-----------------|-------|
| Estimator | `fit` | `BaseEstimator` |
| Transformer | `fit`, `transform` | `TransformerMixin` |
| Predictor | `fit`, `predict` | — |
| Classifier | `fit`, `predict`, `predict_proba` | `ClassifierMixin` |
| Regressor | `fit`, `predict` | `RegressorMixin` |

## The `score` Method

Classifiers and regressors provide a default `score` method:

```python
# ClassifierMixin.score → accuracy
accuracy = clf.score(X_test, y_test)

# RegressorMixin.score → R²
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X_train, y_train)
r2 = reg.score(X_test, y_test)
```

For custom scoring, use `sklearn.metrics` functions or the `scoring` parameter in cross-validation (see [Custom Scorers](../metrics/custom.md)).

## Data Representation

Scikit-learn expects data in a consistent format:

```python
import numpy as np

# Feature matrix: 2D array-like, shape (n_samples, n_features)
X = np.array([[1.0, 2.0],
              [3.0, 4.0],
              [5.0, 6.0]])

# Target vector: 1D array-like, shape (n_samples,)
y = np.array([0, 1, 0])

# Pandas DataFrames work directly (sklearn ≥ 1.0)
import pandas as pd
df = pd.DataFrame(X, columns=['feature_a', 'feature_b'])
clf.fit(df, y)  # feature names propagated
```

Sparse matrices (`scipy.sparse`) are supported by most estimators for high-dimensional data like text features.

## Common Patterns

### Pattern 1: Fit on Train, Transform Both

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # learn + apply
X_test_scaled = scaler.transform(X_test)          # apply only

# WRONG — causes data leakage:
# scaler.fit_transform(X_all)
```

### Pattern 2: Pipeline for Reproducibility

```python
from sklearn.pipeline import make_pipeline

pipe = make_pipeline(StandardScaler(), LogisticRegression())
pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)
# Scaler fitted only on X_train internally
```

### Pattern 3: Persistence

```python
import joblib

joblib.dump(pipe, 'model.pkl')
loaded = joblib.load('model.pkl')
y_pred = loaded.predict(X_new)
```

## Quantitative Finance Context

The uniform API is particularly valuable in finance workflows where you routinely compare many model families on the same dataset:

```python
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

models = {
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.01),
    'RF': RandomForestRegressor(n_estimators=100),
    'GBM': GradientBoostingRegressor(n_estimators=100),
}

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    print(f"{name:8s}: MSE = {-scores.mean():.4f} ± {scores.std():.4f}")
```

Because every model exposes `fit`, `predict`, and `score`, the loop body is identical regardless of the algorithm. This composability extends to pipelines, grid search, and evaluation—the topics of the remaining sections.
