# Estimator Interface

Scikit-learn's extensibility comes from a small set of base classes and mixins. By subclassing them, you can create custom transformers and estimators that integrate seamlessly with pipelines, grid search, and cross-validation.

## Base Classes and Mixins

### `BaseEstimator`

Provides `get_params()` and `set_params()` for free. The only requirement is that constructor arguments match attribute names exactly:

```python
from sklearn.base import BaseEstimator

class MyEstimator(BaseEstimator):
    def __init__(self, alpha=1.0, fit_intercept=True):
        self.alpha = alpha                # must match __init__ parameter name
        self.fit_intercept = fit_intercept

est = MyEstimator(alpha=0.5)
print(est.get_params())
# {'alpha': 0.5, 'fit_intercept': True}

est.set_params(alpha=2.0)
```

### `TransformerMixin`

Adds `fit_transform(X, y)` as a convenience that calls `fit` then `transform`:

```python
from sklearn.base import BaseEstimator, TransformerMixin

class MyTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # learn parameters
        return self
    
    def transform(self, X):
        # apply transformation
        return X_transformed
```

### `ClassifierMixin` and `RegressorMixin`

Provide a default `score` method (accuracy for classifiers, $R^2$ for regressors) and set the `_estimator_type` tag used by cross-validation and scoring utilities:

```python
from sklearn.base import ClassifierMixin, RegressorMixin

class MyClassifier(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        # ...
        return self
    
    def predict(self, X):
        # ...
        return y_pred
    
    # score(X, y) → accuracy is inherited from ClassifierMixin
```

## Writing a Custom Transformer

A robust custom transformer follows this template:

```python
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array

class RobustZScorer(BaseEstimator, TransformerMixin):
    """Scale features using median and IQR, with optional clipping.
    
    Parameters
    ----------
    quantile_range : tuple of float, default=(25.0, 75.0)
        Percentiles used to compute the IQR.
    clip : float or None, default=None
        If set, clip transformed values to [-clip, clip].
    """
    
    def __init__(self, quantile_range=(25.0, 75.0), clip=None):
        self.quantile_range = quantile_range
        self.clip = clip
    
    def fit(self, X, y=None):
        X = check_array(X, accept_sparse=False, dtype=np.float64)
        
        q_lo, q_hi = self.quantile_range
        self.center_ = np.median(X, axis=0)
        q_lower = np.percentile(X, q_lo, axis=0)
        q_upper = np.percentile(X, q_hi, axis=0)
        self.scale_ = q_upper - q_lower
        self.scale_[self.scale_ == 0] = 1.0   # avoid division by zero
        
        self.n_features_in_ = X.shape[1]
        return self
    
    def transform(self, X):
        check_is_fitted(self, ['center_', 'scale_'])
        X = check_array(X, accept_sparse=False, dtype=np.float64)
        
        X_transformed = (X - self.center_) / self.scale_
        
        if self.clip is not None:
            X_transformed = np.clip(X_transformed, -self.clip, self.clip)
        
        return X_transformed
    
    def inverse_transform(self, X):
        check_is_fitted(self, ['center_', 'scale_'])
        X = check_array(X)
        return X * self.scale_ + self.center_
```

### Key Conventions

1. **`__init__` stores parameters as attributes** with identical names—no computation in the constructor.
2. **`fit` returns `self`** to enable method chaining.
3. **Learned attributes end with `_`**: `center_`, `scale_`, `n_features_in_`.
4. **`check_array`** validates input shape and dtype.
5. **`check_is_fitted`** raises `NotFittedError` if `transform` is called before `fit`.

### Using in a Pipeline

```python
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

pipe = make_pipeline(
    RobustZScorer(),
    LogisticRegression()
)

# Grid search over custom transformer parameters
param_grid = {
    'robustzscorer__clip': [None, 3.0, 5.0],
    'logisticregression__C': [0.1, 1.0, 10.0],
}

grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)
```

## Writing a Custom Classifier

```python
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

class NearestCentroidClassifier(BaseEstimator, ClassifierMixin):
    """Classify by distance to class centroids.
    
    Parameters
    ----------
    metric : str, default='euclidean'
        Distance metric ('euclidean' or 'manhattan').
    """
    
    def __init__(self, metric='euclidean'):
        self.metric = metric
    
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.n_features_in_ = X.shape[1]
        
        # Compute centroid for each class
        self.centroids_ = np.array([
            X[y == c].mean(axis=0) for c in self.classes_
        ])
        return self
    
    def predict(self, X):
        check_is_fitted(self, ['centroids_'])
        X = check_array(X)
        
        if self.metric == 'euclidean':
            distances = np.array([
                np.linalg.norm(X - c, axis=1) for c in self.centroids_
            ]).T
        elif self.metric == 'manhattan':
            distances = np.array([
                np.abs(X - c).sum(axis=1) for c in self.centroids_
            ]).T
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
        
        return self.classes_[distances.argmin(axis=1)]
    
    def predict_proba(self, X):
        """Probability estimates via softmax of negative distances."""
        check_is_fitted(self, ['centroids_'])
        X = check_array(X)
        
        distances = np.array([
            np.linalg.norm(X - c, axis=1) for c in self.centroids_
        ]).T
        
        # Softmax of negative distances
        neg_dist = -distances
        exp_vals = np.exp(neg_dist - neg_dist.max(axis=1, keepdims=True))
        proba = exp_vals / exp_vals.sum(axis=1, keepdims=True)
        return proba
```

## Writing a Custom Regressor

```python
from sklearn.base import BaseEstimator, RegressorMixin

class WeightedAverageRegressor(BaseEstimator, RegressorMixin):
    """Predict the (optionally weighted) mean of training targets.
    
    Serves as a trivial baseline for regression benchmarking.
    """
    
    def __init__(self, strategy='mean'):
        self.strategy = strategy
    
    def fit(self, X, y, sample_weight=None):
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]
        
        if self.strategy == 'mean':
            self.prediction_ = np.average(y, weights=sample_weight)
        elif self.strategy == 'median':
            self.prediction_ = np.median(y)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        return self
    
    def predict(self, X):
        check_is_fitted(self, ['prediction_'])
        X = check_array(X)
        return np.full(X.shape[0], self.prediction_)
```

## The `FunctionTransformer` Shortcut

For simple, stateless transformations, skip the class boilerplate:

```python
from sklearn.preprocessing import FunctionTransformer

log_transform = FunctionTransformer(
    func=np.log1p,
    inverse_func=np.expm1,
    validate=True
)

# Use in pipeline
pipe = make_pipeline(log_transform, StandardScaler(), Ridge())
```

## Validation Utilities

| Utility | Purpose |
|---------|---------|
| `check_array(X)` | Validate and convert to NumPy array |
| `check_X_y(X, y)` | Validate feature matrix and target |
| `check_is_fitted(est)` | Raise `NotFittedError` if not fitted |
| `check_estimator(est)` | Run full API compliance tests |
| `unique_labels(y)` | Extract sorted unique class labels |

## Quantitative Finance Example: Winsorised Scaler

A common preprocessing step in factor investing is to winsorise extreme values before standardising:

```python
class WinsorisedScaler(BaseEstimator, TransformerMixin):
    """Winsorise at given percentiles, then z-score standardise.
    
    Standard in cross-sectional factor models to reduce the influence
    of outlier stocks on factor returns.
    """
    
    def __init__(self, limits=(0.01, 0.99)):
        self.limits = limits
    
    def fit(self, X, y=None):
        X = check_array(X)
        lo, hi = self.limits
        self.lower_ = np.percentile(X, lo * 100, axis=0)
        self.upper_ = np.percentile(X, hi * 100, axis=0)
        
        # Compute mean/std after winsorisation
        X_clipped = np.clip(X, self.lower_, self.upper_)
        self.mean_ = X_clipped.mean(axis=0)
        self.std_ = X_clipped.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        
        self.n_features_in_ = X.shape[1]
        return self
    
    def transform(self, X):
        check_is_fitted(self)
        X = check_array(X)
        X_clipped = np.clip(X, self.lower_, self.upper_)
        return (X_clipped - self.mean_) / self.std_
```

This integrates seamlessly with pipelines and grid search, and can be used in walk-forward validation where winsorisation limits are fitted only on the training window.
