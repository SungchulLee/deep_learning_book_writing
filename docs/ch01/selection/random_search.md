# Randomized Search

`RandomizedSearchCV` samples a fixed number of parameter combinations from specified distributions, trading exhaustiveness for efficiency. For the same computational budget, it often finds better configurations than grid search because it explores more of the parameter space.

## Why Randomized Over Grid?

Grid search discretises each parameter independently, so adding one new value to a 5-parameter grid multiplies the total cost. Randomised search fixes the budget at $n$ iterations regardless of parameter dimensionality.

**Bergstra & Bengio (2012)** showed that for most problems, only a few parameters matter. Randomised search allocates more resolution to important parameters automatically because it draws from continuous distributions.

## Basic RandomizedSearchCV

```python
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint, uniform, loguniform
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_distributions = {
    'n_estimators': randint(50, 500),
    'max_depth': randint(3, 50),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': uniform(0.1, 0.9),
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions,
    n_iter=100,          # number of random combinations
    cv=5,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1,
    return_train_score=True,
)

random_search.fit(X_train, y_train)
print(f"Best params: {random_search.best_params_}")
print(f"Best CV score: {random_search.best_score_:.4f}")
```

## Distribution Specification

### Discrete Distributions

```python
from scipy.stats import randint

# Uniform integer in [low, high)
randint(50, 500)        # n_estimators: 50 to 499
randint(2, 20)          # min_samples_split: 2 to 19

# Explicit list (sampled uniformly)
param_distributions = {
    'kernel': ['linear', 'rbf', 'poly'],
    'degree': [2, 3, 4, 5],
}
```

### Continuous Distributions

```python
from scipy.stats import uniform, loguniform

# Uniform on [loc, loc + scale]
uniform(0, 1)           # [0, 1]
uniform(0.1, 0.9)       # [0.1, 1.0]

# Log-uniform (essential for learning rates, regularisation)
loguniform(1e-5, 1e-1)  # samples uniformly in log-space
# Equivalent to: 10^U where U ~ Uniform(-5, -1)
```

### When to Use Log-Uniform

For parameters where the order of magnitude matters more than the exact value:

| Parameter | Distribution | Range |
|-----------|-------------|-------|
| Learning rate | `loguniform(1e-5, 1e-1)` | 0.00001 to 0.1 |
| Regularisation $\alpha$ | `loguniform(1e-4, 1e2)` | 0.0001 to 100 |
| SVM $C$ | `loguniform(1e-2, 1e3)` | 0.01 to 1000 |
| `n_estimators` | `randint(50, 500)` | 50 to 499 |
| `max_depth` | `randint(3, 30)` | 3 to 29 |

## Analysing Results

```python
import pandas as pd

results = pd.DataFrame(random_search.cv_results_)
results_sorted = results.sort_values('rank_test_score')[
    ['params', 'mean_test_score', 'std_test_score', 'mean_fit_time']
]
print(results_sorted.head(10))
```

### Visualising Parameter Importance

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, param in zip(axes, ['param_n_estimators', 'param_max_depth', 'param_min_samples_split']):
    ax.scatter(results[param], results['mean_test_score'], alpha=0.5)
    ax.set_xlabel(param.replace('param_', ''))
    ax.set_ylabel('CV Score')
plt.tight_layout()
```

## Pipeline Integration

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC()),
])

param_distributions = {
    'svm__C': loguniform(1e-2, 1e3),
    'svm__gamma': loguniform(1e-5, 1e1),
    'svm__kernel': ['rbf', 'poly'],
}

random_search = RandomizedSearchCV(
    pipe, param_distributions, n_iter=50, cv=5, random_state=42
)
random_search.fit(X_train, y_train)
```

## Budget Allocation Strategy

A practical workflow:

1. **Coarse search** ($n\_iter = 50\text{–}100$) with wide distributions to identify promising regions
2. **Refined search** ($n\_iter = 50$) with narrower distributions around the best region
3. **Optional grid search** over a small grid near the optimum for fine-tuning

```python
# Step 1: Coarse
coarse = RandomizedSearchCV(model, wide_distributions, n_iter=100, cv=5)
coarse.fit(X_train, y_train)

# Step 2: Refine around best
best = coarse.best_params_
refined_distributions = {
    'n_estimators': randint(max(50, best['n_estimators'] - 50),
                            best['n_estimators'] + 50),
    'max_depth': randint(max(2, best['max_depth'] - 5),
                          best['max_depth'] + 5),
}
refined = RandomizedSearchCV(model, refined_distributions, n_iter=50, cv=5)
refined.fit(X_train, y_train)
```

## Grid Search vs. Randomized Search

| Aspect | Grid Search | Randomized Search |
|--------|------------|-------------------|
| **Combinations tried** | All ($\prod |V_p|$) | Fixed $n\_iter$ |
| **Scaling with \# params** | Exponential | Linear |
| **Continuous params** | Must discretise | Native support |
| **Reproducibility** | Deterministic | Set `random_state` |
| **Best for** | $< 100$ combos | Large or continuous spaces |

## Quantitative Finance: Hyperparameter Search for GBM Alpha Model

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit

param_distributions = {
    'n_estimators': randint(100, 1000),
    'max_depth': randint(2, 8),
    'learning_rate': loguniform(1e-3, 3e-1),
    'subsample': uniform(0.5, 0.5),       # [0.5, 1.0]
    'min_samples_leaf': randint(5, 50),
}

random_search = RandomizedSearchCV(
    GradientBoostingRegressor(random_state=42),
    param_distributions,
    n_iter=200,
    cv=TimeSeriesSplit(n_splits=5),
    scoring='neg_mean_squared_error',
    random_state=42,
    n_jobs=-1,
)

random_search.fit(X_factors, forward_returns)
print(f"Best walk-forward MSE: {-random_search.best_score_:.6f}")
```

## References

1. Bergstra, J. & Bengio, Y. (2012). "Random Search for Hyper-Parameter Optimization." *JMLR*, 13, 281–305.
