# Grid Search and Tuning

Hyperparameter tuning systematically searches parameter space to find optimal model configuration, using grid search, random search, or Bayesian optimization with cross-validation.

---

## GridSearchCV

### 1. Exhaustive Search
```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid = GridSearchCV(
    RandomForestClassifier(),
    param_grid,
    cv=5,
    scoring='accuracy'
)

grid.fit(X_train, y_train)
print(f"Best params: {grid.best_params_}")
print(f"Best score: {grid.best_score_:.3f}")
```

### 2. Results
```python
import pandas as pd
results = pd.DataFrame(grid.cv_results_)
print(results[['params', 'mean_test_score', 'rank_test_score']])

# Best estimator
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
```

---

## RandomizedSearchCV

### 1. Random Sampling
```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

param_dist = {
    'n_estimators': randint(50, 500),
    'max_depth': randint(5, 50),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10)
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(),
    param_distributions=param_dist,
    n_iter=100,  # Number of combinations
    cv=5,
    random_state=42
)

random_search.fit(X_train, y_train)
```

### 2. Continuous Distributions
```python
from scipy.stats import uniform, loguniform

param_dist = {
    'learning_rate': loguniform(1e-4, 1e-1),
    'alpha': uniform(0, 1)
}
```

---

## Pipeline Tuning

### 1. Tune Pipeline Steps
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC())
])

param_grid = {
    'svm__C': [0.1, 1, 10],
    'svm__kernel': ['linear', 'rbf'],
    'svm__gamma': ['scale', 'auto']
}

grid = GridSearchCV(pipeline, param_grid, cv=5)
grid.fit(X_train, y_train)
```

---

## Summary
**Grid Search:** Exhaustive but expensive (tries all combinations)  
**Random Search:** Efficient for large search spaces  
**Best practice:** Start with random search, refine with grid search
