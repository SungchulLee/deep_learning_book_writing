# Ensemble Methods

Ensemble methods combine multiple models to produce better predictions than any single model. They reduce variance (bagging), bias (boosting), or both, and typically achieve state-of-the-art results on tabular data.

---

## Random Forest

### 1. Random Forest Classifier

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np

# Generate data
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                           n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print(f"Train Accuracy: {model.score(X_train, y_train):.4f}")
print(f"Test Accuracy: {model.score(X_test, y_test):.4f}")
```

### 2. Key Hyperparameters

```python
model = RandomForestClassifier(
    n_estimators=100,        # Number of trees
    max_depth=10,            # Maximum depth per tree
    min_samples_split=5,     # Minimum samples to split
    min_samples_leaf=2,      # Minimum samples in leaf
    max_features='sqrt',     # Features per split: 'sqrt', 'log2', int, float
    bootstrap=True,          # Bootstrap sampling
    oob_score=True,          # Out-of-bag score
    n_jobs=-1,               # Parallel processing
    random_state=42
)
model.fit(X_train, y_train)

# Out-of-bag score (validation without holdout)
print(f"OOB Score: {model.oob_score_:.4f}")
print(f"Test Accuracy: {model.score(X_test, y_test):.4f}")
```

### 3. Feature Importance

```python
import matplotlib.pyplot as plt

# Mean decrease in impurity
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(20), importances[indices])
plt.xticks(range(20), [f'F{i}' for i in indices], rotation=45)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Random Forest Feature Importance')
plt.tight_layout()
plt.show()
```

### 4. Permutation Importance (More Reliable)

```python
from sklearn.inspection import permutation_importance

perm_importance = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=42)

sorted_idx = perm_importance.importances_mean.argsort()[::-1]

plt.figure(figsize=(10, 6))
plt.boxplot([perm_importance.importances[i] for i in sorted_idx[:10]],
            labels=[f'F{i}' for i in sorted_idx[:10]])
plt.ylabel('Decrease in Accuracy')
plt.title('Permutation Importance')
plt.tight_layout()
plt.show()
```

### 5. Random Forest Regressor

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

X_reg, y_reg = make_regression(n_samples=1000, n_features=20, noise=10, random_state=42)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

model_reg = RandomForestRegressor(n_estimators=100, random_state=42)
model_reg.fit(X_train_r, y_train_r)

print(f"Train R²: {model_reg.score(X_train_r, y_train_r):.4f}")
print(f"Test R²: {model_reg.score(X_test_r, y_test_r):.4f}")
```

---

## Gradient Boosting

### 1. Basic Usage

```python
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
model.fit(X_train, y_train)

print(f"Train Accuracy: {model.score(X_train, y_train):.4f}")
print(f"Test Accuracy: {model.score(X_test, y_test):.4f}")
```

### 2. Key Hyperparameters

```python
model = GradientBoostingClassifier(
    n_estimators=100,        # Number of boosting stages
    learning_rate=0.1,       # Shrinkage (smaller = more trees needed)
    max_depth=3,             # Depth of individual trees
    min_samples_split=2,     # Minimum samples to split
    min_samples_leaf=1,      # Minimum samples in leaf
    subsample=0.8,           # Fraction of samples per tree (stochastic GB)
    max_features='sqrt',     # Features per split
    random_state=42
)
model.fit(X_train, y_train)
```

### 3. Learning Curve (Staged Predictions)

```python
# Track performance as trees are added
train_scores = []
test_scores = []

for i, (train_pred, test_pred) in enumerate(zip(
    model.staged_predict(X_train), model.staged_predict(X_test)
)):
    train_scores.append(np.mean(train_pred == y_train))
    test_scores.append(np.mean(test_pred == y_test))

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_scores)+1), train_scores, 'b-', label='Train')
plt.plot(range(1, len(test_scores)+1), test_scores, 'r-', label='Test')
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')
plt.title('Gradient Boosting: Learning Curve')
plt.legend()
plt.grid(True)
plt.show()
```

### 4. Gradient Boosting Regressor

```python
from sklearn.ensemble import GradientBoostingRegressor

model_reg = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    loss='squared_error',  # or 'absolute_error', 'huber', 'quantile'
    random_state=42
)
model_reg.fit(X_train_r, y_train_r)

print(f"Test R²: {model_reg.score(X_test_r, y_test_r):.4f}")
```

---

## Histogram-Based Gradient Boosting

### 1. HistGradientBoosting (Faster)

```python
from sklearn.ensemble import HistGradientBoostingClassifier

# Inspired by LightGBM - much faster for large datasets
model = HistGradientBoostingClassifier(
    max_iter=100,
    learning_rate=0.1,
    max_depth=10,
    random_state=42
)
model.fit(X_train, y_train)

print(f"Test Accuracy: {model.score(X_test, y_test):.4f}")
```

### 2. Native Missing Value Support

```python
from sklearn.ensemble import HistGradientBoostingClassifier
import numpy as np

# Create data with missing values
X_with_nan = X_train.copy()
mask = np.random.random(X_with_nan.shape) < 0.1
X_with_nan[mask] = np.nan

# HistGradientBoosting handles NaN natively
model = HistGradientBoostingClassifier(random_state=42)
model.fit(X_with_nan, y_train)

X_test_nan = X_test.copy()
X_test_nan[np.random.random(X_test_nan.shape) < 0.1] = np.nan

print(f"Test Accuracy (with NaN): {model.score(X_test_nan, y_test):.4f}")
```

### 3. Early Stopping

```python
model = HistGradientBoostingClassifier(
    max_iter=1000,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    random_state=42
)
model.fit(X_train, y_train)

print(f"Stopped at iteration: {model.n_iter_}")
print(f"Test Accuracy: {model.score(X_test, y_test):.4f}")
```

---

## AdaBoost

### 1. Basic Usage

```python
from sklearn.ensemble import AdaBoostClassifier

model = AdaBoostClassifier(
    n_estimators=100,
    learning_rate=0.1,
    algorithm='SAMME',  # or 'SAMME.R' for probability estimates
    random_state=42
)
model.fit(X_train, y_train)

print(f"Test Accuracy: {model.score(X_test, y_test):.4f}")
```

### 2. With Different Base Estimator

```python
from sklearn.tree import DecisionTreeClassifier

# Default uses DecisionTreeClassifier(max_depth=1) - decision stumps
# Can use deeper trees
base_estimator = DecisionTreeClassifier(max_depth=3)
model = AdaBoostClassifier(
    estimator=base_estimator,
    n_estimators=50,
    random_state=42
)
model.fit(X_train, y_train)

print(f"Test Accuracy: {model.score(X_test, y_test):.4f}")
```

### 3. Estimator Weights

```python
# AdaBoost assigns weights to each estimator
print(f"Estimator weights: {model.estimator_weights_[:10]}")

# Higher weight = more important estimator
plt.figure(figsize=(10, 4))
plt.bar(range(len(model.estimator_weights_)), model.estimator_weights_)
plt.xlabel('Estimator Index')
plt.ylabel('Weight')
plt.title('AdaBoost Estimator Weights')
plt.show()
```

---

## Bagging

### 1. BaggingClassifier

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# Bagging with decision trees (similar to Random Forest)
model = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=100,
    max_samples=0.8,       # 80% of samples per estimator
    max_features=0.8,      # 80% of features per estimator
    bootstrap=True,        # Sample with replacement
    bootstrap_features=False,
    oob_score=True,
    n_jobs=-1,
    random_state=42
)
model.fit(X_train, y_train)

print(f"OOB Score: {model.oob_score_:.4f}")
print(f"Test Accuracy: {model.score(X_test, y_test):.4f}")
```

### 2. Bagging with Different Base Estimators

```python
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Bagging SVMs
model_svm = BaggingClassifier(
    estimator=SVC(),
    n_estimators=10,
    random_state=42
)

# Bagging KNN
model_knn = BaggingClassifier(
    estimator=KNeighborsClassifier(),
    n_estimators=10,
    random_state=42
)

for name, m in [('SVM', model_svm), ('KNN', model_knn)]:
    m.fit(X_train, y_train)
    print(f"Bagging {name}: {m.score(X_test, y_test):.4f}")
```

---

## Extra Trees

### 1. ExtraTreesClassifier

```python
from sklearn.ensemble import ExtraTreesClassifier

# Extremely Randomized Trees
# More random than Random Forest - splits are random, not optimal
model = ExtraTreesClassifier(
    n_estimators=100,
    max_features='sqrt',
    random_state=42
)
model.fit(X_train, y_train)

print(f"Test Accuracy: {model.score(X_test, y_test):.4f}")
```

### 2. Comparison with Random Forest

```python
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import time

rf = RandomForestClassifier(n_estimators=100, random_state=42)
et = ExtraTreesClassifier(n_estimators=100, random_state=42)

for name, m in [('Random Forest', rf), ('Extra Trees', et)]:
    start = time.time()
    m.fit(X_train, y_train)
    fit_time = time.time() - start
    acc = m.score(X_test, y_test)
    print(f"{name}: Accuracy={acc:.4f}, Time={fit_time:.3f}s")
```

---

## Voting Classifier

### 1. Hard Voting

```python
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Combine different types of models
estimators = [
    ('lr', LogisticRegression(max_iter=1000)),
    ('dt', DecisionTreeClassifier(max_depth=5)),
    ('svc', SVC(probability=True))
]

# Hard voting: majority vote
model = VotingClassifier(estimators=estimators, voting='hard')
model.fit(X_train, y_train)

print(f"Voting Accuracy: {model.score(X_test, y_test):.4f}")

# Individual model accuracies
for name, m in estimators:
    m.fit(X_train, y_train)
    print(f"  {name}: {m.score(X_test, y_test):.4f}")
```

### 2. Soft Voting

```python
# Soft voting: weighted average of probabilities
model = VotingClassifier(
    estimators=estimators,
    voting='soft',
    weights=[1, 1, 2]  # Give more weight to SVC
)
model.fit(X_train, y_train)

print(f"Soft Voting Accuracy: {model.score(X_test, y_test):.4f}")
```

---

## Stacking

### 1. StackingClassifier

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Base estimators
estimators = [
    ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42))
]

# Meta-learner
model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=5,  # Cross-validation for generating meta-features
    passthrough=False  # Don't pass original features to meta-learner
)
model.fit(X_train, y_train)

print(f"Stacking Accuracy: {model.score(X_test, y_test):.4f}")
```

### 2. With Passthrough

```python
# Pass original features to meta-learner
model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=5,
    passthrough=True  # Include original features
)
model.fit(X_train, y_train)

print(f"Stacking (with passthrough): {model.score(X_test, y_test):.4f}")
```

---

## Hyperparameter Tuning

### 1. Random Forest Tuning

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_dist = {
    'n_estimators': randint(50, 500),
    'max_depth': [None, 5, 10, 20, 30],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2', None]
}

rf = RandomForestClassifier(random_state=42)
search = RandomizedSearchCV(
    rf, param_dist, n_iter=50, cv=5, 
    scoring='accuracy', n_jobs=-1, random_state=42
)
search.fit(X_train, y_train)

print(f"Best params: {search.best_params_}")
print(f"Best CV score: {search.best_score_:.4f}")
print(f"Test accuracy: {search.score(X_test, y_test):.4f}")
```

### 2. Gradient Boosting Tuning

```python
param_dist = {
    'n_estimators': randint(50, 300),
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': randint(2, 10),
    'min_samples_split': randint(2, 20),
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0]
}

gb = GradientBoostingClassifier(random_state=42)
search = RandomizedSearchCV(
    gb, param_dist, n_iter=50, cv=5,
    scoring='accuracy', n_jobs=-1, random_state=42
)
search.fit(X_train, y_train)

print(f"Best params: {search.best_params_}")
print(f"Test accuracy: {search.score(X_test, y_test):.4f}")
```

---

## Comparison of Methods

### 1. Bagging vs Boosting

| Aspect | Bagging | Boosting |
|--------|---------|----------|
| **Reduces** | Variance | Bias (and variance) |
| **Training** | Parallel | Sequential |
| **Base learners** | Independent | Dependent |
| **Weighting** | Equal | Weighted by error |
| **Overfitting** | Less prone | More prone |
| **Example** | Random Forest | Gradient Boosting |

### 2. Benchmark Comparison

```python
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, ExtraTreesClassifier,
    HistGradientBoostingClassifier, BaggingClassifier
)
import time

models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'HistGradient Boosting': HistGradientBoostingClassifier(max_iter=100, random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
    'Extra Trees': ExtraTreesClassifier(n_estimators=100, random_state=42),
    'Bagging': BaggingClassifier(n_estimators=100, random_state=42)
}

print(f"{'Model':<25} {'Train Acc':<12} {'Test Acc':<12} {'Time (s)':<10}")
print("-" * 60)

for name, model in models.items():
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    
    print(f"{name:<25} {train_acc:<12.4f} {test_acc:<12.4f} {train_time:<10.3f}")
```

---

## PyTorch Comparison

### 1. Gradient Boosting Concept in PyTorch

```python
import torch
import torch.nn as nn

class SimpleTreeStump(nn.Module):
    """A simple differentiable decision stump"""
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return torch.tanh(self.linear(x))

class GradientBoostingEnsemble(nn.Module):
    """Simplified gradient boosting with neural network base learners"""
    def __init__(self, input_dim, n_estimators=10, learning_rate=0.1):
        super().__init__()
        self.learning_rate = learning_rate
        self.estimators = nn.ModuleList([
            SimpleTreeStump(input_dim) for _ in range(n_estimators)
        ])
    
    def forward(self, x):
        # Additive model
        output = torch.zeros(x.shape[0], 1, device=x.device)
        for estimator in self.estimators:
            output = output + self.learning_rate * estimator(x)
        return output

# Note: Real gradient boosting trains sequentially on residuals
# This is a simplified demonstration of the additive structure
```

### 2. Bagging in PyTorch

```python
class BaggingEnsemble(nn.Module):
    """Simple bagging ensemble"""
    def __init__(self, base_model_class, n_estimators, **model_kwargs):
        super().__init__()
        self.estimators = nn.ModuleList([
            base_model_class(**model_kwargs) for _ in range(n_estimators)
        ])
    
    def forward(self, x):
        # Average predictions from all estimators
        outputs = torch.stack([est(x) for est in self.estimators], dim=0)
        return outputs.mean(dim=0)
    
    def train_with_bootstrap(self, X, y, epochs=100, lr=0.01):
        """Train each estimator on a bootstrap sample"""
        for est in self.estimators:
            # Bootstrap sample
            indices = torch.randint(0, len(X), (len(X),))
            X_boot = X[indices]
            y_boot = y[indices]
            
            optimizer = torch.optim.Adam(est.parameters(), lr=lr)
            criterion = nn.BCEWithLogitsLoss()
            
            for epoch in range(epochs):
                optimizer.zero_grad()
                outputs = est(X_boot)
                loss = criterion(outputs.squeeze(), y_boot)
                loss.backward()
                optimizer.step()
```

---

## When to Use Each Method

### Random Forest

- First choice for most tabular problems
- When interpretability matters (feature importance)
- When training time is limited
- Handles missing values well (with HistGradientBoosting)

### Gradient Boosting

- When maximum accuracy is needed
- When you have time for hyperparameter tuning
- For competition-level performance
- Consider HistGradientBoosting for large datasets

### AdaBoost

- Simple boosting baseline
- When you need a lightweight model
- Good with decision stumps for weak learning

### Voting/Stacking

- When you have diverse models
- For production systems where stability matters
- When individual models are already tuned

---

## Summary

| Method | Type | Parallelizable | Handles NaN | Typical Use |
|--------|------|----------------|-------------|-------------|
| Random Forest | Bagging | Yes | With imputation | General purpose |
| Extra Trees | Bagging | Yes | With imputation | Faster RF alternative |
| Gradient Boosting | Boosting | No | With imputation | High accuracy |
| HistGradient Boosting | Boosting | Partially | Yes | Large datasets |
| AdaBoost | Boosting | No | With imputation | Simple baseline |
| Voting | Meta | Yes | Depends | Model combination |
| Stacking | Meta | Partially | Depends | Advanced combination |

**Key takeaways:**
- **Start with Random Forest** for most problems
- **Use HistGradientBoosting** for large datasets
- **Tune Gradient Boosting** for best accuracy
- **Consider Stacking** when you have diverse models
- **Always use cross-validation** for ensemble methods
