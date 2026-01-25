# Practical Guidelines for Hyperparameter Tuning

## Overview

Effective hyperparameter tuning requires more than choosing an optimization algorithm—it demands strategic planning, proper experimental design, and sound methodology. This section provides comprehensive guidelines for making informed decisions about hyperparameter optimization in practice.

## Method Selection Framework

### Decision Tree for Method Selection

```
                    Start
                      │
                      ▼
            ┌─────────────────────┐
            │  Parameter space    │
            │  size?              │
            └─────────────────────┘
                      │
         ┌────────────┼────────────┐
         ▼            ▼            ▼
      Small       Medium         Large
     (<100)     (100-1000)      (>1000)
         │            │            │
         ▼            ▼            ▼
    Grid Search   Is objective   Random Search
         │        expensive?      or Bayesian
         │            │
         │     ┌──────┴──────┐
         │     ▼             ▼
         │    Yes           No
         │     │             │
         │     ▼             ▼
         │  Bayesian    Random Search
         │  Optimization
         │
         ▼
    Are you doing
    final fine-tuning?
         │
    ┌────┴────┐
    ▼         ▼
   Yes       No
    │         │
    ▼         ▼
  Grid     Random
  Search   Search
```

### Method Comparison Summary

| Criterion | Grid Search | Random Search | Bayesian Opt |
|-----------|-------------|---------------|--------------|
| **Sample Efficiency** | Low | Medium | High |
| **Computational Cost** | Exponential | Linear | Linear + overhead |
| **Continuous Params** | No (discretized) | Yes | Yes |
| **Parallelization** | Excellent | Excellent | Limited |
| **Implementation** | Simple | Simple | Moderate |
| **Best For** | Small spaces, final tuning | Initial exploration | Expensive evaluations |

## Search Space Design

### Parameter Importance by Model Type

Not all hyperparameters are equally important. Prioritize tuning based on typical impact:

| Model Type | High Impact | Medium Impact | Low Impact |
|------------|-------------|---------------|------------|
| **Random Forest** | n_estimators, max_depth | min_samples_split | min_samples_leaf |
| **Gradient Boosting** | learning_rate, n_estimators | max_depth, subsample | min_samples_split |
| **Neural Networks** | learning_rate, architecture | batch_size, dropout | weight_decay |
| **SVM** | C, kernel, gamma | class_weight | shrinking |

### Distribution Selection Guidelines

```python
from scipy.stats import loguniform, randint, uniform

# Parameter type → Distribution mapping
PARAM_DISTRIBUTIONS = {
    # Learning rates: ALWAYS log-scale
    'learning_rate': loguniform(1e-5, 1e-1),
    
    # Regularization strength: log-scale
    'alpha': loguniform(1e-6, 1e0),
    'lambda': loguniform(1e-6, 1e0),
    'C': loguniform(1e-2, 1e3),
    
    # Tree depth: small integers
    'max_depth': randint(3, 30),
    
    # Number of estimators/units: linear integers
    'n_estimators': randint(50, 500),
    'hidden_units': randint(32, 512),
    
    # Dropout/subsample: uniform in [0, 0.5] or [0.5, 1.0]
    'dropout': uniform(0.0, 0.5),
    'subsample': uniform(0.5, 0.5),  # [0.5, 1.0]
    
    # Batch size: powers of 2 (categorical)
    'batch_size': [16, 32, 64, 128, 256],
}
```

### Iterative Refinement Strategy

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform, randint, uniform
import numpy as np

def coarse_to_fine_search(model_class, X, y, 
                          coarse_params, refine_func,
                          coarse_iter=30, fine_iter=50):
    """
    Two-stage hyperparameter search: coarse exploration then fine-tuning.
    """
    # Stage 1: Coarse search
    print("Stage 1: Coarse Search (wide ranges)")
    coarse_search = RandomizedSearchCV(
        model_class(random_state=42),
        coarse_params,
        n_iter=coarse_iter,
        cv=3,
        n_jobs=-1,
        random_state=42
    )
    coarse_search.fit(X, y)
    print(f"  Best: {coarse_search.best_score_:.4f}")
    
    # Stage 2: Fine search around best
    print("\nStage 2: Fine Search (refined ranges)")
    fine_params = refine_func(coarse_search.best_params_)
    fine_search = RandomizedSearchCV(
        model_class(random_state=42),
        fine_params,
        n_iter=fine_iter,
        cv=5,
        n_jobs=-1,
        random_state=42
    )
    fine_search.fit(X, y)
    print(f"  Best: {fine_search.best_score_:.4f}")
    
    return fine_search.best_estimator_, fine_search.best_params_

# Example refinement function
def refine_gb_params(best):
    """Narrow ranges around best Gradient Boosting parameters."""
    return {
        'n_estimators': randint(
            max(50, best['n_estimators'] - 50),
            best['n_estimators'] + 50
        ),
        'learning_rate': loguniform(
            best['learning_rate'] * 0.5,
            best['learning_rate'] * 2.0
        ),
        'max_depth': randint(
            max(2, best['max_depth'] - 2),
            best['max_depth'] + 3
        ),
    }
```

## Cross-Validation Best Practices

### Selecting CV Strategy

| Data Characteristic | Recommended CV | Why |
|---------------------|----------------|-----|
| Balanced classes | KFold | Standard approach |
| Imbalanced classes | StratifiedKFold | Preserves class ratios |
| Time series | TimeSeriesSplit | Respects temporal order |
| Grouped data | GroupKFold | Prevents data leakage |
| Small dataset | Leave-One-Out or RepeatedKFold | Maximum data utilization |

```python
from sklearn.model_selection import (
    KFold, StratifiedKFold, TimeSeriesSplit, 
    GroupKFold, RepeatedStratifiedKFold
)

def select_cv(y, groups=None, is_timeseries=False, n_splits=5):
    """Automatically select appropriate CV strategy."""
    if is_timeseries:
        return TimeSeriesSplit(n_splits=n_splits)
    
    if groups is not None:
        return GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    
    # Check class balance
    _, counts = np.unique(y, return_counts=True)
    if counts.min() / counts.max() < 0.3:
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    return KFold(n_splits=n_splits, shuffle=True, random_state=42)
```

### Nested CV for Unbiased Estimates

Standard hyperparameter tuning produces optimistically biased performance estimates. Use nested CV for unbiased evaluation:

```python
from sklearn.model_selection import cross_val_score, RandomizedSearchCV

def nested_cv_evaluation(model_class, param_dist, X, y,
                         outer_cv=5, inner_cv=3, n_iter=50):
    """
    Nested cross-validation for unbiased performance estimation.
    """
    # Inner CV: hyperparameter tuning
    inner_cv_search = RandomizedSearchCV(
        model_class(random_state=42),
        param_dist,
        n_iter=n_iter,
        cv=inner_cv,
        n_jobs=-1,
        random_state=42
    )
    
    # Outer CV: performance estimation
    outer_scores = cross_val_score(
        inner_cv_search, X, y,
        cv=outer_cv,
        n_jobs=-1
    )
    
    print(f"Nested CV Score: {outer_scores.mean():.4f} (±{outer_scores.std():.4f})")
    return outer_scores
```

## Reproducibility

### Complete Seed Setting

```python
import numpy as np
import random
import os

def set_all_seeds(seed=42):
    """Set seeds for complete reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # PyTorch (if available)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    
    # TensorFlow (if available)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass

# Call at start of every experiment
set_all_seeds(42)
```

### Experiment Tracking

```python
import json
from datetime import datetime
import os

class SimpleExperimentTracker:
    """Lightweight experiment tracking for hyperparameter tuning."""
    
    def __init__(self, name, save_dir='./experiments'):
        self.name = name
        self.save_dir = save_dir
        self.results = []
        os.makedirs(save_dir, exist_ok=True)
    
    def log(self, params, score, extra=None):
        self.results.append({
            'timestamp': datetime.now().isoformat(),
            'params': params,
            'score': score,
            'extra': extra
        })
    
    def save(self):
        path = os.path.join(
            self.save_dir, 
            f"{self.name}_{datetime.now():%Y%m%d_%H%M%S}.json"
        )
        with open(path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"Saved {len(self.results)} results to {path}")
    
    def best(self):
        return max(self.results, key=lambda x: x['score'])

# Usage
tracker = SimpleExperimentTracker('rf_tuning')
# tracker.log({'n_estimators': 100}, 0.95)
# tracker.save()
```

## Common Pitfalls and Solutions

### Pitfall 1: Data Leakage in Preprocessing

**Problem**: Fitting preprocessors on entire dataset before CV split.

```python
# ❌ WRONG: Data leakage
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Leaks test info into training
scores = cross_val_score(model, X_scaled, y, cv=5)

# ✅ CORRECT: Use Pipeline
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier())
])
scores = cross_val_score(pipeline, X, y, cv=5)
```

### Pitfall 2: Overfitting the Validation Set

**Problem**: Selecting hyperparameters based on test performance.

```python
# ❌ WRONG: Using test set for selection
for params in param_grid:
    model.fit(X_train, y_train)
    test_score = model.score(X_test, y_test)
    if test_score > best:
        best_params = params  # Overfitting to test set!

# ✅ CORRECT: Separate validation
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.2
)
# Tune on validation, final eval on test
```

### Pitfall 3: Ignoring Computational Budget

**Problem**: Running exhaustive search without considering time.

```python
# Estimate before running
def estimate_time(n_combinations, time_per_eval, cv_folds):
    total_seconds = n_combinations * cv_folds * time_per_eval
    hours = total_seconds / 3600
    print(f"Estimated time: {hours:.1f} hours")
    if hours > 24:
        print("Consider using Random Search or Bayesian Optimization")
    return hours

# Grid with 4^5 = 1024 combinations, 30s per eval, 5 folds
estimate_time(1024, 30, 5)  # ~42 hours!
```

### Pitfall 4: Wrong Scale for Parameters

**Problem**: Using linear scale for parameters spanning orders of magnitude.

```python
# ❌ WRONG: Linear spacing for learning rate
learning_rates = [0.001, 0.01, 0.1, 0.2, 0.3]  # Oversamples high values

# ✅ CORRECT: Log-uniform distribution
from scipy.stats import loguniform
learning_rates = loguniform(1e-4, 1e-1)  # Equal probability per decade
```

## Recommendations by Scenario

### Scenario 1: Quick Baseline

**Goal**: Get reasonable performance fast

```python
# Random Search with modest iterations
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, loguniform

quick_search = RandomizedSearchCV(
    model,
    {
        'n_estimators': randint(50, 200),
        'max_depth': [5, 10, 20, None],
        'learning_rate': loguniform(0.01, 0.1),
    },
    n_iter=20,  # Quick exploration
    cv=3,       # Fewer folds
    n_jobs=-1
)
```

### Scenario 2: Production Model

**Goal**: Maximize performance, time less critical

```python
# Bayesian Optimization with many trials
import optuna

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'learning_rate': trial.suggest_float('lr', 1e-4, 1e-1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 30),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
    }
    scores = cross_val_score(
        GradientBoostingClassifier(**params), X, y, cv=5
    )
    return scores.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=200)  # Thorough search
```

### Scenario 3: Neural Network Tuning

**Goal**: Tune architecture and training hyperparameters

```python
def nn_objective(trial):
    # Architecture
    n_layers = trial.suggest_int('n_layers', 1, 4)
    hidden_sizes = [
        trial.suggest_int(f'units_{i}', 32, 256, log=True)
        for i in range(n_layers)
    ]
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    
    # Training
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch', [32, 64, 128])
    
    # Build and train (with pruning for early stopping)
    model = build_model(hidden_sizes, dropout)
    for epoch in range(100):
        train_epoch(model, lr, batch_size)
        val_loss = evaluate(model)
        
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return val_loss
```

### Scenario 4: Time Series

**Goal**: Respect temporal ordering

```python
from sklearn.model_selection import TimeSeriesSplit

# Time-aware CV
tscv = TimeSeriesSplit(n_splits=5)

search = RandomizedSearchCV(
    model,
    param_distributions,
    n_iter=50,
    cv=tscv,  # Temporal CV
    n_jobs=-1
)
```

## Summary Checklist

Before starting hyperparameter tuning:

- [ ] **Define budget**: Time and compute constraints
- [ ] **Select method**: Grid/Random/Bayesian based on constraints
- [ ] **Design search space**: Right distributions, reasonable ranges
- [ ] **Choose CV strategy**: Appropriate for data type
- [ ] **Set random seeds**: For reproducibility
- [ ] **Set up tracking**: Log experiments systematically

During tuning:

- [ ] **Monitor progress**: Check intermediate results
- [ ] **Watch for overfitting**: Compare train/val scores
- [ ] **Adjust ranges**: Refine based on initial results

After tuning:

- [ ] **Final evaluation**: On held-out test set
- [ ] **Document results**: Parameters, scores, insights
- [ ] **Validate robustness**: Multiple random seeds if time permits
