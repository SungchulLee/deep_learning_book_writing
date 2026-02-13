# Bayesian Optimization

Bayesian optimization uses a probabilistic surrogate model to guide hyperparameter search, choosing the next point to evaluate based on an acquisition function that balances exploration and exploitation. It typically finds good configurations in far fewer evaluations than grid or random search.

## Core Idea

1. **Surrogate model**: Fit a probabilistic model $p(f \mid \mathcal{D}_{1:t})$ to the observed (hyperparameters, score) pairs
2. **Acquisition function**: Select the next point $\mathbf{x}_{t+1}$ that maximises expected improvement (or similar criterion)
3. **Evaluate**: Train the actual model at $\mathbf{x}_{t+1}$, observe the score, update the surrogate
4. **Repeat** until budget is exhausted

$$\mathbf{x}_{t+1} = \arg\max_{\mathbf{x}} \; \alpha(\mathbf{x} \mid \mathcal{D}_{1:t})$$

## Acquisition Functions

### Expected Improvement (EI)

$$\text{EI}(\mathbf{x}) = \mathbb{E}\left[\max(f(\mathbf{x}) - f^+, 0)\right]$$

where $f^+ = \max_{i=1}^{t} f(\mathbf{x}_i)$ is the best observed value.

Under a Gaussian surrogate with mean $\mu(\mathbf{x})$ and variance $\sigma^2(\mathbf{x})$:

$$\text{EI}(\mathbf{x}) = (\mu(\mathbf{x}) - f^+ - \xi) \, \Phi(Z) + \sigma(\mathbf{x}) \, \phi(Z)$$

where $Z = (\mu(\mathbf{x}) - f^+ - \xi)/\sigma(\mathbf{x})$ and $\xi$ controls exploration.

| Function | Formula | Trade-off |
|----------|---------|-----------|
| Expected Improvement (EI) | $\mathbb{E}[\max(f - f^+, 0)]$ | Balanced |
| Lower Confidence Bound (LCB) | $\mu - \kappa\sigma$ | $\kappa$ controls exploration |
| Probability of Improvement (PI) | $P(f(\mathbf{x}) > f^+)$ | More exploitative |

## Scikit-Optimize (skopt)

```python
# pip install scikit-optimize
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

search_spaces = {
    'n_estimators': Integer(50, 500),
    'max_depth': Integer(2, 10),
    'learning_rate': Real(1e-3, 3e-1, prior='log-uniform'),
    'subsample': Real(0.5, 1.0),
    'min_samples_leaf': Integer(1, 20),
}

bayes_search = BayesSearchCV(
    GradientBoostingClassifier(random_state=42),
    search_spaces,
    n_iter=50,
    cv=5,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1,
)

bayes_search.fit(X_train, y_train)
print(f"Best params: {bayes_search.best_params_}")
print(f"Best CV score: {bayes_search.best_score_:.4f}")
```

## Optuna

Optuna is a modern framework with pruning, multi-objective support, and a define-by-run API:

```python
# pip install optuna
import optuna
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 3e-1, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
    }
    
    model = GradientBoostingClassifier(**params, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    return scores.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50, show_progress_bar=True)

print(f"Best params: {study.best_params}")
print(f"Best CV score: {study.best_value:.4f}")
```

### Pruning (Early Stopping of Bad Trials)

```python
from sklearn.model_selection import StratifiedKFold

def objective_with_pruning(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 3e-1, log=True),
    }
    
    model = GradientBoostingClassifier(**params, random_state=42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        model.fit(X_train[train_idx], y_train[train_idx])
        score = model.score(X_train[val_idx], y_train[val_idx])
        fold_scores.append(score)
        
        trial.report(score, fold)
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return sum(fold_scores) / len(fold_scores)

study = optuna.create_study(
    direction='maximize',
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=2),
)
study.optimize(objective_with_pruning, n_trials=100)
```

### Visualisation

```python
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
)

fig1 = plot_optimization_history(study)
fig2 = plot_param_importances(study)
fig3 = plot_parallel_coordinate(study)
```

## Comparison: Grid vs. Random vs. Bayesian

| Aspect | Grid | Random | Bayesian |
|--------|------|--------|----------|
| **Strategy** | Exhaustive | Random sampling | Informed by surrogate |
| **Budget scaling** | $\prod |V_p|$ | Fixed $n$ | Fixed $n$ |
| **Continuous params** | Discretised | Native | Native |
| **Uses prior results** | No | No | Yes |
| **Parallelisation** | Trivial | Trivial | Limited |
| **Typical budget** | $< 100$ | 50–500 | 20–200 |
| **Best for** | Small, discrete | Medium spaces | Expensive evaluations |

## Practical Guidelines

1. **Start with random search** (50–100 iterations) to identify the general region
2. **Switch to Bayesian** if each evaluation is expensive and the budget is tight
3. **Use pruning** to abandon unpromising trials early
4. **Log-scale** for learning rates, regularisation, and parameters spanning orders of magnitude
5. **Fix irrelevant parameters** to reduce search dimensionality

## Quantitative Finance: Walk-Forward Bayesian Optimisation

```python
import optuna
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 2, 8),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
    }
    
    model = GradientBoostingRegressor(**params, random_state=42)
    tscv = TimeSeriesSplit(n_splits=5)
    scores = cross_val_score(
        model, X_train, y_train,
        cv=tscv, scoring='neg_mean_squared_error'
    )
    return scores.mean()  # negative MSE, so maximise

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print(f"Best walk-forward MSE: {-study.best_value:.6f}")
print(f"Best params: {study.best_params}")
```

## References

1. Snoek, J., Larochelle, H., & Adams, R. P. (2012). "Practical Bayesian Optimization of Machine Learning Algorithms." *NeurIPS*.
2. Bergstra, J., Yamins, D., & Cox, D. D. (2013). "Making a Science of Model Search." *ICML*.
3. Akiba, T., et al. (2019). "Optuna: A Next-generation Hyperparameter Optimization Framework." *KDD*.
