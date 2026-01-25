# Grid Search

## Overview

Grid Search is the most straightforward hyperparameter optimization technique that systematically evaluates every possible combination of hyperparameters within a predefined search space. By exhaustively exploring all configurations, Grid Search guarantees finding the optimal combination within the specified grid.

**Key Characteristics:**

- **Exhaustive Search**: Evaluates every combination in the parameter grid
- **Deterministic**: Same grid always produces the same results
- **Parallelizable**: Independent evaluations can run concurrently
- **Curse of Dimensionality**: Computational cost grows exponentially with parameters

## Mathematical Formulation

### Search Space Definition

Given $d$ hyperparameters $\theta_1, \theta_2, \ldots, \theta_d$, where each hyperparameter $\theta_i$ has a discrete set of candidate values $V_i = \{v_{i,1}, v_{i,2}, \ldots, v_{i,|V_i|}\}$, the parameter grid is the Cartesian product:

$$\mathcal{G} = V_1 \times V_2 \times \cdots \times V_d$$

The total number of configurations to evaluate is:

$$|\mathcal{G}| = \prod_{i=1}^{d} |V_i|$$

### Optimization Objective

Grid Search solves:

$$\boldsymbol{\theta}^* = \underset{\boldsymbol{\theta} \in \mathcal{G}}{\arg\max} \; \hat{J}(\boldsymbol{\theta})$$

where $\hat{J}(\boldsymbol{\theta})$ is typically the cross-validation score:

$$\hat{J}(\boldsymbol{\theta}) = \frac{1}{K} \sum_{k=1}^{K} \mathcal{L}\bigl(f_{\boldsymbol{\theta}}^{(-k)}, \mathcal{D}_k\bigr)$$

Here $K$ is the number of folds, $f_{\boldsymbol{\theta}}^{(-k)}$ is the model trained on all folds except $k$, $\mathcal{D}_k$ is the held-out fold, and $\mathcal{L}$ is the evaluation metric.

### Computational Complexity

For a grid with $N = |\mathcal{G}|$ configurations, $K$-fold cross-validation, and training complexity $\mathcal{O}(T)$ per fold:

$$\text{Total Complexity} = \mathcal{O}(N \cdot K \cdot T)$$

**Example**: With 5 hyperparameters, each having 4 values, using 5-fold CV:
- Total configurations: $4^5 = 1024$
- Total model fits: $1024 \times 5 = 5120$

## Algorithm

```
Algorithm: Grid Search with Cross-Validation
─────────────────────────────────────────────
Input: Parameter grid G, Dataset D, Number of folds K, Scoring metric L
Output: Best parameters θ*, Best score J*

1. Initialize: best_score ← -∞, best_params ← None
2. Generate all configurations C ← CartesianProduct(G)
3. For each configuration θ ∈ C:
   a. Split D into K folds: {(D_train^k, D_val^k)}_{k=1}^K
   b. scores ← []
   c. For k = 1 to K:
      i.   Train model f_θ on D_train^k
      ii.  Evaluate: score_k ← L(f_θ, D_val^k)
      iii. Append score_k to scores
   d. mean_score ← mean(scores)
   e. If mean_score > best_score:
      i.  best_score ← mean_score
      ii. best_params ← θ
4. Return best_params, best_score
```

## Implementation

### Basic Grid Search with scikit-learn

```python
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load and split data
X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
}

# Calculate total combinations
total = np.prod([len(v) for v in param_grid.values()])
print(f"Total configurations: {total}")  # 216 combinations

# Create Grid Search
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,                    # 5-fold cross-validation
    scoring='accuracy',      # Optimization metric
    n_jobs=-1,               # Use all CPU cores
    verbose=1,
    return_train_score=True  # Track training scores
)

# Execute search
grid_search.fit(X_train, y_train)

# Results
print(f"\nBest Parameters: {grid_search.best_params_}")
print(f"Best CV Score: {grid_search.best_score_:.4f}")
print(f"Test Score: {grid_search.score(X_test, y_test):.4f}")
```

### Analyzing Grid Search Results

```python
import pandas as pd
import matplotlib.pyplot as plt

# Convert results to DataFrame
results_df = pd.DataFrame(grid_search.cv_results_)

# View top configurations
top_configs = results_df.nsmallest(10, 'rank_test_score')[
    ['params', 'mean_test_score', 'std_test_score', 'rank_test_score']
]
print("\nTop 10 Configurations:")
print(top_configs.to_string())

# Analyze parameter importance
def plot_param_effect(results_df, param_name):
    """Plot effect of a single parameter on model performance."""
    param_col = f'param_{param_name}'
    
    grouped = results_df.groupby(param_col).agg({
        'mean_test_score': ['mean', 'std']
    }).reset_index()
    grouped.columns = [param_name, 'mean_score', 'std_score']
    
    plt.figure(figsize=(8, 5))
    plt.errorbar(
        range(len(grouped)), 
        grouped['mean_score'], 
        yerr=grouped['std_score'],
        marker='o', capsize=5, linewidth=2, markersize=8
    )
    plt.xticks(range(len(grouped)), grouped[param_name])
    plt.xlabel(param_name)
    plt.ylabel('Mean CV Score')
    plt.title(f'Effect of {param_name} on Performance')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Visualize effect of each parameter
for param in param_grid.keys():
    plot_param_effect(results_df, param)
```

### Multi-Metric Grid Search

```python
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score

# Define multiple scoring metrics
scoring = {
    'accuracy': 'accuracy',
    'f1_weighted': 'f1_weighted',
    'precision': make_scorer(precision_score, average='weighted'),
    'recall': make_scorer(recall_score, average='weighted')
}

# Grid search with multiple metrics
grid_search_multi = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring=scoring,
    refit='f1_weighted',  # Primary metric for best model selection
    n_jobs=-1
)

grid_search_multi.fit(X_train, y_train)

# Compare metrics for best configuration
print("\nBest Configuration Metrics:")
for metric in scoring.keys():
    idx = grid_search_multi.best_index_
    mean_score = grid_search_multi.cv_results_[f'mean_test_{metric}'][idx]
    std_score = grid_search_multi.cv_results_[f'std_test_{metric}'][idx]
    print(f"  {metric}: {mean_score:.4f} (±{std_score:.4f})")
```

## Nested Cross-Validation

Standard Grid Search with a single train-test split can produce optimistically biased estimates. Nested cross-validation provides unbiased performance estimates by separating hyperparameter selection from model evaluation.

### Mathematical Framework

**Outer Loop**: Estimates generalization performance
$$\hat{E}[\text{Error}] = \frac{1}{K_{\text{out}}} \sum_{k=1}^{K_{\text{out}}} \mathcal{L}\bigl(f_{\boldsymbol{\theta}_k^*}^{(-k)}, \mathcal{D}_k\bigr)$$

**Inner Loop**: Selects optimal hyperparameters for each outer fold
$$\boldsymbol{\theta}_k^* = \underset{\boldsymbol{\theta} \in \mathcal{G}}{\arg\max} \; \frac{1}{K_{\text{in}}} \sum_{j=1}^{K_{\text{in}}} \mathcal{L}\bigl(f_{\boldsymbol{\theta}}^{(-k,-j)}, \mathcal{D}_{k,j}\bigr)$$

### Implementation

```python
from sklearn.model_selection import cross_val_score, KFold

# Outer cross-validation
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Inner cross-validation (Grid Search)
inner_grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid={
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    },
    cv=3,  # Inner 3-fold CV
    scoring='accuracy',
    n_jobs=-1
)

# Nested CV scores
nested_scores = cross_val_score(
    inner_grid_search, X_train, y_train,
    cv=outer_cv, scoring='accuracy', n_jobs=-1
)

print(f"\nNested CV Results:")
print(f"  Scores: {nested_scores}")
print(f"  Mean: {nested_scores.mean():.4f}")
print(f"  Std:  {nested_scores.std():.4f}")
print(f"  95% CI: [{nested_scores.mean() - 1.96*nested_scores.std():.4f}, "
      f"{nested_scores.mean() + 1.96*nested_scores.std():.4f}]")

# Final model: fit inner CV on full training data
inner_grid_search.fit(X_train, y_train)
print(f"\nFinal Best Parameters: {inner_grid_search.best_params_}")
print(f"Test Set Score: {inner_grid_search.score(X_test, y_test):.4f}")
```

## PyTorch Integration

For neural network hyperparameter tuning, we can combine Grid Search with PyTorch:

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin
from itertools import product

class PyTorchClassifier(BaseEstimator, ClassifierMixin):
    """sklearn-compatible wrapper for PyTorch neural network."""
    
    def __init__(self, hidden_size=64, learning_rate=0.01, 
                 epochs=100, batch_size=32):
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _build_model(self, input_size, output_size):
        return nn.Sequential(
            nn.Linear(input_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, output_size)
        ).to(self.device)
    
    def fit(self, X, y):
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        self.classes_ = np.unique(y)
        self.model_ = self._build_model(X.shape[1], len(self.classes_))
        
        optimizer = torch.optim.Adam(
            self.model_.parameters(), lr=self.learning_rate
        )
        criterion = nn.CrossEntropyLoss()
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model_.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model_(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        return self
    
    def predict(self, X):
        self.model_.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            outputs = self.model_(X_tensor)
            _, predicted = torch.max(outputs, 1)
        return predicted.cpu().numpy()
    
    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)

# Grid search for PyTorch model
pytorch_param_grid = {
    'hidden_size': [32, 64, 128],
    'learning_rate': [0.001, 0.01, 0.1],
    'epochs': [50, 100],
    'batch_size': [16, 32]
}

grid_search_pytorch = GridSearchCV(
    estimator=PyTorchClassifier(),
    param_grid=pytorch_param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=1,  # PyTorch handles parallelism internally
    verbose=2
)

# Scale features for neural network
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

grid_search_pytorch.fit(X_train_scaled, y_train)
print(f"\nBest PyTorch Parameters: {grid_search_pytorch.best_params_}")
print(f"Best CV Score: {grid_search_pytorch.best_score_:.4f}")
```

## Coarse-to-Fine Search Strategy

For large parameter spaces, use a two-stage approach:

### Stage 1: Coarse Grid

```python
# Coarse grid with wide ranges
coarse_grid = {
    'n_estimators': [50, 200, 500],
    'max_depth': [5, 20, 50],
    'learning_rate': [0.01, 0.1, 1.0],  # For gradient boosting
}

coarse_search = GridSearchCV(
    estimator=GradientBoostingClassifier(random_state=42),
    param_grid=coarse_grid,
    cv=3,  # Fewer folds for speed
    n_jobs=-1
)
coarse_search.fit(X_train, y_train)
print(f"Coarse Best: {coarse_search.best_params_}")
```

### Stage 2: Fine Grid

```python
# Refine around best coarse parameters
best_coarse = coarse_search.best_params_

fine_grid = {
    'n_estimators': [
        best_coarse['n_estimators'] - 50,
        best_coarse['n_estimators'],
        best_coarse['n_estimators'] + 50
    ],
    'max_depth': [
        max(1, best_coarse['max_depth'] - 5),
        best_coarse['max_depth'],
        best_coarse['max_depth'] + 5
    ],
    'learning_rate': [
        best_coarse['learning_rate'] * 0.5,
        best_coarse['learning_rate'],
        best_coarse['learning_rate'] * 2.0
    ],
}

fine_search = GridSearchCV(
    estimator=GradientBoostingClassifier(random_state=42),
    param_grid=fine_grid,
    cv=5,  # More folds for final selection
    n_jobs=-1
)
fine_search.fit(X_train, y_train)
print(f"Fine Best: {fine_search.best_params_}")
print(f"Final CV Score: {fine_search.best_score_:.4f}")
```

## Advantages and Limitations

### Advantages

| Advantage | Description |
|-----------|-------------|
| **Exhaustive** | Guarantees optimal combination within the grid |
| **Reproducible** | Deterministic results given fixed random states |
| **Interpretable** | Easy to understand and analyze results |
| **Parallelizable** | Embarrassingly parallel across configurations |
| **Comprehensive** | Complete picture of parameter interactions |

### Limitations

| Limitation | Description |
|------------|-------------|
| **Exponential Growth** | Computational cost: $\mathcal{O}(|V_1| \cdot |V_2| \cdots |V_d|)$ |
| **Discrete Values** | Cannot directly search continuous spaces |
| **No Adaptivity** | Does not learn from previous evaluations |
| **Grid Sensitivity** | May miss optimal values between grid points |
| **Resource Intensive** | Requires substantial compute for large grids |

## When to Use Grid Search

**Good Use Cases:**

- Small parameter spaces (< 100 configurations)
- Need to understand parameter interactions
- Reproducibility is critical
- Have sufficient computational resources
- Final fine-tuning after coarse search

**Avoid When:**

- Large parameter spaces (> 1000 configurations)
- Many hyperparameters (> 5-6)
- Continuous parameter spaces
- Limited computational budget
- Parameters have vastly different importance

## Summary

Grid Search remains a fundamental hyperparameter optimization technique due to its simplicity, exhaustiveness, and interpretability. Key best practices include using nested cross-validation for unbiased estimates, employing coarse-to-fine strategies for efficiency, and analyzing results to understand parameter effects. For larger search spaces, consider Random Search or Bayesian Optimization as more efficient alternatives.
