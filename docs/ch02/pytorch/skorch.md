# Skorch

Skorch wraps PyTorch modules as scikit-learn estimators, giving you `fit`/`predict`/`score` compatibility with pipelines, cross-validation, and grid search—while retaining full control over the PyTorch training loop.

## Installation

```bash
pip install skorch
```

## NeuralNetClassifier

```python
import torch
import torch.nn as nn
from skorch import NeuralNetClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np

# Define a PyTorch module
class MLP(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=64, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, X):
        return self.net(X)

# Generate data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X = X.astype(np.float32)
y = y.astype(np.int64)

# Wrap as sklearn estimator
net = NeuralNetClassifier(
    module=MLP,
    module__input_dim=20,
    module__hidden_dim=64,
    module__output_dim=2,
    max_epochs=50,
    lr=0.001,
    batch_size=32,
    optimizer=torch.optim.Adam,
    criterion=nn.CrossEntropyLoss,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    verbose=0,
)

# Use exactly like any sklearn estimator
scores = cross_val_score(net, X, y, cv=5, scoring='accuracy')
print(f"CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
```

## NeuralNetRegressor

```python
from skorch import NeuralNetRegressor
from sklearn.datasets import make_regression

class RegressionMLP(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, X):
        return self.net(X)

X_reg, y_reg = make_regression(n_samples=1000, n_features=10, noise=10, random_state=42)
X_reg = X_reg.astype(np.float32)
y_reg = y_reg.astype(np.float32).reshape(-1, 1)

net_reg = NeuralNetRegressor(
    module=RegressionMLP,
    module__input_dim=10,
    max_epochs=100,
    lr=0.001,
    batch_size=64,
    criterion=nn.MSELoss,
    verbose=0,
)

scores = cross_val_score(net_reg, X_reg, y_reg, cv=5, scoring='neg_mean_squared_error')
print(f"CV MSE: {-scores.mean():.4f}")
```

## Pipeline Integration

```python
pipe = make_pipeline(
    StandardScaler(),
    NeuralNetClassifier(
        module=MLP,
        module__input_dim=20,
        max_epochs=50,
        lr=0.001,
        verbose=0,
    )
)

# No data leakage: scaler fits on training folds only
scores = cross_val_score(pipe, X, y, cv=5, scoring='accuracy')
```

## Grid Search Over Network Architecture

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'neuralnetclassifier__module__hidden_dim': [32, 64, 128],
    'neuralnetclassifier__lr': [0.0001, 0.001, 0.01],
    'neuralnetclassifier__max_epochs': [50, 100],
    'neuralnetclassifier__batch_size': [32, 64],
}

grid = GridSearchCV(pipe, param_grid, cv=3, scoring='accuracy', refit=True)
grid.fit(X, y)
print(f"Best: {grid.best_params_}")
print(f"Best CV score: {grid.best_score_:.4f}")
```

## Callbacks

Skorch supports training callbacks for early stopping, learning rate scheduling, and checkpointing:

```python
from skorch.callbacks import EarlyStopping, LRScheduler

net = NeuralNetClassifier(
    module=MLP,
    max_epochs=200,
    lr=0.001,
    callbacks=[
        EarlyStopping(patience=10, monitor='valid_loss'),
        LRScheduler(policy='StepLR', step_size=30, gamma=0.1),
    ],
    verbose=1,
)
```

## Quantitative Finance: Factor Model with Skorch

```python
class FactorNet(nn.Module):
    """Non-linear factor model for cross-sectional return prediction."""
    
    def __init__(self, n_factors=50, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_factors, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
    
    def forward(self, X):
        return self.net(X)

factor_model = NeuralNetRegressor(
    module=FactorNet,
    module__n_factors=50,
    module__hidden_dim=128,
    max_epochs=100,
    lr=0.001,
    criterion=nn.MSELoss,
    callbacks=[EarlyStopping(patience=10)],
    verbose=0,
)

# Use with walk-forward CV
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

pipe = make_pipeline(StandardScaler(), factor_model)
tscv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(pipe, X_factors, y_returns, cv=tscv,
                         scoring='neg_mean_squared_error')
```

## Summary

| Feature | Skorch Approach |
|---------|----------------|
| Cross-validation | `cross_val_score(net, X, y)` |
| Grid search | `GridSearchCV` with `module__param` syntax |
| Pipeline | `make_pipeline(scaler, net)` |
| Early stopping | `EarlyStopping` callback |
| LR scheduling | `LRScheduler` callback |
| GPU support | `device='cuda'` |
