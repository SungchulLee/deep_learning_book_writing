# Custom Estimators

When Skorch's abstractions don't fit your needs—custom training loops, non-standard loss functions, or specialised inference—you can implement the scikit-learn estimator interface directly for a PyTorch model.

## Minimal Custom Classifier

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


class PyTorchClassifier(BaseEstimator, ClassifierMixin):
    """Scikit-learn compatible wrapper for a PyTorch classifier.
    
    Parameters
    ----------
    hidden_dim : int
        Number of hidden units.
    lr : float
        Learning rate.
    epochs : int
        Number of training epochs.
    batch_size : int
        Mini-batch size.
    """
    
    def __init__(self, hidden_dim=64, lr=0.001, epochs=100, batch_size=32):
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
    
    def _build_model(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, output_dim),
        )
    
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.n_features_in_ = X.shape[1]
        n_classes = len(self.classes_)
        
        # Build model
        self.model_ = self._build_model(X.shape[1], n_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr)
        
        # Convert to tensors
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Training loop
        self.model_.train()
        self.history_ = []
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(self.model_(xb), yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
            self.history_.append(epoch_loss / len(X))
        
        return self
    
    def predict(self, X):
        check_is_fitted(self, ['model_'])
        X = check_array(X)
        
        self.model_.eval()
        X_t = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            logits = self.model_(X_t)
            preds = logits.argmax(dim=1).numpy()
        return preds
    
    def predict_proba(self, X):
        check_is_fitted(self, ['model_'])
        X = check_array(X)
        
        self.model_.eval()
        X_t = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            logits = self.model_(X_t)
            proba = torch.softmax(logits, dim=1).numpy()
        return proba
    
    # score() inherited from ClassifierMixin → accuracy
```

### Usage

```python
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

clf = PyTorchClassifier(hidden_dim=64, lr=0.001, epochs=50)

# Cross-validation
scores = cross_val_score(clf, X.astype(np.float32), y, cv=5, scoring='accuracy')
print(f"CV Accuracy: {scores.mean():.4f}")

# Grid search
pipe = make_pipeline(StandardScaler(), PyTorchClassifier())
param_grid = {
    'pytorchclassifier__hidden_dim': [32, 64, 128],
    'pytorchclassifier__lr': [0.0001, 0.001, 0.01],
    'pytorchclassifier__epochs': [50, 100],
}
grid = GridSearchCV(pipe, param_grid, cv=3, scoring='accuracy')
grid.fit(X, y)
```

## Custom Regressor

```python
from sklearn.base import RegressorMixin

class PyTorchRegressor(BaseEstimator, RegressorMixin):
    
    def __init__(self, hidden_dim=64, lr=0.001, epochs=100, batch_size=32):
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
    
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]
        
        self.model_ = nn.Sequential(
            nn.Linear(X.shape[1], self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr)
        
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model_.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                criterion(self.model_(xb), yb).backward()
                optimizer.step()
        return self
    
    def predict(self, X):
        check_is_fitted(self, ['model_'])
        X = check_array(X)
        self.model_.eval()
        with torch.no_grad():
            preds = self.model_(torch.tensor(X, dtype=torch.float32))
        return preds.squeeze().numpy()
    
    # score() inherited from RegressorMixin → R²
```

## Adding Early Stopping

```python
class PyTorchClassifierES(PyTorchClassifier):
    """Extended with validation-based early stopping."""
    
    def __init__(self, hidden_dim=64, lr=0.001, epochs=200,
                 batch_size=32, patience=10, val_fraction=0.1):
        super().__init__(hidden_dim=hidden_dim, lr=lr,
                        epochs=epochs, batch_size=batch_size)
        self.patience = patience
        self.val_fraction = val_fraction
    
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.n_features_in_ = X.shape[1]
        
        # Split off validation set
        n_val = int(len(X) * self.val_fraction)
        X_train, X_val = X[n_val:], X[:n_val]
        y_train, y_val = y[n_val:], y[:n_val]
        
        self.model_ = self._build_model(X.shape[1], len(self.classes_))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr)
        
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.long)
        
        dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long)
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        
        for epoch in range(self.epochs):
            self.model_.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                criterion(self.model_(xb), yb).backward()
                optimizer.step()
            
            # Validation
            self.model_.eval()
            with torch.no_grad():
                val_loss = criterion(self.model_(X_val_t), y_val_t).item()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.clone() for k, v in self.model_.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break
        
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        
        return self
```

## Key Implementation Rules

1. **`__init__` stores only parameters** — no computation, no model building
2. **`fit` returns `self`** — enables method chaining
3. **Learned attributes end with `_`**: `model_`, `classes_`, `history_`
4. **Use `check_X_y` / `check_array`** — validates input
5. **Use `check_is_fitted`** — prevents calling `predict` before `fit`
6. **`clone` compatibility** — `get_params` / `set_params` work automatically via `BaseEstimator`

## Summary

| Approach | When to Use |
|----------|-------------|
| **Skorch** | Standard training loops, quick prototyping |
| **Custom estimator** | Non-standard losses, custom training logic, multi-output |
| **Both** | Full sklearn integration: pipelines, CV, grid search |
