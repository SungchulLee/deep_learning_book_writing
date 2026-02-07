# Hybrid Pipelines

Hybrid pipelines combine scikit-learn preprocessing with PyTorch models, letting you use sklearn's mature infrastructure for data transformation, cross-validation, and evaluation while leveraging PyTorch for the learning component.

## Architecture

```
Raw Data → [sklearn Preprocessing] → [PyTorch Model] → [sklearn Evaluation]
             ColumnTransformer            Skorch/Custom      cross_val_score
             StandardScaler               NeuralNet          GridSearchCV
             OneHotEncoder                                   classification_report
```

## Basic Hybrid Pipeline

```python
import numpy as np
import torch.nn as nn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from skorch import NeuralNetClassifier

class Net(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 2),
        )
    
    def forward(self, X):
        return self.net(X)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('net', NeuralNetClassifier(
        module=Net,
        module__input_dim=20,
        max_epochs=50,
        lr=0.001,
        verbose=0,
    )),
])

# X is raw NumPy float64 — pipeline handles dtype conversion
scores = cross_val_score(pipe, X.astype(np.float32), y, cv=5, scoring='accuracy')
print(f"CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
```

## Mixed Data Types: ColumnTransformer + PyTorch

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_selector
import pandas as pd

# Preprocessing for mixed data
preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ]), make_column_selector(dtype_include='number')),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ]), make_column_selector(dtype_include='object')),
])

# Determine input dimension after preprocessing
preprocessor.fit(df_train)
input_dim = preprocessor.transform(df_train[:1]).shape[1]

# Full hybrid pipeline
hybrid_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('net', NeuralNetClassifier(
        module=Net,
        module__input_dim=input_dim,
        max_epochs=50,
        lr=0.001,
        verbose=0,
    )),
])

hybrid_pipe.fit(df_train, y_train)
y_pred = hybrid_pipe.predict(df_test)
```

## Grid Search Across the Full Pipeline

Search over both preprocessing and model hyperparameters simultaneously:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    # Preprocessing parameters
    'preprocessor__num__scaler': [StandardScaler(), RobustScaler()],
    # Model parameters
    'net__module__hidden_dim': [32, 64, 128],
    'net__lr': [0.0001, 0.001, 0.01],
    'net__max_epochs': [50, 100],
}

grid = GridSearchCV(hybrid_pipe, param_grid, cv=3, scoring='accuracy', n_jobs=1)
grid.fit(df_train, y_train)

print(f"Best params: {grid.best_params_}")
print(f"Best CV score: {grid.best_score_:.4f}")
```

## sklearn Feature Engineering → PyTorch

Combine powerful sklearn feature transformations with neural networks:

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA

feature_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
    ('select', SelectKBest(f_classif, k=50)),
    ('net', NeuralNetClassifier(
        module=Net,
        module__input_dim=50,  # must match k
        max_epochs=50,
        verbose=0,
    )),
])
```

## Comparison: sklearn Model vs. PyTorch Model

A common workflow is to compare classical models against neural networks using the same preprocessing:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

models = {
    'Logistic': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000)),
    ]),
    'GBM': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', GradientBoostingClassifier(n_estimators=100)),
    ]),
    'MLP': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', NeuralNetClassifier(
            module=Net, module__input_dim=20,
            max_epochs=50, lr=0.001, verbose=0,
        )),
    ]),
}

for name, model in models.items():
    scores = cross_val_score(model, X.astype(np.float32), y, cv=5, scoring='accuracy')
    print(f"{name:12s}: {scores.mean():.4f} ± {scores.std():.4f}")
```

## Walk-Forward Hybrid Pipeline

For finance, ensure temporal integrity:

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('net', NeuralNetRegressor(
        module=RegressionNet,
        module__input_dim=X.shape[1],
        max_epochs=100,
        lr=0.001,
        verbose=0,
    )),
])

scores = cross_val_score(
    pipe, X.astype(np.float32), y.astype(np.float32),
    cv=tscv, scoring='neg_mean_squared_error'
)
print(f"Walk-forward MSE: {-scores.mean():.6f}")
```

## Persistence

```python
import joblib

# Save the entire hybrid pipeline (sklearn preprocessing + PyTorch model)
joblib.dump(hybrid_pipe, 'hybrid_model.pkl')

# Load and deploy
loaded = joblib.load('hybrid_model.pkl')
predictions = loaded.predict(new_data)
```

## Summary

| Pattern | Use Case |
|---------|----------|
| `StandardScaler → Skorch` | Basic hybrid |
| `ColumnTransformer → Skorch` | Mixed data types |
| `FeatureEngineering → SelectKBest → Skorch` | Feature pipeline + NN |
| `sklearn vs. Skorch comparison` | Model selection |
| `Pipeline + TimeSeriesSplit` | Walk-forward finance |

The key principle: use sklearn for everything except the learning algorithm, and wrap the PyTorch component so it speaks the `fit`/`predict` interface.
