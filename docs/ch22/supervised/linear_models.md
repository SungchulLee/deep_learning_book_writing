# Linear Models

Linear models predict outputs as linear combinations of input features. They form the foundation of supervised learning and provide interpretable, fast, and often surprisingly effective solutions for both regression and classification tasks.

---

## Linear Regression

### 1. Basic Usage

```python
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import numpy as np

# Generate sample data
X, y = make_regression(n_samples=1000, n_features=10, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
```

### 2. Model Parameters

```python
# Coefficients (weights)
print(f"Coefficients shape: {model.coef_.shape}")
print(f"Coefficients: {model.coef_}")

# Intercept (bias)
print(f"Intercept: {model.intercept_}")

# Feature importance by absolute coefficient value
importance = np.abs(model.coef_)
print(f"Feature importance: {importance}")
```

### 3. Model Evaluation

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# R² score (built-in)
print(f"Train R²: {model.score(X_train, y_train):.4f}")
print(f"Test R²: {model.score(X_test, y_test):.4f}")

# Additional metrics
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
```

### 4. Mathematical Foundation

**Model:** y = Xβ + ε

**Closed-form solution (Normal Equation):**
$$\hat{\beta} = (X^T X)^{-1} X^T y$$

```python
# Manual closed-form solution
X_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]  # Add bias term
beta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y_train

# Compare with sklearn
print(f"Manual intercept: {beta[0]:.4f}")
print(f"Sklearn intercept: {model.intercept_:.4f}")
print(f"Match: {np.allclose(beta[0], model.intercept_)}")
```

### 5. Options and Parameters

```python
# fit_intercept: whether to calculate intercept
model_no_intercept = LinearRegression(fit_intercept=False)
model_no_intercept.fit(X_train, y_train)
print(f"Intercept: {model_no_intercept.intercept_}")  # 0.0

# copy_X: copy data (default True for safety)
model = LinearRegression(copy_X=True)

# n_jobs: parallel computation for multi-target
model = LinearRegression(n_jobs=-1)  # Use all CPU cores
```

### 6. Multi-output Regression

```python
# Predict multiple targets simultaneously
from sklearn.datasets import make_regression

X, Y = make_regression(n_samples=500, n_features=5, n_targets=3, random_state=42)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, Y_train)

print(f"Coefficient shape: {model.coef_.shape}")  # (3, 5)
print(f"Intercept shape: {model.intercept_.shape}")  # (3,)
```

---

## Ridge Regression (L2 Regularization)

### 1. Basic Usage

```python
from sklearn.linear_model import Ridge

# Ridge adds L2 penalty: ||β||² 
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"Test R²: {model.score(X_test, y_test):.4f}")
```

### 2. Mathematical Foundation

**Objective:** minimize ||y - Xβ||² + α||β||²

**Closed-form solution:**
$$\hat{\beta} = (X^T X + \alpha I)^{-1} X^T y$$

```python
# Manual Ridge solution
alpha = 1.0
X_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]
I = np.eye(X_b.shape[1])
I[0, 0] = 0  # Don't regularize intercept
beta_ridge = np.linalg.inv(X_b.T @ X_b + alpha * I) @ X_b.T @ y_train
```

### 3. Effect of Alpha

```python
import matplotlib.pyplot as plt

alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
coefs = []

for alpha in alphas:
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    coefs.append(model.coef_)

coefs = np.array(coefs)

# Plot coefficient paths
plt.figure(figsize=(10, 6))
for i in range(coefs.shape[1]):
    plt.plot(alphas, coefs[:, i], label=f'Feature {i}')
plt.xscale('log')
plt.xlabel('Alpha (regularization strength)')
plt.ylabel('Coefficient value')
plt.title('Ridge Regression: Coefficient Shrinkage')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
```

### 4. Cross-Validation for Alpha Selection

```python
from sklearn.linear_model import RidgeCV

# Built-in cross-validation
alphas = np.logspace(-3, 3, 100)
model = RidgeCV(alphas=alphas, cv=5)
model.fit(X_train, y_train)

print(f"Best alpha: {model.alpha_:.4f}")
print(f"Test R²: {model.score(X_test, y_test):.4f}")
```

### 5. When to Use Ridge

- Multicollinearity (correlated features)
- More features than samples (p > n)
- Prevent overfitting
- When all features are expected to be relevant

---

## Lasso Regression (L1 Regularization)

### 1. Basic Usage

```python
from sklearn.linear_model import Lasso

# Lasso adds L1 penalty: |β|
model = Lasso(alpha=0.1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"Test R²: {model.score(X_test, y_test):.4f}")
```

### 2. Feature Selection (Sparsity)

```python
# Lasso sets some coefficients exactly to zero
print(f"Number of features: {len(model.coef_)}")
print(f"Non-zero coefficients: {np.sum(model.coef_ != 0)}")
print(f"Zero coefficients: {np.sum(model.coef_ == 0)}")

# Get selected features
selected_features = np.where(model.coef_ != 0)[0]
print(f"Selected feature indices: {selected_features}")
```

### 3. Mathematical Foundation

**Objective:** minimize ||y - Xβ||² + α||β||₁

Unlike Ridge, Lasso has no closed-form solution and uses coordinate descent.

```python
# Lasso objective function
def lasso_objective(beta, X, y, alpha):
    mse = np.mean((y - X @ beta) ** 2)
    l1_penalty = alpha * np.sum(np.abs(beta))
    return mse + l1_penalty
```

### 4. Effect of Alpha on Sparsity

```python
alphas = np.logspace(-4, 1, 50)
n_nonzero = []
scores = []

for alpha in alphas:
    model = Lasso(alpha=alpha, max_iter=10000)
    model.fit(X_train, y_train)
    n_nonzero.append(np.sum(model.coef_ != 0))
    scores.append(model.score(X_test, y_test))

# Plot sparsity vs alpha
fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.set_xlabel('Alpha')
ax1.set_ylabel('Non-zero coefficients', color='tab:blue')
ax1.semilogx(alphas, n_nonzero, 'b-')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.set_ylabel('R² Score', color='tab:red')
ax2.semilogx(alphas, scores, 'r-')
ax2.tick_params(axis='y', labelcolor='tab:red')

plt.title('Lasso: Sparsity vs Performance')
plt.tight_layout()
plt.show()
```

### 5. Cross-Validation for Alpha Selection

```python
from sklearn.linear_model import LassoCV

# Built-in cross-validation
model = LassoCV(cv=5, n_alphas=100)
model.fit(X_train, y_train)

print(f"Best alpha: {model.alpha_:.6f}")
print(f"Non-zero features: {np.sum(model.coef_ != 0)}")
print(f"Test R²: {model.score(X_test, y_test):.4f}")
```

### 6. When to Use Lasso

- Feature selection is needed
- Expect only a few features to be important
- Interpretability is crucial
- High-dimensional data with many irrelevant features

---

## ElasticNet (Combined L1 + L2)

### 1. Basic Usage

```python
from sklearn.linear_model import ElasticNet

# Combines L1 and L2 penalties
# l1_ratio: 0 = Ridge, 1 = Lasso
model = ElasticNet(alpha=0.1, l1_ratio=0.5)
model.fit(X_train, y_train)

print(f"Non-zero coefficients: {np.sum(model.coef_ != 0)}")
print(f"Test R²: {model.score(X_test, y_test):.4f}")
```

### 2. Mathematical Foundation

**Objective:** minimize ||y - Xβ||² + α·ρ||β||₁ + α·(1-ρ)||β||²/2

Where ρ is `l1_ratio`.

### 3. Cross-Validation

```python
from sklearn.linear_model import ElasticNetCV

# Search over both alpha and l1_ratio
l1_ratios = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99]
model = ElasticNetCV(l1_ratio=l1_ratios, cv=5, n_alphas=50)
model.fit(X_train, y_train)

print(f"Best alpha: {model.alpha_:.6f}")
print(f"Best l1_ratio: {model.l1_ratio_:.2f}")
print(f"Test R²: {model.score(X_test, y_test):.4f}")
```

### 4. When to Use ElasticNet

- Correlated features AND want feature selection
- Groups of correlated features (Lasso may select only one)
- Compromise between Ridge and Lasso

---

## Logistic Regression

### 1. Binary Classification

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Generate classification data
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5,
                           n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

print(f"Accuracy: {model.score(X_test, y_test):.4f}")
```

### 2. Model Parameters

```python
# Coefficients and intercept
print(f"Coefficients: {model.coef_}")  # Shape: (1, n_features)
print(f"Intercept: {model.intercept_}")  # Shape: (1,)

# Probability predictions
print(f"P(class=0): {y_proba[0, 0]:.4f}")
print(f"P(class=1): {y_proba[0, 1]:.4f}")
```

### 3. Mathematical Foundation

**Model:** P(y=1|x) = σ(xᵀβ) = 1 / (1 + exp(-xᵀβ))

**Loss:** Binary cross-entropy (log loss)

```python
# Manual sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Manual probability prediction
z = X_test @ model.coef_.T + model.intercept_
p_manual = sigmoid(z)
print(f"Match: {np.allclose(p_manual.ravel(), y_proba[:, 1])}")
```

### 4. Regularization

```python
# C = 1/alpha (smaller C = stronger regularization)
model_l2 = LogisticRegression(penalty='l2', C=1.0)  # Default
model_l1 = LogisticRegression(penalty='l1', C=0.1, solver='saga')
model_en = LogisticRegression(penalty='elasticnet', C=0.1, l1_ratio=0.5, solver='saga')
model_none = LogisticRegression(penalty=None)

# Compare sparsity
for name, m in [('L2', model_l2), ('L1', model_l1), ('ElasticNet', model_en)]:
    m.fit(X_train, y_train)
    n_nonzero = np.sum(m.coef_ != 0)
    print(f"{name}: {n_nonzero} non-zero, Accuracy: {m.score(X_test, y_test):.4f}")
```

### 5. Multiclass Classification

```python
# Generate multiclass data
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5,
                           n_classes=4, n_clusters_per_class=1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# One-vs-Rest (default)
model_ovr = LogisticRegression(multi_class='ovr')
model_ovr.fit(X_train, y_train)

# Multinomial (softmax)
model_multinomial = LogisticRegression(multi_class='multinomial')
model_multinomial.fit(X_train, y_train)

print(f"OvR Accuracy: {model_ovr.score(X_test, y_test):.4f}")
print(f"Multinomial Accuracy: {model_multinomial.score(X_test, y_test):.4f}")

# Coefficient shapes
print(f"OvR coef shape: {model_ovr.coef_.shape}")  # (n_classes, n_features)
print(f"Multinomial coef shape: {model_multinomial.coef_.shape}")
```

### 6. Solvers

```python
# Different solvers for different scenarios
solvers = {
    'lbfgs': {'penalty': 'l2'},           # Default, good for small datasets
    'liblinear': {'penalty': 'l1'},       # Good for small datasets, supports L1
    'saga': {'penalty': 'elasticnet', 'l1_ratio': 0.5},  # Supports all penalties
    'newton-cg': {'penalty': 'l2'},       # Good for multiclass
    'sag': {'penalty': 'l2'},             # Fast for large datasets
}

for solver, params in solvers.items():
    try:
        model = LogisticRegression(solver=solver, **params, max_iter=1000)
        model.fit(X_train, y_train)
        print(f"{solver}: Accuracy = {model.score(X_test, y_test):.4f}")
    except Exception as e:
        print(f"{solver}: {e}")
```

### 7. Class Imbalance

```python
# Balanced class weights
model = LogisticRegression(class_weight='balanced')
model.fit(X_train, y_train)

# Custom weights
model = LogisticRegression(class_weight={0: 1, 1: 10})
model.fit(X_train, y_train)
```

---

## SGD Classifier and Regressor

### 1. SGDRegressor

```python
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

# SGD requires feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit with SGD
model = SGDRegressor(loss='squared_error', penalty='l2', alpha=0.0001,
                     max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

print(f"Test R²: {model.score(X_test_scaled, y_test):.4f}")
```

### 2. SGDClassifier

```python
from sklearn.linear_model import SGDClassifier

# SGD for classification
model = SGDClassifier(loss='log_loss', penalty='l2', alpha=0.0001,
                      max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

print(f"Accuracy: {model.score(X_test_scaled, y_test):.4f}")
```

### 3. Loss Functions

```python
# SGDClassifier loss options
losses = ['hinge', 'log_loss', 'modified_huber', 'squared_hinge', 'perceptron']

for loss in losses:
    model = SGDClassifier(loss=loss, max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    print(f"{loss}: Accuracy = {model.score(X_test_scaled, y_test):.4f}")
```

### 4. Partial Fit (Online Learning)

```python
from sklearn.linear_model import SGDClassifier
import numpy as np

# Online learning with partial_fit
model = SGDClassifier(loss='log_loss', max_iter=1, warm_start=False)

# Simulate streaming data
n_batches = 10
batch_size = len(X_train_scaled) // n_batches
classes = np.unique(y_train)

for i in range(n_batches):
    start = i * batch_size
    end = start + batch_size
    X_batch = X_train_scaled[start:end]
    y_batch = y_train[start:end]
    
    model.partial_fit(X_batch, y_batch, classes=classes)
    acc = model.score(X_test_scaled, y_test)
    print(f"Batch {i+1}: Accuracy = {acc:.4f}")
```

---

## Comparison and Selection Guide

### 1. Regression Models

| Model | Regularization | Feature Selection | Use Case |
|-------|---------------|-------------------|----------|
| LinearRegression | None | No | Baseline, small datasets |
| Ridge | L2 | No | Multicollinearity, prevent overfitting |
| Lasso | L1 | Yes | Feature selection, sparse solutions |
| ElasticNet | L1 + L2 | Yes | Correlated features + selection |
| SGDRegressor | L1/L2/ElasticNet | Configurable | Large datasets, online learning |

### 2. Classification Models

| Model | Solver | Use Case |
|-------|--------|----------|
| LogisticRegression | lbfgs | Default, small-medium datasets |
| LogisticRegression | saga | L1/ElasticNet, large datasets |
| LogisticRegression | liblinear | Small datasets, L1 |
| SGDClassifier | SGD | Very large datasets, online learning |

### 3. Quick Selection

```python
def select_linear_model(n_samples, n_features, is_classification=False,
                        need_feature_selection=False, correlated_features=False):
    """Recommend a linear model based on data characteristics."""
    
    if is_classification:
        if n_samples > 100000:
            return "SGDClassifier(loss='log_loss')"
        elif need_feature_selection:
            return "LogisticRegression(penalty='l1', solver='saga')"
        else:
            return "LogisticRegression()"
    else:
        if n_samples > 100000:
            return "SGDRegressor()"
        elif need_feature_selection and correlated_features:
            return "ElasticNetCV()"
        elif need_feature_selection:
            return "LassoCV()"
        elif correlated_features or n_features > n_samples:
            return "RidgeCV()"
        else:
            return "LinearRegression()"

# Example usage
print(select_linear_model(10000, 100, is_classification=True, need_feature_selection=True))
```

---

## PyTorch Comparison

### 1. Linear Regression in PyTorch

```python
import torch
import torch.nn as nn

# Data preparation
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.FloatTensor(y_test).unsqueeze(1)

# Model
class LinearRegression(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return self.linear(x)

model_pt = LinearRegression(X_train.shape[1])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model_pt.parameters(), lr=0.01)

# Training
for epoch in range(1000):
    optimizer.zero_grad()
    outputs = model_pt(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()

# Evaluation
with torch.no_grad():
    y_pred_pt = model_pt(X_test_t)
    mse = criterion(y_pred_pt, y_test_t)
    print(f"PyTorch MSE: {mse.item():.4f}")
```

### 2. Logistic Regression in PyTorch

```python
# Binary classification
class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

model_pt = LogisticRegression(X_train.shape[1])
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model_pt.parameters(), lr=0.01)

# Training loop similar to above
```

### 3. L2 Regularization in PyTorch

```python
# Add weight decay (L2 regularization) to optimizer
optimizer = torch.optim.Adam(model_pt.parameters(), lr=0.01, weight_decay=0.01)
```

### 4. L1 Regularization in PyTorch

```python
# Manual L1 penalty in loss
def l1_penalty(model, lambda_l1=0.01):
    l1_norm = sum(p.abs().sum() for p in model.parameters())
    return lambda_l1 * l1_norm

# In training loop
loss = criterion(outputs, y_train_t) + l1_penalty(model_pt)
```

---

## Summary

Linear models remain fundamental in machine learning for their:

1. **Interpretability**: Coefficients directly show feature importance
2. **Speed**: Fast training and inference
3. **Reliability**: Well-understood statistical properties
4. **Baseline**: Essential comparison point for complex models

**Key takeaways:**
- Use **Ridge** when features are correlated
- Use **Lasso** for automatic feature selection
- Use **ElasticNet** for both benefits
- Use **LogisticRegression** as the first classifier to try
- Scale features for regularized models and SGD variants
- Use cross-validation to select regularization strength
