# Support Vector Machines

Support Vector Machines (SVMs) find the optimal hyperplane that maximizes the margin between classes. With kernel functions, SVMs can efficiently learn non-linear decision boundaries in high-dimensional feature spaces.

---

## Linear SVM Classification

### 1. Basic Usage

```python
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Generate data
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5,
                           n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SVMs require feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Linear SVM
model = SVC(kernel='linear', C=1.0)
model.fit(X_train_scaled, y_train)

print(f"Accuracy: {model.score(X_test_scaled, y_test):.4f}")
```

### 2. Support Vectors

```python
# Get support vectors
print(f"Number of support vectors: {len(model.support_vectors_)}")
print(f"Support vectors per class: {model.n_support_}")

# Indices of support vectors in training data
print(f"Support vector indices: {model.support_[:10]}")

# The actual support vectors
print(f"First support vector: {model.support_vectors_[0]}")
```

### 3. Decision Function and Margins

```python
# Decision function: distance to hyperplane
decision_values = model.decision_function(X_test_scaled)
print(f"Decision values (first 5): {decision_values[:5]}")

# Positive = class 1, negative = class 0
# Magnitude = confidence (distance from boundary)
```

### 4. Mathematical Foundation

**Primal Problem:**
$$\min_{w,b} \frac{1}{2}||w||^2 + C\sum_{i=1}^{n}\xi_i$$

Subject to: $y_i(w^T x_i + b) \geq 1 - \xi_i$, $\xi_i \geq 0$

```python
# Get weights for linear kernel
w = model.coef_[0]
b = model.intercept_[0]

print(f"Weights: {w}")
print(f"Bias: {b}")
```

---

## Regularization (C Parameter)

### 1. Hard vs Soft Margin

```python
# C controls margin softness
# Small C: Larger margin, more violations allowed (softer)
# Large C: Smaller margin, fewer violations (harder)

C_values = [0.001, 0.01, 0.1, 1, 10, 100]

for C in C_values:
    model = SVC(kernel='linear', C=C)
    model.fit(X_train_scaled, y_train)
    train_acc = model.score(X_train_scaled, y_train)
    test_acc = model.score(X_test_scaled, y_test)
    n_sv = len(model.support_vectors_)
    print(f"C={C:>6}: Train={train_acc:.4f}, Test={test_acc:.4f}, SVs={n_sv}")
```

---

## Kernel SVM

### 1. Kernel Types

```python
# Linear kernel: K(x, y) = x^T y
model_linear = SVC(kernel='linear', C=1.0)

# Polynomial kernel: K(x, y) = (γ x^T y + r)^d
model_poly = SVC(kernel='poly', degree=3, gamma='scale', coef0=1)

# RBF (Gaussian) kernel: K(x, y) = exp(-γ ||x-y||²)
model_rbf = SVC(kernel='rbf', gamma='scale', C=1.0)

# Sigmoid kernel: K(x, y) = tanh(γ x^T y + r)
model_sigmoid = SVC(kernel='sigmoid', gamma='scale', coef0=0)

kernels = [('Linear', model_linear), ('Polynomial', model_poly), 
           ('RBF', model_rbf), ('Sigmoid', model_sigmoid)]

for name, model in kernels:
    model.fit(X_train_scaled, y_train)
    print(f"{name}: Accuracy = {model.score(X_test_scaled, y_test):.4f}")
```

### 2. Gamma Parameter (RBF)

```python
# Gamma controls kernel width
# Small gamma: Larger influence radius (smoother boundary)
# Large gamma: Smaller influence radius (more complex boundary)

gamma_values = [0.01, 0.1, 1, 10, 100]

for gamma in gamma_values:
    model = SVC(kernel='rbf', C=1.0, gamma=gamma)
    model.fit(X_train_scaled, y_train)
    print(f"γ={gamma}: Accuracy = {model.score(X_test_scaled, y_test):.4f}")
```

### 3. Custom Kernel

```python
# Define custom kernel function
def custom_kernel(X, Y):
    """Custom kernel: K(x,y) = (x^T y + 1)^2"""
    return (X @ Y.T + 1) ** 2

model_custom = SVC(kernel=custom_kernel)
model_custom.fit(X_train_scaled, y_train)
print(f"Custom kernel accuracy: {model_custom.score(X_test_scaled, y_test):.4f}")
```

---

## SVM Regression (SVR)

### 1. Basic Usage

```python
from sklearn.svm import SVR
from sklearn.datasets import make_regression

# Generate regression data
X_reg, y_reg = make_regression(n_samples=500, n_features=5, noise=10, random_state=42)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Scale features
scaler_r = StandardScaler()
X_train_r_scaled = scaler_r.fit_transform(X_train_r)
X_test_r_scaled = scaler_r.transform(X_test_r)

# SVR with RBF kernel
model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
model.fit(X_train_r_scaled, y_train_r)

print(f"R² Score: {model.score(X_test_r_scaled, y_test_r):.4f}")
```

### 2. Epsilon-Insensitive Loss

```python
# Epsilon defines the tube width
# Points within epsilon distance from prediction have no loss

epsilon_values = [0.01, 0.1, 0.5, 1.0, 2.0]

for eps in epsilon_values:
    model = SVR(kernel='rbf', C=1.0, epsilon=eps)
    model.fit(X_train_r_scaled, y_train_r)
    n_sv = len(model.support_vectors_)
    r2 = model.score(X_test_r_scaled, y_test_r)
    print(f"ε={eps}: R²={r2:.4f}, SVs={n_sv}")
```

### 3. SVR Visualization

```python
import matplotlib.pyplot as plt

# 1D regression for visualization
X_1d = np.sort(np.random.randn(100, 1), axis=0)
y_1d = np.sin(X_1d).ravel() + np.random.randn(100) * 0.1

model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
model.fit(X_1d, y_1d)

X_plot = np.linspace(-3, 3, 100).reshape(-1, 1)
y_pred = model.predict(X_plot)

plt.figure(figsize=(10, 6))
plt.scatter(X_1d, y_1d, c='blue', label='Training data')
plt.plot(X_plot, y_pred, 'r-', linewidth=2, label='SVR prediction')
plt.fill_between(X_plot.ravel(), y_pred - 0.1, y_pred + 0.1, 
                  alpha=0.3, color='red', label=f'ε-tube (ε=0.1)')
plt.scatter(model.support_vectors_, model.predict(model.support_vectors_),
            s=100, facecolors='none', edgecolors='green', linewidths=2,
            label='Support vectors')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Support Vector Regression')
plt.show()
```

---

## Linear SVM (Faster)

### 1. LinearSVC

```python
from sklearn.svm import LinearSVC

# Much faster for large datasets
# Uses liblinear instead of libsvm
model = LinearSVC(C=1.0, max_iter=10000)
model.fit(X_train_scaled, y_train)

print(f"Accuracy: {model.score(X_test_scaled, y_test):.4f}")
```

### 2. Comparison

```python
import time

# Large dataset
X_large, y_large = make_classification(n_samples=10000, n_features=100, random_state=42)
scaler_large = StandardScaler()
X_large_scaled = scaler_large.fit_transform(X_large)
X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(
    X_large_scaled, y_large, test_size=0.2, random_state=42
)

# LinearSVC
start = time.time()
model_linear = LinearSVC(C=1.0, max_iter=10000)
model_linear.fit(X_train_l, y_train_l)
linear_time = time.time() - start

# SVC with linear kernel
start = time.time()
model_svc = SVC(kernel='linear', C=1.0)
model_svc.fit(X_train_l, y_train_l)
svc_time = time.time() - start

print(f"LinearSVC: {linear_time:.2f}s, Acc={model_linear.score(X_test_l, y_test_l):.4f}")
print(f"SVC(linear): {svc_time:.2f}s, Acc={model_svc.score(X_test_l, y_test_l):.4f}")
```

### 3. SGD-based SVM

```python
from sklearn.linear_model import SGDClassifier

# Even faster for very large datasets
# loss='hinge' gives SVM
model = SGDClassifier(loss='hinge', alpha=0.0001, max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

print(f"SGD SVM Accuracy: {model.score(X_test_scaled, y_test):.4f}")
```

---

## Probability Estimation

### 1. Enabling Probabilities

```python
# SVC doesn't provide probabilities by default
model = SVC(kernel='rbf', C=1.0, probability=True)
model.fit(X_train_scaled, y_train)

# Now can get probabilities
y_proba = model.predict_proba(X_test_scaled)
print(f"Class probabilities shape: {y_proba.shape}")
print(f"First sample probabilities: {y_proba[0]}")
```

### 2. Platt Scaling

```python
# Probabilities are obtained via Platt scaling (sigmoid calibration)
# This adds computational cost and may change decision boundary slightly

# Without probability
model_no_prob = SVC(kernel='rbf', C=1.0, probability=False)
model_no_prob.fit(X_train_scaled, y_train)

# With probability
model_with_prob = SVC(kernel='rbf', C=1.0, probability=True)
model_with_prob.fit(X_train_scaled, y_train)

print(f"Without probability: Accuracy = {model_no_prob.score(X_test_scaled, y_test):.4f}")
print(f"With probability: Accuracy = {model_with_prob.score(X_test_scaled, y_test):.4f}")
```

---

## Multiclass Classification

### 1. One-vs-One (Default)

```python
from sklearn.datasets import make_classification

# Multiclass data
X_multi, y_multi = make_classification(n_samples=500, n_features=10, n_informative=5,
                                        n_classes=4, n_clusters_per_class=1, random_state=42)
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_multi, y_multi, test_size=0.2, random_state=42
)

scaler_m = StandardScaler()
X_train_m_scaled = scaler_m.fit_transform(X_train_m)
X_test_m_scaled = scaler_m.transform(X_test_m)

# OvO: trains k(k-1)/2 classifiers
model_ovo = SVC(kernel='rbf', decision_function_shape='ovo')
model_ovo.fit(X_train_m_scaled, y_train_m)

print(f"OvO Accuracy: {model_ovo.score(X_test_m_scaled, y_test_m):.4f}")
```

### 2. One-vs-Rest

```python
from sklearn.multiclass import OneVsRestClassifier

# OvR: trains k classifiers
model_ovr = OneVsRestClassifier(SVC(kernel='rbf'))
model_ovr.fit(X_train_m_scaled, y_train_m)

print(f"OvR Accuracy: {model_ovr.score(X_test_m_scaled, y_test_m):.4f}")
```

---

## Hyperparameter Tuning

### 1. Grid Search

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.01, 0.1, 1],
    'kernel': ['rbf', 'poly']
}

grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")
print(f"Test accuracy: {grid_search.score(X_test_scaled, y_test):.4f}")
```

### 2. Randomized Search

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform, uniform

param_distributions = {
    'C': loguniform(0.01, 100),
    'gamma': loguniform(0.001, 10),
    'kernel': ['rbf', 'poly', 'sigmoid']
}

random_search = RandomizedSearchCV(
    SVC(), param_distributions, n_iter=50, cv=5, 
    scoring='accuracy', n_jobs=-1, random_state=42
)
random_search.fit(X_train_scaled, y_train)

print(f"Best parameters: {random_search.best_params_}")
print(f"Best CV score: {random_search.best_score_:.4f}")
```

---

## Class Imbalance

### 1. Class Weights

```python
# Balanced weights
model = SVC(kernel='rbf', class_weight='balanced')
model.fit(X_train_scaled, y_train)

# Custom weights
model = SVC(kernel='rbf', class_weight={0: 1, 1: 10})
model.fit(X_train_scaled, y_train)
```

### 2. Sample Weights

```python
# Give more weight to certain samples
sample_weights = np.ones(len(y_train))
sample_weights[y_train == 1] = 3

model = SVC(kernel='rbf')
model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
```

---

## Computational Considerations

### 1. Scaling with Data Size

```python
# SVM complexity: O(n² ~ n³) for training
# Not suitable for very large datasets

# Alternatives for large data:
# 1. LinearSVC: O(n)
# 2. SGDClassifier with hinge loss
# 3. Kernel approximation with linear model
```

### 2. Kernel Approximation

```python
from sklearn.kernel_approximation import RBFSampler, Nystroem
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline

# RBF Sampler (Random Fourier Features)
rbf_feature = RBFSampler(gamma=1, n_components=100, random_state=42)
model_rbf_approx = make_pipeline(rbf_feature, SGDClassifier(loss='hinge', max_iter=1000))
model_rbf_approx.fit(X_train_scaled, y_train)

print(f"RBF Approx Accuracy: {model_rbf_approx.score(X_test_scaled, y_test):.4f}")

# Nystroem approximation
nystroem = Nystroem(gamma=1, n_components=100, random_state=42)
model_nystroem = make_pipeline(nystroem, SGDClassifier(loss='hinge', max_iter=1000))
model_nystroem.fit(X_train_scaled, y_train)

print(f"Nystroem Accuracy: {model_nystroem.score(X_test_scaled, y_test):.4f}")
```

---

## PyTorch Comparison

### 1. Linear SVM in PyTorch

```python
import torch
import torch.nn as nn

# Hinge loss for SVM
class LinearSVM(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return self.linear(x)

def hinge_loss(outputs, targets):
    """Multi-class hinge loss"""
    targets = 2 * targets - 1  # Convert 0/1 to -1/1
    return torch.mean(torch.clamp(1 - targets * outputs.squeeze(), min=0))

# Training
X_train_t = torch.FloatTensor(X_train_scaled)
y_train_t = torch.FloatTensor(y_train)

model_pt = LinearSVM(X_train_scaled.shape[1])
optimizer = torch.optim.SGD(model_pt.parameters(), lr=0.01, weight_decay=0.01)

for epoch in range(1000):
    optimizer.zero_grad()
    outputs = model_pt(X_train_t)
    loss = hinge_loss(outputs, y_train_t)
    loss.backward()
    optimizer.step()

# Evaluation
with torch.no_grad():
    X_test_t = torch.FloatTensor(X_test_scaled)
    preds = (model_pt(X_test_t).squeeze() > 0).float()
    accuracy = (preds == torch.FloatTensor(y_test)).float().mean()
    print(f"PyTorch SVM Accuracy: {accuracy:.4f}")
```

### 2. Kernel SVM Approximation in PyTorch

```python
class RBFFeatures(nn.Module):
    """Random Fourier Features for RBF kernel approximation"""
    def __init__(self, input_dim, n_features, gamma=1.0):
        super().__init__()
        self.gamma = gamma
        self.n_features = n_features
        
        # Random weights for Fourier features
        self.W = nn.Parameter(
            torch.randn(input_dim, n_features) * np.sqrt(2 * gamma),
            requires_grad=False
        )
        self.b = nn.Parameter(
            torch.rand(n_features) * 2 * np.pi,
            requires_grad=False
        )
    
    def forward(self, x):
        proj = x @ self.W + self.b
        return np.sqrt(2 / self.n_features) * torch.cos(proj)

class ApproxKernelSVM(nn.Module):
    def __init__(self, input_dim, n_features=100, gamma=1.0):
        super().__init__()
        self.rbf_features = RBFFeatures(input_dim, n_features, gamma)
        self.linear = nn.Linear(n_features, 1)
    
    def forward(self, x):
        features = self.rbf_features(x)
        return self.linear(features)
```

---

## When to Use SVM

### Advantages

1. **Effective in high dimensions**: Works well when n_features > n_samples
2. **Memory efficient**: Only uses support vectors
3. **Versatile**: Different kernels for different problems
4. **Robust**: Max-margin principle provides good generalization

### Disadvantages

1. **Scaling**: O(n²) to O(n³) training complexity
2. **Feature scaling required**: Must standardize features
3. **Probability calibration**: Needs extra step (Platt scaling)
4. **Parameter sensitivity**: Requires careful tuning of C and gamma

### When to Use

- Medium-sized datasets (< 100k samples)
- High-dimensional data
- Clear margin of separation exists
- Non-linear relationships (with kernel)

### When NOT to Use

- Very large datasets (use SGD-based methods)
- Need probability estimates (other methods better calibrated)
- Many features with noise (trees may be better)

---

## Summary

| Aspect | SVC | LinearSVC | SGDClassifier |
|--------|-----|-----------|---------------|
| **Kernel** | Any | Linear only | Linear only |
| **Complexity** | O(n²~n³) | O(n) | O(n) |
| **Max samples** | ~10k | ~100k | Millions |
| **Probabilities** | Platt scaling | No | log_loss |

**Key hyperparameters:**
- `C`: Regularization (smaller = more regularization)
- `gamma`: RBF kernel width (larger = more complex)
- `kernel`: linear, rbf, poly, sigmoid
- `epsilon`: SVR tube width
