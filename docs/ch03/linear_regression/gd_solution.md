# Gradient Descent Solution

## Overview

While the normal equations give an exact closed-form solution, **gradient
descent** offers a scalable, iterative alternative that forms the foundation for
training neural networks.  This page derives the gradient of the MSE loss,
connects it to the negative log-likelihood, and progresses through four levels
of PyTorch implementation — from manual tensor operations to a production-ready
training pipeline.

---

## 1. MSE as Negative Log-Likelihood

### 1.1 Recap: Gaussian Likelihood

Under the probabilistic model
$y_i \mid \mathbf{x}_i \sim \mathcal{N}(\mathbf{w}^\top\mathbf{x}_i + b,\,\sigma^2)$,
the log-likelihood for $n$ observations is

$$
\ell(\mathbf{w}, b, \sigma^2)
= -\frac{n}{2}\ln(2\pi\sigma^2)
  - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2.
$$

For fixed $\sigma^2$, maximising $\ell$ with respect to $(\mathbf{w}, b)$ is
equivalent to minimising

$$
\text{MSE}
= \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2.
$$

!!! tip "Key Insight"
    MSE is the negative log-likelihood (up to an additive constant and
    positive scaling factor) under the Gaussian noise assumption.
    Gradient descent on MSE is therefore statistically justified maximum
    likelihood estimation.

### 1.2 When MSE Breaks Down

| Scenario | Problem | Alternative |
|----------|---------|-------------|
| Outliers | MSE amplifies large errors | Huber loss, MAE |
| Heavy-tailed residuals | Gaussian assumption invalid | Student-$t$ likelihood |
| Heteroscedasticity | Constant-$\sigma^2$ assumption fails | Weighted MSE, model $\sigma(\mathbf{x})$ explicitly |

```python
import torch
import torch.nn as nn

# Robust loss alternatives
criterion_mse   = nn.MSELoss()       # L2 — Gaussian NLL
criterion_mae   = nn.L1Loss()        # L1 — Laplace NLL
criterion_huber = nn.HuberLoss(delta=1.0)  # Smooth L2→L1 transition
```

---

## 2. Gradient Computation

### 2.1 Matrix Form

The MSE loss with the compact notation
$\boldsymbol{\theta} = (b, w_1, \ldots, w_p)^\top$ is

$$
J(\boldsymbol{\theta})
= \frac{1}{n}\|\mathbf{y} - \mathbf{X}\boldsymbol{\theta}\|^2.
$$

Its gradient is (derived in [Closed-Form Solution](closed_form.md)):

$$
\nabla_{\boldsymbol{\theta}} J
= \frac{2}{n}\,\mathbf{X}^\top\!
  \bigl(\mathbf{X}\boldsymbol{\theta} - \mathbf{y}\bigr)
= \frac{2}{n}\,\mathbf{X}^\top\!
  \bigl(\hat{\mathbf{y}} - \mathbf{y}\bigr).
$$

### 2.2 Component-Wise (Separate $\mathbf{w}$, $b$)

$$
\frac{\partial J}{\partial w_j}
= \frac{2}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)\,x_{ij},
\qquad
\frac{\partial J}{\partial b}
= \frac{2}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i).
$$

In vector form:

$$
\nabla_{\mathbf{w}} J
= \frac{2}{n}\,\mathbf{X}^\top(\hat{\mathbf{y}} - \mathbf{y}),
\qquad
\frac{\partial J}{\partial b}
= \frac{2}{n}\,\mathbf{1}^\top(\hat{\mathbf{y}} - \mathbf{y}).
$$

### 2.3 Optimal Learning Rate

For linear regression with MSE, the update is stable when

$$
\eta < \frac{2}{\lambda_{\max}(\mathbf{X}^\top\mathbf{X}/n)},
$$

where $\lambda_{\max}$ is the largest eigenvalue of the normalised Gram matrix.

```python
def compute_max_learning_rate(X: torch.Tensor) -> float:
    """Upper bound for stable gradient descent on MSE."""
    n = X.shape[0]
    eigenvalues = torch.linalg.eigvalsh(X.T @ X / n)
    return (2.0 / eigenvalues.max().item())
```

---

## 3. Gradient Descent Variants

### 3.1 Batch Gradient Descent

Uses the **entire** dataset for each update:

$$
\boldsymbol{\theta}^{(t+1)}
= \boldsymbol{\theta}^{(t)}
  - \eta\,\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}^{(t)}).
$$

```python
def batch_gradient_descent(
    X: torch.Tensor,
    y: torch.Tensor,
    lr: float = 0.01,
    n_epochs: int = 100,
) -> dict:
    """Batch GD with manual gradients (no autograd)."""
    n, d = X.shape
    w = torch.zeros(d, 1)
    b = torch.zeros(1)
    history = []

    for epoch in range(n_epochs):
        y_pred = X @ w + b
        loss = torch.mean((y - y_pred) ** 2)

        error = y_pred - y
        grad_w = (2.0 / n) * (X.T @ error)
        grad_b = (2.0 / n) * error.sum()

        w = w - lr * grad_w
        b = b - lr * grad_b

        history.append(loss.item())

    return {"w": w, "b": b, "history": history}
```

### 3.2 Stochastic Gradient Descent (SGD)

Uses a **single** sample per update — noisy but fast:

```python
def sgd(
    X: torch.Tensor,
    y: torch.Tensor,
    lr: float = 0.01,
    n_epochs: int = 50,
) -> dict:
    n, d = X.shape
    w = torch.zeros(d, 1)
    b = torch.zeros(1)
    history = []

    for epoch in range(n_epochs):
        perm = torch.randperm(n)
        for i in perm:
            xi = X[i : i + 1]
            yi = y[i : i + 1]
            y_pred = xi @ w + b
            error = y_pred - yi
            w = w - lr * 2 * (xi.T @ error)
            b = b - lr * 2 * error.squeeze()

        loss = torch.mean((y - (X @ w + b)) ** 2)
        history.append(loss.item())

    return {"w": w, "b": b, "history": history}
```

### 3.3 Mini-Batch Gradient Descent

The practical choice — balances gradient quality and computation:

```python
def mini_batch_gd(
    X: torch.Tensor,
    y: torch.Tensor,
    lr: float = 0.01,
    n_epochs: int = 100,
    batch_size: int = 32,
) -> dict:
    n, d = X.shape
    w = torch.zeros(d, 1)
    b = torch.zeros(1)
    history = []

    for epoch in range(n_epochs):
        perm = torch.randperm(n)
        X_s, y_s = X[perm], y[perm]

        for start in range(0, n, batch_size):
            X_b = X_s[start : start + batch_size]
            y_b = y_s[start : start + batch_size]
            B = X_b.shape[0]

            y_pred = X_b @ w + b
            error = y_pred - y_b
            w = w - lr * (2.0 / B) * (X_b.T @ error)
            b = b - lr * (2.0 / B) * error.sum()

        loss = torch.mean((y - (X @ w + b)) ** 2)
        history.append(loss.item())

    return {"w": w, "b": b, "history": history}
```

### 3.4 Comparison

| Variant | Gradient Cost | Update Noise | Convergence |
|---------|---------------|--------------|-------------|
| Batch | $O(np)$ per step | None | Smooth, linear rate $O(\kappa^t)$ |
| SGD | $O(p)$ per step | High | Noisy, $O(1/t)$ with decay |
| Mini-batch ($B$) | $O(Bp)$ per step | Moderate | Balanced |

---

## 4. Implementation Levels in PyTorch

### Level 1: Manual Gradients with DataLoader

```python
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(42)
n, p = 500, 3
X = torch.randn(n, p)
w_true = torch.tensor([2.0, -1.5, 0.5])
y = X @ w_true + 3.0 + 0.3 * torch.randn(n)

n_train = int(0.8 * n)
loader = DataLoader(
    TensorDataset(X[:n_train], y[:n_train].unsqueeze(1)),
    batch_size=32,
    shuffle=True,
)

w = torch.zeros(p, 1)
b = torch.zeros(1)
lr = 0.01

for epoch in range(100):
    for X_b, y_b in loader:
        y_pred = X_b @ w + b
        residual = y_pred - y_b
        B = len(y_b)
        w -= lr * (2.0 / B) * (X_b.T @ residual)
        b -= lr * (2.0 / B) * residual.sum()
```

### Level 2: Autograd (`requires_grad`)

```python
w = torch.zeros(p, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
lr = 0.01

for epoch in range(100):
    y_pred = X[:n_train] @ w + b
    loss = ((y_pred - y[:n_train]) ** 2).mean()

    loss.backward()                     # populates w.grad, b.grad

    with torch.no_grad():               # prevent tracking updates
        w -= lr * w.grad
        b -= lr * b.grad

    w.grad.zero_()                      # CRITICAL: PyTorch accumulates
    b.grad.zero_()
```

| Mechanism | Purpose |
|-----------|---------|
| `requires_grad=True` | Track operations for reverse-mode AD |
| `loss.backward()` | Compute $\partial\mathcal{L}/\partial\theta$ |
| `torch.no_grad()` | Disable gradient tracking for in-place updates |
| `.grad.zero_()` | Clear accumulated gradients |
| `.detach()` | Detach tensor from computation graph |

!!! warning "Gradient Accumulation"
    PyTorch **adds** new gradients to `.grad` rather than replacing them.
    Forgetting `zero_()` causes gradients to grow, leading to divergence.

### Level 3: `nn.Module` + Optimiser

```python
class LinearRegression(nn.Module):
    def __init__(self, in_features: int, out_features: int = 1):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)

model = LinearRegression(in_features=p)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(100):
    for X_b, y_b in loader:
        y_pred = model(X_b).unsqueeze(1)
        loss = criterion(y_pred, y_b)

        optimizer.zero_grad()           # 1. clear gradients
        loss.backward()                 # 2. compute gradients
        optimizer.step()                # 3. update parameters
```

This **four-line pattern** — forward, zero_grad, backward, step — is used
verbatim for logistic regression, CNNs, Transformers, and every other
architecture.

### Level 4: Production Pipeline

```python
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class LinearRegressionPipeline:
    """Full pipeline: scaling, train/val split, early stopping."""

    def __init__(self, input_dim, lr=0.01, batch_size=32, patience=10):
        self.model = nn.Linear(input_dim, 1)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=5, factor=0.5
        )
        self.batch_size = batch_size
        self.patience = patience
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

    def fit(self, X_np, y_np, n_epochs=200, val_size=0.2):
        # Scale
        X_train, X_val, y_train, y_val = train_test_split(
            X_np, y_np, test_size=val_size, random_state=42
        )
        Xs = torch.FloatTensor(self.scaler_X.fit_transform(X_train))
        ys = torch.FloatTensor(
            self.scaler_y.fit_transform(y_train.reshape(-1, 1))
        )
        Xv = torch.FloatTensor(self.scaler_X.transform(X_val))
        yv = torch.FloatTensor(
            self.scaler_y.transform(y_val.reshape(-1, 1))
        )

        loader = DataLoader(
            TensorDataset(Xs, ys), batch_size=self.batch_size, shuffle=True
        )

        best_val, wait, best_state = float("inf"), 0, None

        for epoch in range(n_epochs):
            # --- Train ---
            self.model.train()
            for xb, yb in loader:
                loss = self.criterion(self.model(xb), yb)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # --- Validate ---
            self.model.eval()
            with torch.no_grad():
                val_loss = self.criterion(self.model(Xv), yv).item()
            self.scheduler.step(val_loss)

            if val_loss < best_val:
                best_val, wait = val_loss, 0
                best_state = self.model.state_dict().copy()
            else:
                wait += 1
                if wait >= self.patience:
                    break

        if best_state:
            self.model.load_state_dict(best_state)

    def predict(self, X_np):
        self.model.eval()
        Xs = torch.FloatTensor(self.scaler_X.transform(X_np))
        with torch.no_grad():
            y_scaled = self.model(Xs).numpy()
        return self.scaler_y.inverse_transform(y_scaled)
```

---

## 5. Convergence Analysis

### 5.1 Linear Convergence for Batch GD

For a convex quadratic (like MSE in linear regression), batch gradient descent
converges at a **linear rate**:

$$
J(\boldsymbol{\theta}^{(t)}) - J(\boldsymbol{\theta}^*)
\leq \left(\frac{\kappa - 1}{\kappa + 1}\right)^{2t}
     \bigl(J(\boldsymbol{\theta}^{(0)}) - J(\boldsymbol{\theta}^*)\bigr),
$$

where $\kappa = \lambda_{\max} / \lambda_{\min}$ is the **condition number** of
$\mathbf{X}^\top\mathbf{X} / n$.  Large $\kappa$ (ill-conditioning) slows
convergence.

### 5.2 Effect of Learning Rate

| Regime | Behaviour |
|--------|-----------|
| $\eta$ too small | Slow convergence, many epochs needed |
| $\eta$ optimal | Fast, smooth convergence |
| $\eta$ too large | Oscillation around minimum |
| $\eta > 2/\lambda_{\max}$ | Divergence |

### 5.3 Practical Guidelines

| Hyperparameter | Recommendation |
|----------------|----------------|
| Batch size | 32–128 (powers of 2 for GPU efficiency) |
| Learning rate | Start with 0.01 or 0.1; use `ReduceLROnPlateau` |
| Epochs | 100–500; use early stopping on validation loss |
| Optimiser | Adam as default; SGD with momentum for fine control |

---

## 6. Model Persistence

```python
# Save (state_dict is the recommended format)
torch.save(model.state_dict(), "linear_regression.pt")

# Load
model2 = LinearRegression(in_features=p)
model2.load_state_dict(
    torch.load("linear_regression.pt", map_location="cpu")
)
model2.eval()
```

!!! warning "`torch.save(model)` vs `torch.save(model.state_dict())`"
    Saving the entire model uses pickle and embeds the class definition,
    which breaks when code is refactored.  Always save and load the
    `state_dict()`.

---

## 7. When to Use Each Method

| Method | Best For | Complexity |
|--------|----------|------------|
| Normal equations | $p < 10{,}000$, exact solution | $O(np^2 + p^3)$ |
| Batch GD | Medium datasets | $O(knp)$ |
| Mini-batch GD | Large datasets, GPU training | $O(kBp)$ per step |
| SGD | Streaming data, very large $n$ | $O(kp)$ per step |

---

## Summary

### The Canonical Training Loop

```
┌──────────────────────────────────────────────────────┐
│  1. Define model       model = nn.Linear(p, 1)       │
│  2. Define loss        criterion = nn.MSELoss()      │
│  3. Define optimiser   optim.SGD(model.parameters()) │
│  4. Training loop:                                   │
│       for batch in loader:                           │
│           y_pred = model(x)                          │
│           loss = criterion(y_pred, y)                │
│           optimizer.zero_grad()                      │
│           loss.backward()                            │
│           optimizer.step()                           │
│  5. Evaluate           model.eval(); torch.no_grad() │
│  6. Save               torch.save(state_dict)        │
└──────────────────────────────────────────────────────┘
```

### Key Takeaways

1. **MSE = NLL** (up to constants) under Gaussian noise — gradient descent on
   MSE is statistically justified.
2. **Mini-batch GD** is the practical choice for most applications.
3. **Learning rate** is the most important hyperparameter; bound it by
   $2/\lambda_{\max}$.
4. **Autograd** eliminates manual gradient derivation — the math is still
   valuable for understanding, but PyTorch handles the computation.
5. The **four-line pattern** (forward → zero_grad → backward → step) is
   universal across all PyTorch architectures.

---

## References

1. Bottou, L. (2010). "Large-Scale Machine Learning with Stochastic Gradient
   Descent."
2. Ruder, S. (2016). "An Overview of Gradient Descent Optimization Algorithms."
3. Goodfellow, I., Bengio, Y. & Courville, A. (2016). *Deep Learning*, Ch. 8.
4. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*, Ch. 3.
