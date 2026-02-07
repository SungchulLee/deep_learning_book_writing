# Polynomial Features

## Overview

Linear regression is linear in **parameters**, not necessarily in features.  By
constructing polynomial (or other nonlinear) transformations of the original
inputs, we can fit curved relationships while keeping the optimisation machinery
of ordinary least squares.  This page develops the theory, connects it to the
bias–variance trade-off, and demonstrates cross-validated model selection.

---

## 1. Feature Map Idea

### 1.1 The Limitation of a Linear Basis

With raw features $x \in \mathbb{R}$, the model $\hat{y} = w_1 x + b$ can only
represent a straight line.  If the true relationship is nonlinear — say,
quadratic — the best linear fit under-fits systematically.

### 1.2 Polynomial Expansion

Define the degree-$d$ **feature map** $\phi : \mathbb{R} \to \mathbb{R}^{d+1}$:

$$
\phi(x) = [1,\; x,\; x^2,\; \ldots,\; x^d]^\top.
$$

The model becomes

$$
\hat{y}
= \boldsymbol{\theta}^\top \phi(x)
= \theta_0 + \theta_1 x + \theta_2 x^2 + \cdots + \theta_d x^d.
$$

This is still **linear in $\boldsymbol{\theta}$**, so the normal equations and
gradient descent apply without modification — only the design matrix changes.

### 1.3 Multivariate Extension

For $p$ input features and degree $d$, the polynomial feature map includes all
monomials up to total degree $d$:

$$
\phi(\mathbf{x}) = \{x_1^{a_1} x_2^{a_2} \cdots x_p^{a_p}
                    : a_1 + a_2 + \cdots + a_p \leq d\}.
$$

The number of features after expansion is

$$
\binom{p + d}{d} = \frac{(p + d)!}{p!\, d!}.
$$

| Original features $p$ | Degree $d$ | Expanded features |
|:---------------------:|:----------:|:-----------------:|
| 1 | 2 | 3 |
| 1 | 5 | 6 |
| 2 | 2 | 6 |
| 2 | 3 | 10 |
| 5 | 3 | 56 |
| 10 | 3 | 286 |

!!! warning "Curse of Dimensionality"
    The number of polynomial features grows combinatorially.  For moderate $p$
    and high $d$, the expanded feature space can become very large, increasing
    both computation and the risk of overfitting.

---

## 2. NumPy Implementation

### 2.1 Manual Construction (Univariate)

```python
import numpy as np


def polynomial_features_1d(x: np.ndarray, degree: int) -> np.ndarray:
    """Build polynomial design matrix for a single feature.

    Parameters
    ----------
    x : 1-D array of shape (n,)
    degree : polynomial degree d

    Returns
    -------
    X : ndarray of shape (n, d+1) — columns are [1, x, x², …, x^d]
    """
    return np.column_stack([x ** k for k in range(degree + 1)])
```

### 2.2 Using scikit-learn

```python
from sklearn.preprocessing import PolynomialFeatures

# degree=3, include_bias=True adds the constant column
poly = PolynomialFeatures(degree=3, include_bias=True)
X_poly = poly.fit_transform(X_raw)  # (n, C(p+3, 3))

print(poly.get_feature_names_out())  # ['1', 'x0', 'x1', 'x0^2', ...]
```

### 2.3 End-to-End Fit

```python
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ("poly", PolynomialFeatures(degree=3, include_bias=False)),
    ("scaler", StandardScaler()),
    ("lr", LinearRegression()),
])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
```

!!! tip "Scaling After Expansion"
    Always standardise **after** the polynomial expansion.  High-degree terms
    ($x^5$, $x^6$, …) can span orders of magnitude, destabilising both the
    normal equations and gradient descent.

---

## 3. PyTorch Implementation

```python
import torch
import torch.nn as nn


def polynomial_features_torch(
    x: torch.Tensor, degree: int
) -> torch.Tensor:
    """Univariate polynomial features: [x, x², …, x^d].

    Args:
        x: shape (n, 1)
        degree: polynomial degree

    Returns:
        shape (n, degree) — no constant column (nn.Linear adds bias).
    """
    return torch.cat([x ** k for k in range(1, degree + 1)], dim=1)


# Example: fit a degree-4 polynomial
torch.manual_seed(42)
n = 100
x = torch.linspace(-3, 3, n).unsqueeze(1)
y_true = 0.5 * x ** 3 - 2 * x ** 2 + x + 1
y = y_true + 2.0 * torch.randn(n, 1)

degree = 4
X_poly = polynomial_features_torch(x, degree)  # (100, 4)

model = nn.Linear(degree, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

for epoch in range(1000):
    loss = criterion(model(X_poly), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Inspect learned coefficients
# model.bias ≈ θ₀,  model.weight ≈ [θ₁, θ₂, θ₃, θ₄]
print(f"Bias:    {model.bias.item():.3f}")
print(f"Weights: {model.weight.detach().squeeze().tolist()}")
```

---

## 4. Bias–Variance Trade-Off

### 4.1 Conceptual Framework

| Degree | Model Complexity | Bias | Variance | Risk |
|:------:|:----------------:|:----:|:--------:|:----:|
| 1 | Low | High (underfitting) | Low | High |
| 3 | Moderate | Low | Moderate | **Low** |
| 10 | High | Very low | High (overfitting) | High |

- **Bias** measures the systematic error from an overly simple model.
- **Variance** measures sensitivity to the particular training set.
- **Total error** $\approx \text{Bias}^2 + \text{Variance} + \text{Irreducible noise}$.

### 4.2 Mathematical Statement

For the squared-error loss and a model $\hat{f}$ trained on dataset
$\mathcal{D}$:

$$
E_{\mathcal{D}}\!\bigl[(y - \hat{f}(x))^2\bigr]
= \underbrace{\bigl(f(x) - E[\hat{f}(x)]\bigr)^2}_{\text{Bias}^2}
  + \underbrace{E\!\bigl[(\hat{f}(x) - E[\hat{f}(x)])^2\bigr]}_{\text{Variance}}
  + \sigma^2.
$$

Increasing the polynomial degree reduces bias but increases variance.

### 4.3 Visualising the Trade-Off

```python
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

degrees = range(1, 15)
train_errors, val_errors = [], []

for d in degrees:
    pipe = Pipeline([
        ("poly", PolynomialFeatures(degree=d, include_bias=False)),
        ("lr", LinearRegression()),
    ])
    # Negative MSE (sklearn convention)
    cv_scores = cross_val_score(
        pipe, X_train, y_train, cv=5, scoring="neg_mean_squared_error"
    )
    pipe.fit(X_train, y_train)
    train_mse = np.mean((y_train - pipe.predict(X_train)) ** 2)

    train_errors.append(train_mse)
    val_errors.append(-cv_scores.mean())

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(list(degrees), train_errors, "o-", label="Train MSE")
ax.plot(list(degrees), val_errors, "s-", label="CV MSE")
ax.set_xlabel("Polynomial Degree")
ax.set_ylabel("MSE")
ax.set_title("Bias–Variance Trade-Off")
ax.legend()
ax.grid(True, alpha=0.3)
```

---

## 5. Cross-Validation for Degree Selection

### 5.1 $k$-Fold Cross-Validation

The optimal degree minimises the **cross-validated** error, not the training
error.  $k$-fold CV partitions the data into $k$ folds, trains on $k-1$ folds,
and evaluates on the held-out fold, repeating $k$ times.

```python
from sklearn.model_selection import cross_val_score


def select_degree(X, y, max_degree=10, cv=5):
    """Select optimal polynomial degree via cross-validation."""
    best_degree, best_score = 1, -np.inf

    for d in range(1, max_degree + 1):
        pipe = Pipeline([
            ("poly", PolynomialFeatures(degree=d, include_bias=False)),
            ("scaler", StandardScaler()),
            ("lr", LinearRegression()),
        ])
        scores = cross_val_score(
            pipe, X, y, cv=cv, scoring="neg_mean_squared_error"
        )
        mean_score = scores.mean()
        if mean_score > best_score:
            best_score, best_degree = mean_score, d

    print(f"Best degree: {best_degree}  (CV MSE: {-best_score:.4f})")
    return best_degree
```

### 5.2 Information Criteria

For model selection without explicit CV, information criteria penalise model
complexity:

| Criterion | Formula | Notes |
|-----------|---------|-------|
| AIC | $n\ln(\text{MSE}) + 2k$ | Asymptotically equivalent to leave-one-out CV |
| BIC | $n\ln(\text{MSE}) + k\ln(n)$ | Stronger penalty; favours simpler models |

where $k$ is the number of parameters (degree $+$ 1 for univariate).

---

## 6. Regularisation Connection

High-degree polynomials overfit because they have too many free parameters
relative to the data.  Two remedies:

1. **Limit the degree** (model selection via CV — this page).
2. **Penalise large coefficients** (regularisation — see
   [Ridge](ridge_regression.md) and [Lasso](lasso_regression.md)).

In practice, combining moderate-degree polynomial features with Ridge or Lasso
regularisation is more robust than using a very high degree without
regularisation.

```python
from sklearn.linear_model import Ridge

pipe_regularised = Pipeline([
    ("poly", PolynomialFeatures(degree=8, include_bias=False)),
    ("scaler", StandardScaler()),
    ("ridge", Ridge(alpha=1.0)),
])
pipe_regularised.fit(X_train, y_train)
```

---

## 7. Beyond Polynomials

Polynomial features are one choice of basis expansion.  Other common choices
include:

| Basis | Formula | Use Case |
|-------|---------|----------|
| Polynomial | $x, x^2, \ldots, x^d$ | Smooth global trends |
| Radial Basis Functions | $\exp(-\gamma\|x - c_k\|^2)$ | Local patterns, kernel methods |
| Fourier | $\sin(k\omega x), \cos(k\omega x)$ | Periodic data |
| Splines | Piecewise polynomials with knots | Flexible local fits |
| Interaction terms | $x_i x_j$ | Feature cross-effects |

All of these keep the model linear in parameters, so the same OLS / gradient
descent machinery applies.

---

## 8. Complete Example

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# --- Synthetic data: cubic with noise ---
rng = np.random.default_rng(42)
n = 80
x = rng.uniform(-3, 3, n)
y = 0.5 * x ** 3 - x ** 2 + 0.5 * x + 2 + 3 * rng.normal(size=n)

X = x.reshape(-1, 1)

# --- Degree selection via CV ---
degrees = range(1, 12)
cv_mses = []
for d in degrees:
    pipe = Pipeline([
        ("poly", PolynomialFeatures(degree=d, include_bias=False)),
        ("scaler", StandardScaler()),
        ("lr", LinearRegression()),
    ])
    scores = cross_val_score(pipe, X, y, cv=5, scoring="neg_mean_squared_error")
    cv_mses.append(-scores.mean())

best_d = degrees[np.argmin(cv_mses)]
print(f"Best degree: {best_d}  (CV MSE: {min(cv_mses):.2f})")

# --- Final fit ---
final_pipe = Pipeline([
    ("poly", PolynomialFeatures(degree=best_d, include_bias=False)),
    ("scaler", StandardScaler()),
    ("lr", LinearRegression()),
])
final_pipe.fit(X, y)

# --- Plot ---
x_plot = np.linspace(-3.5, 3.5, 200).reshape(-1, 1)
y_plot = final_pipe.predict(x_plot)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].scatter(x, y, alpha=0.6, s=20, label="Data")
axes[0].plot(x_plot, y_plot, "r-", lw=2, label=f"Degree {best_d}")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
axes[0].set_title("Polynomial Fit")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(list(degrees), cv_mses, "o-", lw=2)
axes[1].axvline(best_d, color="r", ls="--", label=f"Best d={best_d}")
axes[1].set_xlabel("Polynomial Degree")
axes[1].set_ylabel("5-Fold CV MSE")
axes[1].set_title("Cross-Validation Curve")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
```

---

## Summary

| Concept | Key Point |
|---------|-----------|
| Feature map | $\phi(x) = [1, x, x^2, \ldots, x^d]^\top$ — nonlinear in $x$, linear in $\boldsymbol{\theta}$ |
| Expanded dimension | $\binom{p+d}{d}$ features for $p$ inputs, degree $d$ |
| Bias–variance | Low degree → high bias; high degree → high variance |
| Model selection | $k$-fold CV or AIC/BIC to choose $d$ |
| Regularisation | Ridge/Lasso with moderate $d$ is more robust than high $d$ alone |
| Scaling | **Always** standardise after polynomial expansion |

---

## References

1. Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of
   Statistical Learning*, §§3.1, 7.10.
2. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*, §§1.1,
   3.1.
3. Murphy, K. P. (2022). *Probabilistic Machine Learning: An Introduction*,
   Ch. 11.
