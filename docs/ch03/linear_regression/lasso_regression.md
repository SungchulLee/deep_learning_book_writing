# Lasso Regression

## Overview

**Lasso** (Least Absolute Shrinkage and Selection Operator) replaces the
$\ell_2$ penalty of Ridge with an $\ell_1$ penalty.  This seemingly small
change has a profound consequence: Lasso drives some coefficients to **exactly
zero**, performing automatic feature selection.  This page derives the Lasso
objective, explains why sparsity arises, introduces the coordinate descent
algorithm, and covers the **ElasticNet** hybrid.

---

## 1. The Lasso Objective

### 1.1 Formulation

$$
\mathcal{L}_{\text{lasso}}(\boldsymbol{\theta})
= \frac{1}{2n}\|\mathbf{y} - \mathbf{X}\boldsymbol{\theta}\|^2
  + \lambda\|\boldsymbol{\theta}\|_1
= \frac{1}{2n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
  + \lambda\sum_{j=1}^{p}|\theta_j|.
$$

!!! note "Scaling Convention"
    The $1/(2n)$ factor on the data term (used by scikit-learn) ensures that
    the optimal $\lambda$ does not depend on $n$.  Some references use $1/2$
    or omit the factor entirely — always check conventions.

### 1.2 No Closed-Form Solution

Unlike Ridge, the $\ell_1$ norm is **not differentiable** at zero, so there is
no closed-form matrix solution.  Lasso requires iterative algorithms:

| Algorithm | Description |
|-----------|-------------|
| Coordinate descent | Update one $\theta_j$ at a time (scikit-learn default) |
| Proximal gradient descent (ISTA/FISTA) | Gradient step + soft thresholding |
| Subgradient methods | Replace gradient with subgradient at non-smooth points |
| LARS | Least Angle Regression — builds entire regularisation path efficiently |

---

## 2. Why $\ell_1$ Produces Sparsity

### 2.1 Geometric Argument

Consider the constrained form:

$$
\min_{\boldsymbol{\theta}}
\|\mathbf{y} - \mathbf{X}\boldsymbol{\theta}\|^2
\quad \text{subject to} \quad
\|\boldsymbol{\theta}\|_1 \leq t.
$$

The $\ell_1$ ball $\{|\theta_1| + |\theta_2| \leq t\}$ is a **diamond** in 2-D
(cross-polytope in higher dimensions).  Its corners lie on the coordinate axes.
The MSE contour ellipsoids are most likely to first touch the constraint at a
**corner**, where one or more coordinates are exactly zero.

In contrast, the $\ell_2$ ball (used by Ridge) is a smooth sphere with no
corners — the contact point generically has all coordinates nonzero.

### 2.2 Soft Thresholding

For a single coordinate $\theta_j$, the Lasso subproblem has the solution

$$
\theta_j^*
= \mathcal{S}_\lambda(z_j)
= \operatorname{sign}(z_j)\,\max(|z_j| - \lambda,\; 0),
$$

where $z_j$ is the partial residual (the OLS update ignoring the penalty) and
$\mathcal{S}_\lambda$ is the **soft-thresholding operator**.  When
$|z_j| \leq \lambda$, the coefficient is set exactly to zero.

```python
def soft_threshold(z: float, lam: float) -> float:
    """Soft-thresholding operator S_λ(z)."""
    if z > lam:
        return z - lam
    elif z < -lam:
        return z + lam
    else:
        return 0.0
```

---

## 3. Coordinate Descent Algorithm

The default solver in scikit-learn.  It cycles through coordinates, updating one
$\theta_j$ at a time while holding others fixed.

### 3.1 Derivation

For coordinate $j$, the partial residual is

$$
r_j = \mathbf{y} - \mathbf{X}_{-j}\boldsymbol{\theta}_{-j},
$$

where $\mathbf{X}_{-j}$ is $\mathbf{X}$ without column $j$.  The
single-variable subproblem becomes

$$
\min_{\theta_j}
\frac{1}{2n}\|r_j - \mathbf{x}_j\theta_j\|^2
+ \lambda|\theta_j|,
$$

whose solution is

$$
\theta_j^*
= \frac{1}{\|\mathbf{x}_j\|^2 / n}\,
  \mathcal{S}_\lambda\!\left(
    \frac{\mathbf{x}_j^\top r_j}{n}
  \right).
$$

When features are standardised ($\|\mathbf{x}_j\|^2 / n = 1$), this simplifies
to $\theta_j^* = \mathcal{S}_\lambda(\mathbf{x}_j^\top r_j / n)$.

### 3.2 NumPy Implementation

```python
import numpy as np


def lasso_coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    lam: float,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> np.ndarray:
    """Lasso via coordinate descent (features assumed standardised).

    Parameters
    ----------
    X : (n, p) standardised design matrix (no intercept column)
    y : (n,) centred target
    lam : regularisation strength
    max_iter : maximum passes through all coordinates
    tol : convergence threshold on max coefficient change

    Returns
    -------
    theta : (p,) Lasso solution
    """
    n, p = X.shape
    theta = np.zeros(p)
    x_sq = np.sum(X ** 2, axis=0) / n  # precompute ‖x_j‖²/n

    for iteration in range(max_iter):
        theta_old = theta.copy()

        for j in range(p):
            # Partial residual excluding feature j
            r_j = y - X @ theta + X[:, j] * theta[j]
            # Correlation
            z_j = X[:, j] @ r_j / n
            # Soft threshold
            theta[j] = soft_threshold(z_j, lam) / x_sq[j]

        # Check convergence
        if np.max(np.abs(theta - theta_old)) < tol:
            break

    return theta
```

---

## 4. Proximal Gradient Descent (ISTA)

An alternative to coordinate descent: take a gradient step on the smooth part
(MSE), then apply the proximal operator (soft thresholding) for the non-smooth
part ($\ell_1$).

### 4.1 Algorithm

$$
\boldsymbol{\theta}^{(t+1)}
= \mathcal{S}_{\eta\lambda}\!\Bigl(
    \boldsymbol{\theta}^{(t)}
    - \eta\,\nabla_{\boldsymbol{\theta}}
      \frac{1}{2n}\|\mathbf{y} - \mathbf{X}\boldsymbol{\theta}^{(t)}\|^2
  \Bigr),
$$

where $\eta$ is the step size and $\mathcal{S}$ is applied element-wise.

### 4.2 PyTorch Implementation

```python
import torch


def lasso_ista(
    X: torch.Tensor,
    y: torch.Tensor,
    lam: float,
    lr: float = 0.01,
    n_iter: int = 1000,
) -> torch.Tensor:
    """Iterative Soft-Thresholding Algorithm (ISTA) for Lasso."""
    n, p = X.shape
    theta = torch.zeros(p)

    for _ in range(n_iter):
        # Gradient of MSE (smooth part)
        residual = X @ theta - y
        grad = X.T @ residual / n

        # Gradient step
        z = theta - lr * grad

        # Proximal step (soft thresholding)
        theta = torch.sign(z) * torch.clamp(torch.abs(z) - lr * lam, min=0)

    return theta
```

---

## 5. Implementations

### 5.1 scikit-learn

```python
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Fixed alpha
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("lasso", Lasso(alpha=0.1)),
])
pipe.fit(X_train, y_train)
coefs = pipe.named_steps["lasso"].coef_
n_nonzero = np.sum(np.abs(coefs) > 1e-8)
print(f"Non-zero coefficients: {n_nonzero} / {len(coefs)}")

# Cross-validated alpha
pipe_cv = Pipeline([
    ("scaler", StandardScaler()),
    ("lasso", LassoCV(cv=5, random_state=42)),
])
pipe_cv.fit(X_train, y_train)
print(f"Best α: {pipe_cv.named_steps['lasso'].alpha_:.4f}")
```

### 5.2 PyTorch (Manual L1 Penalty)

PyTorch's built-in `weight_decay` implements $\ell_2$ only.  For $\ell_1$, add
the penalty manually:

```python
import torch.nn as nn

model = nn.Linear(p, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
lam = 0.01

for epoch in range(100):
    y_pred = model(X_train)
    mse_loss = criterion(y_pred, y_train)

    # L1 penalty on weights (not bias)
    l1_penalty = sum(param.abs().sum() for name, param in model.named_parameters()
                     if "weight" in name)

    loss = mse_loss + lam * l1_penalty

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

!!! warning "Gradient at Zero"
    The $\ell_1$ norm is not differentiable at 0.  PyTorch uses the
    **subgradient** (which is 0 at 0), so SGD will not drive parameters to
    exact zero.  For true sparsity, post-process by thresholding small
    coefficients, or use a proximal optimiser.

---

## 6. ElasticNet ($\ell_1 + \ell_2$)

### 6.1 Motivation

Lasso has two limitations when features are correlated:

1. It arbitrarily selects **one** feature from a correlated group and zeros the
   rest.
2. With $n < p$, Lasso selects at most $n$ features.

**ElasticNet** combines both penalties to get sparsity (from $\ell_1$) and
grouping (from $\ell_2$):

$$
\mathcal{L}_{\text{elastic}}(\boldsymbol{\theta})
= \frac{1}{2n}\|\mathbf{y} - \mathbf{X}\boldsymbol{\theta}\|^2
  + \lambda\Bigl[
    \rho\|\boldsymbol{\theta}\|_1
    + \frac{1-\rho}{2}\|\boldsymbol{\theta}\|_2^2
  \Bigr],
$$

where $\rho \in [0, 1]$ controls the mix (`l1_ratio` in scikit-learn):

| $\rho$ | Behaviour |
|--------|-----------|
| 0 | Pure Ridge |
| 1 | Pure Lasso |
| 0.5 | Equal mix |

### 6.2 scikit-learn

```python
from sklearn.linear_model import ElasticNet, ElasticNetCV

elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic.fit(X_train_scaled, y_train)

elastic_cv = ElasticNetCV(
    cv=5,
    l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95],
    random_state=42,
)
elastic_cv.fit(X_train_scaled, y_train)
print(f"Best α: {elastic_cv.alpha_:.4f}")
print(f"Best l1_ratio: {elastic_cv.l1_ratio_:.2f}")
```

---

## 7. Comparison of Regularisation Methods

| Property | OLS | Ridge | Lasso | ElasticNet |
|----------|-----|-------|-------|------------|
| Penalty | None | $\lambda\|\theta\|_2^2$ | $\lambda\|\theta\|_1$ | $\lambda[\rho\|\theta\|_1 + (1{-}\rho)\|\theta\|_2^2/2]$ |
| Closed-form | ✓ | ✓ | ✗ | ✗ |
| Sparsity | ✗ | ✗ | ✓ | ✓ |
| Correlated features | Unstable | Handles well | Picks one | Groups shared |
| $n < p$ | Fails | ✓ | Selects ≤ $n$ | ✓ |
| Bayesian prior | Uniform | Gaussian | Laplace | Gaussian + Laplace |
| scikit-learn | `LinearRegression` | `Ridge` | `Lasso` | `ElasticNet` |

---

## 8. Regularisation Path

The **regularisation path** traces coefficients as $\lambda$ varies from large
(all zeros) to small (OLS):

```python
from sklearn.linear_model import lasso_path
import matplotlib.pyplot as plt

alphas, coefs_path, _ = lasso_path(
    X_train_scaled, y_train, alphas=np.logspace(-3, 1, 100)
)

fig, ax = plt.subplots(figsize=(10, 5))
for j in range(coefs_path.shape[0]):
    ax.plot(alphas, coefs_path[j], label=f"w_{j}")
ax.set_xscale("log")
ax.invert_xaxis()
ax.set_xlabel("α (regularisation strength)")
ax.set_ylabel("Coefficient value")
ax.set_title("Lasso Regularisation Path")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
```

As $\lambda$ decreases, coefficients "enter" the model one by one — the order
in which they appear reveals their relative importance.

---

## 9. Practical Guidelines

| Decision | Recommendation |
|----------|----------------|
| Many correlated features, no sparsity needed | Ridge |
| Feature selection desired, low correlation | Lasso |
| Feature selection with correlated features | ElasticNet ($\rho \approx 0.5$–$0.9$) |
| Unsure | Start with ElasticNetCV; let CV choose $\alpha$ and $\rho$ |
| PyTorch + sparsity | Add manual L1 penalty + post-hoc thresholding, or use proximal methods |

!!! tip "Feature Scaling"
    As with Ridge, **always standardise** features before applying Lasso or
    ElasticNet.  The $\ell_1$ penalty is sensitive to feature magnitudes.

---

## Summary

| Concept | Key Result |
|---------|------------|
| Lasso objective | $\frac{1}{2n}\|\mathbf{y}-\mathbf{X}\boldsymbol{\theta}\|^2 + \lambda\|\boldsymbol{\theta}\|_1$ |
| Sparsity mechanism | $\ell_1$ ball has corners on coordinate axes |
| Soft thresholding | $\mathcal{S}_\lambda(z) = \mathrm{sign}(z)\max(\|z\|-\lambda, 0)$ |
| Solver | Coordinate descent (scikit-learn) or proximal gradient (ISTA) |
| ElasticNet | Hybrid $\ell_1 + \ell_2$; groups correlated features |
| Bayesian view | MAP under Laplace prior |

---

## References

1. Tibshirani, R. (1996). "Regression Shrinkage and Selection via the Lasso."
   *Journal of the Royal Statistical Society B*.
2. Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of
   Statistical Learning*, §§3.4.2–3.4.3.
3. Zou, H. & Hastie, T. (2005). "Regularization and Variable Selection via the
   Elastic Net." *Journal of the Royal Statistical Society B*.
4. Friedman, J., Hastie, T. & Tibshirani, R. (2010). "Regularization Paths for
   Generalized Linear Models via Coordinate Descent." *Journal of Statistical
   Software*.
