# Ridge Regression

## Overview

When features are correlated or the model overfits, Ordinary Least Squares
(OLS) produces parameter estimates with high variance.  **Ridge regression**
adds an $\ell_2$ penalty to the loss, shrinking coefficients toward zero and
stabilising the solution.  This page derives the closed-form solution, explores
its geometric and Bayesian interpretations, and demonstrates implementations in
NumPy, PyTorch, and scikit-learn.

---

## 1. The Ridge Objective

### 1.1 Formulation

Ridge regression minimises the penalised sum of squared errors:

$$
\mathcal{L}_{\text{ridge}}(\boldsymbol{\theta})
= \|\mathbf{y} - \mathbf{X}\boldsymbol{\theta}\|^2
  + \lambda\|\boldsymbol{\theta}\|_2^2
= \sum_{i=1}^{n}(y_i - \hat{y}_i)^2
  + \lambda\sum_{j=1}^{p}\theta_j^2,
$$

where $\lambda > 0$ is the **regularisation strength**.  By convention, the
**bias** term is usually excluded from the penalty — only the weights are
shrunk.

!!! note "Notation"
    scikit-learn uses `alpha` for $\lambda$.  Some texts write the penalty as
    $\frac{\lambda}{2}\|\boldsymbol{\theta}\|^2$ so that the gradient has no
    extra factor of 2.

### 1.2 Closed-Form Solution

Setting the gradient to zero:

$$
\nabla_{\boldsymbol{\theta}}\mathcal{L}
= -2\mathbf{X}^\top\mathbf{y}
  + 2(\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})\boldsymbol{\theta}
= \mathbf{0},
$$

gives

$$
\boxed{
  \boldsymbol{\theta}^*_{\text{ridge}}
  = (\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})^{-1}
    \mathbf{X}^\top\mathbf{y}.
}
$$

The matrix
$\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I}$ is always invertible for
$\lambda > 0$, since its eigenvalues are $\sigma_j^2 + \lambda > 0$ where
$\sigma_j$ are the singular values of $\mathbf{X}$.  Ridge regression therefore
eliminates the invertibility problems that OLS faces under multicollinearity.

---

## 2. Geometric Interpretation

### 2.1 Eigenvalue Shrinkage

Let the SVD of the centred design matrix be
$\mathbf{X} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$.  The OLS
predictions are

$$
\hat{\mathbf{y}}_{\text{OLS}}
= \mathbf{X}\boldsymbol{\theta}^*_{\text{OLS}}
= \sum_{j=1}^{p} \mathbf{u}_j
  \frac{\sigma_j^2}{\sigma_j^2}\,
  \mathbf{u}_j^\top\mathbf{y}
= \sum_{j=1}^{p} \mathbf{u}_j\,\mathbf{u}_j^\top\mathbf{y},
$$

while the Ridge predictions are

$$
\hat{\mathbf{y}}_{\text{ridge}}
= \sum_{j=1}^{p} \mathbf{u}_j\,
  \underbrace{\frac{\sigma_j^2}{\sigma_j^2 + \lambda}}_{\text{shrinkage factor}}\,
  \mathbf{u}_j^\top\mathbf{y}.
$$

Each component is multiplied by a factor strictly less than 1, with **small
singular values shrunk the most**.  Directions of low variance in the data
(where OLS estimates are most unstable) are penalised most heavily.

### 2.2 Constrained Optimisation View

Ridge regression is equivalent to

$$
\min_{\boldsymbol{\theta}}
\|\mathbf{y} - \mathbf{X}\boldsymbol{\theta}\|^2
\quad \text{subject to} \quad
\|\boldsymbol{\theta}\|_2^2 \leq t,
$$

for some $t$ that depends on $\lambda$.  Geometrically, this is the intersection
of the MSE contour ellipsoids with the $\ell_2$ ball of radius $\sqrt{t}$.  The
constraint surface is **smooth** (a sphere), so the solution lies on the
boundary but is never exactly zero — Ridge shrinks but does not produce sparse
solutions.

---

## 3. Bayesian Interpretation

### 3.1 Gaussian Prior on Weights

Place an isotropic Gaussian prior on the parameters:

$$
\boldsymbol{\theta} \sim \mathcal{N}(\mathbf{0},\, \tau^2\mathbf{I}).
$$

Combined with the Gaussian likelihood
$\mathbf{y} \mid \mathbf{X}, \boldsymbol{\theta}
\sim \mathcal{N}(\mathbf{X}\boldsymbol{\theta},\, \sigma^2\mathbf{I})$,
the posterior is

$$
p(\boldsymbol{\theta} \mid \mathbf{X}, \mathbf{y})
\propto
\exp\!\Bigl(
  -\frac{1}{2\sigma^2}\|\mathbf{y} - \mathbf{X}\boldsymbol{\theta}\|^2
  -\frac{1}{2\tau^2}\|\boldsymbol{\theta}\|^2
\Bigr).
$$

### 3.2 MAP = Ridge

The **maximum a posteriori** (MAP) estimate is

$$
\boldsymbol{\theta}_{\text{MAP}}
= \arg\min_{\boldsymbol{\theta}}
\left[
  \|\mathbf{y} - \mathbf{X}\boldsymbol{\theta}\|^2
  + \frac{\sigma^2}{\tau^2}\|\boldsymbol{\theta}\|^2
\right],
$$

which is exactly Ridge regression with $\lambda = \sigma^2 / \tau^2$.

| Bayesian | Ridge |
|----------|-------|
| Tight prior ($\tau^2$ small) | Strong regularisation ($\lambda$ large) |
| Broad prior ($\tau^2$ large) | Weak regularisation ($\lambda$ small) |
| No prior ($\tau^2 \to \infty$) | OLS ($\lambda = 0$) |

---

## 4. Effect on Bias and Variance

Ridge introduces **bias** (shrinking coefficients away from their true values)
in exchange for reduced **variance** (more stable estimates across different
training sets):

$$
\text{MSE}(\hat{\boldsymbol{\theta}}_{\text{ridge}})
= \underbrace{\text{Bias}^2(\lambda)}_{\nearrow\text{ with }\lambda}
  + \underbrace{\text{Variance}(\lambda)}_{\searrow\text{ with }\lambda}.
$$

The optimal $\lambda$ minimises the total MSE — a bias–variance trade-off.

---

## 5. Implementations

### 5.1 NumPy

```python
import numpy as np


def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Ridge regression: θ = (X^T X + λ I)^{-1} X^T y.

    Parameters
    ----------
    X : design matrix (n, p) — WITHOUT bias column
    y : target (n,)
    lam : regularisation strength λ

    Returns
    -------
    theta : (p,) parameter vector (no intercept)
    """
    p = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(p), X.T @ y)


def ridge_fit_with_intercept(X, y, lam):
    """Centre data to handle intercept separately."""
    X_mean, y_mean = X.mean(axis=0), y.mean()
    X_c = X - X_mean
    y_c = y - y_mean

    w = ridge_fit(X_c, y_c, lam)
    b = y_mean - X_mean @ w
    return w, b
```

### 5.2 PyTorch (Closed-Form)

```python
import torch


def ridge_closed_form(
    X: torch.Tensor, y: torch.Tensor, lam: float
) -> torch.Tensor:
    """Ridge regression via closed-form solution."""
    if y.dim() == 1:
        y = y.reshape(-1, 1)
    p = X.shape[1]
    A = X.T @ X + lam * torch.eye(p)
    return torch.linalg.solve(A, X.T @ y)
```

### 5.3 PyTorch (Gradient Descent with Weight Decay)

In PyTorch, $\ell_2$ regularisation is implemented as **weight decay** in the
optimiser — the penalty gradient $\lambda\boldsymbol{\theta}$ is added to the
parameter update automatically:

```python
import torch.nn as nn

model = nn.Linear(p, 1)
criterion = nn.MSELoss()

# weight_decay = λ / n  (PyTorch convention)
optimizer = torch.optim.SGD(
    model.parameters(), lr=0.01, weight_decay=1e-2
)

for epoch in range(100):
    loss = criterion(model(X_train), y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

!!! warning "Weight Decay Convention"
    PyTorch's `weight_decay` parameter in SGD and Adam adds
    $\text{weight\_decay} \times \theta$ to the gradient.  This is equivalent
    to adding $\frac{\text{weight\_decay}}{2}\|\theta\|^2$ to the loss.
    The relationship to the Ridge $\lambda$ depends on whether your loss
    includes the $1/n$ factor and what convention you use.

### 5.4 scikit-learn

```python
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Fixed alpha
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge(alpha=1.0)),
])
pipe.fit(X_train, y_train)

# Cross-validated alpha selection
pipe_cv = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])),
])
pipe_cv.fit(X_train, y_train)
best_alpha = pipe_cv.named_steps["ridge"].alpha_
print(f"Best α: {best_alpha}")
```

!!! warning "Always Standardise"
    Ridge penalises the **magnitude** of coefficients.  If features are on
    different scales, the penalty disproportionately shrinks coefficients of
    high-variance features.  Always use `StandardScaler` before Ridge.

---

## 6. Selecting $\lambda$

### 6.1 Cross-Validation

The standard approach: evaluate on held-out data for a grid of $\lambda$ values.

```python
import numpy as np
from sklearn.model_selection import cross_val_score

alphas = np.logspace(-3, 3, 50)
cv_scores = []

for a in alphas:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=a)),
    ])
    scores = cross_val_score(
        pipe, X_train, y_train, cv=5, scoring="neg_mean_squared_error"
    )
    cv_scores.append(-scores.mean())

best_idx = np.argmin(cv_scores)
print(f"Best α = {alphas[best_idx]:.4f}, CV MSE = {cv_scores[best_idx]:.4f}")
```

### 6.2 Regularisation Path

Plot coefficients as a function of $\lambda$ to see the shrinkage effect:

```python
import matplotlib.pyplot as plt

alphas = np.logspace(-2, 4, 100)
coefs = []

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

for a in alphas:
    model = Ridge(alpha=a, fit_intercept=True)
    model.fit(X_scaled, y_train)
    coefs.append(model.coef_)

coefs = np.array(coefs)

fig, ax = plt.subplots(figsize=(10, 5))
for j in range(coefs.shape[1]):
    ax.plot(alphas, coefs[:, j], label=f"w_{j}")
ax.set_xscale("log")
ax.set_xlabel("α (regularisation strength)")
ax.set_ylabel("Coefficient value")
ax.set_title("Ridge Regularisation Path")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
```

---

## 7. Ridge vs OLS: When Does Ridge Help?

| Scenario | OLS | Ridge |
|----------|-----|-------|
| $n \gg p$, low correlation | ✓ Works well | Marginal improvement |
| $n \gg p$, high correlation | Inflated variance | ✓ Stabilises estimates |
| $n \approx p$ | Overfits | ✓ Essential |
| $n < p$ | Singular $\mathbf{X}^\top\mathbf{X}$ | ✓ Always invertible |
| Sparse true model | Includes all features | All features remain (no sparsity) |

Ridge **never** produces exactly zero coefficients — if feature selection is
needed, see [Lasso Regression](lasso_regression.md).

---

## 8. Connection to Other Methods

| Method | Penalty | Bayesian Prior | Sparsity |
|--------|---------|----------------|----------|
| OLS | None | Improper uniform | No |
| Ridge | $\lambda\|\boldsymbol{\theta}\|_2^2$ | Gaussian | No |
| Lasso | $\lambda\|\boldsymbol{\theta}\|_1$ | Laplace | Yes |
| ElasticNet | $\lambda[\rho\|\cdot\|_1 + (1-\rho)\|\cdot\|_2^2]$ | Gaussian + Laplace | Yes |

---

## Summary

| Concept | Key Result |
|---------|------------|
| Objective | $\|\mathbf{y} - \mathbf{X}\boldsymbol{\theta}\|^2 + \lambda\|\boldsymbol{\theta}\|_2^2$ |
| Solution | $(\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^\top\mathbf{y}$ |
| Eigenvalue shrinkage | Factor $\sigma_j^2 / (\sigma_j^2 + \lambda)$ per component |
| Bayesian view | MAP under Gaussian prior $\mathcal{N}(0, \tau^2 I)$ with $\lambda = \sigma^2/\tau^2$ |
| PyTorch | `weight_decay` parameter in optimiser |
| scikit-learn | `Ridge(alpha=λ)` or `RidgeCV` for automatic selection |
| Key property | Shrinks but never zeros coefficients |

---

## References

1. Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of
   Statistical Learning*, §3.4.1.
2. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*, §3.1.4.
3. Hoerl, A. E. & Kennard, R. W. (1970). "Ridge Regression: Biased Estimation
   for Nonorthogonal Problems." *Technometrics*.
