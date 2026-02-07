# Closed-Form Solution

## Overview

Unlike most machine learning problems that require iterative optimisation,
linear regression has a **closed-form** solution known as the **normal
equations**.  This page develops the solution from first principles: the
required vector-calculus identities, the derivation itself, its geometric
interpretation as an orthogonal projection, and efficient numerical
implementations in NumPy and PyTorch.

---

## 1. Vector Calculus Prerequisites

### 1.1 Notation

| Symbol | Meaning |
|--------|---------|
| $\mathbf{X} \in \mathbb{R}^{n \times (p+1)}$ | Design matrix (bias column included) |
| $\mathbf{y} \in \mathbb{R}^{n}$ | Target vector |
| $\boldsymbol{\theta} \in \mathbb{R}^{p+1}$ | Parameter vector |
| $\|\mathbf{v}\|^2 = \mathbf{v}^\top\mathbf{v}$ | Squared Euclidean norm |

### 1.2 Gradient of a Linear Form

For a constant vector $\mathbf{a}$:

$$
\frac{\partial}{\partial \boldsymbol{\theta}}\,
\mathbf{a}^\top \boldsymbol{\theta}
= \mathbf{a}.
$$

**Proof.** $\mathbf{a}^\top\boldsymbol{\theta} = \sum_j a_j \theta_j$.
Differentiating with respect to $\theta_k$ gives $a_k$. $\square$

### 1.3 Gradient of a Quadratic Form

For a **symmetric** matrix $\mathbf{A}$:

$$
\frac{\partial}{\partial \boldsymbol{\theta}}\,
\boldsymbol{\theta}^\top \mathbf{A}\,\boldsymbol{\theta}
= 2\mathbf{A}\boldsymbol{\theta}.
$$

**Proof.** Expanding
$\boldsymbol{\theta}^\top\mathbf{A}\boldsymbol{\theta}
= \sum_j \sum_k A_{jk}\theta_j\theta_k$ and differentiating with respect to
$\theta_i$:

$$
\frac{\partial}{\partial \theta_i}
= \sum_k A_{ik}\theta_k + \sum_j A_{ji}\theta_j
= (\mathbf{A}\boldsymbol{\theta})_i
  + (\mathbf{A}^\top\boldsymbol{\theta})_i
= 2(\mathbf{A}\boldsymbol{\theta})_i,
$$

where the last step uses $\mathbf{A} = \mathbf{A}^\top$. $\square$

!!! warning "Non-Symmetric Case"
    If $\mathbf{A}$ is not symmetric, the gradient becomes
    $(\mathbf{A} + \mathbf{A}^\top)\boldsymbol{\theta}$.  The Gram matrix
    $\mathbf{X}^\top\mathbf{X}$ is always symmetric, so this distinction
    does not arise in ordinary least squares.

### 1.4 Trace Identities

Several trace identities appear in the MLE derivation and regularised losses:

| Identity | Formula |
|----------|---------|
| Cyclic property | $\mathrm{tr}(\mathbf{ABC}) = \mathrm{tr}(\mathbf{CAB}) = \mathrm{tr}(\mathbf{BCA})$ |
| Scalar as trace | $\mathbf{v}^\top\mathbf{v} = \mathrm{tr}(\mathbf{v}\mathbf{v}^\top)$ |
| Derivative of trace | $\frac{\partial}{\partial \mathbf{A}}\mathrm{tr}(\mathbf{BA}) = \mathbf{B}^\top$ |
| Derivative of log-det | $\frac{\partial}{\partial \mathbf{A}}\ln|\det\mathbf{A}| = (\mathbf{A}^{-1})^\top$ |

The scalar-as-trace identity lets us write the MSE loss as

$$
\mathcal{L}
= \frac{1}{n}\,\mathrm{tr}\!\bigl[
  (\mathbf{y} - \mathbf{X}\boldsymbol{\theta})
  (\mathbf{y} - \mathbf{X}\boldsymbol{\theta})^\top
\bigr],
$$

which is useful when simultaneously optimising over the noise variance
$\sigma^2$.

---

## 2. Derivation of the Normal Equations

### 2.1 Expanding the Loss

The MSE loss is

$$
\mathcal{L}(\boldsymbol{\theta})
= \frac{1}{n}\|\mathbf{y} - \mathbf{X}\boldsymbol{\theta}\|^2
= \frac{1}{n}\bigl(
  \mathbf{y}^\top\mathbf{y}
  - 2\boldsymbol{\theta}^\top\mathbf{X}^\top\mathbf{y}
  + \boldsymbol{\theta}^\top\mathbf{X}^\top\mathbf{X}\boldsymbol{\theta}
\bigr).
$$

### 2.2 Computing the Gradient

Applying the identities from §1:

$$
\nabla_{\boldsymbol{\theta}}\mathcal{L}
= \frac{1}{n}\bigl(
  -2\mathbf{X}^\top\mathbf{y}
  + 2\mathbf{X}^\top\mathbf{X}\boldsymbol{\theta}
\bigr)
= \frac{2}{n}\,\mathbf{X}^\top\!
  \bigl(\mathbf{X}\boldsymbol{\theta} - \mathbf{y}\bigr).
$$

### 2.3 Setting the Gradient to Zero

$$
\boxed{
  \mathbf{X}^\top\mathbf{X}\,\boldsymbol{\theta}^*
  = \mathbf{X}^\top\mathbf{y}
}
\qquad \text{(Normal Equations)}
$$

If $\mathbf{X}^\top\mathbf{X}$ is invertible (i.e.\ $\mathbf{X}$ has full
column rank), then

$$
\boldsymbol{\theta}^*
= (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}.
$$

### 2.4 The Hessian and Convexity

The Hessian is

$$
\mathbf{H}
= \frac{2}{n}\,\mathbf{X}^\top\mathbf{X}.
$$

Since
$\mathbf{v}^\top\mathbf{X}^\top\mathbf{X}\mathbf{v}
= \|\mathbf{X}\mathbf{v}\|^2 \geq 0$ for all $\mathbf{v}$, the Hessian is
positive semi-definite and the loss is **convex**.  Strict convexity holds when
$\mathbf{X}$ has full column rank ($n \geq p + 1$, no perfect
multicollinearity), guaranteeing a unique global minimum.

---

## 3. Why "Normal" Equations?

The residual vector
$\mathbf{r} = \mathbf{y} - \mathbf{X}\boldsymbol{\theta}^*$ is **orthogonal**
(normal) to the column space of $\mathbf{X}$:

$$
\mathbf{X}^\top\mathbf{r}
= \mathbf{X}^\top(\mathbf{y} - \mathbf{X}\boldsymbol{\theta}^*)
= \mathbf{X}^\top\mathbf{y} - \mathbf{X}^\top\mathbf{X}\boldsymbol{\theta}^*
= \mathbf{0}.
$$

This orthogonality condition gives the equations their name.

---

## 4. Geometric Interpretation

### 4.1 Projection onto the Column Space

The **column space** of $\mathbf{X}$ is the set of all possible predictions:

$$
\mathrm{Col}(\mathbf{X})
= \{\mathbf{X}\boldsymbol{\theta} : \boldsymbol{\theta} \in \mathbb{R}^{p+1}\}.
$$

Linear regression finds the point in $\mathrm{Col}(\mathbf{X})$ **closest** to
$\mathbf{y}$ — the orthogonal projection:

$$
\hat{\mathbf{y}}
= \mathrm{proj}_{\mathrm{Col}(\mathbf{X})}(\mathbf{y}).
$$

### 4.2 The Projection (Hat) Matrix

The **projection matrix** is

$$
\mathbf{P}
= \mathbf{X}(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top,
\qquad
\hat{\mathbf{y}} = \mathbf{P}\mathbf{y}.
$$

| Property | Formula | Interpretation |
|----------|---------|----------------|
| Idempotent | $\mathbf{P}^2 = \mathbf{P}$ | Projecting twice gives the same result |
| Symmetric | $\mathbf{P}^\top = \mathbf{P}$ | Self-adjoint operator |
| Rank | $\mathrm{rank}(\mathbf{P}) = p + 1$ | Dimension of the column space |
| Eigenvalues | 0 or 1 only | Pure projection |
| Trace | $\mathrm{tr}(\mathbf{P}) = p + 1$ | Equals the rank |

### 4.3 The Pythagorean Decomposition

The target vector decomposes into orthogonal components:

$$
\mathbf{y} = \underbrace{\hat{\mathbf{y}}}_{\in\,\mathrm{Col}(\mathbf{X})}
           + \underbrace{\mathbf{r}}_{\perp\,\mathrm{Col}(\mathbf{X})},
\qquad
\|\mathbf{y}\|^2 = \|\hat{\mathbf{y}}\|^2 + \|\mathbf{r}\|^2.
$$

This is the geometric basis for the ANOVA decomposition:

$$
\underbrace{\|\mathbf{y} - \bar{y}\mathbf{1}\|^2}_{\text{SS}_{\text{tot}}}
= \underbrace{\|\hat{\mathbf{y}} - \bar{y}\mathbf{1}\|^2}_{\text{SS}_{\text{reg}}}
+ \underbrace{\|\mathbf{r}\|^2}_{\text{SS}_{\text{res}}},
$$

and therefore $R^2 = \text{SS}_{\text{reg}} / \text{SS}_{\text{tot}}$ measures
the fraction of variance explained.

### 4.4 Geometric $R^2$

$$
R^2 = \cos^2\!\theta,
$$

where $\theta$ is the angle between the centred target
$(\mathbf{y} - \bar{y}\mathbf{1})$ and its projection
$(\hat{\mathbf{y}} - \bar{y}\mathbf{1})$.

### 4.5 Leverage

The diagonal elements of $\mathbf{P}$, denoted $h_{ii}$, are called
**leverage values**:

$$
h_{ii} = \mathbf{x}_i^\top(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{x}_i.
$$

High-leverage points have unusual feature values and disproportionate influence
on the fit.  Key properties:

- $\sum_i h_{ii} = p + 1$ (trace of $\mathbf{P}$).
- $1/n \leq h_{ii} \leq 1$.
- Common rule of thumb for flagging: $h_{ii} > 2(p+1)/n$.

---

## 5. NumPy Implementation

### 5.1 Design Matrix Construction

```python
import numpy as np


def make_design_matrix(x: np.ndarray) -> np.ndarray:
    """Prepend a column of ones: X = [1 | x].

    Parameters
    ----------
    x : ndarray of shape (n, p) or (n,)

    Returns
    -------
    X : ndarray of shape (n, p+1)
    """
    x = np.atleast_2d(x) if x.ndim == 1 else x
    return np.hstack([np.ones((x.shape[0], 1)), x])
```

### 5.2 Normal Equation Solver

```python
def fit_normal_equation(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve θ* = (X^T X)^{-1} X^T y.

    Parameters
    ----------
    X : design matrix (n, p+1)
    y : target vector (n,)

    Returns
    -------
    theta : parameter vector (p+1,)
    """
    return np.linalg.solve(X.T @ X, X.T @ y)
```

!!! tip "`solve` vs `inv`"
    `np.linalg.solve(A, b)` is preferred over `np.linalg.inv(A) @ b` because
    it is numerically more stable and faster — it avoids explicitly forming the
    inverse.

### 5.3 Prediction and Evaluation

```python
def predict(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Return predictions ŷ = X θ."""
    return X @ theta


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute regression metrics."""
    residual = y_true - y_pred
    ss_res = np.sum(residual ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    mse = np.mean(residual ** 2)
    return {
        "mse": mse,
        "rmse": np.sqrt(mse),
        "mae": np.mean(np.abs(residual)),
        "r2": 1.0 - ss_res / ss_tot,
    }
```

### 5.4 End-to-End Example

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

x, y = make_regression(n_samples=200, n_features=3, noise=10, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

X_train = make_design_matrix(x_train)
X_test = make_design_matrix(x_test)

theta = fit_normal_equation(X_train, y_train)
y_pred = predict(X_test, theta)
print(evaluate(y_test, y_pred))
```

---

## 6. PyTorch Implementation

### 6.1 Direct Solution

```python
import torch


def normal_equations(X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Solve θ = (X^T X)^{-1} X^T y via Cholesky / LU.

    Args:
        X: Design matrix (n, p+1) — include bias column.
        y: Target vector (n,) or (n, 1).

    Returns:
        Parameter vector θ of shape (p+1,) or (p+1, 1).
    """
    if y.dim() == 1:
        y = y.reshape(-1, 1)
    return torch.linalg.solve(X.T @ X, X.T @ y)
```

### 6.2 Numerically Stable Alternatives

```python
def normal_equations_qr(X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """QR decomposition: X = QR → R θ = Q^T y."""
    if y.dim() == 1:
        y = y.reshape(-1, 1)
    Q, R = torch.linalg.qr(X)
    return torch.linalg.solve_triangular(R, Q.T @ y, upper=True)


def normal_equations_svd(X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """SVD-based solution (most robust, handles rank deficiency)."""
    if y.dim() == 1:
        y = y.reshape(-1, 1)
    return torch.linalg.lstsq(X, y).solution
```

### 6.3 Projection Matrix Analysis

```python
def compute_projection_matrix(X: torch.Tensor) -> torch.Tensor:
    """Compute the hat matrix P = X (X^T X)^{-1} X^T."""
    return X @ torch.linalg.inv(X.T @ X) @ X.T


def verify_projection_properties(X: torch.Tensor):
    """Check idempotency, symmetry, eigenvalues, and trace."""
    P = compute_projection_matrix(X)
    n, d = X.shape

    is_symmetric = torch.allclose(P, P.T, atol=1e-6)
    is_idempotent = torch.allclose(P @ P, P, atol=1e-6)
    trace = torch.trace(P).item()
    eigenvalues = torch.linalg.eigvalsh(P)

    print(f"Symmetric:  {is_symmetric}")
    print(f"Idempotent: {is_idempotent}")
    print(f"Trace:      {trace:.2f} (expected {d})")
    print(f"Eigenvalues: {sorted(eigenvalues.tolist(), reverse=True)}")
```

### 6.4 ANOVA Decomposition

```python
def anova_decomposition(X: torch.Tensor, y: torch.Tensor):
    """Verify SST = SSR + SSE."""
    theta = torch.linalg.lstsq(X, y).solution
    y_hat = X @ theta
    y_mean = y.mean()

    SST = torch.sum((y - y_mean) ** 2)
    SSR = torch.sum((y_hat - y_mean) ** 2)
    SSE = torch.sum((y - y_hat) ** 2)
    R2 = (SSR / SST).item()

    print(f"SST = {SST.item():.4f}")
    print(f"SSR = {SSR.item():.4f},  SSE = {SSE.item():.4f}")
    print(f"SSR + SSE = {(SSR + SSE).item():.4f}")
    print(f"R² = {R2:.4f}")
```

---

## 7. Handling Special Cases

### 7.1 Near-Multicollinearity

When $\mathbf{X}^\top\mathbf{X}$ is nearly singular, add a small
regularisation term (this is Ridge regression with tiny $\lambda$):

$$
\boldsymbol{\theta}^*
= (\mathbf{X}^\top\mathbf{X} + \alpha\mathbf{I})^{-1}
  \mathbf{X}^\top\mathbf{y}.
$$

```python
def fit_regularised(
    X: torch.Tensor, y: torch.Tensor, alpha: float = 1e-6
) -> torch.Tensor:
    """Normal equations with Tikhonov regularisation for stability."""
    if y.dim() == 1:
        y = y.reshape(-1, 1)
    d = X.shape[1]
    return torch.linalg.solve(X.T @ X + alpha * torch.eye(d), X.T @ y)
```

### 7.2 Underdetermined Systems ($n < p$)

When there are more parameters than observations, infinitely many solutions
exist.  The **minimum-norm** solution is

$$
\boldsymbol{\theta}^*
= \mathbf{X}^\top(\mathbf{X}\mathbf{X}^\top)^{-1}\mathbf{y},
$$

which gives the solution with smallest $\|\boldsymbol{\theta}\|$.

---

## 8. Complexity Comparison

| Method | Time | Space | Notes |
|--------|------|-------|-------|
| Normal equations | $O(np^2 + p^3)$ | $O(p^2)$ | Best for $p < 10{,}000$ |
| QR decomposition | $O(np^2)$ | $O(np)$ | More stable than direct inverse |
| SVD | $O(np^2)$ | $O(np)$ | Most robust; handles rank deficiency |
| Gradient descent | $O(knp)$ | $O(p)$ | Best for $p > 10{,}000$ or very large $n$ |

**Rule of thumb:** use the closed-form solution when $p < 10{,}000$; switch to
gradient descent for larger feature spaces or when mini-batch training is
desirable.

---

## 9. Verification Against `nn.Linear`

```python
import torch.nn as nn


def verify_against_nn_linear():
    """Confirm normal equations match a fully-converged nn.Linear."""
    torch.manual_seed(42)
    n, p = 1000, 5
    X = torch.randn(n, p)
    true_w = torch.tensor([2.0, -1.5, 0.5, 1.0, -0.8])
    y = X @ true_w + 0.5 + 0.1 * torch.randn(n)

    # Normal equations
    ones = torch.ones(n, 1)
    X_aug = torch.cat([ones, X], dim=1)
    theta_ne = normal_equations_qr(X_aug, y.reshape(-1, 1))

    # nn.Linear trained with LBFGS (converges in few steps)
    model = nn.Linear(p, 1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.LBFGS(
        model.parameters(), line_search_fn="strong_wolfe"
    )

    y_col = y.reshape(-1, 1)
    for _ in range(10):
        def closure():
            optimizer.zero_grad()
            loss = criterion(model(X), y_col)
            loss.backward()
            return loss
        optimizer.step(closure)

    print(f"Normal eq bias:  {theta_ne[0].item():.6f}")
    print(f"nn.Linear bias:  {model.bias.item():.6f}")
    print(f"Close: {torch.allclose(theta_ne[1:].squeeze(), model.weight.squeeze(), atol=1e-4)}")
```

---

## Summary

| Concept | Key Formula |
|---------|-------------|
| Normal equations | $\mathbf{X}^\top\mathbf{X}\boldsymbol{\theta}^* = \mathbf{X}^\top\mathbf{y}$ |
| Closed-form solution | $\boldsymbol{\theta}^* = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}$ |
| Projection matrix | $\mathbf{P} = \mathbf{X}(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top$ |
| Orthogonality | $\mathbf{X}^\top(\mathbf{y} - \hat{\mathbf{y}}) = \mathbf{0}$ |
| ANOVA | $\text{SS}_{\text{tot}} = \text{SS}_{\text{reg}} + \text{SS}_{\text{res}}$ |
| $R^2$ (geometric) | $\cos^2\theta$ between centred $\mathbf{y}$ and $\hat{\mathbf{y}}$ |
| Recommended solver | `torch.linalg.lstsq()` / `np.linalg.solve()` |

---

## References

1. Strang, G. (2019). *Linear Algebra and Learning from Data*, Ch. I.4.
2. Golub, G. H. & Van Loan, C. F. (2013). *Matrix Computations*.
3. Petersen, K. B. & Pedersen, M. S. *The Matrix Cookbook*, §§2–5.
4. Lay, D. C. (2016). *Linear Algebra and Its Applications*, Ch. 6.
