# Linear Regression

## Overview

Linear regression is the foundational supervised learning algorithm that models
the relationship between input features and a continuous target variable as a
linear function.  It admits both a **closed-form** solution (normal equations)
and an **iterative** solution (gradient descent), making it an ideal starting
point for understanding optimisation in machine learning.

This section develops the theory from first principles — model specification,
probabilistic interpretation, solution methods, and extensions — with
accompanying NumPy, PyTorch, and scikit-learn implementations throughout.

---

## 1. The Linear Model

### 1.1 Univariate Case

For a single input feature $x$, the model is

$$
\hat{y} = wx + b,
$$

where $w$ is the **weight** (slope) and $b$ is the **bias** (intercept).

### 1.2 Multivariate Case

For $p$ input features $\mathbf{x} = [x_1, x_2, \ldots, x_p]^\top$, the model
generalises to

$$
\hat{y}
= \mathbf{w}^\top \mathbf{x} + b
= \sum_{j=1}^{p} w_j x_j + b.
$$

In matrix form for $n$ samples with design matrix
$\mathbf{X} \in \mathbb{R}^{n \times p}$:

$$
\hat{\mathbf{y}} = \mathbf{X}\mathbf{w} + b\mathbf{1}.
$$

### 1.3 Compact Notation (Bias Absorption)

Augmenting the feature vector with a constant 1 absorbs the bias into the
parameter vector:

$$
\tilde{\mathbf{x}} = [1,\; x_1,\; \ldots,\; x_p]^\top,
\qquad
\boldsymbol{\theta} = [b,\; w_1,\; \ldots,\; w_p]^\top,
$$

so that $\hat{y} = \boldsymbol{\theta}^\top \tilde{\mathbf{x}}$.  For the full
dataset the design matrix becomes
$\mathbf{X} = [\mathbf{1} \mid \mathbf{X}_{\text{raw}}]
\in \mathbb{R}^{n \times (p+1)}$ and

$$
\hat{\mathbf{y}} = \mathbf{X}\boldsymbol{\theta}.
$$

We use this compact notation for the closed-form derivations and switch to
separate $(\mathbf{w}, b)$ notation for gradient-descent implementations where
PyTorch handles the bias automatically via `nn.Linear`.

---

## 2. Probabilistic Formulation

### 2.1 Generative Story

We assume each observation is generated as

$$
y_i = \mathbf{w}^\top \mathbf{x}_i + b + \epsilon_i,
\qquad
\epsilon_i \sim \mathcal{N}(0, \sigma^2),
$$

with the noise terms $\epsilon_i$ independent and identically distributed.
This implies the conditional distribution

$$
y_i \mid \mathbf{x}_i \;\sim\; \mathcal{N}\!\bigl(\mathbf{w}^\top \mathbf{x}_i + b,\;\sigma^2\bigr).
$$

### 2.2 From MLE to MSE

The log-likelihood for $n$ observations is

$$
\ell(\mathbf{w}, b, \sigma^2)
= -\frac{n}{2}\ln(2\pi\sigma^2)
  - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(y_i - \mathbf{w}^\top\mathbf{x}_i - b)^2.
$$

For fixed $\sigma^2$, maximising $\ell$ with respect to $(\mathbf{w}, b)$ is
equivalent to minimising the **Mean Squared Error** (MSE):

$$
\mathcal{L}(\mathbf{w}, b)
= \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
= \frac{1}{n}\|\mathbf{y} - \mathbf{X}\boldsymbol{\theta}\|^2.
$$

!!! tip "Key Insight"
    MSE is **not** an arbitrary loss — it is the negative log-likelihood (up
    to constants) under Gaussian noise.  This gives the MSE estimator all the
    desirable asymptotic properties of maximum likelihood (consistency,
    efficiency).

### 2.3 MLE for Noise Variance

Setting $\partial \ell / \partial \sigma^2 = 0$ yields the MLE estimate

$$
\hat{\sigma}^2_{\text{MLE}}
= \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2.
$$

This is **biased**; the unbiased estimator divides by $n - p - 1$ (degrees of
freedom).

---

## 3. Classical Assumptions (Gauss–Markov)

For the OLS estimator to be the **Best Linear Unbiased Estimator** (BLUE), the
following assumptions must hold.

### 3.1 Linearity in Parameters

$$
E[y \mid \mathbf{x}] = \mathbf{w}^\top \mathbf{x} + b.
$$

The model may include nonlinear *features* (e.g.\ polynomial terms) as long as
it remains linear in the *parameters*.

### 3.2 Strict Exogeneity

$$
E[\epsilon \mid \mathbf{X}] = \mathbf{0}.
$$

Features are uncorrelated with the error term — no omitted variable bias, no
measurement error in features.

### 3.3 Homoscedasticity

$$
\operatorname{Var}(\epsilon_i \mid \mathbf{x}_i) = \sigma^2
\quad \forall\, i.
$$

Constant error variance across observations.  Violations (heteroscedasticity)
are common in financial data where variance scales with magnitude.

### 3.4 No Autocorrelation

$$
\operatorname{Cov}(\epsilon_i, \epsilon_j) = 0
\quad \forall\, i \neq j.
$$

Crucial for time-series data; violations detected via the Durbin–Watson test.

### 3.5 Full Column Rank

$$
\operatorname{rank}(\mathbf{X}) = p + 1.
$$

Ensures $\mathbf{X}^\top\mathbf{X}$ is invertible; violated by perfect
multicollinearity.  Near-multicollinearity causes numerical instability and
inflated parameter variance.

---

## 4. Simple Linear Regression ($p = 1$)

Before the matrix formulation, consider the scalar case.  We seek the line
$\hat{y}_i = wx_i + b$ minimising

$$
\mathcal{L}(w, b)
= \frac{1}{n}\sum_{i=1}^{n}(y_i - wx_i - b)^2.
$$

### 4.1 Optimal Parameters

Setting partial derivatives to zero gives

$$
b^* = \bar{y} - w^*\bar{x},
\qquad
\boxed{
  w^* = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}
             {\sum_{i=1}^{n}(x_i - \bar{x})^2}
      = \frac{S_{xy}}{S_{xx}}.
}
$$

### 4.2 Connection to Correlation

Define the sample standard deviations $s_x$, $s_y$ and the sample correlation
coefficient

$$
\rho = \frac{S_{xy}}{n\, s_x\, s_y}.
$$

Then

$$
w^* = \rho\,\frac{s_y}{s_x},
$$

so the optimal slope is the correlation scaled by the ratio of standard
deviations.  The prediction for a new point $x_*$ is

$$
\hat{y}_* = \bar{y} + \rho\,\frac{s_y}{s_x}\,(x_* - \bar{x}).
$$

### 4.3 Coefficient of Determination

For simple linear regression the $R^2$ statistic satisfies

$$
R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}} = \rho^2.
$$

!!! warning "Multiple Regression"
    The identity $R^2 = \rho^2$ holds **only** for simple linear regression.
    For $p > 1$, $R^2$ equals the squared *multiple* correlation coefficient.

### 4.4 Second-Order Condition

The Hessian of $\mathcal{L}$ with respect to $(w, b)$ is

$$
\mathbf{H} = \frac{2}{n}
\begin{pmatrix}
\sum x_i^2 & \sum x_i \\
\sum x_i   & n
\end{pmatrix},
$$

which is positive definite whenever the data have non-zero variance, confirming
the critical point is a global minimum.

### 4.5 Python Implementation

```python
import numpy as np


def simple_linear_regression(x: np.ndarray, y: np.ndarray):
    """Fit y = w*x + b using the correlation-form formula.

    Parameters
    ----------
    x, y : 1-D arrays of shape (n,)

    Returns
    -------
    w, b, rho, r_squared : float
    """
    x_bar, y_bar = x.mean(), y.mean()
    s_xy = np.sum((x - x_bar) * (y - y_bar))
    s_xx = np.sum((x - x_bar) ** 2)
    s_yy = np.sum((y - y_bar) ** 2)

    w = s_xy / s_xx
    b = y_bar - w * x_bar

    rho = s_xy / np.sqrt(s_xx * s_yy)
    r_squared = rho ** 2

    return w, b, rho, r_squared
```

---

## 5. From Scalar to Matrix Form

Writing the model with a bias column
$\mathbf{X} = [\mathbf{1} \mid \mathbf{x}] \in \mathbb{R}^{n \times 2}$ and
parameter vector $\boldsymbol{\theta} = (b, w)^\top$, the optimality condition
$\nabla_{\boldsymbol{\theta}} \mathcal{L} = \mathbf{0}$ yields the **normal
equations**

$$
\mathbf{X}^\top\mathbf{X}\,\boldsymbol{\theta}^*
= \mathbf{X}^\top\mathbf{y}.
$$

This is the $p$-dimensional generalisation of the scalar formulas derived
above.  Two solution strategies follow:

| Strategy | Page | When to Use |
|----------|------|-------------|
| Closed-form (normal equations) | [Closed-Form Solution](closed_form.md) | $p < 10{,}000$, exact solution needed |
| Gradient descent | [Gradient Descent Solution](gd_solution.md) | Large $p$ or $n$, mini-batch training |

---

## 6. Multiple Outputs

Standard linear regression maps $\mathbf{x} \in \mathbb{R}^p$ to a scalar
$y$.  When predicting $q$ targets simultaneously, the model becomes

$$
\hat{\mathbf{Y}} = \mathbf{X}\mathbf{W} + \mathbf{1}\mathbf{b}^\top,
\qquad
\mathbf{W} \in \mathbb{R}^{p \times q},\;\;
\hat{\mathbf{Y}} \in \mathbb{R}^{n \times q},
$$

with Frobenius-norm MSE loss

$$
\mathcal{L}
= \frac{1}{nq}\|\mathbf{Y} - \hat{\mathbf{Y}}\|_F^2.
$$

The normal-equation solution extends column-wise:

$$
\mathbf{B}^*
= (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{Y}
\;\in\; \mathbb{R}^{(p+1) \times q}.
$$

Each column of $\mathbf{B}^*$ is the single-output solution for that target —
the outputs **decouple** because they share the same design matrix.

!!! info "When Multi-Output Helps"
    When outputs are independent, fitting $q$ separate models gives identical
    results.  The multi-output formulation becomes beneficial when you add
    **shared structure** — e.g.\ a shared hidden layer in a neural network or a
    common covariance structure in multivariate regression.

### PyTorch: `nn.Linear(p, q)`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(42)
n, p, q = 500, 3, 2
X = torch.randn(n, p)
W_true = torch.tensor([[2.0, -1.0], [0.5, 1.5], [-0.3, 0.8]])
b_true = torch.tensor([1.0, -2.0])
Y = X @ W_true + b_true + 0.2 * torch.randn(n, q)

model = nn.Linear(p, q)   # weight shape: (q, p) = (2, 3)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loader = DataLoader(TensorDataset(X, Y), batch_size=32, shuffle=True)

for epoch in range(200):
    for X_b, Y_b in loader:
        loss = F.mse_loss(model(X_b), Y_b)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## 7. Data Requirements

### 7.1 Feature Scaling

Linear regression is sensitive to feature scales when using gradient-based
optimisation.  Standardisation ($z = (x - \mu) / \sigma$) ensures all features
contribute equally to the gradient and makes learning-rate selection easier.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # use training statistics!
```

!!! warning "Regularised Models"
    Scaling is **essential** before Ridge, Lasso, or ElasticNet — the penalty
    is magnitude-dependent and would otherwise disproportionately shrink
    coefficients of high-variance features.

### 7.2 Tensor Shapes in PyTorch

```python
# X: (n_samples, n_features) — always 2-D
# y: (n_samples, 1)          — column vector for consistency

X = torch.randn(100, 8)
y = torch.randn(100, 1)

# Common mistake: y as shape (100,) leads to broadcasting issues
y_flat = torch.randn(100)
y_correct = y_flat.reshape(-1, 1)
```

---

## 8. Verification of Assumptions

### 8.1 Residual Diagnostics

```python
import matplotlib.pyplot as plt

def plot_residual_diagnostics(y_true, y_pred):
    """Residuals-vs-fitted and histogram."""
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].scatter(y_pred, residuals, alpha=0.5, s=10)
    axes[0].axhline(0, color="r", ls="--")
    axes[0].set_xlabel("Fitted Values")
    axes[0].set_ylabel("Residuals")
    axes[0].set_title("Residuals vs Fitted")

    axes[1].hist(residuals, bins=30, edgecolor="black", density=True)
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Residual Distribution")

    plt.tight_layout()
    return fig
```

---

## 9. Section Roadmap

| Page | Content |
|------|---------|
| [Closed-Form Solution](closed_form.md) | Vector calculus prerequisites, normal equations, geometric interpretation, QR/SVD solvers |
| [Gradient Descent Solution](gd_solution.md) | MSE–NLL connection, batch/mini-batch/SGD, autograd, `nn.Linear` pipeline |
| [Polynomial Features](polynomial_features.md) | Nonlinear feature maps, bias–variance trade-off, cross-validation |
| [Ridge Regression](ridge_regression.md) | $\ell_2$ regularisation, Bayesian interpretation, shrinkage geometry |
| [Lasso Regression](lasso_regression.md) | $\ell_1$ regularisation, sparsity, ElasticNet, coordinate descent |

---

## 10. Financial Application: CAPM Beta

A classic application of simple linear regression in finance is the **Capital
Asset Pricing Model** (CAPM):

$$
R_{\text{WMT}} = \alpha + \beta\, R_{\text{SPY}} + \varepsilon,
$$

where $\beta$ measures the sensitivity of Walmart's return to the broad market.

| $\beta$ Value | Interpretation |
|---------------|----------------|
| $\beta = 1$ | Moves with the market |
| $\beta > 1$ | More volatile than the market (aggressive) |
| $\beta < 1$ | Less volatile than the market (defensive) |

WMT is typically defensive with $\beta \approx 0.4$–$0.6$.  See
`code/wmt_on_spy.py` for a complete implementation using NumPy, scikit-learn,
and PyTorch, including rolling-window beta estimation.

---

## References

1. Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of
   Statistical Learning*, Ch. 3.
2. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*, Ch. 3.
3. Freedman, D., Pisani, R. & Purves, R. *Statistics* (4th ed.), Ch. 10–12.
4. Murphy, K. P. (2022). *Probabilistic Machine Learning: An Introduction*,
   Ch. 4.
