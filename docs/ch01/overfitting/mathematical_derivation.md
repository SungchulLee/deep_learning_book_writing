# Mathematical Derivation of Bias-Variance Decomposition

## Learning Objectives

By the end of this section, you will be able to:

- Follow the rigorous mathematical derivation of the bias-variance decomposition
- Understand alternative derivation approaches including matrix formulations
- Derive bias-variance for specific model classes (linear regression, ridge regression)
- Connect the decomposition to statistical concepts like the Cramér-Rao bound
- Appreciate the assumptions required and when they break down

## Prerequisites

- Linear algebra (matrix operations, eigendecomposition)
- Probability theory (expectation, variance, covariance)
- Basic calculus (partial derivatives)
- Understanding of linear regression

---

## 1. Rigorous Setup and Assumptions

### 1.1 Formal Problem Statement

Let $(\mathcal{X}, \mathcal{Y})$ be the input-output space where $\mathcal{X} \subseteq \mathbb{R}^d$ and $\mathcal{Y} \subseteq \mathbb{R}$.

**Assumption 1 (Data Generating Process):**
Data is generated according to:
$$Y = f(X) + \varepsilon$$

where $f: \mathcal{X} \to \mathbb{R}$ is a fixed, unknown function.

**Assumption 2 (Noise Properties):**
The noise $\varepsilon$ satisfies:
- $\mathbb{E}[\varepsilon | X] = 0$ (zero conditional mean)
- $\text{Var}(\varepsilon | X) = \sigma^2$ (homoscedasticity)
- $\varepsilon \perp X$ (independence from features)

**Assumption 3 (Random Training Data):**
The training set $\mathcal{D} = \{(X_i, Y_i)\}_{i=1}^n$ consists of i.i.d. samples from the joint distribution of $(X, Y)$.

**Assumption 4 (Learner):**
A learning algorithm $\mathcal{A}$ maps training data to a hypothesis:
$$\hat{f} = \mathcal{A}(\mathcal{D})$$

The learned function $\hat{f}$ is thus a random variable through its dependence on $\mathcal{D}$.

### 1.2 Sources of Randomness

It's crucial to distinguish three sources of randomness:

1. **Training data randomness**: $\mathcal{D}$ is random
2. **Test input randomness**: Test point $X_0$ is random  
3. **Test noise randomness**: Test noise $\varepsilon_0$ is random

These are all **independent** of each other.

---

## 2. Main Derivation: Conditional on Test Point

### 2.1 Decomposition for Fixed $x_0$

**Theorem 2.1 (Bias-Variance Decomposition):**
For a fixed test point $x_0 \in \mathcal{X}$:

$$\mathbb{E}_{\mathcal{D}, \varepsilon_0}\left[(Y_0 - \hat{f}(x_0))^2\right] = \text{Bias}^2[\hat{f}(x_0)] + \text{Var}[\hat{f}(x_0)] + \sigma^2$$

where:
- $\text{Bias}[\hat{f}(x_0)] = \mathbb{E}_{\mathcal{D}}[\hat{f}(x_0)] - f(x_0)$
- $\text{Var}[\hat{f}(x_0)] = \mathbb{E}_{\mathcal{D}}\left[(\hat{f}(x_0) - \mathbb{E}_{\mathcal{D}}[\hat{f}(x_0)])^2\right]$

**Proof:**

**Step 1: Introduce convenient notation.**

Let $\bar{f}(x_0) = \mathbb{E}_{\mathcal{D}}[\hat{f}(x_0)]$ be the expected prediction.

**Step 2: Decompose the squared error.**

\begin{align}
(Y_0 - \hat{f}(x_0))^2 &= (f(x_0) + \varepsilon_0 - \hat{f}(x_0))^2 \\
&= ((f(x_0) - \bar{f}(x_0)) + (\bar{f}(x_0) - \hat{f}(x_0)) + \varepsilon_0)^2
\end{align}

**Step 3: Expand the square.**

Let $a = f(x_0) - \bar{f}(x_0)$, $b = \bar{f}(x_0) - \hat{f}(x_0)$, $c = \varepsilon_0$.

Then:
$$(a + b + c)^2 = a^2 + b^2 + c^2 + 2ab + 2ac + 2bc$$

**Step 4: Take expectation over $\mathcal{D}$ and $\varepsilon_0$.**

$$\mathbb{E}[(a + b + c)^2] = \mathbb{E}[a^2] + \mathbb{E}[b^2] + \mathbb{E}[c^2] + 2\mathbb{E}[ab] + 2\mathbb{E}[ac] + 2\mathbb{E}[bc]$$

**Step 5: Evaluate each term.**

**Term $\mathbb{E}[a^2]$:** $a = f(x_0) - \bar{f}(x_0)$ is a constant.
$$\mathbb{E}[a^2] = (f(x_0) - \bar{f}(x_0))^2 = \text{Bias}^2$$

**Term $\mathbb{E}[b^2]$:** $b = \bar{f}(x_0) - \hat{f}(x_0)$
$$\mathbb{E}[b^2] = \mathbb{E}[(\bar{f}(x_0) - \hat{f}(x_0))^2] = \text{Var}[\hat{f}(x_0)]$$

**Term $\mathbb{E}[c^2]$:**
$$\mathbb{E}[\varepsilon_0^2] = \text{Var}(\varepsilon_0) + (\mathbb{E}[\varepsilon_0])^2 = \sigma^2 + 0 = \sigma^2$$

**Term $\mathbb{E}[ab]$:** Since $a$ is constant w.r.t. $\mathcal{D}$:
$$\mathbb{E}[ab] = a \cdot \mathbb{E}[b] = (f(x_0) - \bar{f}(x_0)) \cdot \mathbb{E}[\bar{f}(x_0) - \hat{f}(x_0)] = a \cdot 0 = 0$$

**Term $\mathbb{E}[ac]$:** Since $a$ is constant and $\varepsilon_0 \perp \mathcal{D}$:
$$\mathbb{E}[ac] = a \cdot \mathbb{E}[\varepsilon_0] = 0$$

**Term $\mathbb{E}[bc]$:** Since $\varepsilon_0 \perp \mathcal{D}$:
$$\mathbb{E}[bc] = \mathbb{E}[b] \cdot \mathbb{E}[c] = 0 \cdot 0 = 0$$

**Step 6: Combine terms.**

$$\mathbb{E}[(Y_0 - \hat{f}(x_0))^2] = \text{Bias}^2 + \text{Var} + \sigma^2 \quad \blacksquare$$

---

## 3. Integrated Risk Formulation

### 3.1 Average Over Test Distribution

When we also average over the test point distribution:

$$R(\hat{f}) = \mathbb{E}_{X_0, \mathcal{D}, \varepsilon_0}[(Y_0 - \hat{f}(X_0))^2]$$

**Theorem 3.1 (Integrated Bias-Variance):**

$$R(\hat{f}) = \mathbb{E}_{X_0}[\text{Bias}^2[\hat{f}(X_0)]] + \mathbb{E}_{X_0}[\text{Var}[\hat{f}(X_0)]] + \sigma^2$$

**Proof:** Apply the tower property and Theorem 2.1:

\begin{align}
R(\hat{f}) &= \mathbb{E}_{X_0}\left[\mathbb{E}_{\mathcal{D}, \varepsilon_0}\left[(Y_0 - \hat{f}(X_0))^2 | X_0\right]\right] \\
&= \mathbb{E}_{X_0}\left[\text{Bias}^2[\hat{f}(X_0)] + \text{Var}[\hat{f}(X_0)] + \sigma^2\right] \\
&= \mathbb{E}_{X_0}[\text{Bias}^2] + \mathbb{E}_{X_0}[\text{Var}] + \sigma^2 \quad \blacksquare
\end{align}

---

## 4. Linear Regression Case Study

### 4.1 Setup

Consider linear regression with design matrix $\mathbf{X} \in \mathbb{R}^{n \times d}$ and targets $\mathbf{y} \in \mathbb{R}^n$.

True model: $\mathbf{y} = \mathbf{X}\boldsymbol{\beta}^* + \boldsymbol{\varepsilon}$

where $\boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I}_n)$.

OLS estimator: $\hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$

### 4.2 Bias of OLS

**Proposition 4.1:** The OLS estimator is unbiased when the true model is linear.

**Proof:**
\begin{align}
\mathbb{E}[\hat{\boldsymbol{\beta}}] &= \mathbb{E}[(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}] \\
&= (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbb{E}[\mathbf{y}] \\
&= (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T(\mathbf{X}\boldsymbol{\beta}^*) \\
&= \boldsymbol{\beta}^* \quad \blacksquare
\end{align}

For predictions at test point $\mathbf{x}_0$:
$$\text{Bias}[\hat{f}(\mathbf{x}_0)] = \mathbf{x}_0^T\mathbb{E}[\hat{\boldsymbol{\beta}}] - \mathbf{x}_0^T\boldsymbol{\beta}^* = 0$$

!!! note "Model Misspecification"
    If the true function $f$ is not linear, the OLS estimator is biased. The bias equals $\mathbb{E}[\hat{f}(\mathbf{x}_0)] - f(\mathbf{x}_0)$.

### 4.3 Variance of OLS

**Proposition 4.2:** The covariance matrix of the OLS estimator is:
$$\text{Cov}(\hat{\boldsymbol{\beta}}) = \sigma^2 (\mathbf{X}^T\mathbf{X})^{-1}$$

**Proof:**
\begin{align}
\text{Cov}(\hat{\boldsymbol{\beta}}) &= \text{Cov}((\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}) \\
&= (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T \text{Cov}(\mathbf{y}) \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1} \\
&= (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T (\sigma^2\mathbf{I}) \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1} \\
&= \sigma^2 (\mathbf{X}^T\mathbf{X})^{-1} \quad \blacksquare
\end{align}

For predictions:
$$\text{Var}[\hat{f}(\mathbf{x}_0)] = \text{Var}[\mathbf{x}_0^T\hat{\boldsymbol{\beta}}] = \mathbf{x}_0^T \text{Cov}(\hat{\boldsymbol{\beta}}) \mathbf{x}_0 = \sigma^2 \mathbf{x}_0^T(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{x}_0$$

### 4.4 Total Expected Error for OLS

**Theorem 4.3:** For OLS with correct model specification:

$$\mathbb{E}[(Y_0 - \hat{f}(\mathbf{x}_0))^2] = \sigma^2\left(1 + \mathbf{x}_0^T(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{x}_0\right)$$

This decomposes as:
- **Bias²** = 0
- **Variance** = $\sigma^2 \mathbf{x}_0^T(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{x}_0$
- **Irreducible Error** = $\sigma^2$

---

## 5. Ridge Regression: Explicit Tradeoff

### 5.1 Ridge Estimator

Ridge regression adds L2 regularization:

$$\hat{\boldsymbol{\beta}}_{\text{ridge}} = (\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}$$

### 5.2 Bias of Ridge

**Proposition 5.1:** Ridge regression is biased:

$$\mathbb{E}[\hat{\boldsymbol{\beta}}_{\text{ridge}}] = (\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{X}\boldsymbol{\beta}^* \neq \boldsymbol{\beta}^*$$

The bias is:
$$\text{Bias}[\hat{\boldsymbol{\beta}}_{\text{ridge}}] = -\lambda(\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\boldsymbol{\beta}^*$$

As $\lambda \to 0$, bias → 0.
As $\lambda \to \infty$, $\hat{\boldsymbol{\beta}}_{\text{ridge}} \to \mathbf{0}$.

### 5.3 Variance of Ridge

**Proposition 5.2:** The covariance matrix is:

$$\text{Cov}(\hat{\boldsymbol{\beta}}_{\text{ridge}}) = \sigma^2(\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{X}(\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}$$

As $\lambda$ increases, variance decreases.

### 5.4 Eigenvalue Analysis

To gain deeper insight, consider the SVD of $\mathbf{X} = \mathbf{U}\mathbf{D}\mathbf{V}^T$ where $\mathbf{D} = \text{diag}(d_1, \ldots, d_p)$ contains the singular values.

The ridge estimator becomes:
$$\hat{\boldsymbol{\beta}}_{\text{ridge}} = \sum_{j=1}^{p} \frac{d_j}{d_j^2 + \lambda} \mathbf{u}_j^T\mathbf{y} \cdot \mathbf{v}_j$$

Compare to OLS:
$$\hat{\boldsymbol{\beta}}_{\text{OLS}} = \sum_{j=1}^{p} \frac{1}{d_j} \mathbf{u}_j^T\mathbf{y} \cdot \mathbf{v}_j$$

**Key insight:** Ridge shrinks coefficients more for directions with small singular values (poorly determined directions).

### 5.5 Optimal Ridge Parameter

The total MSE for ridge can be minimized by choosing:

$$\lambda^* = \frac{p\sigma^2}{\boldsymbol{\beta}^{*T}\boldsymbol{\beta}^*}$$

This trades off bias and variance optimally.

---

## 6. Matrix Formulation of Total MSE

### 6.1 General Linear Estimator

Consider any linear estimator of the form:
$$\hat{\boldsymbol{\beta}} = \mathbf{C}\mathbf{y}$$

where $\mathbf{C} \in \mathbb{R}^{d \times n}$ is a fixed matrix.

**Total MSE Matrix:**
$$\text{MSE}(\hat{\boldsymbol{\beta}}) = \mathbb{E}[(\hat{\boldsymbol{\beta}} - \boldsymbol{\beta}^*)(\hat{\boldsymbol{\beta}} - \boldsymbol{\beta}^*)^T]$$

**Decomposition:**
$$\text{MSE}(\hat{\boldsymbol{\beta}}) = \text{Cov}(\hat{\boldsymbol{\beta}}) + \text{Bias}(\hat{\boldsymbol{\beta}})\text{Bias}(\hat{\boldsymbol{\beta}})^T$$

where:
- $\text{Cov}(\hat{\boldsymbol{\beta}}) = \sigma^2 \mathbf{C}\mathbf{C}^T$
- $\text{Bias}(\hat{\boldsymbol{\beta}}) = (\mathbf{C}\mathbf{X} - \mathbf{I})\boldsymbol{\beta}^*$

### 6.2 Scalar Summary: Total MSE

The scalar total MSE is:
$$\text{Total MSE} = \text{tr}(\text{MSE}(\hat{\boldsymbol{\beta}})) = \sigma^2\text{tr}(\mathbf{C}\mathbf{C}^T) + \|\text{Bias}\|^2$$

| Estimator | $\mathbf{C}$ | Bias | Variance ($\propto$ tr) |
|-----------|--------------|------|-------------------------|
| OLS | $(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T$ | 0 | $\sigma^2 \text{tr}((\mathbf{X}^T\mathbf{X})^{-1})$ |
| Ridge | $(\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T$ | $\neq 0$ | $<$ OLS variance |

---

## 7. Beyond Squared Error Loss

### 7.1 General Loss Functions

The bias-variance decomposition is specific to squared error loss. For other losses, analogous decompositions exist but are more complex.

**0-1 Loss (Classification):**
$$\mathbb{E}[\mathbf{1}\{\hat{Y} \neq Y\}] = P(\hat{Y} \neq Y)$$

This doesn't decompose cleanly, but a related decomposition exists for the **Bregman divergence**.

### 7.2 Bregman Divergence Decomposition

For Bregman divergence $D_\phi(y, \hat{y}) = \phi(y) - \phi(\hat{y}) - \phi'(\hat{y})(y - \hat{y})$:

$$\mathbb{E}[D_\phi(Y, \hat{f})] = \mathbb{E}[D_\phi(f, \bar{f})] + \mathbb{E}[D_\phi(\bar{f}, \hat{f})] + \mathbb{E}[D_\phi(Y, f)]$$

This generalizes the squared error case (where $\phi(x) = x^2$).

---

## 8. Connection to Information Theory

### 8.1 Cramér-Rao Bound

The variance of any unbiased estimator is bounded below by:

$$\text{Var}(\hat{\theta}) \geq \frac{1}{I(\theta)}$$

where $I(\theta)$ is the Fisher information.

**Connection:** This shows that reducing bias to zero forces a minimum level of variance. The bias-variance tradeoff says we might prefer some bias to reduce variance below this bound.

### 8.2 James-Stein Estimator

The James-Stein theorem shows that for $d \geq 3$ dimensions, biased estimators can dominate (have lower total MSE than) unbiased estimators uniformly over all parameter values.

$$\hat{\boldsymbol{\mu}}_{JS} = \left(1 - \frac{(d-2)\sigma^2}{\|\mathbf{y}\|^2}\right)\mathbf{y}$$

This is a concrete example where introducing bias reduces total error.

---

## 9. PyTorch Implementation: Matrix Computation

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def compute_linear_regression_bias_variance(X_train, y_train, X_test, 
                                             beta_true, sigma_sq):
    """
    Compute exact bias and variance for linear regression.
    
    Uses closed-form formulas rather than Monte Carlo.
    
    Args:
        X_train: Design matrix (n, d)
        y_train: Targets (n,)
        X_test: Test points (m, d)
        beta_true: True parameters (d,)
        sigma_sq: Noise variance
        
    Returns:
        bias_squared: (m,) squared bias at each test point
        variance: (m,) variance at each test point
        predictions: (m,) predicted values
    """
    # Compute OLS estimator
    XtX = X_train.T @ X_train
    XtX_inv = torch.linalg.inv(XtX)
    beta_hat = XtX_inv @ X_train.T @ y_train
    
    # Predictions
    predictions = X_test @ beta_hat
    
    # True values (at test points, assuming linear model)
    f_true = X_test @ beta_true
    
    # Bias = E[f_hat] - f_true
    # For OLS, E[beta_hat] = beta_true, so bias = 0
    bias = X_test @ beta_true - f_true  # Should be 0
    bias_squared = bias ** 2
    
    # Variance = sigma^2 * x^T (X^T X)^{-1} x for each test point
    variance = sigma_sq * torch.sum((X_test @ XtX_inv) * X_test, dim=1)
    
    return bias_squared, variance, predictions

def compute_ridge_bias_variance(X_train, y_train, X_test, 
                                 beta_true, sigma_sq, lambda_reg):
    """
    Compute exact bias and variance for ridge regression.
    
    Ridge introduces bias but reduces variance.
    """
    n, d = X_train.shape
    XtX = X_train.T @ X_train
    ridge_matrix = XtX + lambda_reg * torch.eye(d)
    ridge_inv = torch.linalg.inv(ridge_matrix)
    
    # Ridge estimator
    beta_ridge = ridge_inv @ X_train.T @ y_train
    
    # Expected value of ridge estimator
    # E[beta_ridge] = (XtX + lambda*I)^{-1} XtX beta_true
    E_beta_ridge = ridge_inv @ XtX @ beta_true
    
    # Bias at each test point
    f_true = X_test @ beta_true
    E_f_hat = X_test @ E_beta_ridge
    bias = E_f_hat - f_true
    bias_squared = bias ** 2
    
    # Variance: Cov(beta_ridge) = sigma^2 * ridge_inv @ XtX @ ridge_inv
    cov_beta = sigma_sq * ridge_inv @ XtX @ ridge_inv
    
    # Variance at test points: Var(x^T beta) = x^T Cov(beta) x
    variance = torch.sum((X_test @ cov_beta) * X_test, dim=1)
    
    return bias_squared, variance, beta_ridge

def analyze_ridge_tradeoff():
    """
    Demonstrate the bias-variance tradeoff in ridge regression.
    """
    torch.manual_seed(42)
    
    # Generate synthetic data
    n, d = 50, 10
    sigma = 0.5
    sigma_sq = sigma ** 2
    
    # True parameters (some large, some small)
    beta_true = torch.randn(d)
    beta_true = beta_true / torch.norm(beta_true) * 3  # Normalize
    
    # Design matrix
    X_train = torch.randn(n, d)
    epsilon = torch.randn(n) * sigma
    y_train = X_train @ beta_true + epsilon
    
    # Test points
    X_test = torch.randn(100, d)
    
    # Analyze across lambda values
    lambdas = torch.logspace(-3, 3, 50)
    
    results = {
        'lambda': lambdas.numpy(),
        'avg_bias_sq': [],
        'avg_variance': [],
        'avg_total_mse': []
    }
    
    for lam in lambdas:
        bias_sq, var, _ = compute_ridge_bias_variance(
            X_train, y_train, X_test, beta_true, sigma_sq, lam
        )
        
        results['avg_bias_sq'].append(bias_sq.mean().item())
        results['avg_variance'].append(var.mean().item())
        results['avg_total_mse'].append((bias_sq + var + sigma_sq).mean().item())
    
    return results

def plot_ridge_tradeoff(results):
    """Plot the bias-variance tradeoff for ridge regression."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.loglog(results['lambda'], results['avg_bias_sq'], 
              'b-', linewidth=2, label='Bias²')
    ax.loglog(results['lambda'], results['avg_variance'], 
              'r-', linewidth=2, label='Variance')
    ax.loglog(results['lambda'], results['avg_total_mse'], 
              'g-', linewidth=2, label='Total MSE')
    
    # Find optimal lambda
    opt_idx = np.argmin(results['avg_total_mse'])
    opt_lambda = results['lambda'][opt_idx]
    ax.axvline(opt_lambda, color='purple', linestyle='--', 
               label=f'Optimal λ = {opt_lambda:.4f}')
    
    ax.set_xlabel('Regularization Parameter λ (log scale)', fontsize=12)
    ax.set_ylabel('Error (log scale)', fontsize=12)
    ax.set_title('Ridge Regression: Bias-Variance Tradeoff', 
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# Run analysis
if __name__ == "__main__":
    import numpy as np
    
    results = analyze_ridge_tradeoff()
    fig = plot_ridge_tradeoff(results)
    plt.savefig('ridge_bias_variance.png', dpi=150, bbox_inches='tight')
    plt.show()
```

---

## 10. When the Decomposition Breaks Down

### 10.1 Model Misspecification

If we fit a linear model but the true function is nonlinear:

$$Y = g(X) + \varepsilon$$

where $g$ is nonlinear, the bias becomes:

$$\text{Bias}[\hat{f}(x_0)] = \mathbf{x}_0^T\mathbb{E}[\hat{\boldsymbol{\beta}}] - g(x_0) \neq 0$$

The bias now depends on how well a linear function can approximate $g$.

### 10.2 Heteroscedastic Noise

If $\text{Var}(\varepsilon | X = x) = \sigma^2(x)$ varies with $x$, the decomposition still holds but the irreducible error becomes:

$$\mathbb{E}[\sigma^2(X_0)]$$

when averaged over test points.

### 10.3 Non-i.i.d. Data

For dependent data (time series, spatial data), the derivation requires modification. The key complication is that $\text{Cov}(\varepsilon_i, \varepsilon_j) \neq 0$, which affects the variance calculations.

### 10.4 Double Descent

In highly overparameterized models (parameters >> samples), classical bias-variance analysis fails. The variance can actually decrease as we add more parameters beyond the interpolation threshold, leading to double descent behavior.

---

## 11. Summary

### Key Mathematical Results

1. **The fundamental decomposition** expresses expected squared error as the sum of three independent terms

2. **For linear regression**, we have closed-form expressions:
   - OLS: Zero bias, variance = $\sigma^2 \mathbf{x}_0^T(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{x}_0$
   - Ridge: Bias = $-\lambda(\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\boldsymbol{\beta}^*$

3. **The tradeoff is explicit** in ridge regression: larger $\lambda$ increases bias but decreases variance

4. **Matrix formulation** provides a unified view through the MSE matrix

5. **Extensions** to general losses use Bregman divergence

### Assumptions to Remember

- Noise is zero-mean and independent of features
- Training data is i.i.d.
- The decomposition is for squared error loss
- Modern deep learning can violate classical predictions

---

## Exercises

### Exercise 1: Ridge Optimal Lambda
Derive the optimal $\lambda^*$ that minimizes total MSE for ridge regression in the case where $\mathbf{X}^T\mathbf{X} = \mathbf{I}$ (orthonormal design).

### Exercise 2: PCA Regression
Derive the bias-variance decomposition for principal component regression, where we first project onto the top $k$ principal components.

### Exercise 3: LASSO
Explain qualitatively why the bias-variance tradeoff for LASSO is more complex than ridge (hint: LASSO doesn't have a closed-form solution).

### Exercise 4: Bayes Optimal
Show that the Bayes optimal predictor $f^*(x) = \mathbb{E}[Y|X=x]$ has zero bias. What is its variance?

---

## References

1. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer. Chapter 7.

2. Wasserman, L. (2004). *All of Statistics*. Springer. Chapter 7.

3. Hoerl, A. E., & Kennard, R. W. (1970). Ridge Regression: Biased Estimation for Nonorthogonal Problems. *Technometrics*, 12(1), 55-67.

4. Stein, C. (1956). Inadmissibility of the usual estimator for the mean of a multivariate normal distribution. *Proceedings of the Third Berkeley Symposium*, 1, 197-206.
