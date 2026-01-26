# Normal Equations (Closed-Form Solution)

## Overview

Unlike many machine learning problems that require iterative optimization, linear regression has a beautiful closed-form solution known as the **Normal Equations**. This analytical solution provides the optimal parameters directly, offering both theoretical insight and practical utility for small to medium-sized datasets.

## Derivation

### Setting Up the Problem

Given:
- Design matrix $\mathbf{X} \in \mathbb{R}^{n \times d}$ (with bias column prepended)
- Target vector $\mathbf{y} \in \mathbb{R}^{n}$
- Parameters $\boldsymbol{\theta} \in \mathbb{R}^{d}$ (includes bias)

The objective is to minimize the Sum of Squared Errors (SSE):

$$J(\boldsymbol{\theta}) = \|\mathbf{y} - \mathbf{X}\boldsymbol{\theta}\|_2^2 = (\mathbf{y} - \mathbf{X}\boldsymbol{\theta})^T(\mathbf{y} - \mathbf{X}\boldsymbol{\theta})$$

### Expanding the Objective

$$J(\boldsymbol{\theta}) = \mathbf{y}^T\mathbf{y} - 2\boldsymbol{\theta}^T\mathbf{X}^T\mathbf{y} + \boldsymbol{\theta}^T\mathbf{X}^T\mathbf{X}\boldsymbol{\theta}$$

### Computing the Gradient

Using matrix calculus identities:
- $\nabla_{\boldsymbol{\theta}}(\mathbf{a}^T\boldsymbol{\theta}) = \mathbf{a}$
- $\nabla_{\boldsymbol{\theta}}(\boldsymbol{\theta}^T\mathbf{A}\boldsymbol{\theta}) = 2\mathbf{A}\boldsymbol{\theta}$ (for symmetric $\mathbf{A}$)

$$\nabla_{\boldsymbol{\theta}} J = -2\mathbf{X}^T\mathbf{y} + 2\mathbf{X}^T\mathbf{X}\boldsymbol{\theta}$$

### Setting Gradient to Zero

At the optimum:

$$\mathbf{X}^T\mathbf{X}\boldsymbol{\theta} = \mathbf{X}^T\mathbf{y}$$

These are the **Normal Equations**.

### Solving for Optimal Parameters

If $\mathbf{X}^T\mathbf{X}$ is invertible:

$$\boxed{\hat{\boldsymbol{\theta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}}$$

The matrix $(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T$ is called the **Moore-Penrose pseudoinverse** (in this case, equivalent to the left inverse).

## Why "Normal" Equations?

The residual vector $\mathbf{r} = \mathbf{y} - \mathbf{X}\hat{\boldsymbol{\theta}}$ is **orthogonal** (normal) to the column space of $\mathbf{X}$:

$$\mathbf{X}^T\mathbf{r} = \mathbf{X}^T(\mathbf{y} - \mathbf{X}\hat{\boldsymbol{\theta}}) = \mathbf{X}^T\mathbf{y} - \mathbf{X}^T\mathbf{X}\hat{\boldsymbol{\theta}} = \mathbf{0}$$

This orthogonality condition gives the equations their name.

## PyTorch Implementation

### Direct Implementation

```python
import torch
import numpy as np

def normal_equations(X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Solve linear regression using Normal Equations
    
    θ = (X^T X)^(-1) X^T y
    
    Args:
        X: Design matrix (n_samples, n_features)
           Should include a column of ones for bias
        y: Target vector (n_samples,) or (n_samples, 1)
    
    Returns:
        Optimal parameters θ (n_features,) or (n_features, 1)
    """
    # Ensure y is a column vector
    if y.dim() == 1:
        y = y.reshape(-1, 1)
    
    # X^T X
    XtX = X.T @ X
    
    # X^T y
    Xty = X.T @ y
    
    # Solve (X^T X) θ = X^T y
    # Using torch.linalg.solve is more numerically stable than explicit inverse
    theta = torch.linalg.solve(XtX, Xty)
    
    return theta

# Example usage
torch.manual_seed(42)

# Generate data
n_samples = 100
X_raw = torch.randn(n_samples, 3)

# Add bias column (column of ones)
ones = torch.ones(n_samples, 1)
X = torch.cat([ones, X_raw], dim=1)  # (100, 4)

# True parameters [bias, w1, w2, w3]
true_theta = torch.tensor([[1.0], [2.0], [-1.5], [0.5]])
noise = 0.3 * torch.randn(n_samples, 1)
y = X @ true_theta + noise

# Solve using normal equations
theta_hat = normal_equations(X, y)

print("Normal Equations Solution:")
print(f"True θ:     {true_theta.squeeze().tolist()}")
print(f"Estimated θ: {theta_hat.squeeze().tolist()}")
print(f"Max error:   {torch.max(torch.abs(theta_hat - true_theta)).item():.6f}")
```

### Numerically Stable Implementation

```python
def normal_equations_stable(X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    More numerically stable solution using QR decomposition
    
    Instead of computing (X^T X)^(-1), we use:
    X = QR  =>  R θ = Q^T y
    """
    if y.dim() == 1:
        y = y.reshape(-1, 1)
    
    # QR decomposition
    Q, R = torch.linalg.qr(X)
    
    # Solve R θ = Q^T y using back substitution
    theta = torch.linalg.solve_triangular(R, Q.T @ y, upper=True)
    
    return theta

def normal_equations_svd(X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Solution using SVD (most robust, handles rank deficiency)
    
    X = U Σ V^T  =>  θ = V Σ^(-1) U^T y
    """
    if y.dim() == 1:
        y = y.reshape(-1, 1)
    
    # Use torch.linalg.lstsq which handles all edge cases
    solution = torch.linalg.lstsq(X, y)
    
    return solution.solution

# Compare all methods
print("\nComparing numerical methods:")
theta_direct = normal_equations(X, y)
theta_qr = normal_equations_stable(X, y)
theta_svd = normal_equations_svd(X, y)

print(f"Direct:  {theta_direct.squeeze().tolist()}")
print(f"QR:      {theta_qr.squeeze().tolist()}")
print(f"SVD:     {theta_svd.squeeze().tolist()}")
print(f"All close: {torch.allclose(theta_direct, theta_qr) and torch.allclose(theta_qr, theta_svd)}")
```

### With Separate Bias Term

```python
def normal_equations_with_bias(
    X: torch.Tensor, 
    y: torch.Tensor
) -> tuple:
    """
    Solve normal equations, returning separate weight and bias
    
    Args:
        X: Features WITHOUT bias column (n_samples, n_features)
        y: Targets (n_samples,) or (n_samples, 1)
    
    Returns:
        (weights, bias) tuple
    """
    n_samples = X.shape[0]
    
    # Add bias column
    ones = torch.ones(n_samples, 1)
    X_augmented = torch.cat([ones, X], dim=1)
    
    # Solve
    theta = normal_equations_stable(X_augmented, y)
    
    # Split into bias and weights
    bias = theta[0]
    weights = theta[1:]
    
    return weights, bias

# Example
X_features = torch.randn(100, 3)
y_target = 2 * X_features[:, 0] - X_features[:, 1] + 0.5 * X_features[:, 2] + 1.0 + 0.2 * torch.randn(100)

weights, bias = normal_equations_with_bias(X_features, y_target.reshape(-1, 1))

print("\nWith Separate Bias:")
print(f"Weights: {weights.squeeze().tolist()}")
print(f"Bias:    {bias.item():.4f}")
print(f"Expected: [2.0, -1.0, 0.5], bias ≈ 1.0")
```

## Comparison with Gradient Descent

### When to Use Normal Equations

```python
import time

def compare_methods(n_samples: int, n_features: int, n_epochs: int = 1000):
    """Compare normal equations vs gradient descent"""
    torch.manual_seed(42)
    
    # Generate data
    X = torch.randn(n_samples, n_features)
    ones = torch.ones(n_samples, 1)
    X_aug = torch.cat([ones, X], dim=1)
    
    true_theta = torch.randn(n_features + 1, 1)
    y = X_aug @ true_theta + 0.1 * torch.randn(n_samples, 1)
    
    # Method 1: Normal Equations
    start = time.time()
    theta_ne = normal_equations_stable(X_aug, y)
    time_ne = time.time() - start
    
    # Method 2: Gradient Descent
    theta_gd = torch.zeros(n_features + 1, 1, requires_grad=True)
    optimizer = torch.optim.SGD([theta_gd], lr=0.01)
    
    start = time.time()
    for _ in range(n_epochs):
        y_pred = X_aug @ theta_gd
        loss = torch.mean((y - y_pred) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    time_gd = time.time() - start
    
    # Compare
    error_ne = torch.mean((theta_ne - true_theta) ** 2).item()
    error_gd = torch.mean((theta_gd.detach() - true_theta) ** 2).item()
    
    print(f"\nn={n_samples}, d={n_features}:")
    print(f"  Normal Equations: {time_ne:.4f}s, MSE={error_ne:.2e}")
    print(f"  Gradient Descent: {time_gd:.4f}s, MSE={error_gd:.2e}")
    
    return time_ne, time_gd, error_ne, error_gd

# Test with different sizes
print("Method Comparison:")
compare_methods(100, 10)
compare_methods(1000, 100)
compare_methods(10000, 50)
```

### Complexity Analysis

| Method | Time Complexity | Space Complexity |
|--------|-----------------|------------------|
| Normal Equations | $O(nd^2 + d^3)$ | $O(d^2)$ |
| Gradient Descent | $O(knd)$ | $O(d)$ |

where $n$ = samples, $d$ = features, $k$ = iterations.

**Rule of thumb:**
- $d < 10,000$: Normal equations often faster
- $d > 10,000$ or very large $n$: Gradient descent preferred
- $n >> d$: Normal equations very efficient

## Handling Special Cases

### Multicollinearity and Rank Deficiency

```python
def handle_rank_deficiency(X: torch.Tensor, y: torch.Tensor, alpha: float = 1e-6):
    """
    Handle rank-deficient design matrices with regularization
    
    Instead of: θ = (X^T X)^(-1) X^T y
    Use:        θ = (X^T X + αI)^(-1) X^T y
    
    This is Ridge Regression with small α
    """
    if y.dim() == 1:
        y = y.reshape(-1, 1)
    
    n_features = X.shape[1]
    
    # Add small regularization for numerical stability
    XtX = X.T @ X + alpha * torch.eye(n_features)
    Xty = X.T @ y
    
    theta = torch.linalg.solve(XtX, Xty)
    
    return theta

# Example with collinear features
n = 100
X_indep = torch.randn(n, 2)
X_collinear = torch.cat([X_indep, X_indep[:, 0:1] + 0.001 * torch.randn(n, 1)], dim=1)
ones = torch.ones(n, 1)
X_bad = torch.cat([ones, X_collinear], dim=1)

y_test = X_bad @ torch.tensor([[1.0], [2.0], [-1.0], [0.5]]) + 0.1 * torch.randn(n, 1)

print("\nHandling Near-Multicollinearity:")
print(f"Condition number of X^T X: {torch.linalg.cond(X_bad.T @ X_bad).item():.2e}")

# Try standard vs regularized
try:
    theta_standard = normal_equations(X_bad, y_test)
    print(f"Standard solution: {theta_standard.squeeze().tolist()}")
except Exception as e:
    print(f"Standard solution failed: {e}")

theta_regularized = handle_rank_deficiency(X_bad, y_test)
print(f"Regularized solution: {theta_regularized.squeeze().tolist()}")
```

### Underdetermined Systems (n < d)

```python
def minimum_norm_solution(X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Minimum norm solution for underdetermined systems (n < d)
    
    Uses: θ = X^T (X X^T)^(-1) y
    This gives the solution with minimum L2 norm
    """
    if y.dim() == 1:
        y = y.reshape(-1, 1)
    
    # When n < d, use right inverse: X^T (X X^T)^(-1)
    XXt = X @ X.T
    theta = X.T @ torch.linalg.solve(XXt, y)
    
    return theta

# Example: more features than samples
n, d = 10, 50  # Underdetermined
X_under = torch.randn(n, d)
y_under = torch.randn(n, 1)

theta_min_norm = minimum_norm_solution(X_under, y_under)
print(f"\nUnderdetermined System (n={n}, d={d}):")
print(f"Solution norm: {torch.norm(theta_min_norm).item():.4f}")
print(f"Residual norm: {torch.norm(X_under @ theta_min_norm - y_under).item():.6f}")
```

## Comparison with PyTorch nn.Linear

```python
import torch.nn as nn

def verify_against_nn_linear():
    """Verify normal equations give same result as trained nn.Linear"""
    torch.manual_seed(42)
    
    # Generate data
    n_samples = 1000
    n_features = 5
    
    X = torch.randn(n_samples, n_features)
    true_w = torch.tensor([2.0, -1.5, 0.5, 1.0, -0.8])
    true_b = 0.5
    y = X @ true_w + true_b + 0.1 * torch.randn(n_samples)
    
    # Method 1: Normal Equations
    ones = torch.ones(n_samples, 1)
    X_aug = torch.cat([ones, X], dim=1)
    theta_ne = normal_equations_stable(X_aug, y.reshape(-1, 1))
    
    # Method 2: Train nn.Linear to convergence
    model = nn.Linear(n_features, 1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.LBFGS(model.parameters(), line_search_fn='strong_wolfe')
    
    y_reshaped = y.reshape(-1, 1)
    
    def closure():
        optimizer.zero_grad()
        loss = criterion(model(X), y_reshaped)
        loss.backward()
        return loss
    
    for _ in range(10):
        optimizer.step(closure)
    
    # Compare
    print("\nVerification against nn.Linear:")
    print(f"Normal Equations:")
    print(f"  Bias:    {theta_ne[0].item():.6f}")
    print(f"  Weights: {theta_ne[1:].squeeze().tolist()}")
    print(f"\nTrained nn.Linear:")
    print(f"  Bias:    {model.bias.item():.6f}")
    print(f"  Weights: {model.weight.squeeze().tolist()}")
    print(f"\nClose match: {torch.allclose(theta_ne[1:].squeeze(), model.weight.squeeze(), atol=1e-4)}")

verify_against_nn_linear()
```

## Summary

### Normal Equations at a Glance

| Aspect | Details |
|--------|---------|
| Formula | $\hat{\boldsymbol{\theta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$ |
| Complexity | $O(nd^2 + d^3)$ |
| Requires | $\mathbf{X}^T\mathbf{X}$ invertible (full rank) |
| Best for | $d < 10,000$, small to medium datasets |
| PyTorch | `torch.linalg.lstsq()` or `torch.linalg.solve()` |

### Practical Recommendations

1. **Small datasets**: Use normal equations for exact, one-shot solution
2. **Large features**: Switch to gradient descent when $d > 10,000$
3. **Numerical stability**: Use QR or SVD decomposition
4. **Multicollinearity**: Add small regularization ($\alpha I$)
5. **Production code**: Use `torch.linalg.lstsq()` which handles all cases

### Key Takeaways

- Normal equations provide **closed-form** solution to linear regression
- Solution is **globally optimal** (convex problem)
- **Orthogonality** of residuals to column space gives the name
- For **numerical stability**, use QR decomposition or SVD
- **Trade-off** with gradient descent depends on problem size

## References

- Strang, G. (2019). *Linear Algebra and Learning from Data*
- Golub, G. H., & Van Loan, C. F. (2013). *Matrix Computations*
- PyTorch Documentation: [torch.linalg.lstsq](https://pytorch.org/docs/stable/generated/torch.linalg.lstsq.html)
