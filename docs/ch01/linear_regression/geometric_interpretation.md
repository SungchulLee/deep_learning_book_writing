# Geometric Interpretation of Linear Regression

## Overview

Understanding linear regression geometrically provides deep intuition about what the algorithm actually computes. The key insight is that linear regression finds the **orthogonal projection** of the target vector onto the subspace spanned by the features.

## The Column Space Perspective

### Setup

Consider the linear model in matrix form:

$$\hat{\mathbf{y}} = \mathbf{X}\boldsymbol{\theta}$$

where:
- $\mathbf{X} \in \mathbb{R}^{n \times d}$ is the design matrix
- $\boldsymbol{\theta} \in \mathbb{R}^{d}$ is the parameter vector
- $\hat{\mathbf{y}} \in \mathbb{R}^{n}$ is the prediction vector

### Column Space

The **column space** (or range) of $\mathbf{X}$ is:

$$\text{Col}(\mathbf{X}) = \{\mathbf{X}\boldsymbol{\theta} : \boldsymbol{\theta} \in \mathbb{R}^d\}$$

This is the set of all possible predictions—a $d$-dimensional subspace of $\mathbb{R}^n$.

### The Projection Interpretation

Linear regression finds the point in $\text{Col}(\mathbf{X})$ **closest** to $\mathbf{y}$:

$$\hat{\mathbf{y}} = \text{proj}_{\text{Col}(\mathbf{X})}(\mathbf{y})$$

This is the orthogonal projection of $\mathbf{y}$ onto the column space of $\mathbf{X}$.

## Mathematical Foundation

### Orthogonality Condition

At the optimal solution, the residual is orthogonal to every column of $\mathbf{X}$:

$$\mathbf{X}^T(\mathbf{y} - \hat{\mathbf{y}}) = \mathbf{0}$$

This is equivalent to the Normal Equations:

$$\mathbf{X}^T\mathbf{X}\hat{\boldsymbol{\theta}} = \mathbf{X}^T\mathbf{y}$$

### Projection Matrix

The **projection matrix** (or hat matrix) is:

$$\mathbf{P} = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T$$

Then:

$$\hat{\mathbf{y}} = \mathbf{P}\mathbf{y}$$

### Properties of the Projection Matrix

| Property | Mathematical Form | Interpretation |
|----------|------------------|----------------|
| Idempotent | $\mathbf{P}^2 = \mathbf{P}$ | Projecting twice gives same result |
| Symmetric | $\mathbf{P}^T = \mathbf{P}$ | Self-adjoint operator |
| Rank | $\text{rank}(\mathbf{P}) = d$ | Dimension of column space |
| Eigenvalues | 0 or 1 only | Projection to subspace or zero |
| Trace | $\text{tr}(\mathbf{P}) = d$ | Sum of eigenvalues = rank |

## PyTorch Implementation

### Computing Projections

```python
import torch
import numpy as np

def compute_projection_matrix(X: torch.Tensor) -> torch.Tensor:
    """
    Compute the projection (hat) matrix P = X(X'X)^(-1)X'
    
    Args:
        X: Design matrix (n_samples, n_features)
    
    Returns:
        Projection matrix P (n_samples, n_samples)
    """
    XtX_inv = torch.linalg.inv(X.T @ X)
    P = X @ XtX_inv @ X.T
    return P

def project_onto_column_space(X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Project y onto the column space of X
    
    Args:
        X: Design matrix (n_samples, n_features)
        y: Target vector (n_samples,) or (n_samples, 1)
    
    Returns:
        Projected vector ŷ
    """
    P = compute_projection_matrix(X)
    y_hat = P @ y
    return y_hat

# Example
torch.manual_seed(42)
n, d = 10, 3
X = torch.randn(n, d)
y = torch.randn(n, 1)

y_hat = project_onto_column_space(X, y)
residual = y - y_hat

print("Projection onto Column Space:")
print(f"||y||: {torch.norm(y).item():.4f}")
print(f"||ŷ||: {torch.norm(y_hat).item():.4f}")
print(f"||residual||: {torch.norm(residual).item():.4f}")

# Verify Pythagorean theorem: ||y||² = ||ŷ||² + ||r||²
y_norm_sq = torch.norm(y) ** 2
y_hat_norm_sq = torch.norm(y_hat) ** 2
r_norm_sq = torch.norm(residual) ** 2
print(f"\nPythagorean Theorem: {y_norm_sq.item():.4f} = {y_hat_norm_sq.item():.4f} + {r_norm_sq.item():.4f}")
print(f"Check: {torch.isclose(y_norm_sq, y_hat_norm_sq + r_norm_sq)}")
```

### Verifying Orthogonality

```python
def verify_orthogonality(X: torch.Tensor, y: torch.Tensor, y_hat: torch.Tensor):
    """
    Verify that residual is orthogonal to column space
    """
    residual = y - y_hat
    
    # Residual should be orthogonal to each column of X
    orthogonality_check = X.T @ residual
    
    print("Orthogonality Verification:")
    print(f"X' @ residual = {orthogonality_check.squeeze().tolist()}")
    print(f"All close to zero: {torch.allclose(orthogonality_check, torch.zeros_like(orthogonality_check), atol=1e-6)}")
    
    # Residual should be orthogonal to y_hat
    dot_product = torch.dot(residual.squeeze(), y_hat.squeeze())
    print(f"residual · ŷ = {dot_product.item():.2e} (should be ≈ 0)")

verify_orthogonality(X, y, y_hat)
```

### Analyzing the Projection Matrix

```python
def analyze_projection_matrix(X: torch.Tensor):
    """
    Compute and analyze properties of the projection matrix
    """
    P = compute_projection_matrix(X)
    n, d = X.shape
    
    print("Projection Matrix Analysis:")
    print(f"Shape: {P.shape}")
    
    # Property 1: Symmetric
    is_symmetric = torch.allclose(P, P.T, atol=1e-6)
    print(f"\n1. Symmetric (P = P'): {is_symmetric}")
    
    # Property 2: Idempotent
    P_squared = P @ P
    is_idempotent = torch.allclose(P_squared, P, atol=1e-6)
    print(f"2. Idempotent (P² = P): {is_idempotent}")
    
    # Property 3: Eigenvalues
    eigenvalues = torch.linalg.eigvalsh(P)
    print(f"3. Eigenvalues: {sorted(eigenvalues.tolist(), reverse=True)}")
    print(f"   Expected: {d} ones and {n-d} zeros")
    
    # Property 4: Trace = rank
    trace = torch.trace(P)
    print(f"4. Trace: {trace.item():.4f} (expected: {d})")
    
    # Property 5: Diagonal elements (leverage)
    leverage = torch.diag(P)
    print(f"5. Leverage (diagonal elements):")
    print(f"   Mean: {leverage.mean().item():.4f} (expected: {d/n:.4f})")
    print(f"   Range: [{leverage.min().item():.4f}, {leverage.max().item():.4f}]")
    
    return P

P = analyze_projection_matrix(X)
```

## Leverage and Influence

### Hat Matrix Diagonal

The diagonal elements of $\mathbf{P}$, denoted $h_{ii}$, are called **leverage values**:

$$h_{ii} = \mathbf{x}_i^T(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{x}_i$$

### Properties of Leverage

```python
def compute_leverage(X: torch.Tensor) -> torch.Tensor:
    """
    Compute leverage values for each observation
    
    High leverage points have unusual feature values and
    disproportionate influence on the fit.
    """
    P = compute_projection_matrix(X)
    leverage = torch.diag(P)
    return leverage

def analyze_leverage(X: torch.Tensor, y: torch.Tensor):
    """
    Analyze leverage and its relationship to influence
    """
    n, d = X.shape
    leverage = compute_leverage(X)
    
    print("Leverage Analysis:")
    print(f"Sum of leverage: {leverage.sum().item():.4f} (should equal d = {d})")
    print(f"Average leverage: {leverage.mean().item():.4f} (= d/n = {d/n:.4f})")
    
    # High leverage threshold (common rule: 2*(d+1)/n)
    threshold = 2 * (d + 1) / n
    high_leverage = leverage > threshold
    print(f"\nHigh leverage threshold: {threshold:.4f}")
    print(f"High leverage points: {high_leverage.sum().item()}")
    
    # Leverage bounds
    print(f"\nLeverage bounds: 1/n ≤ h_ii ≤ 1")
    print(f"Actual range: [{leverage.min().item():.4f}, {leverage.max().item():.4f}]")
    
    return leverage

leverage_values = analyze_leverage(X, y)
```

## Residual Geometry

### Decomposition of Target Vector

The target vector $\mathbf{y}$ decomposes into orthogonal components:

$$\mathbf{y} = \hat{\mathbf{y}} + \mathbf{r}$$

where:
- $\hat{\mathbf{y}} \in \text{Col}(\mathbf{X})$: fitted values
- $\mathbf{r} \in \text{Col}(\mathbf{X})^\perp$: residuals (orthogonal complement)

### Pythagorean Theorem

$$\|\mathbf{y}\|^2 = \|\hat{\mathbf{y}}\|^2 + \|\mathbf{r}\|^2$$

This decomposition underlies the ANOVA (Analysis of Variance) decomposition:

$$\text{SST} = \text{SSR} + \text{SSE}$$

where:
- **SST** (Total Sum of Squares): $\|\mathbf{y} - \bar{y}\mathbf{1}\|^2$
- **SSR** (Regression Sum of Squares): $\|\hat{\mathbf{y}} - \bar{y}\mathbf{1}\|^2$
- **SSE** (Error Sum of Squares): $\|\mathbf{r}\|^2$

```python
def anova_decomposition(X: torch.Tensor, y: torch.Tensor):
    """
    Perform ANOVA decomposition and verify Pythagorean relationship
    """
    # Add intercept column
    n = X.shape[0]
    ones = torch.ones(n, 1)
    X_aug = torch.cat([ones, X], dim=1)
    
    # Fit
    theta = torch.linalg.lstsq(X_aug, y).solution
    y_hat = X_aug @ theta
    residual = y - y_hat
    
    # Means
    y_mean = y.mean()
    
    # Sum of squares
    SST = torch.sum((y - y_mean) ** 2)
    SSR = torch.sum((y_hat - y_mean) ** 2)
    SSE = torch.sum(residual ** 2)
    
    print("ANOVA Decomposition:")
    print(f"SST (Total):      {SST.item():.4f}")
    print(f"SSR (Regression): {SSR.item():.4f}")
    print(f"SSE (Error):      {SSE.item():.4f}")
    print(f"SSR + SSE:        {(SSR + SSE).item():.4f}")
    print(f"Match: {torch.isclose(SST, SSR + SSE, rtol=1e-4)}")
    
    # R-squared
    R_squared = SSR / SST
    print(f"\nR² = SSR/SST = {R_squared.item():.4f}")
    
    return SST, SSR, SSE

SST, SSR, SSE = anova_decomposition(X, y)
```

## Visualization

### 2D Projection Visualization

```python
import matplotlib.pyplot as plt

def visualize_2d_projection():
    """
    Visualize linear regression as projection for simple 2D case
    """
    torch.manual_seed(42)
    
    # Simple case: 2 data points, predicting with 1 feature + intercept
    x = torch.tensor([1.0, 3.0])
    y = torch.tensor([2.0, 5.0])
    
    # Design matrix
    X = torch.stack([torch.ones(2), x], dim=1)
    
    # Solve
    theta = torch.linalg.lstsq(X, y.reshape(-1, 1)).solution.squeeze()
    y_hat = X @ theta
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Standard regression view
    ax1 = axes[0]
    ax1.scatter(x.numpy(), y.numpy(), color='red', s=100, zorder=5, label='Data')
    x_line = torch.linspace(0, 4, 100)
    y_line = theta[0] + theta[1] * x_line
    ax1.plot(x_line.numpy(), y_line.numpy(), 'b-', linewidth=2, 
             label=f'y = {theta[0]:.2f} + {theta[1]:.2f}x')
    
    # Residuals
    for i in range(len(x)):
        ax1.plot([x[i], x[i]], [y[i], y_hat[i]], 'g--', linewidth=2)
    
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Standard View: Line Fitting')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right: Projection view in R²
    ax2 = axes[1]
    
    # Column vectors
    col1 = X[:, 0].numpy()  # [1, 1]
    col2 = X[:, 1].numpy()  # [1, 3]
    
    # Plot columns as vectors from origin
    ax2.arrow(0, 0, col1[0], col1[1], head_width=0.15, head_length=0.1,
              fc='blue', ec='blue', linewidth=2)
    ax2.arrow(0, 0, col2[0], col2[1], head_width=0.15, head_length=0.1,
              fc='green', ec='green', linewidth=2)
    
    # Plot y
    ax2.scatter([y[0].item()], [y[1].item()], color='red', s=150, zorder=5)
    ax2.annotate('y', (y[0].item() + 0.1, y[1].item() + 0.1), fontsize=12)
    
    # Plot y_hat
    ax2.scatter([y_hat[0].item()], [y_hat[1].item()], color='purple', 
                s=150, zorder=5, marker='s')
    ax2.annotate('ŷ', (y_hat[0].item() + 0.1, y_hat[1].item() - 0.3), fontsize=12)
    
    # Residual vector
    ax2.annotate('', xy=(y[0].item(), y[1].item()), 
                 xytext=(y_hat[0].item(), y_hat[1].item()),
                 arrowprops=dict(arrowstyle='->', color='red', lw=2))
    
    # Labels
    ax2.text(col1[0]/2 - 0.2, col1[1]/2 + 0.1, 'col₁ = [1,1]', fontsize=10, color='blue')
    ax2.text(col2[0]/2 + 0.1, col2[1]/2 - 0.2, 'col₂ = [1,3]', fontsize=10, color='green')
    
    ax2.set_xlabel('Component 1 (data point 1)')
    ax2.set_ylabel('Component 2 (data point 2)')
    ax2.set_title('Projection View: y projected onto Col(X)')
    ax2.set_xlim(-0.5, 5.5)
    ax2.set_ylim(-0.5, 6)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# visualize_2d_projection()
```

## Connection to R-squared

### Geometric Interpretation of R²

$$R^2 = \frac{\|\hat{\mathbf{y}} - \bar{y}\mathbf{1}\|^2}{\|\mathbf{y} - \bar{y}\mathbf{1}\|^2} = \cos^2(\theta)$$

where $\theta$ is the angle between $(\mathbf{y} - \bar{y}\mathbf{1})$ and its projection.

```python
def geometric_r_squared(X: torch.Tensor, y: torch.Tensor):
    """
    Compute R² and interpret geometrically
    """
    n = X.shape[0]
    
    # Add intercept
    X_aug = torch.cat([torch.ones(n, 1), X], dim=1)
    
    # Fit
    theta = torch.linalg.lstsq(X_aug, y).solution
    y_hat = X_aug @ theta
    
    y_mean = y.mean()
    
    # Centered vectors
    y_centered = y - y_mean
    y_hat_centered = y_hat - y_mean
    
    # R² as squared cosine
    cos_theta = torch.dot(y_centered.squeeze(), y_hat_centered.squeeze()) / (
        torch.norm(y_centered) * torch.norm(y_hat_centered)
    )
    R_squared_cos = cos_theta ** 2
    
    # R² traditional
    SST = torch.sum(y_centered ** 2)
    SSR = torch.sum(y_hat_centered ** 2)
    R_squared_trad = SSR / SST
    
    print("Geometric R² Interpretation:")
    print(f"R² (traditional): {R_squared_trad.item():.4f}")
    print(f"R² (cos²θ):       {R_squared_cos.item():.4f}")
    print(f"cos(θ):           {cos_theta.item():.4f}")
    print(f"θ (degrees):      {torch.arccos(cos_theta).item() * 180 / np.pi:.2f}°")
    
    return R_squared_trad

R_sq = geometric_r_squared(X, y)
```

## Summary

### Key Geometric Insights

| Concept | Geometric Interpretation |
|---------|-------------------------|
| Prediction $\hat{\mathbf{y}}$ | Projection of $\mathbf{y}$ onto $\text{Col}(\mathbf{X})$ |
| Residual $\mathbf{r}$ | Component of $\mathbf{y}$ orthogonal to $\text{Col}(\mathbf{X})$ |
| Normal Equations | Orthogonality condition: $\mathbf{X}^T\mathbf{r} = \mathbf{0}$ |
| Projection Matrix | $\mathbf{P} = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T$ |
| R² | Squared cosine of angle between centered vectors |
| Leverage | Diagonal of projection matrix |

### Why Geometry Matters

1. **Intuition**: Understand what regression actually computes
2. **Diagnostics**: High leverage points have geometric interpretation
3. **Uniqueness**: Projection is unique, explaining unique OLS solution
4. **Extensions**: Basis for understanding ridge regression, PCA, etc.

## References

- Strang, G. (2019). *Linear Algebra and Learning from Data*, Chapter I.4
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*, Chapter 3.2
- Lay, D. C. (2016). *Linear Algebra and Its Applications*, Chapter 6
