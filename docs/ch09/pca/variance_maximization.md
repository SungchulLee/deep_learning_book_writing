# Variance Maximization

Derive PCA by finding the directions that maximize the variance of projected data.

---

## Overview

**Key Idea:** Project high-dimensional data onto directions where it "spreads out" the most.

**Mathematical Goal:** Find unit vector $w$ that maximizes $\text{Var}(Xw)$.

**Time:** ~30 minutes  
**Level:** Intermediate

---

## Mathematical Derivation

### Problem Setup

Given centered data matrix $X \in \mathbb{R}^{n \times d}$ (each row is a sample, columns have zero mean), we want to find the direction $w \in \mathbb{R}^d$ that maximizes the variance of the projection $z = Xw$.

### Variance of Projection

The projected data is:

$$z = Xw \in \mathbb{R}^n$$

The variance of this projection is:

$$\text{Var}(z) = \frac{1}{n-1} z^T z = \frac{1}{n-1} w^T X^T X w = w^T \Sigma w$$

where $\Sigma = \frac{1}{n-1} X^T X$ is the sample covariance matrix.

### Constrained Optimization

To prevent $w$ from growing arbitrarily large, we constrain it to be a unit vector:

$$\max_w \quad w^T \Sigma w \quad \text{subject to} \quad w^T w = 1$$

### Lagrange Multiplier Solution

Form the Lagrangian:

$$\mathcal{L}(w, \lambda) = w^T \Sigma w - \lambda (w^T w - 1)$$

Take the derivative and set to zero:

$$\frac{\partial \mathcal{L}}{\partial w} = 2 \Sigma w - 2 \lambda w = 0$$

This gives us:

$$\Sigma w = \lambda w$$

**This is the eigenvalue equation!** The optimal $w$ is an eigenvector of $\Sigma$.

### Which Eigenvector?

Substituting back into the objective:

$$w^T \Sigma w = w^T (\lambda w) = \lambda w^T w = \lambda$$

The variance equals the eigenvalue. To maximize variance, choose the **largest eigenvalue** $\lambda_1$ and its corresponding eigenvector $v_1$.

**First Principal Component:** $v_1$ is the direction of maximum variance, with variance $\lambda_1$.

---

## Sequential Component Extraction

### Finding Subsequent Components

After finding $v_1$, we want the next direction of maximum variance that is **orthogonal** to $v_1$.

**Modified Problem:**

$$\max_w \quad w^T \Sigma w \quad \text{subject to} \quad w^T w = 1, \quad w^T v_1 = 0$$

**Solution:** The $k$-th principal component is the eigenvector corresponding to the $k$-th largest eigenvalue.

### Proof of Orthogonality

Since $\Sigma$ is symmetric and positive semi-definite, its eigenvectors corresponding to distinct eigenvalues are orthogonal:

$$v_i^T v_j = \begin{cases} 1 & \text{if } i = j \\ 0 & \text{if } i \neq j \end{cases}$$

This follows from the spectral theorem for symmetric matrices.

---

## Geometric Interpretation

### Variance Ellipsoid

The covariance matrix $\Sigma$ defines an ellipsoid in $d$-dimensional space:

$$\{x : x^T \Sigma^{-1} x = 1\}$$

- **Principal axes:** eigenvectors of $\Sigma$
- **Axis lengths:** proportional to $\sqrt{\lambda_i}$

The first principal component points along the longest axis of the ellipsoid.

### Projection Visualization

```
Original 2D Data         Projected onto PC1
      •                      
    • • •                    |----•-•-•----•-|
  •   •   •                      (1D line)
    • • •
      •
```

Projecting onto $v_1$ captures the most spread in the data.

---

## The Rayleigh Quotient

### Definition

The **Rayleigh quotient** for a symmetric matrix $\Sigma$ and vector $w$ is:

$$R(w) = \frac{w^T \Sigma w}{w^T w}$$

This equals the variance of the projection when $w$ is normalized.

### Key Properties

1. **Range:** $\lambda_{\min} \leq R(w) \leq \lambda_{\max}$
2. **Maximum:** Achieved at the eigenvector for $\lambda_{\max}$
3. **Minimum:** Achieved at the eigenvector for $\lambda_{\min}$
4. **Stationary points:** All eigenvectors

The Rayleigh quotient provides a characterization of PCA without explicit constraints.

---

## PyTorch Implementation

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def pca_variance_maximization(X: torch.Tensor, n_components: int) -> tuple:
    """
    Compute PCA by maximizing projected variance.
    
    Parameters:
    -----------
    X : torch.Tensor
        Data matrix of shape (n_samples, n_features)
    n_components : int
        Number of principal components to compute
    
    Returns:
    --------
    components : torch.Tensor
        Principal components (eigenvectors) of shape (n_features, n_components)
    explained_variance : torch.Tensor
        Variance explained by each component (eigenvalues)
    """
    # Center the data
    X_centered = X - X.mean(dim=0)
    
    # Compute covariance matrix
    n_samples = X.shape[0]
    cov_matrix = (X_centered.T @ X_centered) / (n_samples - 1)
    
    # Eigendecomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
    
    # Sort by descending eigenvalue
    idx = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Select top k components
    components = eigenvectors[:, :n_components]
    explained_variance = eigenvalues[:n_components]
    
    return components, explained_variance


def verify_variance_maximization(X: torch.Tensor, components: torch.Tensor, 
                                  explained_variance: torch.Tensor):
    """
    Verify that PCA maximizes variance.
    
    Shows that projecting onto principal components gives maximum variance,
    and that variance equals the corresponding eigenvalue.
    """
    X_centered = X - X.mean(dim=0)
    
    print("Verification of Variance Maximization:")
    print("=" * 50)
    
    for i in range(components.shape[1]):
        # Project onto component i
        projection = X_centered @ components[:, i]
        
        # Compute variance
        actual_variance = torch.var(projection, unbiased=True)
        expected_variance = explained_variance[i]
        
        print(f"PC{i+1}: Actual Var = {actual_variance:.6f}, "
              f"Eigenvalue = {expected_variance:.6f}")
    
    # Show that random directions have less variance
    print("\nComparison with random directions:")
    for _ in range(3):
        random_direction = torch.randn(X.shape[1])
        random_direction = random_direction / torch.norm(random_direction)
        random_projection = X_centered @ random_direction
        random_variance = torch.var(random_projection, unbiased=True)
        print(f"Random direction variance: {random_variance:.6f}")


def visualize_variance_maximization(X: torch.Tensor):
    """
    Visualize PCA on 2D data showing variance maximization.
    """
    assert X.shape[1] == 2, "Visualization requires 2D data"
    
    X_centered = X - X.mean(dim=0)
    components, variances = pca_variance_maximization(X, 2)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Original data with principal components
    ax = axes[0]
    ax.scatter(X_centered[:, 0].numpy(), X_centered[:, 1].numpy(), 
               alpha=0.5, s=20)
    
    # Draw principal component arrows
    for i in range(2):
        pc = components[:, i].numpy()
        scale = 2 * np.sqrt(variances[i].item())
        ax.arrow(0, 0, pc[0] * scale, pc[1] * scale,
                head_width=0.2, head_length=0.1, fc=f'C{i}', ec=f'C{i}')
        ax.text(pc[0] * scale * 1.1, pc[1] * scale * 1.1, 
                f'PC{i+1} (λ={variances[i]:.2f})')
    
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('Data with Principal Components')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Variance as function of angle
    ax = axes[1]
    angles = np.linspace(0, np.pi, 100)
    variance_by_angle = []
    
    for angle in angles:
        direction = torch.tensor([np.cos(angle), np.sin(angle)], dtype=X.dtype)
        projection = X_centered @ direction
        variance_by_angle.append(torch.var(projection, unbiased=True).item())
    
    ax.plot(np.degrees(angles), variance_by_angle, 'b-', linewidth=2)
    
    # Mark principal components
    pc1_angle = np.arctan2(components[1, 0].item(), components[0, 0].item())
    if pc1_angle < 0:
        pc1_angle += np.pi
    ax.axvline(np.degrees(pc1_angle), color='C0', linestyle='--', 
               label=f'PC1 (max var)')
    
    ax.set_xlabel('Projection Angle (degrees)')
    ax.set_ylabel('Variance of Projection')
    ax.set_title('Variance vs Projection Direction')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Projected data onto PC1
    ax = axes[2]
    projected = X_centered @ components[:, 0]
    ax.scatter(projected.numpy(), np.zeros_like(projected.numpy()), 
               alpha=0.5, s=20)
    ax.set_xlabel('PC1')
    ax.set_title(f'Data Projected onto PC1 (Var = {variances[0]:.2f})')
    ax.set_yticks([])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('variance_maximization.png', dpi=150)
    plt.show()


# Example usage
def main():
    torch.manual_seed(42)
    
    # Generate correlated 2D data
    n_samples = 500
    
    # Create data with known covariance structure
    cov = torch.tensor([[2.0, 1.5], [1.5, 1.5]])
    L = torch.linalg.cholesky(cov)
    X = torch.randn(n_samples, 2) @ L.T
    
    # Compute PCA
    components, variances = pca_variance_maximization(X, 2)
    
    print("Principal Components:")
    print(components)
    print("\nExplained Variance (Eigenvalues):")
    print(variances)
    
    # Verify
    verify_variance_maximization(X, components, variances)
    
    # Visualize
    visualize_variance_maximization(X)


if __name__ == "__main__":
    main()
```

---

## Key Takeaways

1. **PCA finds maximum variance directions** — The first principal component is the direction along which the data has maximum variance.

2. **Eigenvalue equals variance** — The eigenvalue $\lambda_i$ equals the variance of data projected onto eigenvector $v_i$.

3. **Orthogonal components** — Principal components are mutually orthogonal, providing uncorrelated projections.

4. **Sequential extraction** — Components are found by iteratively maximizing variance subject to orthogonality constraints.

5. **Rayleigh quotient** — Provides an elegant characterization linking variance maximization to eigenvalue problems.

---

## Exercises

### Exercise 1: Manual Derivation

Derive the first principal component for 2D data with covariance matrix:

$$\Sigma = \begin{pmatrix} 4 & 2 \\ 2 & 3 \end{pmatrix}$$

a) Find eigenvalues and eigenvectors  
b) Identify PC1 and PC2  
c) What fraction of variance does PC1 explain?

### Exercise 2: Geometric Intuition

For the ellipse defined by $\frac{x^2}{4} + \frac{y^2}{1} = 1$:

a) What are the principal directions?  
b) What are the corresponding variances?  
c) How does rotating the ellipse affect the PCs?

### Exercise 3: Constrained Optimization

Verify the Lagrange multiplier solution by:

a) Parameterizing $w = (\cos\theta, \sin\theta)$ for 2D  
b) Taking derivative of variance w.r.t. $\theta$  
c) Showing critical points correspond to eigenvectors

### Exercise 4: Multiple Components

Extend the derivation to show that the $k$-th principal component maximizes variance subject to orthogonality to all previous components.

---

## Summary

| Concept | Description |
|---------|-------------|
| **Objective** | Maximize $w^T \Sigma w$ subject to $\|w\| = 1$ |
| **Solution** | Eigenvector of $\Sigma$ with largest eigenvalue |
| **Variance captured** | Equals the eigenvalue $\lambda$ |
| **Multiple PCs** | Subsequent eigenvectors in decreasing eigenvalue order |
| **Key insight** | Variance maximization leads to eigenvalue problem |

---

## Next: Eigendecomposition

The next section covers the computational approach to PCA using eigendecomposition of the covariance matrix.
