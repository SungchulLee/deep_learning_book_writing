# Eigendecomposition for PCA

Compute principal components using eigenvalue decomposition of the covariance matrix.

---

## Overview

**Key Idea:** The covariance matrix contains all information needed for PCA; its eigenvectors are the principal components.

**Computational Approach:** Center data → Compute covariance → Eigendecompose → Select top components.

**Time:** ~25 minutes  
**Level:** Intermediate

---

## Algorithm

### Step-by-Step PCA via Eigendecomposition

**Input:** Data matrix $X \in \mathbb{R}^{n \times d}$, number of components $k$

**Output:** Principal components $V_k$, projected data $Z$

```
1. Center the data:
   X_centered = X - mean(X)

2. Compute covariance matrix:
   Σ = (1/(n-1)) * X_centered^T @ X_centered

3. Eigendecomposition:
   Σ = V Λ V^T
   
4. Sort eigenvectors by decreasing eigenvalue:
   V = [v_1, v_2, ..., v_d] where λ_1 ≥ λ_2 ≥ ... ≥ λ_d

5. Select top k eigenvectors:
   V_k = [v_1, v_2, ..., v_k]

6. Project data:
   Z = X_centered @ V_k
```

---

## Mathematical Details

### Covariance Matrix

For centered data $X$ with $n$ samples and $d$ features:

$$\Sigma = \frac{1}{n-1} X^T X \in \mathbb{R}^{d \times d}$$

**Properties of $\Sigma$:**

| Property | Implication |
|----------|-------------|
| Symmetric | $\Sigma = \Sigma^T$ |
| Positive semi-definite | All eigenvalues $\lambda_i \geq 0$ |
| Real eigenvalues | Guaranteed by symmetry |
| Orthogonal eigenvectors | For distinct eigenvalues |

### Eigenvalue Problem

Find $\lambda$ and $v$ such that:

$$\Sigma v = \lambda v$$

**Interpretation:**
- $v$: direction (principal component)
- $\lambda$: variance along that direction

### The Spectral Theorem

For any symmetric matrix $\Sigma$:

$$\Sigma = V \Lambda V^T = \sum_{i=1}^{d} \lambda_i v_i v_i^T$$

where:
- $V = [v_1, \ldots, v_d]$: orthonormal matrix of eigenvectors
- $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_d)$: diagonal matrix of eigenvalues

---

## PyTorch Implementation

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional

class PCA_Eigen:
    """
    PCA implementation using eigendecomposition.
    
    Follows scikit-learn API conventions.
    """
    
    def __init__(self, n_components: Optional[int] = None):
        """
        Parameters:
        -----------
        n_components : int or None
            Number of components to keep. If None, keep all.
        """
        self.n_components = n_components
        self.components_ = None  # Principal components (eigenvectors)
        self.explained_variance_ = None  # Eigenvalues
        self.explained_variance_ratio_ = None
        self.mean_ = None
        self.n_features_ = None
    
    def fit(self, X: torch.Tensor) -> 'PCA_Eigen':
        """
        Fit PCA model by computing eigenvectors of covariance matrix.
        
        Parameters:
        -----------
        X : torch.Tensor
            Data matrix of shape (n_samples, n_features)
        
        Returns:
        --------
        self : PCA_Eigen
            Fitted model
        """
        n_samples, n_features = X.shape
        self.n_features_ = n_features
        
        if self.n_components is None:
            self.n_components = min(n_samples, n_features)
        
        # Step 1: Center the data
        self.mean_ = X.mean(dim=0)
        X_centered = X - self.mean_
        
        # Step 2: Compute covariance matrix
        # Using (n-1) for unbiased estimate
        cov_matrix = (X_centered.T @ X_centered) / (n_samples - 1)
        
        # Step 3: Eigendecomposition
        # torch.linalg.eigh returns eigenvalues in ascending order
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
        
        # Step 4: Sort by descending eigenvalue
        idx = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Step 5: Select top k components
        self.components_ = eigenvectors[:, :self.n_components].T  # (k, d)
        self.explained_variance_ = eigenvalues[:self.n_components]
        
        # Compute explained variance ratio
        total_variance = eigenvalues.sum()
        self.explained_variance_ratio_ = self.explained_variance_ / total_variance
        
        return self
    
    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        Project data onto principal components.
        
        Parameters:
        -----------
        X : torch.Tensor
            Data matrix of shape (n_samples, n_features)
        
        Returns:
        --------
        Z : torch.Tensor
            Projected data of shape (n_samples, n_components)
        """
        X_centered = X - self.mean_
        return X_centered @ self.components_.T
    
    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        """Fit model and transform data in one step."""
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct data from principal component representation.
        
        Parameters:
        -----------
        Z : torch.Tensor
            Projected data of shape (n_samples, n_components)
        
        Returns:
        --------
        X_reconstructed : torch.Tensor
            Reconstructed data of shape (n_samples, n_features)
        """
        return Z @ self.components_ + self.mean_
    
    def get_covariance(self) -> torch.Tensor:
        """
        Compute covariance matrix from fitted model.
        
        Returns:
        --------
        cov : torch.Tensor
            Covariance matrix approximation
        """
        return self.components_.T @ torch.diag(self.explained_variance_) @ self.components_


def demonstrate_eigendecomposition():
    """
    Demonstrate PCA via eigendecomposition with visualizations.
    """
    torch.manual_seed(42)
    
    # Generate data with known covariance structure
    n_samples = 300
    
    # True covariance matrix
    true_cov = torch.tensor([
        [3.0, 1.5, 0.5],
        [1.5, 2.0, 0.3],
        [0.5, 0.3, 1.0]
    ])
    
    # Generate data
    L = torch.linalg.cholesky(true_cov)
    X = torch.randn(n_samples, 3) @ L.T
    
    # Add mean
    true_mean = torch.tensor([1.0, 2.0, 3.0])
    X = X + true_mean
    
    # Fit PCA
    pca = PCA_Eigen(n_components=3)
    Z = pca.fit_transform(X)
    
    print("PCA via Eigendecomposition")
    print("=" * 50)
    
    # Compare estimated vs true covariance
    print("\n1. Covariance Matrix Comparison:")
    print("True covariance:\n", true_cov.numpy())
    estimated_cov = pca.get_covariance()
    print("Estimated covariance:\n", estimated_cov.numpy())
    
    # Show eigenvalues
    print("\n2. Eigenvalues (Explained Variance):")
    for i, (ev, evr) in enumerate(zip(pca.explained_variance_, 
                                       pca.explained_variance_ratio_)):
        print(f"   PC{i+1}: λ = {ev:.4f} ({evr*100:.2f}% of variance)")
    
    print(f"\n   Total explained: {pca.explained_variance_ratio_.sum()*100:.2f}%")
    
    # Show eigenvectors
    print("\n3. Principal Components (Eigenvectors):")
    print(pca.components_.numpy())
    
    # Verify orthogonality
    print("\n4. Orthogonality Check (V^T V should be I):")
    orthogonality = pca.components_ @ pca.components_.T
    print(orthogonality.numpy().round(6))
    
    # Reconstruction error
    X_reconstructed = pca.inverse_transform(Z)
    recon_error = torch.mean((X - X_reconstructed) ** 2)
    print(f"\n5. Reconstruction Error (all components): {recon_error:.6f}")
    
    # Test with fewer components
    for k in [1, 2]:
        pca_k = PCA_Eigen(n_components=k)
        Z_k = pca_k.fit_transform(X)
        X_recon_k = pca_k.inverse_transform(Z_k)
        error_k = torch.mean((X - X_recon_k) ** 2)
        print(f"   Reconstruction Error (k={k}): {error_k:.6f}")
    
    return pca, X, Z


def visualize_eigendecomposition(X: torch.Tensor, pca: PCA_Eigen):
    """
    Create visualizations for eigendecomposition-based PCA.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Scree plot (eigenvalue spectrum)
    ax = axes[0, 0]
    n_components = len(pca.explained_variance_)
    x = range(1, n_components + 1)
    ax.bar(x, pca.explained_variance_.numpy(), alpha=0.7, label='Eigenvalue')
    ax.plot(x, pca.explained_variance_.numpy(), 'ro-', markersize=8)
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Eigenvalue (Variance)')
    ax.set_title('Scree Plot')
    ax.set_xticks(x)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Cumulative explained variance
    ax = axes[0, 1]
    cumsum = torch.cumsum(pca.explained_variance_ratio_, dim=0).numpy()
    ax.plot(x, cumsum, 'bo-', markersize=8, linewidth=2)
    ax.fill_between(x, 0, cumsum, alpha=0.3)
    ax.axhline(0.95, color='r', linestyle='--', label='95% threshold')
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Cumulative Explained Variance Ratio')
    ax.set_title('Explained Variance')
    ax.set_xticks(x)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: 2D projection
    ax = axes[1, 0]
    Z = pca.transform(X)
    ax.scatter(Z[:, 0].numpy(), Z[:, 1].numpy(), alpha=0.5, s=20)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title('Data in Principal Component Space')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Plot 4: Reconstruction error vs components
    ax = axes[1, 1]
    errors = []
    for k in range(1, len(pca.explained_variance_) + 1):
        pca_k = PCA_Eigen(n_components=k)
        Z_k = pca_k.fit_transform(X)
        X_recon = pca_k.inverse_transform(Z_k)
        error = torch.mean((X - X_recon) ** 2).item()
        errors.append(error)
    
    ax.plot(range(1, len(errors) + 1), errors, 'go-', markersize=8, linewidth=2)
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Mean Squared Reconstruction Error')
    ax.set_title('Reconstruction Error vs Dimensionality')
    ax.set_xticks(range(1, len(errors) + 1))
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('eigendecomposition_pca.png', dpi=150)
    plt.show()


def compare_with_numpy(X: torch.Tensor):
    """
    Compare our implementation with NumPy's eigendecomposition.
    """
    # Our implementation
    pca = PCA_Eigen(n_components=X.shape[1])
    pca.fit(X)
    
    # NumPy implementation
    X_np = X.numpy()
    X_centered = X_np - X_np.mean(axis=0)
    cov = np.cov(X_centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    print("\nComparison with NumPy:")
    print("=" * 50)
    print("Eigenvalues (ours):", pca.explained_variance_.numpy())
    print("Eigenvalues (numpy):", eigenvalues)
    print("Max difference:", np.max(np.abs(pca.explained_variance_.numpy() - eigenvalues)))


# Main execution
if __name__ == "__main__":
    pca, X, Z = demonstrate_eigendecomposition()
    visualize_eigendecomposition(X, pca)
    compare_with_numpy(X)
```

---

## Computational Considerations

### Time Complexity

| Operation | Complexity |
|-----------|------------|
| Centering | $O(nd)$ |
| Covariance matrix | $O(nd^2)$ |
| Eigendecomposition | $O(d^3)$ |
| **Total** | $O(nd^2 + d^3)$ |

### When to Use Eigendecomposition

**Advantages:**
- Simple and direct
- Gives all eigenvalues at once
- Good when $d$ is small

**Disadvantages:**
- Requires forming $d \times d$ covariance matrix
- $O(d^3)$ eigendecomposition is expensive for large $d$
- Numerical issues when $n < d$

**Rule of thumb:** Use eigendecomposition when $d < 1000$ and $n > d$.

---

## Numerical Stability

### Potential Issues

1. **Near-zero eigenvalues:** Can cause division issues
2. **Repeated eigenvalues:** Eigenvectors not uniquely defined
3. **Ill-conditioned covariance:** Small perturbations cause large changes

### Best Practices

```python
# Use symmetric eigendecomposition (exploits symmetry)
eigenvalues, eigenvectors = torch.linalg.eigh(cov)  # Preferred

# Not:
eigenvalues, eigenvectors = torch.linalg.eig(cov)  # For general matrices

# Handle near-zero eigenvalues
eigenvalues = torch.clamp(eigenvalues, min=1e-10)
```

---

## Exercises

### Exercise 1: Implementation

Implement PCA from scratch without using `torch.linalg.eigh`:

a) Use power iteration to find the largest eigenvector  
b) Use deflation to find subsequent eigenvectors  
c) Compare accuracy with the built-in method

### Exercise 2: Convergence Analysis

For the power iteration method:

a) Plot eigenvalue estimate vs iteration number  
b) Analyze how the ratio $\lambda_1/\lambda_2$ affects convergence  
c) Implement a convergence criterion

### Exercise 3: Numerical Precision

Investigate numerical issues:

a) Create data where $n < d$ (more features than samples)  
b) What happens to the covariance matrix rank?  
c) How do small eigenvalues affect reconstruction?

### Exercise 4: Real Data

Apply PCA to MNIST digits:

a) Load MNIST dataset  
b) Compute PCA with different numbers of components  
c) Plot cumulative explained variance  
d) Visualize reconstructions at different compression levels

---

## Summary

| Concept | Key Point |
|---------|-----------|
| **Covariance matrix** | Symmetric, positive semi-definite |
| **Eigendecomposition** | $\Sigma = V \Lambda V^T$ |
| **Principal components** | Eigenvectors sorted by eigenvalue |
| **Explained variance** | Equals eigenvalue |
| **Complexity** | $O(nd^2 + d^3)$ |
| **Best for** | Small to medium $d$, $n > d$ |

---

## Next: SVD Approach

The next section covers computing PCA using Singular Value Decomposition, which is often more numerically stable and efficient for large datasets.
