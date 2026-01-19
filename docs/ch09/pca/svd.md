# Singular Value Decomposition for PCA

Compute PCA using SVD for improved numerical stability and efficiency.

---

## Overview

**Key Idea:** SVD of the data matrix directly gives principal components without forming the covariance matrix.

**Advantages:** More numerically stable, works when $n < d$, computationally efficient for sparse data.

**Time:** ~25 minutes  
**Level:** Intermediate

---

## SVD Fundamentals

### Definition

For any matrix $X \in \mathbb{R}^{n \times d}$, the **Singular Value Decomposition** is:

$$X = U \Sigma V^T$$

where:
- $U \in \mathbb{R}^{n \times n}$: left singular vectors (orthonormal)
- $\Sigma \in \mathbb{R}^{n \times d}$: diagonal matrix of singular values
- $V \in \mathbb{R}^{d \times d}$: right singular vectors (orthonormal)

### Reduced SVD

For $n > d$, we can use the **thin SVD**:

$$X = U_r \Sigma_r V^T$$

where:
- $U_r \in \mathbb{R}^{n \times d}$: first $d$ columns of $U$
- $\Sigma_r \in \mathbb{R}^{d \times d}$: diagonal with non-zero singular values
- $V \in \mathbb{R}^{d \times d}$: same as before

---

## Connection to PCA

### Key Relationship

For centered data $X$, the covariance matrix is:

$$\Sigma_{cov} = \frac{1}{n-1} X^T X$$

Using SVD $X = U \Sigma V^T$:

$$X^T X = V \Sigma^T U^T U \Sigma V^T = V \Sigma^2 V^T$$

Therefore:

$$\Sigma_{cov} = \frac{1}{n-1} V \Sigma^2 V^T$$

**This is the eigendecomposition of the covariance matrix!**

| PCA via Covariance | PCA via SVD |
|-------------------|-------------|
| Eigenvectors of $\Sigma_{cov}$ | Right singular vectors $V$ |
| Eigenvalues $\lambda_i$ | $\sigma_i^2 / (n-1)$ |

### Principal Component Scores

The projected data (PC scores) can be computed as:

$$Z = XV = U\Sigma$$

This means $U\Sigma$ contains the principal component scores directly!

---

## Algorithm Comparison

### Via Covariance Eigendecomposition

```
1. Center X
2. Compute Σ = X^T X / (n-1)     [O(nd²)]
3. Eigendecompose Σ = VΛV^T     [O(d³)]
4. Project: Z = XV_k            [O(ndk)]

Total: O(nd² + d³)
```

### Via SVD

```
1. Center X
2. Compute SVD: X = UΣV^T       [O(min(nd², n²d))]
3. PC scores: Z = U_k Σ_k       [O(nk)]

Total: O(min(nd², n²d))
```

**SVD is faster when $n < d$ (more features than samples).**

---

## PyTorch Implementation

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple

class PCA_SVD:
    """
    PCA implementation using Singular Value Decomposition.
    
    More numerically stable than covariance eigendecomposition,
    especially when n < d or when data has near-zero variance directions.
    """
    
    def __init__(self, n_components: Optional[int] = None):
        """
        Parameters:
        -----------
        n_components : int or None
            Number of components. If None, keep all.
        """
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None
        self.mean_ = None
        self.n_samples_ = None
        self.n_features_ = None
    
    def fit(self, X: torch.Tensor) -> 'PCA_SVD':
        """
        Fit PCA using SVD of centered data matrix.
        """
        self.n_samples_, self.n_features_ = X.shape
        
        if self.n_components is None:
            self.n_components = min(self.n_samples_, self.n_features_)
        
        # Center the data
        self.mean_ = X.mean(dim=0)
        X_centered = X - self.mean_
        
        # Compute SVD
        # full_matrices=False gives thin SVD
        U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
        
        # Singular values and explained variance
        self.singular_values_ = S[:self.n_components]
        
        # Eigenvalues of covariance = σ²/(n-1)
        self.explained_variance_ = (S ** 2) / (self.n_samples_ - 1)
        self.explained_variance_ = self.explained_variance_[:self.n_components]
        
        # Explained variance ratio
        total_var = (S ** 2).sum() / (self.n_samples_ - 1)
        self.explained_variance_ratio_ = self.explained_variance_ / total_var
        
        # Principal components (right singular vectors)
        self.components_ = Vh[:self.n_components]  # (k, d)
        
        # Store U and S for efficient transform
        self._U = U[:, :self.n_components]
        
        return self
    
    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        Project data onto principal components.
        
        For the training data: Z = U_k * S_k (already computed)
        For new data: Z = (X - mean) @ V^T
        """
        X_centered = X - self.mean_
        return X_centered @ self.components_.T
    
    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        Fit and transform in one step.
        
        More efficient than fit() then transform() because
        we can use U*S directly instead of recomputing X @ V.
        """
        self.fit(X)
        # Use precomputed U * S for training data
        return self._U * self.singular_values_
    
    def inverse_transform(self, Z: torch.Tensor) -> torch.Tensor:
        """Reconstruct from principal component representation."""
        return Z @ self.components_ + self.mean_


def compare_svd_vs_eigen(X: torch.Tensor):
    """
    Compare SVD and eigendecomposition approaches to PCA.
    """
    from pca_eigendecomposition import PCA_Eigen  # Assuming previous file
    
    n_components = min(X.shape)
    
    # SVD approach
    pca_svd = PCA_SVD(n_components=n_components)
    Z_svd = pca_svd.fit_transform(X)
    
    # Eigendecomposition approach
    pca_eigen = PCA_Eigen(n_components=n_components)
    Z_eigen = pca_eigen.fit_transform(X)
    
    print("SVD vs Eigendecomposition Comparison")
    print("=" * 50)
    
    # Compare eigenvalues
    print("\nExplained Variance (should be equal):")
    print("SVD:   ", pca_svd.explained_variance_.numpy()[:5])
    print("Eigen: ", pca_eigen.explained_variance_.numpy()[:5])
    
    # Compare components (may differ by sign)
    print("\nPrincipal Components (may differ by sign):")
    for i in range(min(3, n_components)):
        svd_pc = pca_svd.components_[i].numpy()
        eigen_pc = pca_eigen.components_[i].numpy()
        
        # Check if they're equal or opposite
        same = np.allclose(svd_pc, eigen_pc, atol=1e-5)
        opposite = np.allclose(svd_pc, -eigen_pc, atol=1e-5)
        
        print(f"PC{i+1}: {'Same' if same else 'Opposite' if opposite else 'Different'}")
    
    # Compare projections (account for sign flips)
    print("\nProjection Comparison:")
    for i in range(min(3, n_components)):
        corr = torch.corrcoef(torch.stack([Z_svd[:, i], Z_eigen[:, i]]))[0, 1]
        print(f"PC{i+1} correlation: {abs(corr.item()):.6f} (should be 1.0)")


def demonstrate_svd_stability():
    """
    Demonstrate numerical stability advantages of SVD.
    """
    torch.manual_seed(42)
    
    print("\nNumerical Stability Demonstration")
    print("=" * 50)
    
    # Case 1: More features than samples (n < d)
    print("\n1. n < d case (500 samples, 1000 features):")
    n, d = 500, 1000
    X = torch.randn(n, d)
    
    # SVD works fine
    pca_svd = PCA_SVD(n_components=100)
    Z_svd = pca_svd.fit_transform(X)
    print(f"   SVD: Success, shape = {Z_svd.shape}")
    
    # Eigendecomposition may have issues
    try:
        X_centered = X - X.mean(dim=0)
        cov = X_centered.T @ X_centered / (n - 1)
        eigenvalues, _ = torch.linalg.eigh(cov)
        n_negative = (eigenvalues < -1e-10).sum().item()
        print(f"   Eigen: {n_negative} negative eigenvalues (numerical error)")
    except Exception as e:
        print(f"   Eigen: Failed - {e}")
    
    # Case 2: Near-singular covariance
    print("\n2. Near-singular covariance (highly correlated features):")
    n, d = 1000, 10
    
    # Create data with some features that are linear combinations
    X_base = torch.randn(n, 5)
    noise = torch.randn(n, 5) * 1e-6
    X = torch.cat([X_base, X_base + noise], dim=1)  # Features 6-10 ≈ features 1-5
    
    pca_svd = PCA_SVD(n_components=10)
    pca_svd.fit(X)
    
    print("   Singular values:", pca_svd.singular_values_.numpy())
    print("   Last 5 are near-zero (redundant dimensions)")
    
    # Case 3: Speed comparison
    print("\n3. Speed comparison (large matrix):")
    import time
    
    n, d = 5000, 500
    X = torch.randn(n, d)
    
    # Time SVD
    start = time.time()
    pca_svd = PCA_SVD(n_components=50)
    pca_svd.fit_transform(X)
    svd_time = time.time() - start
    
    # Time Eigendecomposition
    start = time.time()
    X_centered = X - X.mean(dim=0)
    cov = X_centered.T @ X_centered / (n - 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    eigen_time = time.time() - start
    
    print(f"   SVD time: {svd_time:.4f}s")
    print(f"   Eigen time: {eigen_time:.4f}s")


def visualize_svd_components(X: torch.Tensor, n_components: int = 3):
    """
    Visualize the SVD decomposition.
    """
    # Center and compute SVD
    X_centered = X - X.mean(dim=0)
    U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Singular values and variance
    # Plot 1: Singular values
    ax = axes[0, 0]
    ax.semilogy(range(1, len(S) + 1), S.numpy(), 'bo-', markersize=4)
    ax.set_xlabel('Component')
    ax.set_ylabel('Singular Value (log scale)')
    ax.set_title('Singular Value Spectrum')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Explained variance
    ax = axes[0, 1]
    explained_var_ratio = (S ** 2) / (S ** 2).sum()
    cumsum = torch.cumsum(explained_var_ratio, dim=0).numpy()
    ax.plot(range(1, len(cumsum) + 1), cumsum, 'g-', linewidth=2)
    ax.fill_between(range(1, len(cumsum) + 1), 0, cumsum, alpha=0.3)
    ax.axhline(0.95, color='r', linestyle='--', alpha=0.7)
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Cumulative Explained Variance')
    ax.set_title('Explained Variance Ratio')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Low-rank approximation error
    ax = axes[0, 2]
    errors = []
    for k in range(1, min(50, len(S)) + 1):
        # Reconstruction error = sum of squared singular values not used
        error = (S[k:] ** 2).sum().item()
        errors.append(error)
    
    total_energy = (S ** 2).sum().item()
    relative_errors = [e / total_energy for e in errors]
    
    ax.semilogy(range(1, len(relative_errors) + 1), relative_errors, 'r-', linewidth=2)
    ax.set_xlabel('Rank k')
    ax.set_ylabel('Relative Reconstruction Error')
    ax.set_title('Low-Rank Approximation Error')
    ax.grid(True, alpha=0.3)
    
    # Row 2: Component visualizations (for image data)
    if X.shape[1] == 784:  # MNIST
        for i in range(3):
            ax = axes[1, i]
            component = Vh[i].numpy().reshape(28, 28)
            im = ax.imshow(component, cmap='RdBu_r')
            ax.set_title(f'PC{i+1} (σ={S[i]:.2f})')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
    else:
        # For non-image data, show component loadings
        for i in range(min(3, Vh.shape[0])):
            ax = axes[1, i]
            loadings = Vh[i].numpy()
            ax.bar(range(len(loadings)), loadings)
            ax.set_xlabel('Feature')
            ax.set_ylabel('Loading')
            ax.set_title(f'PC{i+1} Loadings')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('svd_pca.png', dpi=150)
    plt.show()


def truncated_svd_demo():
    """
    Demonstrate truncated SVD for efficiency.
    """
    print("\nTruncated SVD Demonstration")
    print("=" * 50)
    
    # For very large matrices, use randomized SVD
    n, d = 10000, 5000
    k = 50  # Number of components
    
    print(f"Matrix size: {n} × {d}")
    print(f"Components: {k}")
    
    # Create random data
    torch.manual_seed(42)
    X = torch.randn(n, d)
    X_centered = X - X.mean(dim=0)
    
    import time
    
    # Full SVD (expensive)
    print("\nFull SVD (computing all singular values)...")
    start = time.time()
    # This would be very slow, so we skip
    # U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
    print("   Skipped (too expensive for demo)")
    
    # Truncated SVD via randomized algorithm (scikit-learn style)
    print("\nTruncated SVD (randomized, k components only)...")
    start = time.time()
    
    # Simple randomized SVD implementation
    # Step 1: Random projection
    random_matrix = torch.randn(d, k + 10)
    Y = X_centered @ random_matrix
    
    # Step 2: QR decomposition
    Q, _ = torch.linalg.qr(Y)
    
    # Step 3: Project and SVD of smaller matrix
    B = Q.T @ X_centered
    U_small, S, Vh = torch.linalg.svd(B, full_matrices=False)
    
    # Step 4: Recover U
    U = Q @ U_small
    
    elapsed = time.time() - start
    print(f"   Time: {elapsed:.2f}s")
    print(f"   Top singular values: {S[:5].numpy()}")


# Main execution
if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Generate test data
    n_samples = 500
    n_features = 20
    
    # Create data with known structure
    X = torch.randn(n_samples, n_features)
    
    # Make some features correlated
    X[:, 5:10] = X[:, :5] + 0.1 * torch.randn(n_samples, 5)
    
    # Demonstrate SVD PCA
    pca = PCA_SVD(n_components=10)
    Z = pca.fit_transform(X)
    
    print("PCA via SVD")
    print("=" * 50)
    print(f"Input shape: {X.shape}")
    print(f"Output shape: {Z.shape}")
    print(f"Singular values: {pca.singular_values_.numpy()[:5]}")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_.numpy()[:5]}")
    
    # Run demonstrations
    demonstrate_svd_stability()
    visualize_svd_components(X, n_components=3)
```

---

## Key Advantages of SVD

### 1. Numerical Stability

SVD avoids forming $X^T X$, which can amplify numerical errors:

| Issue | Covariance Method | SVD Method |
|-------|------------------|------------|
| Condition number | Squared | Original |
| Near-zero eigenvalues | Can become negative | Stay positive |
| Precision loss | Higher | Lower |

### 2. Works When $n < d$

When there are more features than samples:
- Covariance matrix is rank-deficient
- SVD naturally handles this by having at most $\min(n,d)$ non-zero singular values

### 3. Computational Efficiency

For different regimes:

| Regime | Best Method | Complexity |
|--------|-------------|------------|
| $n \gg d$ | Either | $O(nd^2)$ |
| $n \ll d$ | SVD | $O(n^2 d)$ |
| $n \approx d$ | SVD | Often faster |

---

## Low-Rank Approximation

### Eckart-Young-Mirsky Theorem

The best rank-$k$ approximation to matrix $X$ (in Frobenius norm) is:

$$X_k = U_k \Sigma_k V_k^T = \sum_{i=1}^{k} \sigma_i u_i v_i^T$$

**Approximation error:**

$$\|X - X_k\|_F^2 = \sum_{i=k+1}^{r} \sigma_i^2$$

where $r = \text{rank}(X)$.

This theorem guarantees PCA gives the optimal linear dimensionality reduction!

---

## Exercises

### Exercise 1: Verify SVD-Eigenvalue Connection

For a random matrix $X$:

a) Compute eigenvalues of $X^T X$  
b) Compute singular values of $X$  
c) Verify $\lambda_i = \sigma_i^2$

### Exercise 2: Truncated SVD

Implement a simple randomized truncated SVD:

a) Use random projection to reduce dimensionality  
b) Compute SVD of the smaller matrix  
c) Compare accuracy with full SVD

### Exercise 3: Image Compression

Apply SVD to compress an image:

a) Treat image as matrix  
b) Compute SVD  
c) Reconstruct using different numbers of singular values  
d) Plot compression ratio vs PSNR

### Exercise 4: When SVD Beats Eigendecomposition

Create a scenario where:

a) Eigendecomposition gives numerical errors  
b) SVD gives correct results  
c) Explain why

---

## Summary

| Aspect | SVD Approach |
|--------|--------------|
| **Decomposition** | $X = U\Sigma V^T$ |
| **Principal components** | Columns of $V$ |
| **Singular values** | $\sigma_i = \sqrt{(n-1)\lambda_i}$ |
| **PC scores** | $U_k \Sigma_k$ |
| **Advantages** | Stability, works for $n < d$ |
| **Best for** | Large $d$, ill-conditioned data |

---

## Next: Reconstruction Error

The next section derives PCA from the perspective of minimizing reconstruction error.
