# Reconstruction Error Minimization

Derive PCA by minimizing the squared reconstruction error.

---

## Overview

**Key Idea:** Find the linear subspace that best approximates the data in terms of reconstruction error.

**Mathematical Goal:** Minimize $\|X - \hat{X}\|_F^2$ where $\hat{X}$ is the reconstructed data.

**Time:** ~25 minutes  
**Level:** Intermediate

---

## Problem Formulation

### Setup

Given centered data $X \in \mathbb{R}^{n \times d}$, we want to find:

1. **Encoding matrix** $W \in \mathbb{R}^{d \times k}$: projects data to $k$-dimensional subspace
2. **Decoding matrix** $D \in \mathbb{R}^{k \times d}$: reconstructs from low-dimensional representation

### The Reconstruction Process

```
Original: x ∈ ℝᵈ
    ↓ Encode: z = Wᵀx
Latent: z ∈ ℝᵏ
    ↓ Decode: x̂ = Dᵀz
Reconstructed: x̂ ∈ ℝᵈ
```

### Objective Function

Minimize total reconstruction error:

$$\mathcal{L}(W, D) = \sum_{i=1}^{n} \|x_i - \hat{x}_i\|^2 = \|X - XWD\|_F^2$$

where $\|\cdot\|_F$ is the Frobenius norm.

---

## Derivation

### Step 1: Optimal Decoder Given Encoder

For fixed $W$, the optimal decoder that minimizes reconstruction error is:

$$D^* = W^T$$

**Proof:** The reconstruction is $\hat{X} = XWD$. For each sample $x_i$:

$$\hat{x}_i = D^T W^T x_i = D^T z_i$$

This is a linear regression problem: given $z_i$, predict $x_i$ with $D^T$.

The optimal solution is $D^T = (Z^T Z)^{-1} Z^T X^T$, which simplifies to $D = W$ when $W$ has orthonormal columns.

### Step 2: Constraint on Encoder

To get a unique solution, we constrain $W$ to have orthonormal columns:

$$W^T W = I_k$$

With this constraint and $D = W$:

$$\mathcal{L}(W) = \|X - XWW^T\|_F^2$$

### Step 3: Expanding the Objective

$$\mathcal{L} = \text{tr}((X - XWW^T)^T(X - XWW^T))$$

$$= \text{tr}(X^TX) - 2\text{tr}(X^TXWW^T) + \text{tr}(WW^TX^TXWW^T)$$

Since $W^TW = I$:

$$= \text{tr}(X^TX) - 2\text{tr}(W^TX^TXW) + \text{tr}(W^TX^TXW)$$

$$= \text{tr}(X^TX) - \text{tr}(W^TX^TXW)$$

### Step 4: Minimization Becomes Maximization

Minimizing $\mathcal{L}$ is equivalent to **maximizing**:

$$\text{tr}(W^T X^T X W) = \text{tr}(W^T \Sigma W) \cdot (n-1)$$

where $\Sigma = \frac{1}{n-1}X^TX$ is the covariance matrix.

### Step 5: Connection to Variance Maximization

The objective $\text{tr}(W^T \Sigma W)$ is the sum of variances along the $k$ projection directions!

$$\text{tr}(W^T \Sigma W) = \sum_{j=1}^{k} w_j^T \Sigma w_j = \sum_{j=1}^{k} \text{Var}(\text{projection onto } w_j)$$

**This proves the equivalence of the two PCA perspectives!**

### Step 6: Solution

The optimal $W$ consists of the top $k$ eigenvectors of $\Sigma$:

$$W^* = [v_1, v_2, \ldots, v_k]$$

where $\Sigma v_i = \lambda_i v_i$ and $\lambda_1 \geq \lambda_2 \geq \cdots$.

---

## Reconstruction Error Formula

### Closed-Form Expression

The minimum reconstruction error is:

$$\mathcal{L}^* = \|X - XW^*W^{*T}\|_F^2 = \sum_{i=k+1}^{d} \lambda_i \cdot (n-1)$$

**Interpretation:** The reconstruction error equals the sum of variances along the **discarded** directions.

### Explained Variance Ratio

The fraction of variance captured by $k$ components:

$$\text{EVR}_k = \frac{\sum_{i=1}^{k} \lambda_i}{\sum_{i=1}^{d} \lambda_i} = 1 - \frac{\mathcal{L}^*}{\|X\|_F^2}$$

---

## PyTorch Implementation

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

def reconstruction_error_pca(X: torch.Tensor, k: int) -> dict:
    """
    PCA by minimizing reconstruction error.
    
    Parameters:
    -----------
    X : torch.Tensor
        Data matrix (n_samples, n_features)
    k : int
        Number of components
    
    Returns:
    --------
    dict with components, reconstruction, errors
    """
    n, d = X.shape
    
    # Center data
    mean = X.mean(dim=0)
    X_centered = X - mean
    
    # Covariance matrix
    cov = (X_centered.T @ X_centered) / (n - 1)
    
    # Eigendecomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    
    # Sort descending
    idx = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Select top k
    W = eigenvectors[:, :k]  # (d, k)
    
    # Project and reconstruct
    Z = X_centered @ W  # (n, k)
    X_reconstructed = Z @ W.T + mean  # (n, d)
    
    # Compute reconstruction error
    reconstruction_error = torch.sum((X - X_reconstructed) ** 2)
    
    # Verify with eigenvalue formula
    theoretical_error = eigenvalues[k:].sum() * (n - 1)
    
    return {
        'components': W,
        'eigenvalues': eigenvalues,
        'projected': Z,
        'reconstructed': X_reconstructed,
        'reconstruction_error': reconstruction_error.item(),
        'theoretical_error': theoretical_error.item(),
        'explained_variance_ratio': eigenvalues[:k].sum() / eigenvalues.sum()
    }


def verify_equivalence():
    """
    Verify that minimizing reconstruction error gives same result
    as maximizing variance.
    """
    torch.manual_seed(42)
    
    # Generate data
    n, d = 500, 10
    X = torch.randn(n, d)
    
    # Add correlations
    X[:, 5:] = X[:, :5] @ torch.randn(5, 5) * 0.5 + torch.randn(n, 5) * 0.3
    
    print("Verifying Equivalence of PCA Perspectives")
    print("=" * 50)
    
    for k in [1, 3, 5, 8]:
        result = reconstruction_error_pca(X, k)
        
        # Variance captured
        variance_captured = result['explained_variance_ratio'].item()
        
        # Reconstruction error ratio
        total_variance = torch.var(X - X.mean(dim=0)).item() * d * (n - 1)
        error_ratio = result['reconstruction_error'] / total_variance
        
        print(f"\nk = {k}:")
        print(f"  Variance captured: {variance_captured:.4f}")
        print(f"  Error ratio: {error_ratio:.4f}")
        print(f"  Sum (should be 1): {variance_captured + error_ratio:.4f}")
        print(f"  Computed error: {result['reconstruction_error']:.2f}")
        print(f"  Theoretical error: {result['theoretical_error']:.2f}")


def visualize_reconstruction(X: torch.Tensor, ks: list = [1, 2, 5, 10]):
    """
    Visualize reconstruction quality for different k values.
    """
    n, d = X.shape
    
    fig, axes = plt.subplots(2, len(ks), figsize=(4*len(ks), 8))
    
    results = {}
    for i, k in enumerate(ks):
        result = reconstruction_error_pca(X, k)
        results[k] = result
        
        # Top row: Actual vs reconstructed for first few dimensions
        ax = axes[0, i]
        ax.scatter(X[:50, 0].numpy(), result['reconstructed'][:50, 0].numpy(), 
                   alpha=0.5, s=20)
        ax.plot([X[:, 0].min(), X[:, 0].max()], 
                [X[:, 0].min(), X[:, 0].max()], 'r--', linewidth=2)
        ax.set_xlabel('Original')
        ax.set_ylabel('Reconstructed')
        ax.set_title(f'k={k}, Dim 1')
        ax.grid(True, alpha=0.3)
        
        # Bottom row: Error distribution
        ax = axes[1, i]
        errors = torch.sum((X - result['reconstructed']) ** 2, dim=1)
        ax.hist(errors.numpy(), bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(errors.mean().item(), color='r', linestyle='--', 
                   label=f'Mean: {errors.mean().item():.2f}')
        ax.set_xlabel('Sample Reconstruction Error')
        ax.set_ylabel('Count')
        ax.set_title(f'EVR: {result["explained_variance_ratio"]:.2%}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('reconstruction_error.png', dpi=150)
    plt.show()
    
    return results


def reconstruction_vs_components():
    """
    Plot reconstruction error as function of number of components.
    """
    torch.manual_seed(42)
    
    # Generate data with known structure
    n, d = 500, 50
    
    # Create data with decaying variance structure
    true_eigenvalues = torch.exp(-torch.arange(d).float() / 10)
    true_eigenvalues = true_eigenvalues / true_eigenvalues.sum() * d
    
    # Generate data
    V = torch.linalg.qr(torch.randn(d, d))[0]  # Random orthogonal matrix
    L = torch.diag(torch.sqrt(true_eigenvalues))
    X = torch.randn(n, d) @ L @ V.T
    
    # Compute reconstruction error for each k
    errors = []
    evrs = []
    
    for k in range(1, d + 1):
        result = reconstruction_error_pca(X, k)
        errors.append(result['reconstruction_error'])
        evrs.append(result['explained_variance_ratio'].item())
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot 1: Reconstruction error vs k
    ax = axes[0]
    ax.semilogy(range(1, d + 1), errors, 'b-', linewidth=2)
    ax.set_xlabel('Number of Components (k)')
    ax.set_ylabel('Reconstruction Error (log scale)')
    ax.set_title('Reconstruction Error vs k')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Explained variance ratio
    ax = axes[1]
    ax.plot(range(1, d + 1), evrs, 'g-', linewidth=2)
    ax.axhline(0.95, color='r', linestyle='--', label='95% threshold')
    ax.axhline(0.99, color='orange', linestyle='--', label='99% threshold')
    
    # Find k for 95% and 99%
    k_95 = next(k for k, evr in enumerate(evrs, 1) if evr >= 0.95)
    k_99 = next(k for k, evr in enumerate(evrs, 1) if evr >= 0.99)
    
    ax.axvline(k_95, color='r', linestyle=':', alpha=0.7)
    ax.axvline(k_99, color='orange', linestyle=':', alpha=0.7)
    
    ax.set_xlabel('Number of Components (k)')
    ax.set_ylabel('Explained Variance Ratio')
    ax.set_title(f'95%: k={k_95}, 99%: k={k_99}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: True vs estimated eigenvalues
    ax = axes[2]
    result = reconstruction_error_pca(X, d)
    estimated_eigenvalues = result['eigenvalues'].numpy()
    
    ax.semilogy(range(1, d + 1), true_eigenvalues.numpy(), 'b-', 
                linewidth=2, label='True')
    ax.semilogy(range(1, d + 1), estimated_eigenvalues, 'r--', 
                linewidth=2, label='Estimated')
    ax.set_xlabel('Component')
    ax.set_ylabel('Eigenvalue (log scale)')
    ax.set_title('Eigenvalue Spectrum')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('reconstruction_vs_k.png', dpi=150)
    plt.show()


def mnist_reconstruction_demo():
    """
    Demonstrate reconstruction on MNIST images.
    """
    from torchvision import datasets, transforms
    
    # Load MNIST
    transform = transforms.ToTensor()
    mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    
    # Get subset
    n_samples = 1000
    X = mnist.data[:n_samples].float().view(n_samples, -1) / 255.0
    
    print("MNIST Reconstruction Demo")
    print("=" * 50)
    
    fig, axes = plt.subplots(6, 10, figsize=(15, 9))
    
    # Original images (first row)
    for i in range(10):
        axes[0, i].imshow(X[i].numpy().reshape(28, 28), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('Original', fontsize=10)
    
    # Reconstructions with different k
    k_values = [5, 10, 20, 50, 100]
    
    for row, k in enumerate(k_values, 1):
        result = reconstruction_error_pca(X, k)
        evr = result['explained_variance_ratio'].item()
        
        for i in range(10):
            recon = result['reconstructed'][i].numpy().reshape(28, 28)
            axes[row, i].imshow(recon, cmap='gray', vmin=0, vmax=1)
            axes[row, i].axis('off')
            if i == 0:
                axes[row, i].set_ylabel(f'k={k}\n({evr:.1%})', fontsize=10)
    
    plt.suptitle('MNIST Reconstruction with Different Numbers of Components', fontsize=14)
    plt.tight_layout()
    plt.savefig('mnist_reconstruction.png', dpi=150)
    plt.show()


# Main execution
if __name__ == "__main__":
    verify_equivalence()
    
    # Generate test data
    torch.manual_seed(42)
    n, d = 500, 20
    X = torch.randn(n, d)
    X[:, 10:] = X[:, :10] @ torch.randn(10, 10) * 0.3 + torch.randn(n, 10) * 0.5
    
    visualize_reconstruction(X, ks=[1, 5, 10, 15])
    reconstruction_vs_components()
    
    # MNIST demo (uncomment if data available)
    # mnist_reconstruction_demo()
```

---

## Theoretical Results

### Eckart-Young Theorem

PCA gives the **optimal** linear dimensionality reduction:

**Theorem:** Among all rank-$k$ approximations to $X$, the PCA reconstruction minimizes $\|X - \hat{X}\|_F$.

$$X_k^{PCA} = \arg\min_{\text{rank}(\hat{X}) \leq k} \|X - \hat{X}\|_F$$

### Error Decomposition

Total variance = Captured variance + Reconstruction error

$$\sum_{i=1}^d \lambda_i = \sum_{i=1}^k \lambda_i + \sum_{i=k+1}^d \lambda_i$$

$$\text{Total} = \text{Explained} + \text{Residual}$$

---

## Key Insights

### Why the Perspectives Are Equivalent

| Maximize Variance | Minimize Error |
|------------------|----------------|
| $\max \text{tr}(W^T\Sigma W)$ | $\min \|X - XWW^T\|_F^2$ |
| Keep most information | Lose least information |
| Same solution: top eigenvectors | Same solution: top eigenvectors |

### Geometric Intuition

- **Variance maximization:** Find directions of greatest spread
- **Error minimization:** Find subspace closest to all points

Both lead to the principal subspace because projecting onto directions of high variance minimizes the distance to the subspace.

---

## Exercises

### Exercise 1: Analytical Verification

For 2D data with covariance $\Sigma = \begin{pmatrix} 2 & 1 \\ 1 & 1 \end{pmatrix}$:

a) Compute eigenvalues and eigenvectors  
b) For $k=1$, compute the reconstruction matrix $WW^T$  
c) Verify reconstruction error equals $\lambda_2$

### Exercise 2: Optimal Decoder

Prove that when $W$ has orthonormal columns, the optimal decoder is $D = W$.

### Exercise 3: Non-Orthogonal Projection

What happens if we don't require $W^TW = I$?

a) Can we still achieve zero reconstruction error?  
b) What's wrong with using $W = I$?  
c) Why is the orthogonality constraint necessary?

### Exercise 4: Probabilistic Interpretation

Show that minimizing reconstruction error under Gaussian noise is equivalent to maximum likelihood PCA.

---

## Summary

| Concept | Formula/Description |
|---------|---------------------|
| **Objective** | $\min \|X - XWW^T\|_F^2$ |
| **Constraint** | $W^TW = I_k$ |
| **Solution** | Top $k$ eigenvectors of covariance |
| **Min error** | $\sum_{i=k+1}^d \lambda_i$ |
| **EVR** | $\sum_{i=1}^k \lambda_i / \sum_{i=1}^d \lambda_i$ |
| **Key insight** | Variance maximization ≡ Error minimization |

---

## Next: PCA as Linear Autoencoder

The next section shows how PCA relates to linear autoencoders, bridging to neural network approaches.
