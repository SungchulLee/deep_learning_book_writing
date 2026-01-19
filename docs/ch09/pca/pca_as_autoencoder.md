# PCA as Linear Autoencoder

Understanding the connection between PCA and neural network autoencoders.

---

## Overview

**Key Insight:** A linear autoencoder (no activation functions) learns to span the same subspace as PCA.

**Bridge to Deep Learning:** This connection motivates nonlinear autoencoders for capturing complex data structure.

**Time:** ~30 minutes  
**Level:** Intermediate-Advanced

---

## The Linear Autoencoder

### Architecture

A linear autoencoder with:
- Input dimension: $d$
- Latent dimension: $k$ (where $k < d$)
- No activation functions

**Encoder:** $z = W_e^T x$ where $W_e \in \mathbb{R}^{d \times k}$

**Decoder:** $\hat{x} = W_d z$ where $W_d \in \mathbb{R}^{d \times k}$

**Full mapping:** $\hat{x} = W_d W_e^T x$

### Loss Function

Mean squared reconstruction error:

$$\mathcal{L}(W_e, W_d) = \frac{1}{n} \sum_{i=1}^n \|x_i - W_d W_e^T x_i\|^2$$

---

## Theorem: Linear AE ≈ PCA

### Statement

**Theorem:** At any local minimum of the linear autoencoder loss:

1. The columns of $W_e$ span the principal subspace (top $k$ eigenvectors of covariance)
2. The reconstruction $\hat{X} = XW_eW_e^T$ equals the PCA reconstruction
3. The reconstruction error equals the PCA reconstruction error

### Important Caveat

The linear autoencoder finds **the same subspace** as PCA, but:
- May not find the **same basis vectors**
- Any rotation within the subspace is also optimal
- Weight tying ($W_d = W_e$) and orthogonality constraints are needed for exact PCA

---

## PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

class LinearAutoencoder(nn.Module):
    """
    Linear autoencoder (no activation functions).
    At convergence, learns to span the PCA subspace.
    """
    
    def __init__(self, input_dim: int, latent_dim: int, tie_weights: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.tie_weights = tie_weights
        
        self.encoder = nn.Linear(input_dim, latent_dim, bias=False)
        
        if tie_weights:
            self.decoder = None
        else:
            self.decoder = nn.Linear(latent_dim, input_dim, bias=False)
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        if self.tie_weights:
            return z @ self.encoder.weight
        else:
            return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        x_reconstructed = self.decode(z)
        return x_reconstructed, z


def compare_linear_ae_with_pca(X: torch.Tensor, k: int):
    """Compare linear autoencoder with PCA."""
    n, d = X.shape
    X_centered = X - X.mean(dim=0)
    
    # PCA via eigendecomposition
    cov = (X_centered.T @ X_centered) / (n - 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    idx = torch.argsort(eigenvalues, descending=True)
    pca_components = eigenvectors[:, idx[:k]]
    
    # PCA reconstruction
    Z_pca = X_centered @ pca_components
    X_pca_recon = Z_pca @ pca_components.T + X.mean(dim=0)
    pca_error = torch.mean((X - X_pca_recon) ** 2).item()
    
    # Linear autoencoder
    model = LinearAutoencoder(d, k, tie_weights=True)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(2000):
        optimizer.zero_grad()
        x_recon, _ = model(X_centered)
        loss = torch.mean((x_recon - X_centered) ** 2)
        loss.backward()
        optimizer.step()
    
    X_ae_recon, _ = model(X_centered)
    ae_error = torch.mean((X_centered - X_ae_recon) ** 2).item()
    
    print(f"PCA reconstruction error: {pca_error:.6f}")
    print(f"Linear AE reconstruction error: {ae_error:.6f}")
    print(f"Errors match: {np.isclose(pca_error, ae_error, rtol=0.01)}")
```

---

## Why Linear AE Learns PCA Subspace

### Gradient Analysis

The gradient of the loss with respect to encoder weights:

$$\frac{\partial \mathcal{L}}{\partial W_e} = -\frac{2}{n} X^T (X - XW_e W_d^T) W_d$$

At equilibrium, this equals zero, which implies the solution lies in the principal subspace.

### Subspace vs Basis

| Property | PCA | Linear AE |
|----------|-----|-----------|
| Subspace | Top-$k$ principal subspace | Same subspace |
| Basis | Orthonormal eigenvectors | Any basis for subspace |
| Ordered | Yes (by variance) | No |
| Unique | Yes | No (infinitely many) |

---

## Key Takeaways

1. **Linear autoencoders converge to PCA subspace** — optimal reconstruction requires principal directions

2. **Nonlinearity is essential for going beyond PCA** — activation functions enable capturing nonlinear structure

3. **This motivates deep autoencoders** — stacking nonlinear layers can learn complex manifolds

---

## Exercises

### Exercise 1: Verify Subspace Equivalence
Train a linear autoencoder and verify that its encoder weights span the same subspace as PCA components.

### Exercise 2: Effect of Nonlinearity
Add ReLU activations and compare reconstruction error with linear version.

### Exercise 3: Weight Tying
Compare tied vs untied weights in terms of convergence and final solution.

---

## Next: Limitations of Linear Methods

The next section discusses why linear methods fail on nonlinear data manifolds.
