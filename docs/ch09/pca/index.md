# Principal Component Analysis

Linear dimensionality reduction through variance maximization.

---

## Overview

**Principal Component Analysis (PCA)** is a fundamental technique for dimensionality reduction that finds orthogonal directions of maximum variance in data. Understanding PCA provides essential background for autoencoders and VAEs.

---

## Learning Objectives

- Understand PCA from multiple perspectives (variance, projection, eigenvectors)
- Implement PCA using eigendecomposition and SVD
- Compute and interpret reconstruction error
- Connect PCA to linear autoencoders
- Recognize limitations motivating nonlinear methods

---

## Why PCA Before Autoencoders?

| PCA Concept | Autoencoder Connection |
|-------------|------------------------|
| Linear compression | Encoder function |
| Reconstruction | Decoder function |
| Minimize reconstruction error | Training objective |
| Principal components | Learned latent directions |

PCA is a **linear autoencoder** with an analytical solution.

---

## Key Equations

| Concept | Formula |
|---------|---------|
| **Covariance matrix** | $C = \frac{1}{n}X^TX$ |
| **Eigenproblem** | $Cv = \lambda v$ |
| **Projection** | $z = W^T x$ |
| **Reconstruction** | $\hat{x} = Wz = WW^T x$ |
| **Reconstruction error** | $\sum_{i=k+1}^{d} \lambda_i$ |

---

## Section Contents

1. **Variance Maximization** - PCA as finding directions of maximum variance
2. **Eigendecomposition** - Computing PCA via covariance eigenvectors
3. **SVD** - Efficient computation via Singular Value Decomposition
4. **Reconstruction Error** - Quantifying information loss
5. **PCA as Linear Autoencoder** - Connecting to neural networks
6. **Limitations** - Why we need nonlinear methods

---

## Quick Implementation

```python
import torch
import numpy as np

def pca(X, n_components):
    """
    PCA via eigendecomposition.
    
    Args:
        X: Data matrix [n_samples, n_features]
        n_components: Number of principal components
    
    Returns:
        W: Principal components [n_features, n_components]
        z: Projected data [n_samples, n_components]
    """
    # Center data
    X_centered = X - X.mean(dim=0)
    
    # Covariance matrix
    cov = X_centered.T @ X_centered / (X.shape[0] - 1)
    
    # Eigendecomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    
    # Sort by descending eigenvalue
    idx = torch.argsort(eigenvalues, descending=True)
    W = eigenvectors[:, idx[:n_components]]
    
    # Project
    z = X_centered @ W
    
    return W, z
```

---

## Summary

PCA provides the theoretical foundation for understanding autoencoders as nonlinear generalizations of linear dimensionality reduction.

---

## What's Next

The following sections provide detailed derivations and implementations of each PCA aspect.
