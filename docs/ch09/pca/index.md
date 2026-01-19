# Principal Component Analysis

A comprehensive introduction to PCA as a foundation for understanding autoencoders and dimensionality reduction.

---

## Overview

**What you'll learn:**

- Theoretical foundations of PCA
- Multiple derivations (variance maximization, reconstruction error)
- Relationship between PCA and linear autoencoders
- Computational methods (eigendecomposition, SVD)
- Limitations that motivate nonlinear methods

**Prerequisites:**

- Linear algebra (matrices, eigenvalues, eigenvectors)
- Basic statistics (variance, covariance)
- Calculus (optimization, Lagrange multipliers)

---

## Mathematical Foundation

### The Dimensionality Reduction Problem

Given high-dimensional data $X \in \mathbb{R}^{n \times d}$ with $n$ samples and $d$ features, find a lower-dimensional representation $Z \in \mathbb{R}^{n \times k}$ where $k < d$ that preserves the essential structure of the data.

**Key Question:** What makes a "good" lower-dimensional representation?

### PCA Objectives

PCA can be derived from two equivalent perspectives:

| Perspective | Objective | Result |
|-------------|-----------|--------|
| **Variance Maximization** | Find directions of maximum variance | Principal components ordered by variance |
| **Reconstruction Error** | Minimize squared reconstruction error | Same principal components |

### Core Result

The optimal linear projection is given by the eigenvectors of the covariance matrix:

$$\Sigma = \frac{1}{n-1} X^T X$$

where $X$ is centered (mean-subtracted).

**Eigendecomposition:**

$$\Sigma = V \Lambda V^T$$

- $V = [v_1, v_2, \ldots, v_d]$: orthonormal eigenvectors (principal directions)
- $\Lambda = \text{diag}(\lambda_1, \lambda_2, \ldots, \lambda_d)$: eigenvalues (variances)
- Convention: $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_d$

---

## Section Contents

### 9.1.1 Variance Maximization

Derive PCA by finding directions that maximize projected variance.

**Key concepts:**
- Lagrange multiplier optimization
- Rayleigh quotient
- Sequential extraction of components

### 9.1.2 Eigendecomposition

Computational approach using eigenvalue decomposition of the covariance matrix.

**Key concepts:**
- Covariance matrix construction
- Eigenvalue problem solution
- Component selection criteria

### 9.1.3 Singular Value Decomposition

Alternative computation via SVD, often more numerically stable.

**Key concepts:**
- SVD formulation: $X = U \Sigma V^T$
- Relationship to eigendecomposition
- Computational advantages

### 9.1.4 Reconstruction Error

Derive PCA by minimizing squared reconstruction error.

**Key concepts:**
- Projection and reconstruction
- Error decomposition
- Equivalence to variance maximization

### 9.1.5 PCA as Linear Autoencoder

Connection between PCA and linear autoencoders.

**Key concepts:**
- Encoder-decoder interpretation
- Why linear autoencoders learn PCA subspace
- Weight tying and orthogonality

### 9.1.6 Limitations of Linear Methods

Motivate nonlinear approaches (autoencoders, VAEs).

**Key concepts:**
- Linearity assumption
- Failure on nonlinear manifolds
- Curse of dimensionality

---

## Quick Reference

### Notation

| Symbol | Meaning |
|--------|---------|
| $X \in \mathbb{R}^{n \times d}$ | Data matrix (centered) |
| $\Sigma \in \mathbb{R}^{d \times d}$ | Covariance matrix |
| $v_i \in \mathbb{R}^d$ | $i$-th principal component |
| $\lambda_i$ | Variance along $i$-th component |
| $k$ | Number of components retained |
| $z_i \in \mathbb{R}^k$ | Projected representation of sample $i$ |

### Key Formulas

**Projection to $k$ dimensions:**

$$Z = X V_k$$

where $V_k = [v_1, \ldots, v_k]$ contains the top $k$ eigenvectors.

**Reconstruction:**

$$\hat{X} = Z V_k^T = X V_k V_k^T$$

**Explained variance ratio:**

$$\text{EVR}_k = \frac{\sum_{i=1}^{k} \lambda_i}{\sum_{i=1}^{d} \lambda_i}$$

**Reconstruction error:**

$$\mathcal{L} = \|X - \hat{X}\|_F^2 = \sum_{i=k+1}^{d} \lambda_i$$

---

## Learning Path

```
Variance Maximization (9.1.1)
        ↓
Eigendecomposition (9.1.2) ←→ SVD (9.1.3)
        ↓
Reconstruction Error (9.1.4)
        ↓
PCA as Linear AE (9.1.5)
        ↓
Limitations (9.1.6) → Autoencoders (9.2)
```

---

## Connection to Autoencoders

PCA provides the theoretical foundation for understanding autoencoders:

| Aspect | PCA | Autoencoder |
|--------|-----|-------------|
| Transformation | Linear | Nonlinear |
| Optimization | Closed-form | Iterative (gradient descent) |
| Basis vectors | Orthogonal | Not necessarily |
| Computation | $O(d^3)$ or $O(nd^2)$ | Flexible (batch training) |
| Expressivity | Limited to linear subspaces | Can capture nonlinear manifolds |

**Key Insight:** A linear autoencoder (no activation functions) learns to span the same subspace as PCA, though it may not find the same basis vectors unless weights are tied and orthogonality is enforced.

---

## PyTorch Preview

```python
import torch
import numpy as np
from sklearn.decomposition import PCA

# Using sklearn
pca = PCA(n_components=k)
Z = pca.fit_transform(X)
X_reconstructed = pca.inverse_transform(Z)

# Manual implementation
X_centered = X - X.mean(axis=0)
cov = np.cov(X_centered.T)
eigenvalues, eigenvectors = np.linalg.eigh(cov)

# Sort by descending eigenvalue
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Project
V_k = eigenvectors[:, :k]
Z = X_centered @ V_k
X_reconstructed = Z @ V_k.T + X.mean(axis=0)
```

---

## References

### Foundational

1. Pearson, K. (1901). "On lines and planes of closest fit to systems of points in space." *Philosophical Magazine*.
2. Hotelling, H. (1933). "Analysis of a complex of statistical variables into principal components." *Journal of Educational Psychology*.

### Modern Treatment

- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Chapter 12.
- Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. Chapter 12.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. Chapter 5.

---

## Next Steps

After completing this section:

1. **Section 9.2: Autoencoders** — Extend to nonlinear dimensionality reduction
2. **Section 9.3: Variational Autoencoders** — Add probabilistic structure
