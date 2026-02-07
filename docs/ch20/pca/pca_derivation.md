# PCA Derivation

Rigorous derivation of PCA from the variance-maximization and minimum-error perspectives.

---

## Overview

PCA admits two equivalent formulations: **maximum variance** (find directions along which projected data has the greatest spread) and **minimum reconstruction error** (find the rank-$k$ linear projection that best approximates the original data). This section derives both, proves their equivalence, and establishes the connection to linear autoencoders.

---

## Setup and Notation

Let $\mathbf{X} \in \mathbb{R}^{n \times d}$ be a centered data matrix with $n$ observations in $d$ dimensions, where each row $\mathbf{x}^{(i)}$ has been mean-subtracted: $\frac{1}{n}\sum_{i=1}^n \mathbf{x}^{(i)} = \mathbf{0}$.

The sample covariance matrix is:

$$\boldsymbol{\Sigma} = \frac{1}{n}\mathbf{X}^T\mathbf{X} \in \mathbb{R}^{d \times d}$$

Since $\boldsymbol{\Sigma}$ is real symmetric and positive semi-definite, it admits a spectral decomposition:

$$\boldsymbol{\Sigma} = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^T = \sum_{i=1}^d \lambda_i \mathbf{v}_i \mathbf{v}_i^T$$

where $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_d \geq 0$ are eigenvalues and $\{\mathbf{v}_1, \ldots, \mathbf{v}_d\}$ is an orthonormal eigenbasis.

!!! note "Convention: $1/n$ vs. $1/(n-1)$"
    The derivations use $1/n$ (population-style). Using $1/(n-1)$ (Bessel correction) changes eigenvalue magnitudes but not eigenvector directions, so the principal components are identical under either convention.

---

## Derivation 1: Variance Maximization

### Single Component ($k = 1$)

We seek a unit vector $\mathbf{v} \in \mathbb{R}^d$ that maximizes the variance of the projected data $\{z^{(i)} = {\mathbf{x}^{(i)}}^T \mathbf{v}\}_{i=1}^n$.

Since the data is centered, the projected mean is zero:

$$\bar{z} = \frac{1}{n}\sum_{i=1}^n {\mathbf{x}^{(i)}}^T \mathbf{v} = \left(\frac{1}{n}\sum_{i=1}^n \mathbf{x}^{(i)}\right)^T \mathbf{v} = \mathbf{0}^T \mathbf{v} = 0$$

The projected variance is:

$$\operatorname{Var}(z) = \frac{1}{n}\sum_{i=1}^n \left({\mathbf{x}^{(i)}}^T \mathbf{v}\right)^2 = \frac{1}{n}\sum_{i=1}^n \mathbf{v}^T \mathbf{x}^{(i)} {\mathbf{x}^{(i)}}^T \mathbf{v} = \mathbf{v}^T \left(\frac{1}{n}\mathbf{X}^T\mathbf{X}\right) \mathbf{v} = \mathbf{v}^T \boldsymbol{\Sigma} \mathbf{v}$$

The optimization problem is:

$$\max_{\mathbf{v}} \; \mathbf{v}^T \boldsymbol{\Sigma} \mathbf{v} \quad \text{subject to} \quad \mathbf{v}^T \mathbf{v} = 1$$

**Solution via Lagrange multipliers.** Form the Lagrangian:

$$\mathcal{L}(\mathbf{v}, \lambda) = \mathbf{v}^T \boldsymbol{\Sigma} \mathbf{v} - \lambda(\mathbf{v}^T \mathbf{v} - 1)$$

Setting the gradient to zero:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{v}} = 2\boldsymbol{\Sigma}\mathbf{v} - 2\lambda\mathbf{v} = \mathbf{0}$$

$$\boldsymbol{\Sigma}\mathbf{v} = \lambda\mathbf{v}$$

This is the eigenvalue equation for $\boldsymbol{\Sigma}$. The optimal $\mathbf{v}$ must be an eigenvector, and the corresponding eigenvalue $\lambda$ gives the projected variance:

$$\mathbf{v}^T \boldsymbol{\Sigma} \mathbf{v} = \mathbf{v}^T \lambda \mathbf{v} = \lambda \|\mathbf{v}\|^2 = \lambda$$

To maximize variance, we choose the eigenvector corresponding to the **largest eigenvalue** $\lambda_1$. Thus $\mathbf{v}_1^* = \mathbf{v}_1$ (the first principal component).

### Multiple Components ($k > 1$)

For the second principal component, we maximize variance subject to orthogonality with $\mathbf{v}_1$:

$$\max_{\mathbf{v}} \; \mathbf{v}^T \boldsymbol{\Sigma} \mathbf{v} \quad \text{s.t.} \quad \mathbf{v}^T \mathbf{v} = 1, \; \mathbf{v}^T \mathbf{v}_1 = 0$$

The Lagrangian is:

$$\mathcal{L} = \mathbf{v}^T \boldsymbol{\Sigma} \mathbf{v} - \lambda(\mathbf{v}^T \mathbf{v} - 1) - \mu(\mathbf{v}^T \mathbf{v}_1)$$

Setting $\partial \mathcal{L}/\partial \mathbf{v} = 0$:

$$2\boldsymbol{\Sigma}\mathbf{v} - 2\lambda\mathbf{v} - \mu\mathbf{v}_1 = \mathbf{0}$$

Left-multiplying by $\mathbf{v}_1^T$ and using $\mathbf{v}_1^T \mathbf{v} = 0$ and $\boldsymbol{\Sigma}\mathbf{v}_1 = \lambda_1 \mathbf{v}_1$:

$$2\lambda_1 \underbrace{\mathbf{v}_1^T \mathbf{v}}_{= 0} - 2\lambda \underbrace{\mathbf{v}_1^T \mathbf{v}}_{= 0} - \mu \underbrace{\mathbf{v}_1^T \mathbf{v}_1}_{= 1} = 0 \implies \mu = 0$$

With $\mu = 0$, the stationarity condition reduces to $\boldsymbol{\Sigma}\mathbf{v} = \lambda\mathbf{v}$, so $\mathbf{v}$ is again an eigenvector. Excluding $\mathbf{v}_1$ by the orthogonality constraint, the variance-maximizing choice is $\mathbf{v}_2$ (eigenvalue $\lambda_2$).

By induction, the $k$-th principal component is the eigenvector with the $k$-th largest eigenvalue.

### Simultaneous Formulation

The $k$-component problem can also be stated as a single optimization over a matrix $\mathbf{W} \in \mathbb{R}^{d \times k}$:

$$\max_{\mathbf{W}} \; \operatorname{tr}\!\left(\mathbf{W}^T \boldsymbol{\Sigma} \mathbf{W}\right) \quad \text{s.t.} \quad \mathbf{W}^T \mathbf{W} = \mathbf{I}_k$$

The trace objective sums the variance along each projected direction. The solution is $\mathbf{W}^* = [\mathbf{v}_1, \ldots, \mathbf{v}_k]$, and the maximum value is $\sum_{i=1}^k \lambda_i$.

---

## Derivation 2: Minimum Reconstruction Error

### Formulation

Instead of maximizing variance, we can seek the rank-$k$ projection that minimizes the average squared reconstruction error:

$$\min_{\mathbf{W}} \; \frac{1}{n}\sum_{i=1}^n \left\|\mathbf{x}^{(i)} - \mathbf{W}\mathbf{W}^T \mathbf{x}^{(i)}\right\|^2 \quad \text{s.t.} \quad \mathbf{W}^T \mathbf{W} = \mathbf{I}_k$$

Here $\mathbf{W}\mathbf{W}^T$ is the orthogonal projection onto the column space of $\mathbf{W}$.

### Derivation

Expand the squared norm:

$$\left\|\mathbf{x} - \mathbf{W}\mathbf{W}^T \mathbf{x}\right\|^2 = \mathbf{x}^T\mathbf{x} - 2\mathbf{x}^T\mathbf{W}\mathbf{W}^T\mathbf{x} + \mathbf{x}^T\mathbf{W}\underbrace{\mathbf{W}^T\mathbf{W}}_{=\mathbf{I}}\mathbf{W}^T\mathbf{x} = \mathbf{x}^T\mathbf{x} - \mathbf{x}^T\mathbf{W}\mathbf{W}^T\mathbf{x}$$

Summing over samples:

$$\frac{1}{n}\sum_i \left\|\mathbf{x}^{(i)} - \mathbf{W}\mathbf{W}^T\mathbf{x}^{(i)}\right\|^2 = \underbrace{\frac{1}{n}\sum_i \left\|\mathbf{x}^{(i)}\right\|^2}_{\text{constant}} - \operatorname{tr}\!\left(\mathbf{W}^T \boldsymbol{\Sigma} \mathbf{W}\right)$$

Minimizing this is equivalent to maximizing $\operatorname{tr}(\mathbf{W}^T \boldsymbol{\Sigma} \mathbf{W})$, which is exactly the variance-maximization problem. The two formulations are therefore equivalent.

### Closed-Form Error

Using the eigenbasis $\{\mathbf{v}_1, \ldots, \mathbf{v}_d\}$, the total data variance decomposes as:

$$\frac{1}{n}\sum_i \left\|\mathbf{x}^{(i)}\right\|^2 = \operatorname{tr}(\boldsymbol{\Sigma}) = \sum_{j=1}^d \lambda_j$$

The variance captured by the top-$k$ components is $\sum_{j=1}^k \lambda_j$, so the reconstruction error is:

$$\mathcal{E}_k = \sum_{j=1}^d \lambda_j - \sum_{j=1}^k \lambda_j = \sum_{j=k+1}^d \lambda_j$$

**The reconstruction error equals the sum of discarded eigenvalues.**

---

## Equivalence Summary

| Formulation | Objective | Solution |
|-------------|-----------|----------|
| **Max variance** | $\max_{\mathbf{W}} \operatorname{tr}(\mathbf{W}^T\boldsymbol{\Sigma}\mathbf{W})$ | Top-$k$ eigenvectors |
| **Min error** | $\min_{\mathbf{W}} \frac{1}{n}\sum_i \|\mathbf{x}^{(i)} - \mathbf{W}\mathbf{W}^T\mathbf{x}^{(i)}\|^2$ | Top-$k$ eigenvectors |

Both yield the same $\mathbf{W}^* = [\mathbf{v}_1, \ldots, \mathbf{v}_k]$.

---

## The $k$-Dimensional PCA Algorithm

Given centered data $\mathbf{X} \in \mathbb{R}^{n \times d}$ and target dimension $k$:

**Step 1.** Compute the covariance matrix $\boldsymbol{\Sigma} = \frac{1}{n}\mathbf{X}^T\mathbf{X}$.

**Step 2.** Find the eigendecomposition $\boldsymbol{\Sigma} = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^T$ with eigenvalues sorted in descending order.

**Step 3.** Form the loading matrix $\mathbf{W} = [\mathbf{v}_1, \ldots, \mathbf{v}_k] \in \mathbb{R}^{d \times k}$.

**Step 4.** Compute scores: $\mathbf{Z} = \mathbf{X}\mathbf{W} \in \mathbb{R}^{n \times k}$.

**Step 5.** Reconstruct: $\hat{\mathbf{X}} = \mathbf{Z}\mathbf{W}^T = \mathbf{X}\mathbf{W}\mathbf{W}^T \in \mathbb{R}^{n \times d}$.

The reconstruction $\hat{\mathbf{x}}^{(i)}$ is a sum of weighted principal directions:

$$\hat{\mathbf{x}}^{(i)} = \sum_{j=1}^k \underbrace{\left({\mathbf{x}^{(i)}}^T \mathbf{v}_j\right)}_{\text{score}} \, \mathbf{v}_j$$

```python
import numpy as np

def pca(X, k):
    """PCA: variance maximization / minimum reconstruction error.

    Args:
        X: Centered data [n, d]
        k: Number of components

    Returns:
        W: Loadings [d, k]
        Z: Scores [n, k]
        eigenvalues: Variance per component [k]
    """
    n = X.shape[0]
    Sigma = X.T @ X / n

    eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
    idx = np.argsort(eigenvalues)[::-1]

    W = eigenvectors[:, idx[:k]]
    eigenvalues = eigenvalues[idx[:k]]
    Z = X @ W

    return W, Z, eigenvalues
```

---

## Derivation 3: Linear Autoencoder Equivalence

### Setup

Consider a linear autoencoder with encoder $\mathbf{W}_e \in \mathbb{R}^{d \times k}$ and decoder $\mathbf{W}_d \in \mathbb{R}^{d \times k}$:

$$\text{Encode:} \quad \mathbf{z} = \mathbf{W}_e^T \mathbf{x}, \qquad \text{Decode:} \quad \hat{\mathbf{x}} = \mathbf{W}_d \mathbf{z}$$

The reconstruction is $\hat{\mathbf{x}} = \mathbf{W}_d \mathbf{W}_e^T \mathbf{x}$, and the MSE loss is:

$$\mathcal{L}(\mathbf{W}_e, \mathbf{W}_d) = \frac{1}{n}\sum_{i=1}^n \left\|\mathbf{x}^{(i)} - \mathbf{W}_d \mathbf{W}_e^T \mathbf{x}^{(i)}\right\|^2$$

### Optimal Decoder Given Encoder

For fixed $\mathbf{W}_e$, the loss is quadratic in $\mathbf{W}_d$. Taking the derivative and setting to zero:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_d} = -\frac{2}{n}\sum_i \left(\mathbf{x}^{(i)} - \mathbf{W}_d \mathbf{W}_e^T \mathbf{x}^{(i)}\right) {\mathbf{x}^{(i)}}^T \mathbf{W}_e = \mathbf{0}$$

$$\boldsymbol{\Sigma}\mathbf{W}_e = \mathbf{W}_d (\mathbf{W}_e^T \boldsymbol{\Sigma} \mathbf{W}_e)$$

If $\mathbf{W}_e^T \boldsymbol{\Sigma} \mathbf{W}_e$ is invertible, then:

$$\mathbf{W}_d = \boldsymbol{\Sigma}\mathbf{W}_e (\mathbf{W}_e^T \boldsymbol{\Sigma} \mathbf{W}_e)^{-1}$$

### At the Global Optimum

At the global minimum of $\mathcal{L}$, the column spaces of $\mathbf{W}_e$ and $\mathbf{W}_d$ coincide with the subspace spanned by the top-$k$ eigenvectors of $\boldsymbol{\Sigma}$. When $\mathbf{W}_e$ has orthonormal columns aligned with the eigenvectors, $\mathbf{W}_d = \mathbf{W}_e$, and the reconstruction matrix becomes:

$$\mathbf{W}_d \mathbf{W}_e^T = \mathbf{W}\mathbf{W}^T$$

which is identical to the PCA projection.

### Practical Implication

Training a linear autoencoder (no activation functions, no bias) with MSE loss by gradient descent converges to the PCA solution. The loss at convergence equals the PCA reconstruction error $\sum_{j=k+1}^d \lambda_j$.

```python
import torch
import torch.nn as nn

class LinearAutoencoder(nn.Module):
    """Learns PCA solution via gradient descent."""
    def __init__(self, d, k):
        super().__init__()
        self.encoder = nn.Linear(d, k, bias=False)
        self.decoder = nn.Linear(k, d, bias=False)

    def forward(self, x):
        return self.decoder(self.encoder(x))


def verify_equivalence(X, k, epochs=5000, lr=0.01):
    """Compare PCA and linear AE reconstruction errors."""
    X_centered = X - X.mean(axis=0)

    # Analytical PCA
    Sigma = X_centered.T @ X_centered / X.shape[0]
    eigvals = np.linalg.eigh(Sigma)[0]
    eigvals = np.sort(eigvals)[::-1]
    pca_error = eigvals[k:].sum()

    # Linear autoencoder
    X_t = torch.tensor(X_centered, dtype=torch.float32)
    model = LinearAutoencoder(X.shape[1], k)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        opt.zero_grad()
        loss = ((X_t - model(X_t)) ** 2).mean()
        loss.backward()
        opt.step()

    ae_error = loss.item() * X.shape[1]  # MSE -> total error

    print(f"PCA error:       {pca_error:.6f}")
    print(f"Linear AE error: {ae_error:.6f}")
```

---

## Explained Variance Analysis

### Per-Component Ratio

$$\text{EVR}_k = \frac{\lambda_k}{\sum_{i=1}^d \lambda_i} = \frac{\lambda_k}{\operatorname{tr}(\boldsymbol{\Sigma})}$$

### Cumulative Ratio

$$\text{CEVR}_k = \frac{\sum_{i=1}^k \lambda_i}{\sum_{i=1}^d \lambda_i} = 1 - \frac{\mathcal{E}_k}{\operatorname{tr}(\boldsymbol{\Sigma})}$$

The cumulative ratio directly measures the fraction of total variance retained. Its complement gives the fractional reconstruction error.

```python
def explained_variance_analysis(eigenvalues):
    """Compute explained variance ratios and reconstruction errors."""
    total = eigenvalues.sum()
    evr = eigenvalues / total
    cumulative_evr = np.cumsum(evr)
    reconstruction_error = total - np.cumsum(eigenvalues)

    return evr, cumulative_evr, reconstruction_error
```

---

## Properties of the PCA Solution

### Uncorrelated Scores

The score vectors are uncorrelated:

$$\operatorname{Cov}(\mathbf{z}) = \frac{1}{n}\mathbf{Z}^T\mathbf{Z} = \frac{1}{n}\mathbf{W}^T \mathbf{X}^T \mathbf{X} \mathbf{W} = \mathbf{W}^T \boldsymbol{\Sigma} \mathbf{W} = \boldsymbol{\Lambda}_k$$

where $\boldsymbol{\Lambda}_k = \operatorname{diag}(\lambda_1, \ldots, \lambda_k)$. The covariance of the projected data is diagonal — the principal components decorrelate the data.

### Optimality

**Eckart–Young–Mirsky theorem.** Among all rank-$k$ matrices, $\hat{\mathbf{X}} = \mathbf{X}\mathbf{W}\mathbf{W}^T$ minimizes $\|\mathbf{X} - \hat{\mathbf{X}}\|_F^2$. This is a stronger statement than PCA optimality among orthogonal projections: no rank-$k$ matrix (projection or otherwise) achieves lower Frobenius-norm error.

### Variance Decomposition

Total variance decomposes into retained and lost components:

$$\underbrace{\operatorname{tr}(\boldsymbol{\Sigma})}_{\text{total}} = \underbrace{\sum_{i=1}^k \lambda_i}_{\text{retained}} + \underbrace{\sum_{i=k+1}^d \lambda_i}_{\text{reconstruction error}}$$

---

## Summary

| Result | Statement |
|--------|-----------|
| **First PC** | Eigenvector of $\boldsymbol{\Sigma}$ with largest eigenvalue |
| **$k$-th PC** | Eigenvector with $k$-th largest eigenvalue |
| **Projected variance** | Equals the eigenvalue: $\operatorname{Var}(z_k) = \lambda_k$ |
| **Reconstruction error** | Sum of discarded eigenvalues: $\sum_{j > k} \lambda_j$ |
| **Max variance ≡ Min error** | Same solution: top-$k$ eigenvectors |
| **Linear AE ≡ PCA** | MSE-trained linear AE converges to PCA solution |
| **Scores uncorrelated** | $\operatorname{Cov}(\mathbf{z}) = \boldsymbol{\Lambda}_k$ |
| **Optimality** | Best rank-$k$ approximation (Eckart–Young–Mirsky) |
