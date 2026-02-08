# 29.3.2 Graph Fourier Transform

## Introduction

The **Graph Fourier Transform (GFT)** generalizes the classical Fourier transform to graph-structured data. It decomposes graph signals into frequency components defined by the eigenvectors of the graph Laplacian, enabling spectral analysis and filtering of signals on graphs.

## Classical Fourier Transform Analogy

In classical signal processing, the Fourier transform decomposes a signal into sinusoidal basis functions (eigenfunctions of the Laplacian operator $\nabla^2$). The GFT extends this by using the eigenvectors of the graph Laplacian as the basis.

| Classical | Graph |
|-----------|-------|
| Signal $f(t)$ | Graph signal $\mathbf{f} \in \mathbb{R}^n$ |
| Frequency $\omega$ | Eigenvalue $\lambda_k$ |
| Basis $e^{i\omega t}$ | Eigenvector $\mathbf{u}_k$ |
| $\hat{f}(\omega) = \int f(t) e^{-i\omega t} dt$ | $\hat{f}(\lambda_k) = \mathbf{u}_k^T \mathbf{f}$ |

## Forward Transform

$$\hat{\mathbf{f}} = U^T \mathbf{f}$$

Each component $\hat{f}_k = \mathbf{u}_k^T \mathbf{f}$ measures the projection of the signal onto the $k$-th eigenvector. Small eigenvalues correspond to smooth (low-frequency) components; large eigenvalues correspond to oscillatory (high-frequency) components.

## Inverse Transform

$$\mathbf{f} = U \hat{\mathbf{f}} = \sum_{k=0}^{n-1} \hat{f}_k \mathbf{u}_k$$

The signal is reconstructed as a weighted sum of the graph Fourier modes.

## Graph Convolution in Spectral Domain

The **convolution theorem** on graphs states that convolution in the vertex domain equals pointwise multiplication in the spectral domain:

$$\mathbf{f} *_G \mathbf{g} = U \left( (U^T \mathbf{f}) \odot (U^T \mathbf{g}) \right)$$

A learnable spectral filter with parameters $\theta$:

$$\mathbf{f}_{out} = U \text{diag}(\theta) U^T \mathbf{f}_{in} = g_\theta(L) \mathbf{f}_{in}$$

## Parseval's Theorem on Graphs

Energy is preserved under the GFT:

$$\|\mathbf{f}\|^2 = \sum_{i} f_i^2 = \sum_{k} |\hat{f}_k|^2 = \|\hat{\mathbf{f}}\|^2$$

## Computational Considerations

- **Full eigendecomposition**: $O(n^3)$ — impractical for large graphs
- **Spectral filtering**: $O(n^2)$ per filter application (matrix-vector product with $U$)
- **Polynomial approximation**: Avoid explicit eigendecomposition by approximating $g_\theta(\Lambda)$ with polynomials of $L$

## Summary

The GFT provides the theoretical basis for spectral graph convolutions. While computationally expensive for direct computation, it motivates efficient polynomial approximations (ChebNet, GCN) that enable scalable graph neural networks.
