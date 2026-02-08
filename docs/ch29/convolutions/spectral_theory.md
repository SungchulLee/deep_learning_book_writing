# 29.3.1 Spectral Graph Theory

## Introduction

**Spectral graph theory** studies graphs through the eigenvalues and eigenvectors of matrices associated with graphs, particularly the graph Laplacian. These spectral properties provide deep insights into graph structure and form the mathematical foundation for spectral graph convolutions.

## Graph Laplacian Recap

The **unnormalized Laplacian** $L = D - A$ has eigendecomposition:

$$L = U \Lambda U^T$$

where $U = [\mathbf{u}_0, \mathbf{u}_1, \ldots, \mathbf{u}_{n-1}]$ are orthonormal eigenvectors and $\Lambda = \text{diag}(\lambda_0, \lambda_1, \ldots, \lambda_{n-1})$ with $0 = \lambda_0 \leq \lambda_1 \leq \cdots \leq \lambda_{n-1}$.

The **normalized Laplacian** $\hat{L} = I - D^{-1/2}AD^{-1/2}$ has eigenvalues in $[0, 2]$.

## Graph Signals

A **graph signal** is a function $f: V \rightarrow \mathbb{R}$ assigning a real value to each node. It can be represented as a vector $\mathbf{f} \in \mathbb{R}^n$.

The **smoothness** of a signal on the graph is measured by:

$$\mathbf{f}^T L \mathbf{f} = \frac{1}{2} \sum_{(i,j) \in E} w_{ij}(f_i - f_j)^2$$

Signals aligned with low-frequency eigenvectors (small $\lambda$) are smooth; those aligned with high-frequency eigenvectors (large $\lambda$) vary rapidly across edges.

## Graph Fourier Transform

The **Graph Fourier Transform (GFT)** decomposes a signal into its frequency components:

$$\hat{\mathbf{f}} = U^T \mathbf{f}$$

The **inverse GFT** reconstructs the signal:

$$\mathbf{f} = U \hat{\mathbf{f}}$$

The eigenvalues $\lambda_k$ correspond to frequencies: $\lambda_0 = 0$ is the DC component, and larger $\lambda_k$ represent higher frequencies.

## Spectral Filtering

A **spectral filter** $g(\Lambda)$ operates on the graph signal in the spectral domain:

$$\mathbf{f}_{out} = U g(\Lambda) U^T \mathbf{f}_{in}$$

where $g(\Lambda) = \text{diag}(g(\lambda_0), g(\lambda_1), \ldots, g(\lambda_{n-1}))$.

This is the graph analog of convolution in the spectral domain. A low-pass filter ($g$ attenuates high $\lambda$) smooths the signal; a high-pass filter amplifies differences.

## Key Spectral Properties

### Cheeger Inequality
Relates the algebraic connectivity $\lambda_1$ to graph partitioning:
$$\frac{h^2}{2d_{max}} \leq \lambda_1 \leq 2h$$
where $h$ is the Cheeger constant (conductance).

### Expander Graphs
Graphs with large spectral gap have good expansion and mixing properties.

### Fiedler Vector
The eigenvector corresponding to $\lambda_1$ (the **Fiedler vector**) can be used for graph bisection—its sign partitions the graph into two well-connected halves.

## Limitations of Spectral Methods

1. **Computational cost**: Eigendecomposition is $O(n^3)$, prohibitive for large graphs
2. **Non-transferable**: Spectral filters depend on the specific graph's eigenvectors
3. **Not localized**: Spectral filters are generally not localized in the spatial domain

These limitations motivate polynomial approximations (ChebNet) and spatial methods (GCN, GAT).

## Summary

Spectral graph theory provides the mathematical foundation for understanding graph convolutions. While direct spectral methods are computationally expensive, they motivate efficient approximations that form the basis of practical GNN architectures.
