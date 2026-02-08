# 29.3.3 ChebNet

## Introduction

**ChebNet** (Defferrard et al., 2016) addresses the computational bottleneck of spectral graph convolutions by approximating spectral filters using **Chebyshev polynomials** of the graph Laplacian. This avoids explicit eigendecomposition while maintaining spectral filtering capabilities.

## Motivation

Direct spectral filtering requires:
1. Full eigendecomposition: $O(n^3)$
2. Matrix multiplication with $U$: $O(n^2)$ per filter

ChebNet reduces this by approximating $g_\theta(\Lambda)$ with a $K$-th order polynomial:

$$g_\theta(\Lambda) \approx \sum_{k=0}^{K} \theta_k T_k(\tilde{\Lambda})$$

where $T_k$ are Chebyshev polynomials and $\tilde{\Lambda} = \frac{2\Lambda}{\lambda_{max}} - I$ scales eigenvalues to $[-1, 1]$.

## Chebyshev Polynomials

Chebyshev polynomials of the first kind are defined by the recurrence:

$$T_0(x) = 1, \quad T_1(x) = x, \quad T_k(x) = 2x T_{k-1}(x) - T_{k-2}(x)$$

Key properties:
- Orthogonal on $[-1, 1]$ with weight $\frac{1}{\sqrt{1-x^2}}$
- Best polynomial approximation in the minimax sense
- Bounded: $|T_k(x)| \leq 1$ for $x \in [-1, 1]$

## ChebNet Convolution

The ChebNet spectral filter applied to a signal $\mathbf{x}$:

$$\mathbf{y} = g_\theta(L) \mathbf{x} = \sum_{k=0}^{K} \theta_k T_k(\tilde{L}) \mathbf{x}$$

where $\tilde{L} = \frac{2L}{\lambda_{max}} - I$.

### Key Advantages

1. **No eigendecomposition**: Only requires matrix-vector products with $L$
2. **$K$-localized**: Filter depends only on $K$-hop neighborhood
3. **$O(K|E|)$ complexity**: Linear in edges for each filter
4. **Learnable**: Parameters $\theta_0, \theta_1, \ldots, \theta_K$ are trained via backpropagation

### Connection to GCN

Setting $K = 1$ and $\lambda_{max} = 2$ in ChebNet, Kipf & Welling (2017) derived GCN:

$$\mathbf{y} = \theta_0 \mathbf{x} + \theta_1 (L - I) \mathbf{x} = \theta_0 \mathbf{x} - \theta_1 D^{-1/2} A D^{-1/2} \mathbf{x}$$

Further setting $\theta = \theta_0 = -\theta_1$ gives the GCN layer.

## Computation

```
Algorithm: ChebNet Convolution
Input: Signal x, Laplacian L, parameters θ_0,...,θ_K
1. Compute L_tilde = 2L/λ_max - I
2. T_0 = x
3. T_1 = L_tilde @ x
4. y = θ_0 * T_0 + θ_1 * T_1
5. For k = 2 to K:
     T_k = 2 * L_tilde @ T_{k-1} - T_{k-2}
     y = y + θ_k * T_k
6. Return y
```

## Summary

ChebNet bridges spectral graph theory and practical GNN computation. By leveraging Chebyshev polynomial approximation, it achieves spatially localized, computationally efficient spectral filtering that scales to large graphs. It directly inspired the GCN architecture and remains relevant for applications requiring explicit control over spectral filter shape.
