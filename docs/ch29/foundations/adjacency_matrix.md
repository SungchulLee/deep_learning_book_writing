# 29.1.3 Adjacency Matrix

## Introduction

The adjacency matrix is the most fundamental matrix representation of a graph and serves as the starting point for spectral graph theory, graph signal processing, and many GNN formulations. This section provides a deep dive into the adjacency matrix and its derived matrices.

## Definition

Given a graph $G = (V, E)$ with $n = |V|$ nodes, the **adjacency matrix** $A \in \mathbb{R}^{n \times n}$ is defined as:

$$A_{ij} = \begin{cases} w_{ij} & \text{if } (i, j) \in E \\ 0 & \text{otherwise} \end{cases}$$

For unweighted graphs, $A_{ij} \in \{0, 1\}$.

## Properties

### Symmetry
For undirected graphs: $A = A^T$. This implies all eigenvalues are real.

### Degree Matrix
The **degree matrix** $D$ is a diagonal matrix:
$$D_{ii} = \sum_{j} A_{ij} = d(i)$$

### Powers of the Adjacency Matrix
The entry $(A^k)_{ij}$ counts the number of walks of length $k$ from node $i$ to node $j$.

- $A^2_{ij}$: number of common neighbors of $i$ and $j$ (for unweighted undirected graphs)
- $\text{tr}(A^3) / 6$: number of triangles in the graph

### Spectral Properties
The eigenvalues $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_n$ of $A$ satisfy:
- The largest eigenvalue $\lambda_1$ is bounded: $d_{avg} \leq \lambda_1 \leq d_{max}$
- For $d$-regular graphs: $\lambda_1 = d$
- The number of connected components equals the multiplicity of $\lambda_1$

## Graph Laplacian

The **graph Laplacian** $L$ is defined as:

$$L = D - A$$

Properties:
- $L$ is positive semi-definite
- The smallest eigenvalue is always 0 (with eigenvector $\mathbf{1}$)
- The multiplicity of eigenvalue 0 equals the number of connected components
- $\mathbf{x}^T L \mathbf{x} = \frac{1}{2} \sum_{(i,j) \in E} w_{ij}(x_i - x_j)^2$ (smoothness measure)

## Normalized Laplacians

### Symmetric Normalized Laplacian
$$\hat{L} = D^{-1/2} L D^{-1/2} = I - D^{-1/2} A D^{-1/2}$$

Eigenvalues lie in $[0, 2]$. Used in spectral graph convolutions.

### Random Walk Normalized Laplacian
$$L_{rw} = D^{-1}L = I - D^{-1}A$$

$D^{-1}A$ is the transition matrix of a random walk on the graph.

## Normalized Adjacency Matrix

The **symmetrically normalized adjacency matrix** used in GCN:

$$\hat{A} = D^{-1/2} A D^{-1/2}$$

With self-loops (renormalization trick):
$$\tilde{A} = A + I_n, \quad \tilde{D}_{ii} = \sum_j \tilde{A}_{ij}, \quad \hat{A} = \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$$

## Transition Matrix

The **row-normalized adjacency matrix** represents random walk transition probabilities:

$$P = D^{-1}A, \quad P_{ij} = \frac{A_{ij}}{d(i)}$$

$P_{ij}$ is the probability of walking from node $i$ to node $j$ in one step.

## Quantitative Finance: Correlation as Adjacency

In financial networks, the correlation matrix $C$ between assets can be transformed into an adjacency matrix:

1. **Thresholding**: $A_{ij} = \mathbb{1}[|C_{ij}| > \tau]$
2. **Soft weighting**: $A_{ij} = |C_{ij}|^p$ for some power $p > 0$
3. **MST-based**: Extract the minimum spanning tree from distance matrix $d_{ij} = \sqrt{2(1 - C_{ij})}$

These transformations are crucial for building financial graph neural networks.

## Summary

The adjacency matrix and its normalizations (Laplacian, normalized Laplacian, transition matrix) form the mathematical backbone of graph neural networks. Understanding their spectral properties is essential for both spectral and spatial GNN methods.
