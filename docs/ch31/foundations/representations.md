# Graph Representations for Generation

## Overview

The choice of graph representation fundamentally determines which generation strategies are feasible. A representation must balance expressiveness (capturing all graph properties), compactness (enabling efficient learning), and compatibility with neural network architectures. This section surveys the primary representations used in graph generation and their implications for model design.

## Adjacency Matrix Representation

The most direct representation of a graph $\mathcal{G}$ with $n$ nodes is its adjacency matrix $\mathbf{A} \in \{0,1\}^{n \times n}$, where $A_{ij} = 1$ if edge $(i,j)$ exists. For attributed graphs, we augment with node features $\mathbf{X} \in \mathbb{R}^{n \times d_n}$ and edge features $\mathbf{E} \in \mathbb{R}^{n \times n \times d_e}$.

**Properties:**
- Space complexity: $O(n^2)$ for edges, $O(n \cdot d_n)$ for node features
- Naturally symmetric for undirected graphs: $\mathbf{A} = \mathbf{A}^\top$
- Sparse in practice: most real graphs have $|\mathcal{E}| \ll n^2$

**Permutation ambiguity.** The same graph has $n!$ adjacency matrix representations under node relabeling. For a permutation $\pi$, the permuted adjacency matrix is:

$$
\mathbf{A}' = \mathbf{P}_\pi \mathbf{A} \mathbf{P}_\pi^\top
$$

where $\mathbf{P}_\pi$ is the permutation matrix. This ambiguity is the central challenge for one-shot generation methods.

## Upper-Triangular Representation

For undirected graphs without self-loops, only the upper-triangular entries of $\mathbf{A}$ are needed. Flattening this triangle produces a binary vector $\mathbf{a} \in \{0,1\}^{\binom{n}{2}}$:

$$
\mathbf{a} = \text{triu\_flatten}(\mathbf{A}) = (A_{12}, A_{13}, \ldots, A_{1n}, A_{23}, \ldots, A_{(n-1)n})
$$

This reduces the output dimension by half and eliminates the redundancy of the full matrix. Autoregressive models over this flattened vector generate edges in a fixed order.

## Edge List Representation

An edge list $\mathcal{E} = \{(i_1, j_1), \ldots, (i_m, j_m)\}$ stores only existing edges. This is memory-efficient for sparse graphs ($m \ll n^2$) and naturally interfaces with PyTorch Geometric's `edge_index` format:

$$
\texttt{edge\_index} = \begin{bmatrix} i_1 & i_2 & \cdots & i_m \\ j_1 & j_2 & \cdots & j_m \end{bmatrix} \in \mathbb{N}^{2 \times m}
$$

For generation, edge list representations require predicting both the number of edges and each endpoint pair, making autoregressive approaches natural.

## Canonical Orderings

To break permutation symmetry, autoregressive methods impose a canonical node ordering. Common choices:

**BFS ordering.** Starting from a random node, visit neighbors layer by layer. Produces orderings where recently added nodes connect to nearby predecessors — reducing the effective bandwidth of the adjacency matrix.

**DFS ordering.** Starting from a random node, explore as deep as possible before backtracking. Produces orderings with strong locality: the edge structure is concentrated near the diagonal of $\mathbf{A}$.

**Degree ordering.** Sort nodes by degree (descending). Hubs appear first, creating a structured adjacency where high-degree nodes establish the backbone.

The BFS bandwidth $B$ of a graph under BFS ordering determines the maximum lookback distance for edge decisions:

$$
B = \max_{(i,j) \in \mathcal{E}} |i - j|
$$

GraphRNN exploits this by truncating edge predictions to a window of size $B$, reducing complexity from $O(n^2)$ to $O(n \cdot B)$.

## Sequence Representations

Some methods represent graphs as sequences of tokens:

**SMILES strings** encode molecular graphs as character sequences (e.g., `CC(=O)Oc1ccccc1C(=O)O` for aspirin). Generation reduces to language modeling, but the mapping from strings to graphs is many-to-one and syntactic validity is not guaranteed.

**Adjacency sequence.** Flatten the adjacency matrix row by row, yielding a binary sequence of length $n^2$ (or $\binom{n}{2}$ for the upper triangle). Each position is a Bernoulli variable conditioned on previous positions.

## Spectral Representations

Rather than operating on $\mathbf{A}$ directly, spectral methods represent graphs through the eigendecomposition of the graph Laplacian:

$$
\mathbf{L} = \mathbf{D} - \mathbf{A} = \mathbf{U} \boldsymbol{\Lambda} \mathbf{U}^\top
$$

The eigenvalues $\boldsymbol{\Lambda} = \text{diag}(\lambda_1, \ldots, \lambda_n)$ encode global structural properties (connectivity, expansion, clustering), while eigenvectors $\mathbf{U}$ capture node positions in the spectral embedding space. Generating $(\boldsymbol{\Lambda}, \mathbf{U})$ and reconstructing $\mathbf{A} = \mathbf{D} - \mathbf{U}\boldsymbol{\Lambda}\mathbf{U}^\top$ naturally produces permutation-equivariant representations.

## Implementation: Representation Conversions

```python
"""
Graph representation conversions for generation pipelines.
"""
import torch
import numpy as np
from collections import deque
from typing import Optional


def adjacency_to_upper_triangular(adj: torch.Tensor) -> torch.Tensor:
    """
    Flatten upper triangle of adjacency matrix to vector.
    
    Args:
        adj: (n, n) adjacency matrix
        
    Returns:
        (n*(n-1)/2,) binary vector
    """
    n = adj.size(0)
    indices = torch.triu_indices(n, n, offset=1)
    return adj[indices[0], indices[1]]


def upper_triangular_to_adjacency(vec: torch.Tensor, n: int) -> torch.Tensor:
    """
    Reconstruct adjacency matrix from upper triangular vector.
    
    Args:
        vec: (n*(n-1)/2,) vector
        n: number of nodes
        
    Returns:
        (n, n) symmetric adjacency matrix
    """
    adj = torch.zeros(n, n, dtype=vec.dtype, device=vec.device)
    indices = torch.triu_indices(n, n, offset=1)
    adj[indices[0], indices[1]] = vec
    adj = adj + adj.t()
    return adj


def bfs_ordering(adj: torch.Tensor, start: Optional[int] = None) -> list[int]:
    """
    Compute BFS ordering of nodes.
    
    Args:
        adj: (n, n) adjacency matrix
        start: starting node (random if None)
        
    Returns:
        List of node indices in BFS order
    """
    n = adj.size(0)
    if start is None:
        # Start from highest-degree node
        start = adj.sum(dim=1).argmax().item()

    visited = set()
    order = []
    queue = deque([start])
    visited.add(start)

    while queue:
        node = queue.popleft()
        order.append(node)
        neighbors = torch.where(adj[node] > 0)[0].tolist()
        # Sort neighbors by degree (descending) for determinism
        neighbors.sort(key=lambda x: -adj[x].sum().item())
        for nb in neighbors:
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)

    # Add disconnected nodes
    for i in range(n):
        if i not in visited:
            order.append(i)

    return order


def permute_adjacency(adj: torch.Tensor, perm: list[int]) -> torch.Tensor:
    """Reorder adjacency matrix according to permutation."""
    perm_tensor = torch.tensor(perm)
    return adj[perm_tensor][:, perm_tensor]


def compute_bfs_bandwidth(adj: torch.Tensor) -> int:
    """
    Compute the BFS bandwidth: max |i - j| for edges (i, j) 
    under BFS ordering.
    """
    order = bfs_ordering(adj)
    adj_bfs = permute_adjacency(adj, order)
    n = adj_bfs.size(0)
    max_bw = 0
    for i in range(n):
        for j in range(i + 1, n):
            if adj_bfs[i, j] > 0:
                max_bw = max(max_bw, j - i)
    return max_bw


def laplacian_eigendecomposition(
    adj: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute graph Laplacian eigendecomposition.
    
    Returns:
        eigenvalues: (n,) sorted eigenvalues
        eigenvectors: (n, n) corresponding eigenvectors
    """
    degree = adj.sum(dim=1)
    laplacian = torch.diag(degree) - adj
    eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)
    return eigenvalues, eigenvectors


def reconstruct_from_spectrum(
    eigenvalues: torch.Tensor,
    eigenvectors: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """
    Reconstruct adjacency matrix from spectral decomposition.
    
    The Laplacian L = U Λ U^T, and A = D - L.
    Since D is unknown, we estimate it from the reconstructed L.
    """
    laplacian = eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.t()
    # A_ij = -L_ij for i != j
    adj = -laplacian
    adj.fill_diagonal_(0)
    # Threshold to binary
    adj = (adj > threshold).float()
    return adj


if __name__ == "__main__":
    # Create a sample graph (Erdős–Rényi)
    n = 8
    p = 0.4
    adj = (torch.rand(n, n) < p).float()
    adj = torch.triu(adj, diagonal=1)
    adj = adj + adj.t()

    print("=== Adjacency Matrix ===")
    print(adj.int())

    # Upper triangular
    vec = adjacency_to_upper_triangular(adj)
    print(f"\nUpper-tri vector ({vec.shape[0]} entries): {vec.int().tolist()}")
    adj_reconstructed = upper_triangular_to_adjacency(vec, n)
    assert torch.allclose(adj, adj_reconstructed), "Reconstruction failed"

    # BFS ordering
    order = bfs_ordering(adj)
    print(f"\nBFS order: {order}")
    bw = compute_bfs_bandwidth(adj)
    print(f"BFS bandwidth: {bw}")

    # Spectral
    eigenvalues, eigenvectors = laplacian_eigendecomposition(adj)
    print(f"\nLaplacian eigenvalues: {eigenvalues.numpy().round(3)}")
    adj_spectral = reconstruct_from_spectrum(eigenvalues, eigenvectors)
    match = (adj == adj_spectral).float().mean()
    print(f"Spectral reconstruction accuracy: {match:.1%}")
```
