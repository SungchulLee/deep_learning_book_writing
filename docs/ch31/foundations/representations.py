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
