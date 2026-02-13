#!/usr/bin/env python3
"""
GCN - Graph Convolutional Network
Paper: "Semi-Supervised Classification with Graph Convolutional Networks" (2017)
Authors: Thomas N. Kipf, Max Welling
Key idea:
  - Node features are updated by aggregating (normalized) neighbor features
  - Uses normalized adjacency:  D^{-1/2} (A + I) D^{-1/2}

File: appendix/gnn/gcn.py
Note: Educational implementation using dense adjacency for clarity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_adjacency(A: torch.Tensor) -> torch.Tensor:
    """
    Compute normalized adjacency:  D^{-1/2} (A + I) D^{-1/2}

    A: (N, N) adjacency matrix (0/1 or weighted)
    Returns:
      A_norm: (N, N)
    """
    N = A.size(0)

    # Add self-loops: A_hat = A + I
    A_hat = A + torch.eye(N, device=A.device)

    # Degree matrix: D_hat[i] = sum_j A_hat[i, j]
    deg = A_hat.sum(dim=1)  # (N,)

    # D^{-1/2}: careful about division by zero
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    # Normalize: D^{-1/2} A_hat D^{-1/2}
    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt
    return A_norm


class GCNLayer(nn.Module):
    """
    One GCN layer:
      H^{(l+1)} = sigma( A_norm H^{(l)} W )

    Where:
      - H^{(l)} is node feature matrix (N, Fin)
      - W is learnable weight (Fin, Fout)
      - A_norm is normalized adjacency (N, N)
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, X: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        # Multiply features by weight, then propagate via graph structure
        return A_norm @ self.lin(X)  # (N, out_dim)


class GCN(nn.Module):
    """
    A simple 2-layer GCN for node classification.

    Inputs:
      X: (N, Fin) node features
      A: (N, N) adjacency
    Output:
      logits: (N, num_classes)
    """
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.gcn1 = GCNLayer(in_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, num_classes)

    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        A_norm = normalize_adjacency(A)
        H = F.relu(self.gcn1(X, A_norm))
        logits = self.gcn2(H, A_norm)
        return logits


if __name__ == "__main__":
    # Toy example with 4 nodes
    N, Fin, C = 4, 8, 3
    X = torch.randn(N, Fin)

    # Simple undirected adjacency
    A = torch.tensor([
        [0, 1, 1, 0],
        [1, 0, 1, 0],
        [1, 1, 0, 1],
        [0, 0, 1, 0],
    ], dtype=torch.float32)

    model = GCN(in_dim=Fin, hidden_dim=16, num_classes=C)
    logits = model(X, A)
    print("logits:", logits.shape)  # (4, 3)
