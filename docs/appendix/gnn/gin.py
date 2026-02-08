#!/usr/bin/env python3
"""
GIN - Graph Isomorphism Network
Paper: "How Powerful are Graph Neural Networks?" (2019)
Authors: Keyulu Xu et al.
Key idea:
  - Use sum aggregation + MLP
  - Update: h'_i = MLP( (1 + eps) * h_i + sum_{j in N(i)} h_j )
  - Proven as powerful as the Weisfeiler-Lehman test (under assumptions)

File: appendix/gnn/gin.py
Note: Educational implementation with dense adjacency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Small MLP used inside GIN."""
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class GINLayer(nn.Module):
    """
    One GIN layer:
      h'_i = MLP( (1 + eps) * h_i + sum_{j in N(i)} h_j )

    eps can be fixed or learnable; here we make it learnable.
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = MLP(in_dim, out_dim, hidden_dim=2 * out_dim)

    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # For sum aggregation, we can use A @ X if A contains edges (optionally without self-loops)
        neigh_sum = A @ X  # (N, in_dim)
        out = (1.0 + self.eps) * X + neigh_sum
        return self.mlp(out)


class GIN(nn.Module):
    """2-layer GIN for node classification."""
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.gin1 = GINLayer(in_dim, hidden_dim)
        self.gin2 = GINLayer(hidden_dim, num_classes)

    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        H = F.relu(self.gin1(X, A))
        logits = self.gin2(H, A)
        return logits


if __name__ == "__main__":
    N, Fin, C = 4, 6, 3
    X = torch.randn(N, Fin)

    # Adjacency without self-loops is okay for GIN because (1+eps)*X provides self contribution
    A = torch.tensor([
        [0, 1, 1, 0],
        [1, 0, 1, 0],
        [1, 1, 0, 1],
        [0, 0, 1, 0],
    ], dtype=torch.float32)

    model = GIN(in_dim=Fin, hidden_dim=16, num_classes=C)
    logits = model(X, A)
    print("logits:", logits.shape)  # (4, 3)
