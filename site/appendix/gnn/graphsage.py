#!/usr/bin/env python3
"""
GraphSAGE - Inductive Representation Learning on Large Graphs
Paper: "Inductive Representation Learning on Large Graphs" (2017)
Authors: Will Hamilton, Zhitao Ying, Jure Leskovec
Key idea:
  - Sample and aggregate neighbors
  - Inductive: can generalize to unseen nodes/graphs
  - Typical update: h'_i = sigma( W [h_i || AGG({h_j, j in N(i)})] )

File: appendix/gnn/graphsage.py
Note: Educational implementation using mean aggregation with dense adjacency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MeanAggregator(nn.Module):
    """Compute mean of neighbor features."""
    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # A: (N, N) adjacency (0/1), assume includes self-loops if desired
        deg = A.sum(dim=1, keepdim=True).clamp(min=1.0)  # avoid divide-by-zero
        neigh_mean = (A @ X) / deg
        return neigh_mean


class GraphSAGELayer(nn.Module):
    """
    One GraphSAGE layer with mean aggregation.

    h'_i = sigma( W [h_i || mean_{j in N(i)} h_j] )
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.agg = MeanAggregator()
        self.lin = nn.Linear(2 * in_dim, out_dim)

    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        neigh = self.agg(X, A)                    # (N, in_dim)
        h_cat = torch.cat([X, neigh], dim=1)      # (N, 2*in_dim)
        return self.lin(h_cat)                    # (N, out_dim)


class GraphSAGE(nn.Module):
    """2-layer GraphSAGE for node classification."""
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.sage1 = GraphSAGELayer(in_dim, hidden_dim)
        self.sage2 = GraphSAGELayer(hidden_dim, num_classes)

    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        H = F.relu(self.sage1(X, A))
        logits = self.sage2(H, A)
        return logits


if __name__ == "__main__":
    N, Fin, C = 5, 8, 4
    X = torch.randn(N, Fin)

    # Include self-loops to make neighbor mean include node itself (common trick)
    A = torch.tensor([
        [1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1],
    ], dtype=torch.float32)

    model = GraphSAGE(in_dim=Fin, hidden_dim=16, num_classes=C)
    logits = model(X, A)
    print("logits:", logits.shape)  # (5, 4)
