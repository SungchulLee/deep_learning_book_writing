#!/usr/bin/env python3
"""
GAT - Graph Attention Network
Paper: "Graph Attention Networks" (2018)
Authors: Petar Veličković et al.
Key idea:
  - Learn attention weights over neighbors instead of fixed normalization
  - For node i: aggregate neighbors j with alpha_{ij} learned from features

File: appendix/gnn/gat.py
Note: Educational implementation with dense adjacency, single-head attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    """
    Single-head GAT layer (dense adjacency).

    Steps:
      1) Linear transform: Wh_i = W x_i
      2) Attention scores: e_{ij} = LeakyReLU(a^T [Wh_i || Wh_j])
      3) Mask non-edges
      4) alpha_{ij} = softmax_j(e_{ij})
      5) h'_i = sum_j alpha_{ij} Wh_j
    """
    def __init__(self, in_dim: int, out_dim: int, leaky_slope: float = 0.2):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)

        # a is split into two parts (equivalent to a^T [Wh_i || Wh_j])
        self.attn_l = nn.Linear(out_dim, 1, bias=False)
        self.attn_r = nn.Linear(out_dim, 1, bias=False)

        self.leaky_relu = nn.LeakyReLU(leaky_slope)

    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # X: (N, Fin), A: (N, N) adjacency (0/1)
        H = self.W(X)  # (N, Fout)

        # Compute attention logits efficiently:
        # e_{ij} = LeakyReLU( a_l(H_i) + a_r(H_j) )
        e_l = self.attn_l(H)  # (N, 1)
        e_r = self.attn_r(H)  # (N, 1)
        e = e_l + e_r.T        # (N, N)
        e = self.leaky_relu(e)

        # Mask out non-neighbors:
        # Use a very negative value so softmax ~ 0 for non-edges
        mask = (A == 0)
        e = e.masked_fill(mask, float("-inf"))

        # Normalize across neighbors j
        alpha = F.softmax(e, dim=1)  # (N, N)

        # Weighted sum of neighbor features
        H_out = alpha @ H  # (N, Fout)
        return H_out


class GAT(nn.Module):
    """2-layer GAT for node classification (single-head)."""
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.gat1 = GATLayer(in_dim, hidden_dim)
        self.gat2 = GATLayer(hidden_dim, num_classes)

    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        H = F.elu(self.gat1(X, A))
        logits = self.gat2(H, A)
        return logits


if __name__ == "__main__":
    N, Fin, C = 4, 8, 3
    X = torch.randn(N, Fin)
    A = torch.tensor([
        [1, 1, 1, 0],  # include self-loop in GAT for stability
        [1, 1, 1, 0],
        [1, 1, 1, 1],
        [0, 0, 1, 1],
    ], dtype=torch.float32)

    model = GAT(in_dim=Fin, hidden_dim=16, num_classes=C)
    logits = model(X, A)
    print("logits:", logits.shape)  # (4, 3)
