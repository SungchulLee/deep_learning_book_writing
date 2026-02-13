#!/usr/bin/env python3
"""
MPNN - Message Passing Neural Network (General Framework)
Paper: "Neural Message Passing for Quantum Chemistry" (2017)
Authors: Justin Gilmer et al.
Key idea:
  - Separate *message* function and *update* function
  - Iterative propagation for T steps:
      m_i^{t+1} = sum_{j in N(i)} M(h_i^t, h_j^t, e_{ij})
      h_i^{t+1} = U(h_i^t, m_i^{t+1})

File: appendix/gnn/mpnn.py
Note: Educational implementation with:
  - Dense adjacency
  - Optional edge features matrix E (N, N, E_dim)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MessageFn(nn.Module):
    """
    Message function M(h_i, h_j, e_ij).
    Here: concatenate and pass through an MLP.
    """
    def __init__(self, node_dim: int, edge_dim: int, msg_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, msg_dim),
            nn.ReLU(inplace=True),
            nn.Linear(msg_dim, msg_dim),
        )

    def forward(self, h_i, h_j, e_ij):
        # h_i, h_j: (msg_dim?) but here node_dim
        # e_ij: (edge_dim)
        x = torch.cat([h_i, h_j, e_ij], dim=-1)
        return self.net(x)


class UpdateFn(nn.Module):
    """
    Update function U(h_i, m_i).
    Here: GRUCell-like update (simple and common).
    """
    def __init__(self, node_dim: int, msg_dim: int):
        super().__init__()
        self.gru = nn.GRUCell(input_size=msg_dim, hidden_size=node_dim)

    def forward(self, h_i, m_i):
        # m_i: (node_dim?) here msg_dim
        return self.gru(m_i, h_i)


class MPNN(nn.Module):
    """
    Generic message passing network.

    Inputs:
      X: (N, node_dim) node features
      A: (N, N) adjacency (0/1)
      E: (N, N, edge_dim) edge features (optional; if None, use zeros)

    Output:
      H: (N, node_dim) updated node representations after T steps
    """
    def __init__(self, node_dim: int, edge_dim: int = 0, msg_dim: int = 64, T: int = 3):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.msg_dim = msg_dim
        self.T = T

        self.message = MessageFn(node_dim, edge_dim, msg_dim)
        self.update = UpdateFn(node_dim, msg_dim)

    def forward(self, X: torch.Tensor, A: torch.Tensor, E: torch.Tensor | None = None):
        N = X.size(0)
        device = X.device

        # If no edge features are provided, treat edges as having empty/zero features
        if E is None:
            E = torch.zeros(N, N, self.edge_dim, device=device)

        H = X  # current node states (N, node_dim)

        # Perform T rounds of message passing
        for _ in range(self.T):
            messages = []

            # For each node i, aggregate messages from neighbors j
            for i in range(N):
                m_i_list = []

                for j in range(N):
                    # Only send message if edge exists (A[i, j] == 1)
                    if A[i, j] > 0:
                        m_ij = self.message(H[i], H[j], E[i, j])  # (msg_dim,)
                        m_i_list.append(m_ij)

                # Sum aggregation (common in MPNN)
                if len(m_i_list) == 0:
                    m_i = torch.zeros(self.msg_dim, device=device)
                else:
                    m_i = torch.stack(m_i_list, dim=0).sum(dim=0)

                messages.append(m_i)

            M = torch.stack(messages, dim=0)  # (N, msg_dim)

            # Update node states using update function
            H = self.update(H, M)  # (N, node_dim)

        return H


if __name__ == "__main__":
    N, node_dim, edge_dim = 4, 8, 3
    X = torch.randn(N, node_dim)

    A = torch.tensor([
        [0, 1, 1, 0],
        [1, 0, 1, 0],
        [1, 1, 0, 1],
        [0, 0, 1, 0],
    ], dtype=torch.float32)

    E = torch.randn(N, N, edge_dim)  # random edge features (only used where A=1)

    model = MPNN(node_dim=node_dim, edge_dim=edge_dim, msg_dim=16, T=2)
    H = model(X, A, E)
    print("H:", H.shape)  # (4, 8)
