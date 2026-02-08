"""
One-shot adjacency matrix generation with various decoder architectures.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from scipy.optimize import linear_sum_assignment
import numpy as np


class MLPDecoder(nn.Module):
    """Simple MLP decoder: latent -> flattened upper-triangular adjacency."""

    def __init__(self, latent_dim: int, max_nodes: int, hidden_dim: int = 256):
        super().__init__()
        self.max_nodes = max_nodes
        n_edges = max_nodes * (max_nodes - 1) // 2

        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_edges),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.size(0)
        n = self.max_nodes
        edge_logits = self.mlp(z)

        adj = torch.zeros(B, n, n, device=z.device)
        idx = torch.triu_indices(n, n, offset=1)
        adj[:, idx[0], idx[1]] = torch.sigmoid(edge_logits)
        adj = adj + adj.transpose(1, 2)
        return adj


class FactoredDecoder(nn.Module):
    """Factored decoder: latent -> node embeddings -> bilinear edge prediction."""

    def __init__(
        self,
        latent_dim: int,
        max_nodes: int,
        node_embed_dim: int = 64,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.max_nodes = max_nodes
        self.node_embed_dim = node_embed_dim

        self.node_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_nodes * node_embed_dim),
        )
        self.edge_weight = nn.Parameter(
            torch.randn(node_embed_dim, node_embed_dim) * 0.01
        )
        self.edge_bias = nn.Parameter(torch.zeros(1))
        self.node_mask_predictor = nn.Sequential(
            nn.Linear(node_embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B = z.size(0)
        n = self.max_nodes

        node_embeds = self.node_decoder(z).view(B, n, self.node_embed_dim)
        W_sym = (self.edge_weight + self.edge_weight.t()) / 2
        edge_logits = torch.einsum("bid,de,bje->bij", node_embeds, W_sym, node_embeds)
        edge_logits = edge_logits + self.edge_bias
        adj = torch.sigmoid(edge_logits)

        mask = 1 - torch.eye(n, device=z.device).unsqueeze(0)
        adj = adj * mask

        node_mask = torch.sigmoid(
            self.node_mask_predictor(node_embeds).squeeze(-1)
        )
        adj = adj * node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        return adj, node_mask


def hungarian_matching(
    adj_pred: torch.Tensor,
    adj_target: torch.Tensor,
) -> torch.Tensor:
    """Find best node permutation using Hungarian algorithm."""
    cost = torch.cdist(adj_pred.float(), adj_target.float(), p=2)
    cost_np = cost.detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost_np)
    return torch.tensor(col_ind, dtype=torch.long, device=adj_pred.device)


def permutation_aligned_loss(
    adj_pred: torch.Tensor,
    adj_target: torch.Tensor,
    use_matching: bool = True,
) -> torch.Tensor:
    """Compute reconstruction loss with optional permutation alignment."""
    B = adj_pred.size(0)
    total_loss = 0.0

    for b in range(B):
        pred = adj_pred[b]
        target = adj_target[b]

        if use_matching:
            perm = hungarian_matching(pred, target)
            target = target[perm][:, perm]

        n = pred.size(0)
        idx = torch.triu_indices(n, n, offset=1)
        pred_edges = pred[idx[0], idx[1]]
        target_edges = target[idx[0], idx[1]]

        loss = F.binary_cross_entropy(
            pred_edges.clamp(1e-6, 1 - 1e-6),
            target_edges,
            reduction="mean",
        )
        total_loss += loss

    return total_loss / B


if __name__ == "__main__":
    torch.manual_seed(42)
    max_n = 12
    latent_dim = 32

    print("=== One-Shot Adjacency Generation Demo ===\n")

    train_adjs = []
    for _ in range(100):
        n = torch.randint(6, max_n + 1, (1,)).item()
        adj = torch.zeros(max_n, max_n)
        for i in range(n):
            for j in range(i + 1, n):
                if torch.rand(1) < 0.25:
                    adj[i, j] = adj[j, i] = 1
        train_adjs.append(adj)

    train_adjs = torch.stack(train_adjs)
    print(f"Training data: {train_adjs.shape}")

    # Test MLP decoder
    print("\n--- MLP Decoder ---")
    mlp_dec = MLPDecoder(latent_dim, max_n, hidden_dim=128)
    z = torch.randn(5, latent_dim)
    adj_pred = mlp_dec(z)
    print(f"Output shape: {adj_pred.shape}")
    print(f"Symmetry check: {torch.allclose(adj_pred, adj_pred.transpose(1,2))}")
    print(f"Value range: [{adj_pred.min():.3f}, {adj_pred.max():.3f}]")

    # Test factored decoder
    print("\n--- Factored Decoder ---")
    fac_dec = FactoredDecoder(latent_dim, max_n, node_embed_dim=32)
    adj_pred, node_mask = fac_dec(z)
    print(f"Adj shape: {adj_pred.shape}, Mask shape: {node_mask.shape}")
    print(f"Avg predicted nodes: {(node_mask > 0.5).float().sum(1).mean():.1f}")

    # Test Hungarian matching
    print("\n--- Hungarian Matching ---")
    adj_a = torch.zeros(6, 6)
    adj_a[0, 1] = adj_a[1, 0] = 1
    adj_a[1, 2] = adj_a[2, 1] = 1
    adj_a[2, 3] = adj_a[3, 2] = 1

    perm = [3, 2, 1, 0, 4, 5]
    adj_b = adj_a[perm][:, perm]

    recovered_perm = hungarian_matching(adj_a, adj_b)
    adj_recovered = adj_b[recovered_perm][:, recovered_perm]
    print(f"Match quality: {(adj_a == adj_recovered).float().mean():.1%}")

    # Test aligned loss
    print("\n--- Permutation-Aligned Loss ---")
    pred_batch = torch.rand(4, max_n, max_n) * 0.3
    pred_batch = (pred_batch + pred_batch.transpose(1, 2)) / 2
    pred_batch.diagonal(dim1=1, dim2=2).zero_()

    loss_no_match = permutation_aligned_loss(pred_batch, train_adjs[:4], use_matching=False)
    loss_match = permutation_aligned_loss(pred_batch, train_adjs[:4], use_matching=True)
    print(f"Loss without matching: {loss_no_match:.4f}")
    print(f"Loss with matching: {loss_match:.4f}")
    print(f"Matching reduces loss: {loss_match < loss_no_match}")
