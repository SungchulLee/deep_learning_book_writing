"""
GRAN: Graph Recurrent Attention Network for block-wise graph generation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class GRANAttentionLayer(nn.Module):
    """Single attention-based GNN layer for GRAN."""

    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.W_o = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h: torch.Tensor,
        adj: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            h: (n_total, hidden_dim) node embeddings
            adj: (n_total, n_total) adjacency (existing + candidate)
            candidate_mask: (n_total, n_total) candidate edge mask
        """
        n = h.size(0)

        Q = self.W_q(h).view(n, self.num_heads, self.head_dim)
        K = self.W_k(h).view(n, self.num_heads, self.head_dim)
        V = self.W_v(h).view(n, self.num_heads, self.head_dim)

        scores = torch.einsum("ihd,jhd->ijh", Q, K) / math.sqrt(self.head_dim)

        attn_mask = (adj + candidate_mask).clamp(max=1.0)
        attn_mask = attn_mask + torch.eye(n, device=h.device)
        attn_mask = attn_mask.unsqueeze(-1).expand_as(scores)
        scores = scores.masked_fill(attn_mask == 0, -1e9)

        attn = F.softmax(scores, dim=1)
        out = torch.einsum("ijh,jhd->ihd", attn, V)
        out = out.reshape(n, self.hidden_dim)
        out = self.W_o(out)

        h = self.norm(h + out)
        h = self.norm2(h + self.mlp(h))
        return h


class MixtureBernoulliDecoder(nn.Module):
    """Mixture of Bernoullis edge predictor."""

    def __init__(self, hidden_dim: int, num_components: int = 4):
        super().__init__()
        self.K = num_components
        self.edge_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
            for _ in range(num_components)
        ])
        self.mix_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_components),
        )

    def forward(self, h_u: torch.Tensor, h_v: torch.Tensor) -> torch.Tensor:
        """Predict edge probabilities for node pairs."""
        pair_feat = torch.cat([h_u, h_v], dim=-1)
        mix_weights = F.softmax(self.mix_predictor(pair_feat), dim=-1)

        component_probs = torch.stack([
            torch.sigmoid(pred(pair_feat).squeeze(-1))
            for pred in self.edge_predictors
        ], dim=-1)

        return (mix_weights * component_probs).sum(dim=-1)


class GRAN(nn.Module):
    """
    Graph Recurrent Attention Network.
    
    Generates graphs block-by-block using GNN attention and
    mixture of Bernoullis edge prediction.
    """

    def __init__(
        self,
        max_nodes: int,
        block_size: int = 1,
        hidden_dim: int = 128,
        num_gnn_layers: int = 3,
        num_mix_components: int = 4,
        num_heads: int = 4,
    ):
        super().__init__()
        self.max_nodes = max_nodes
        self.block_size = block_size
        self.hidden_dim = hidden_dim

        self.node_embed = nn.Embedding(2, hidden_dim)  # 0=existing, 1=new
        self.gnn_layers = nn.ModuleList([
            GRANAttentionLayer(hidden_dim, num_heads)
            for _ in range(num_gnn_layers)
        ])
        self.edge_decoder = MixtureBernoulliDecoder(hidden_dim, num_mix_components)

    def _predict_block_edges(
        self,
        adj_padded: torch.Tensor,
        n_existing: int,
        n_new: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict edges for a block of new nodes."""
        n_total = n_existing + n_new
        device = adj_padded.device

        node_types = torch.zeros(n_total, dtype=torch.long, device=device)
        node_types[n_existing:] = 1
        h = self.node_embed(node_types)

        candidate_mask = torch.zeros(n_total, n_total, device=device)
        candidate_mask[n_existing:, :n_total] = 1.0
        candidate_mask[:n_total, n_existing:] = 1.0
        candidate_mask.fill_diagonal_(0)

        for layer in self.gnn_layers:
            h = layer(h, adj_padded[:n_total, :n_total], candidate_mask)

        pairs_i, pairs_j = torch.where(
            torch.triu(candidate_mask, diagonal=1) > 0
        )
        candidate_pairs = torch.stack([pairs_i, pairs_j], dim=1)

        if len(candidate_pairs) == 0:
            return torch.tensor([], device=device), candidate_pairs

        h_u = h[candidate_pairs[:, 0]]
        h_v = h[candidate_pairs[:, 1]]
        edge_probs = self.edge_decoder(h_u, h_v)

        return edge_probs, candidate_pairs

    def forward(self, adj_list: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        """Training forward pass over a list of graphs."""
        total_loss = torch.tensor(0.0)
        num_preds = 0

        for adj in adj_list:
            n = adj.size(0)
            adj_padded = torch.zeros(self.max_nodes, self.max_nodes)
            adj_padded[:n, :n] = adj

            for step_start in range(self.block_size, n, self.block_size):
                n_existing = step_start
                n_new = min(self.block_size, n - step_start)

                edge_probs, candidate_pairs = self._predict_block_edges(
                    adj_padded, n_existing, n_new
                )
                if len(candidate_pairs) == 0:
                    continue

                targets = adj_padded[candidate_pairs[:, 0], candidate_pairs[:, 1]]
                loss = F.binary_cross_entropy(
                    edge_probs.clamp(1e-6, 1 - 1e-6), targets, reduction="sum"
                )
                total_loss = total_loss + loss
                num_preds += len(candidate_pairs)

        return {"total_loss": total_loss / max(num_preds, 1)}

    @torch.no_grad()
    def generate(
        self,
        num_graphs: int = 1,
        num_nodes: int = 10,
        temperature: float = 1.0,
    ) -> list[torch.Tensor]:
        """Generate graphs block by block."""
        self.eval()
        graphs = []

        for _ in range(num_graphs):
            adj = torch.zeros(self.max_nodes, self.max_nodes)

            for step_start in range(self.block_size, num_nodes, self.block_size):
                n_existing = step_start
                n_new = min(self.block_size, num_nodes - step_start)

                edge_probs, candidate_pairs = self._predict_block_edges(
                    adj, n_existing, n_new
                )
                if len(candidate_pairs) == 0:
                    continue

                adjusted_probs = torch.sigmoid(
                    torch.log(edge_probs / (1 - edge_probs + 1e-8)) / temperature
                )
                sampled = torch.bernoulli(adjusted_probs)

                for idx in range(len(candidate_pairs)):
                    if sampled[idx] > 0:
                        i, j = candidate_pairs[idx]
                        adj[i, j] = 1.0
                        adj[j, i] = 1.0

            graphs.append(adj[:num_nodes, :num_nodes])

        return graphs


if __name__ == "__main__":
    torch.manual_seed(42)

    max_n = 16
    block_size = 2

    # Create training data
    print("=== Preparing Training Data ===")
    train_graphs = []
    for _ in range(100):
        n = torch.randint(6, max_n, (1,)).item()
        n = (n // block_size) * block_size
        if n < 4:
            n = 4
        adj = torch.zeros(n, n)
        mid = n // 2
        for i in range(mid):
            for j in range(i + 1, mid):
                if torch.rand(1) < 0.5:
                    adj[i, j] = adj[j, i] = 1
        for i in range(mid, n):
            for j in range(i + 1, n):
                if torch.rand(1) < 0.5:
                    adj[i, j] = adj[j, i] = 1
        for i in range(mid):
            for j in range(mid, n):
                if torch.rand(1) < 0.1:
                    adj[i, j] = adj[j, i] = 1
        train_graphs.append(adj)

    print(f"Training graphs: {len(train_graphs)}")
    print(f"Avg nodes: {sum(g.size(0) for g in train_graphs)/len(train_graphs):.1f}")

    # Train GRAN
    print("\n=== Training GRAN ===")
    model = GRAN(
        max_nodes=max_n,
        block_size=block_size,
        hidden_dim=64,
        num_gnn_layers=2,
        num_mix_components=3,
        num_heads=4,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    for epoch in range(40):
        model.train()
        batch = train_graphs[:32]
        result = model(batch)
        loss = result["total_loss"]

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: loss={loss.item():.4f}")

    # Generate
    print("\n=== Generation ===")
    generated = model.generate(num_graphs=10, num_nodes=10)
    for i, g in enumerate(generated):
        n = g.size(0)
        e = int(g.sum().item()) // 2
        density = 2 * e / (n * (n - 1)) if n > 1 else 0
        print(f"Graph {i}: {n} nodes, {e} edges, density={density:.3f}")
