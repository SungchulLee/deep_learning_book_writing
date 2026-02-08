# GraphVAE

## Overview

GraphVAE (Simonovsky & Komodakis, 2018) applies the variational autoencoder framework to one-shot graph generation. The encoder maps a graph to a latent distribution using a GNN, and the decoder reconstructs the full adjacency matrix and node features from a sampled latent vector. The key innovations are: (1) a graph-matching loss that handles permutation invariance without requiring a canonical ordering, and (2) a probabilistic decoder that generates both the graph topology and node/edge attributes simultaneously.

## Architecture

### Encoder

The encoder maps an input graph $\mathcal{G} = (\mathbf{A}, \mathbf{X})$ to a Gaussian posterior in latent space. A GNN computes node embeddings, which are aggregated into a graph-level representation:

$$
\mathbf{H} = \text{GNN}_{\text{enc}}(\mathbf{A}, \mathbf{X}) \in \mathbb{R}^{n \times d_h}
$$

$$
\mathbf{h}_{\mathcal{G}} = \text{READOUT}(\mathbf{H}) \in \mathbb{R}^{d_h}
$$

The posterior parameters are computed from the graph embedding:

$$
\boldsymbol{\mu} = \text{MLP}_\mu(\mathbf{h}_{\mathcal{G}}), \quad \log \boldsymbol{\sigma}^2 = \text{MLP}_\sigma(\mathbf{h}_{\mathcal{G}})
$$

$$
q_\phi(\mathbf{z} \mid \mathcal{G}) = \mathcal{N}(\mathbf{z}; \boldsymbol{\mu}, \text{diag}(\boldsymbol{\sigma}^2))
$$

### Decoder

The decoder maps a latent vector $\mathbf{z}$ to a probabilistic graph with $n_{\max}$ nodes. It predicts three components:

**Node existence.** A probability vector $\hat{\mathbf{m}} \in [0,1]^{n_{\max}}$ indicating which nodes are present:

$$
\hat{\mathbf{m}} = \sigma(\text{MLP}_{\text{node}}(\mathbf{z}))
$$

**Adjacency matrix.** Edge probabilities for all node pairs:

$$
\hat{\mathbf{A}} = \sigma(\text{MLP}_{\text{edge}}(\mathbf{z})) \in [0,1]^{n_{\max} \times n_{\max}}
$$

**Node features.** Predicted features for each node:

$$
\hat{\mathbf{X}} = \text{MLP}_{\text{feat}}(\mathbf{z}) \in \mathbb{R}^{n_{\max} \times d_x}
$$

## Graph Matching Loss

The central contribution of GraphVAE is handling permutation invariance through explicit graph matching. Given a predicted graph $\hat{\mathcal{G}}$ and target graph $\mathcal{G}^*$, the optimal permutation $\pi^*$ is found by solving a maximum weight bipartite matching problem.

Define the node similarity matrix $\mathbf{S} \in \mathbb{R}^{n_{\max} \times n_{\max}}$:

$$
S_{ij} = \mathbf{1}[\hat{m}_i > 0.5] \cdot \mathbf{1}[j \leq n^*] \cdot \left( \lambda_x \cdot \text{sim}(\hat{\mathbf{x}}_i, \mathbf{x}_j^*) + \lambda_e \cdot \text{sim}(\hat{\mathbf{a}}_i, \mathbf{a}_j^*) \right)
$$

where $\hat{\mathbf{a}}_i$ and $\mathbf{a}_j^*$ are the $i$-th and $j$-th rows of the predicted and target adjacency matrices respectively. The optimal matching is:

$$
\pi^* = \arg\max_{\pi} \sum_{i} S_{i, \pi(i)}
$$

computed via the Hungarian algorithm in $O(n_{\max}^3)$ time.

## ELBO Objective

The training loss is the evidence lower bound (ELBO) with the graph-matching reconstruction term:

$$
\mathcal{L} = \underbrace{-\mathbb{E}_{q_\phi(\mathbf{z} \mid \mathcal{G})} \left[ \log p_\theta(\mathcal{G} \mid \mathbf{z}) \right]}_{\text{reconstruction}} + \underbrace{D_{\text{KL}}(q_\phi(\mathbf{z} \mid \mathcal{G}) \| p(\mathbf{z}))}_{\text{regularization}}
$$

The reconstruction term decomposes (after matching) into:

$$
\log p_\theta(\mathcal{G} \mid \mathbf{z}) = \underbrace{\sum_{i} \log p(\text{node}_i \mid \mathbf{z})}_{\text{node existence}} + \underbrace{\sum_{i<j} \log p(A_{ij} \mid \mathbf{z})}_{\text{edge reconstruction}} + \underbrace{\sum_{i} \log p(\mathbf{x}_i \mid \mathbf{z})}_{\text{feature reconstruction}}
$$

Each term uses the permutation-aligned targets from the matching step.

## Limitations

**Scalability.** The $O(n_{\max}^2)$ adjacency output and $O(n_{\max}^3)$ matching computation limit GraphVAE to relatively small graphs (typically $n_{\max} \leq 40$).

**Independence assumption.** The decoder predicts each edge independently given $\mathbf{z}$, ignoring conditional dependencies between edges. This can produce graphs with inconsistent local structure.

**Posterior collapse.** As with standard VAEs, the model may learn to ignore the latent variable and produce average-looking graphs, especially when the decoder is too powerful or the $\beta$ weight on KL is too high.

## Finance Application: Latent Space of Financial Networks

GraphVAE's latent space provides a continuous representation of financial network topologies. Interpolating between two encoded financial networks $\mathbf{z}_1$ and $\mathbf{z}_2$ produces intermediate topologies that smoothly transition between configurations — useful for stress testing scenarios that gradually shift network structure from normal to stressed regimes.

## Implementation: GraphVAE

```python
"""
GraphVAE: Variational Autoencoder for one-shot graph generation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import numpy as np


class GraphEncoder(nn.Module):
    """GNN-based graph encoder for GraphVAE."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_layers: int = 3,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.conv_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.conv_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

    def forward(
        self,
        adj: torch.Tensor,
        x: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            adj: (B, n, n) adjacency
            x: (B, n, d) node features
            node_mask: (B, n) binary mask for valid nodes
        """
        h = self.input_proj(x)  # (B, n, hidden)

        for conv, norm in zip(self.conv_layers, self.norms):
            # Simple GCN-style message passing: h = σ(A h W)
            h_msg = torch.bmm(adj, h)  # (B, n, hidden)
            h = norm(F.relu(conv(h_msg)) + h)

        # Masked mean pooling
        mask = node_mask.unsqueeze(-1)  # (B, n, 1)
        h_graph = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        mu = self.mu_head(h_graph)
        logvar = self.logvar_head(h_graph)
        return mu, logvar


class GraphDecoder(nn.Module):
    """Decoder that generates adjacency, features, and node mask."""

    def __init__(
        self,
        latent_dim: int,
        max_nodes: int,
        node_feature_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.max_nodes = max_nodes
        n_edges = max_nodes * (max_nodes - 1) // 2

        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Adjacency head
        self.adj_head = nn.Linear(hidden_dim, n_edges)

        # Node existence head
        self.node_head = nn.Linear(hidden_dim, max_nodes)

        # Node feature head
        self.feat_head = nn.Linear(hidden_dim, max_nodes * node_feature_dim)
        self.node_feature_dim = node_feature_dim

    def forward(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            adj_prob: (B, n, n)
            node_prob: (B, n)
            node_feat: (B, n, d)
        """
        B = z.size(0)
        n = self.max_nodes
        h = self.backbone(z)

        # Adjacency
        edge_logits = self.adj_head(h)
        adj = torch.zeros(B, n, n, device=z.device)
        idx = torch.triu_indices(n, n, offset=1)
        adj[:, idx[0], idx[1]] = torch.sigmoid(edge_logits)
        adj = adj + adj.transpose(1, 2)

        # Node mask
        node_prob = torch.sigmoid(self.node_head(h))

        # Node features
        node_feat = self.feat_head(h).view(B, n, self.node_feature_dim)

        return adj, node_prob, node_feat


class GraphVAE(nn.Module):
    """
    Complete GraphVAE model with graph-matching loss.
    """

    def __init__(
        self,
        max_nodes: int,
        node_feature_dim: int = 1,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        beta: float = 1.0,
    ):
        super().__init__()
        self.max_nodes = max_nodes
        self.latent_dim = latent_dim
        self.beta = beta

        self.encoder = GraphEncoder(
            input_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
        )
        self.decoder = GraphDecoder(
            latent_dim=latent_dim,
            max_nodes=max_nodes,
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim,
        )

    def reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _compute_matching(
        self,
        adj_pred: torch.Tensor,
        adj_target: torch.Tensor,
        node_prob: torch.Tensor,
        n_target: int,
    ) -> torch.Tensor:
        """Hungarian matching for a single graph pair."""
        n = self.max_nodes

        # Build cost matrix from adjacency row similarity
        with torch.no_grad():
            cost = torch.zeros(n, n)
            for i in range(n):
                for j in range(n_target):
                    # Combine adjacency similarity and node existence
                    adj_sim = -F.mse_loss(
                        adj_pred[i], adj_target[j], reduction="sum"
                    )
                    node_sim = node_prob[i] if j < n_target else 0
                    cost[i, j] = adj_sim + node_sim

            cost_np = (-cost).cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost_np)

        perm = torch.tensor(col_ind, dtype=torch.long, device=adj_pred.device)
        return perm

    def forward(
        self,
        adj: torch.Tensor,
        x: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Training forward pass.
        
        Args:
            adj: (B, n, n) padded adjacency
            x: (B, n, d) padded node features
            node_mask: (B, n) valid node indicator
        """
        B = adj.size(0)

        # Encode
        mu, logvar = self.encoder(adj, x, node_mask)
        z = self.reparameterize(mu, logvar)

        # Decode
        adj_pred, node_pred, feat_pred = self.decoder(z)

        # KL divergence
        kl_loss = -0.5 * torch.mean(
            1 + logvar - mu.pow(2) - logvar.exp()
        )

        # Reconstruction with matching
        recon_loss = torch.tensor(0.0, device=adj.device)
        for b in range(B):
            n_actual = int(node_mask[b].sum().item())

            # Find best matching
            perm = self._compute_matching(
                adj_pred[b], adj[b], node_pred[b], n_actual
            )

            # Aligned targets
            adj_aligned = adj[b][perm][:, perm]
            mask_aligned = node_mask[b][perm]

            # Edge reconstruction loss
            idx = torch.triu_indices(self.max_nodes, self.max_nodes, offset=1)
            edge_pred = adj_pred[b][idx[0], idx[1]]
            edge_target = adj_aligned[idx[0], idx[1]]
            recon_loss = recon_loss + F.binary_cross_entropy(
                edge_pred.clamp(1e-6, 1 - 1e-6),
                edge_target,
                reduction="mean",
            )

            # Node existence loss
            recon_loss = recon_loss + F.binary_cross_entropy(
                node_pred[b].clamp(1e-6, 1 - 1e-6),
                mask_aligned,
                reduction="mean",
            )

        recon_loss = recon_loss / B
        total_loss = recon_loss + self.beta * kl_loss

        return {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }

    @torch.no_grad()
    def generate(
        self,
        num_graphs: int = 1,
        temperature: float = 1.0,
    ) -> list[torch.Tensor]:
        """Generate graphs from prior."""
        self.eval()
        z = torch.randn(num_graphs, self.latent_dim) * temperature
        adj_pred, node_pred, _ = self.decoder(z)

        graphs = []
        for b in range(num_graphs):
            # Determine active nodes
            active = node_pred[b] > 0.5
            n_active = active.sum().item()
            if n_active < 2:
                n_active = 2
                active[:2] = True

            # Sample edges for active nodes
            adj_b = adj_pred[b]
            adj_b = (adj_b > 0.5).float()
            adj_b = adj_b * active.unsqueeze(0).float() * active.unsqueeze(1).float()
            adj_b.fill_diagonal_(0)

            # Extract subgraph
            active_idx = torch.where(active)[0]
            sub_adj = adj_b[active_idx][:, active_idx]
            graphs.append(sub_adj)

        return graphs

    @torch.no_grad()
    def interpolate(
        self,
        adj1: torch.Tensor,
        x1: torch.Tensor,
        mask1: torch.Tensor,
        adj2: torch.Tensor,
        x2: torch.Tensor,
        mask2: torch.Tensor,
        steps: int = 5,
    ) -> list[torch.Tensor]:
        """Interpolate between two graphs in latent space."""
        self.eval()
        mu1, _ = self.encoder(adj1.unsqueeze(0), x1.unsqueeze(0), mask1.unsqueeze(0))
        mu2, _ = self.encoder(adj2.unsqueeze(0), x2.unsqueeze(0), mask2.unsqueeze(0))

        graphs = []
        for alpha in torch.linspace(0, 1, steps):
            z = (1 - alpha) * mu1 + alpha * mu2
            adj_pred, node_pred, _ = self.decoder(z)
            adj_b = (adj_pred[0] > 0.5).float()
            active = node_pred[0] > 0.5
            adj_b = adj_b * active.unsqueeze(0).float() * active.unsqueeze(1).float()
            adj_b.fill_diagonal_(0)
            active_idx = torch.where(active)[0]
            if len(active_idx) >= 2:
                graphs.append(adj_b[active_idx][:, active_idx])
            else:
                graphs.append(adj_b[:2, :2])

        return graphs


if __name__ == "__main__":
    torch.manual_seed(42)

    max_n = 12
    feat_dim = 4

    # Create training data
    print("=== GraphVAE Demo ===\n")
    train_data = []
    for _ in range(150):
        n = torch.randint(4, max_n, (1,)).item()
        adj = torch.zeros(max_n, max_n)
        for i in range(n):
            for j in range(i + 1, n):
                if torch.rand(1) < 0.3:
                    adj[i, j] = adj[j, i] = 1
        x = torch.randn(max_n, feat_dim)
        mask = torch.zeros(max_n)
        mask[:n] = 1.0
        train_data.append((adj, x, mask))

    adjs = torch.stack([d[0] for d in train_data])
    feats = torch.stack([d[1] for d in train_data])
    masks = torch.stack([d[2] for d in train_data])

    # Train
    print("Training GraphVAE...")
    model = GraphVAE(
        max_nodes=max_n,
        node_feature_dim=feat_dim,
        hidden_dim=64,
        latent_dim=16,
        beta=0.1,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    batch_size = 32
    for epoch in range(30):
        model.train()
        idx = torch.randperm(len(train_data))[:batch_size]
        result = model(adjs[idx], feats[idx], masks[idx])
        loss = result["total_loss"]

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: loss={loss.item():.4f} "
                  f"(recon={result['recon_loss'].item():.4f}, "
                  f"kl={result['kl_loss'].item():.4f})")

    # Generate
    print("\n=== Generation ===")
    generated = model.generate(num_graphs=10)
    for i, g in enumerate(generated):
        n = g.size(0)
        e = int(g.sum().item()) // 2
        print(f"Graph {i}: {n} nodes, {e} edges")

    # Interpolation
    print("\n=== Latent Interpolation ===")
    interp = model.interpolate(
        adjs[0], feats[0], masks[0],
        adjs[1], feats[1], masks[1],
        steps=5,
    )
    for i, g in enumerate(interp):
        n = g.size(0)
        e = int(g.sum().item()) // 2
        density = 2 * e / (n * (n - 1)) if n > 1 else 0
        print(f"Step {i}: {n} nodes, {e} edges, density={density:.3f}")
```
