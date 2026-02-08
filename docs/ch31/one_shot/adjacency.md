# One-Shot Adjacency Generation

## Overview

One-shot graph generation produces the entire adjacency matrix $\mathbf{A} \in \{0,1\}^{n \times n}$ and node features $\mathbf{X} \in \mathbb{R}^{n \times d}$ in a single forward pass, avoiding the sequential bottleneck of autoregressive methods. The core challenge is that one-shot generation must handle permutation invariance, discrete structure, and variable graph sizes simultaneously — problems that autoregressive methods sidestep through ordered decomposition.

## Formulation

Given a maximum graph size $n_{\max}$, one-shot methods learn a mapping from a latent vector $\mathbf{z} \in \mathbb{R}^{d_z}$ to a graph:

$$
(\hat{\mathbf{A}}, \hat{\mathbf{X}}) = f_\theta(\mathbf{z}), \quad \mathbf{z} \sim p(\mathbf{z})
$$

where $\hat{\mathbf{A}} \in [0,1]^{n_{\max} \times n_{\max}}$ is a continuous relaxation of the adjacency matrix and $\hat{\mathbf{X}} \in \mathbb{R}^{n_{\max} \times d}$ are predicted node features. The generated graph is obtained by thresholding or sampling: $A_{ij} \sim \text{Bernoulli}(\hat{A}_{ij})$.

## Permutation Invariance Challenge

The fundamental difficulty of one-shot generation is that graph likelihood is defined up to permutation:

$$
p(\mathcal{G}) = \frac{1}{n!} \sum_{\pi \in S_n} p(\mathbf{P}_\pi \mathbf{A} \mathbf{P}_\pi^\top, \mathbf{P}_\pi \mathbf{X})
$$

This marginalization over all $n!$ permutations is intractable for all but trivially small graphs. Three strategies address this:

**Canonical ordering.** Fix a deterministic ordering (e.g., by degree) so each graph has a unique matrix representation. This converts the problem to standard density estimation but may introduce artifacts from the chosen ordering.

**Permutation-equivariant architectures.** Design the decoder so that any permutation of $\mathbf{z}$ (or its structured version) produces a correspondingly permuted output. This ensures the model assigns equal probability to all representations of the same graph.

**Matching-based loss.** During training, find the optimal permutation $\pi^*$ that aligns the generated graph to the target before computing the loss:

$$
\pi^* = \arg\min_{\pi \in S_n} \|\hat{\mathbf{A}} - \mathbf{P}_\pi \mathbf{A}^* \mathbf{P}_\pi^\top\|_F^2
$$

This is equivalent to the graph matching problem (in general NP-hard), but practical approximations using the Hungarian algorithm on node features provide effective solutions.

## Decoder Architectures

### MLP Decoder

The simplest approach maps the latent vector to a flattened upper-triangular adjacency vector:

$$
\hat{\mathbf{a}} = \sigma\left(\text{MLP}(\mathbf{z})\right) \in [0,1]^{\binom{n_{\max}}{2}}
$$

This requires $O(n_{\max}^2 \cdot d_z)$ parameters in the final layer and does not exploit any structural priors about graphs.

### Factored Decoder

Factor the adjacency prediction through node embeddings to reduce parameter count:

$$
\mathbf{Z}_{\text{nodes}} = \text{MLP}_{\text{node}}(\mathbf{z}) \in \mathbb{R}^{n_{\max} \times d_h}
$$
$$
\hat{A}_{ij} = \sigma\left(\mathbf{z}_i^\top \mathbf{W} \mathbf{z}_j + b\right)
$$

where $\mathbf{z}_i$ is the $i$-th row of $\mathbf{Z}_{\text{nodes}}$. This bilinear form requires only $O(n_{\max} \cdot d_h + d_h^2)$ parameters and naturally produces symmetric adjacency matrices when $\mathbf{W}$ is symmetric.

### Graph-Conditioned Decoder

Use a GNN on the latent node representations to iteratively refine edge predictions:

$$
\mathbf{Z}^{(\ell+1)} = \text{GNN}^{(\ell)}(\mathbf{Z}^{(\ell)}, \hat{\mathbf{A}}^{(\ell)})
$$

where $\hat{\mathbf{A}}^{(\ell)}$ is computed from $\mathbf{Z}^{(\ell)}$ at each iteration. This iterative refinement allows edge predictions to depend on the emerging graph structure.

## Size Handling

One-shot methods must handle graphs of different sizes. Common approaches:

**Padding with node mask.** Generate graphs of fixed size $n_{\max}$ and predict a node existence mask $\mathbf{m} \in [0,1]^{n_{\max}}$. The actual graph uses nodes where $m_i > 0.5$. Edges involving masked nodes are zeroed out:

$$
\hat{A}_{ij}^{\text{final}} = \hat{A}_{ij} \cdot m_i \cdot m_j
$$

**Size conditioning.** Sample graph size $n \sim p(n)$ from the empirical size distribution, then condition generation on $n$. This can be implemented by providing $n$ as an additional input to the decoder.

## Training Loss Components

A complete one-shot generation loss typically combines:

$$
\mathcal{L} = \mathcal{L}_{\text{recon}} + \beta \mathcal{L}_{\text{KL}} + \lambda_{\text{match}} \mathcal{L}_{\text{match}}
$$

where $\mathcal{L}_{\text{recon}}$ is the edge reconstruction loss (binary cross-entropy), $\mathcal{L}_{\text{KL}}$ is the KL divergence for VAE-based methods, and $\mathcal{L}_{\text{match}}$ handles permutation alignment.

## Finance Application: Instantaneous Network Snapshots

One-shot generation is naturally suited for producing complete financial network snapshots — for instance, generating a full cross-sectional view of interbank exposures at a regulatory reporting date. Unlike autoregressive methods that model network formation, one-shot methods produce equilibrium configurations directly, which is appropriate when the temporal formation process is not of interest.

## Implementation: One-Shot Adjacency Generator

```python
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
        """
        Args:
            z: (B, latent_dim)
        Returns:
            adj: (B, max_nodes, max_nodes) predicted adjacency probabilities
        """
        B = z.size(0)
        n = self.max_nodes
        edge_logits = self.mlp(z)  # (B, n_edges)

        # Reconstruct symmetric adjacency
        adj = torch.zeros(B, n, n, device=z.device)
        idx = torch.triu_indices(n, n, offset=1)
        adj[:, idx[0], idx[1]] = torch.sigmoid(edge_logits)
        adj = adj + adj.transpose(1, 2)

        return adj


class FactoredDecoder(nn.Module):
    """
    Factored decoder: latent -> node embeddings -> bilinear edge prediction.
    More parameter-efficient than MLP decoder.
    """

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

        # Latent to node embeddings
        self.node_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_nodes * node_embed_dim),
        )

        # Bilinear edge predictor
        self.edge_weight = nn.Parameter(
            torch.randn(node_embed_dim, node_embed_dim) * 0.01
        )
        self.edge_bias = nn.Parameter(torch.zeros(1))

        # Node existence predictor
        self.node_mask_predictor = nn.Sequential(
            nn.Linear(node_embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            adj: (B, n, n) adjacency probabilities
            node_mask: (B, n) node existence probabilities
        """
        B = z.size(0)
        n = self.max_nodes

        # Decode node embeddings
        node_embeds = self.node_decoder(z).view(B, n, self.node_embed_dim)

        # Bilinear edge prediction: A_ij = σ(z_i^T W z_j + b)
        # Make W symmetric for undirected graphs
        W_sym = (self.edge_weight + self.edge_weight.t()) / 2
        edge_logits = torch.einsum("bid,de,bje->bij", node_embeds, W_sym, node_embeds)
        edge_logits = edge_logits + self.edge_bias
        adj = torch.sigmoid(edge_logits)

        # Zero diagonal
        mask = 1 - torch.eye(n, device=z.device).unsqueeze(0)
        adj = adj * mask

        # Node existence mask
        node_mask = torch.sigmoid(
            self.node_mask_predictor(node_embeds).squeeze(-1)
        )

        # Apply node mask to adjacency
        adj = adj * node_mask.unsqueeze(1) * node_mask.unsqueeze(2)

        return adj, node_mask


def hungarian_matching(
    adj_pred: torch.Tensor,
    adj_target: torch.Tensor,
) -> torch.Tensor:
    """
    Find best node permutation using Hungarian algorithm.
    
    Args:
        adj_pred: (n, n) predicted adjacency
        adj_target: (n, n) target adjacency
        
    Returns:
        perm: (n,) permutation indices
    """
    n = adj_pred.size(0)
    # Cost matrix: ||pred_i - target_j||^2 for all (i, j) pairs
    # Using rows of adjacency as node features
    cost = torch.cdist(adj_pred.float(), adj_target.float(), p=2)
    cost_np = cost.detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost_np)
    perm = torch.tensor(col_ind, dtype=torch.long, device=adj_pred.device)
    return perm


def permutation_aligned_loss(
    adj_pred: torch.Tensor,
    adj_target: torch.Tensor,
    use_matching: bool = True,
) -> torch.Tensor:
    """
    Compute reconstruction loss with optional permutation alignment.
    
    Args:
        adj_pred: (B, n, n) predicted adjacencies
        adj_target: (B, n, n) target adjacencies
        use_matching: whether to apply Hungarian matching
    """
    B = adj_pred.size(0)
    total_loss = 0.0

    for b in range(B):
        pred = adj_pred[b]
        target = adj_target[b]

        if use_matching:
            perm = hungarian_matching(pred, target)
            target = target[perm][:, perm]

        # Binary cross-entropy on upper triangle
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

    # Create training data
    print("=== One-Shot Adjacency Generation Demo ===\n")

    train_adjs = []
    for _ in range(100):
        n = torch.randint(6, max_n + 1, (1,)).item()
        adj = torch.zeros(max_n, max_n)
        # Random graph
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

    # Permuted version
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
```
