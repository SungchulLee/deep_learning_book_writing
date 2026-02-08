# Introduction to Graph Generation

## The Generation Problem

Graph generation seeks to learn a distribution $p_\theta(\mathcal{G})$ over graphs from a training set $\{\mathcal{G}_1, \ldots, \mathcal{G}_N\}$ and sample new graphs that are statistically indistinguishable from the training distribution. This problem is fundamentally more challenging than generating images or sequences due to three structural properties unique to graphs.

**Combinatorial output space.** An undirected graph on $n$ labeled nodes requires specifying $\binom{n}{2}$ binary edge variables, yielding $2^{\binom{n}{2}}$ possible graphs. For $n = 50$, this exceeds $10^{368}$ — far larger than the number of atoms in the observable universe. Any tractable generation method must impose structure on this space.

**Permutation symmetry.** A graph $\mathcal{G}$ with $n$ nodes has up to $n!$ equivalent representations under node relabeling. The generative model must either:
1. Be explicitly permutation-invariant: $p_\theta(\mathcal{G}) = p_\theta(\pi(\mathcal{G}))$ for all $\pi \in S_n$
2. Use a canonical ordering that breaks the symmetry
3. Marginalize over permutations (often intractable)

**Variable dimensionality.** Unlike images with fixed resolution, graphs in a dataset may have different numbers of nodes and edges. The model must handle this heterogeneity either by conditioning on size or using architectures with natural stopping criteria.

## Formal Problem Statement

Given a dataset $\mathcal{D} = \{\mathcal{G}_i\}_{i=1}^N$ where each graph $\mathcal{G}_i = (\mathcal{V}_i, \mathcal{E}_i, \mathbf{X}_i, \mathbf{E}_i)$ consists of nodes $\mathcal{V}_i$, edges $\mathcal{E}_i$, node features $\mathbf{X}_i \in \mathbb{R}^{|\mathcal{V}_i| \times d_n}$, and edge features $\mathbf{E}_i \in \mathbb{R}^{|\mathcal{E}_i| \times d_e}$, the objective is:

$$
\theta^* = \arg\max_\theta \sum_{i=1}^N \log p_\theta(\mathcal{G}_i)
$$

The likelihood $p_\theta(\mathcal{G})$ must be invariant to node permutations:

$$
p_\theta(\mathcal{G}) = p_\theta(\mathbf{P}\mathcal{G}\mathbf{P}^\top) \quad \forall \mathbf{P} \in \Pi_n
$$

where $\mathbf{P}\mathcal{G}\mathbf{P}^\top$ denotes simultaneous permutation of the adjacency matrix and node features.

## Two Paradigms

### Autoregressive Decomposition

Factor the joint distribution using the chain rule over a chosen ordering $\sigma$:

$$
p_\theta(\mathcal{G}) = \prod_{t=1}^{T} p_\theta(a_t \mid a_{1:t-1})
$$

where $a_t$ represents the $t$-th generation action (adding a node, edge, or block). This naturally handles variable-sized graphs through a termination action but introduces ordering dependence.

### One-Shot Generation

Generate all nodes and edges simultaneously:

$$
p_\theta(\mathcal{G}) = p_\theta(\mathbf{A}, \mathbf{X})
$$

This is fully parallel but requires handling fixed maximum size $n_{\max}$ and ensuring permutation invariance. Diffusion-based methods extend this paradigm through iterative refinement.

## Generation Pipeline

```
Training Data ──→ Graph Encoder ──→ Latent Space ──→ Graph Decoder ──→ Generated Graph
                        │                                    │
                   Node ordering              Validity enforcement
                   Feature extraction         Post-processing
                   Size normalization         Constraint satisfaction
```

A typical graph generation pipeline involves:

1. **Preprocessing**: Canonicalize node ordering (e.g., BFS, DFS), pad to uniform size
2. **Encoding**: Map input graphs to latent representations via GNNs
3. **Latent modeling**: Learn the latent distribution (VAE, flow, diffusion)
4. **Decoding**: Map latent samples to graph structures
5. **Post-processing**: Enforce validity constraints, remove padding

## Connection to Quantitative Finance

Financial graph generation serves as a critical infrastructure for risk management. Consider the interbank lending network: each bank is a node with attributes (assets, liabilities, capital ratios), and each directed edge represents a lending exposure with amount and maturity. The true network is confidential and only partially observable. Generating plausible completions of this network is essential for:

- **Contagion modeling**: Simulating default cascades under different network topologies
- **Portfolio stress testing**: Evaluating portfolio performance under synthetic market regimes
- **Counterparty risk**: Estimating exposure to indirect connections through generated network paths

The methods in this chapter provide the technical foundations for all of these applications.

## Implementation: Graph Generation Base Class

```python
"""
Graph generation base class and utilities.
"""
import torch
import torch.nn as nn
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Optional


@dataclass
class GeneratedGraph:
    """Container for a generated graph."""
    adjacency: torch.Tensor       # (n, n) binary or weighted
    node_features: Optional[torch.Tensor] = None  # (n, d_n)
    edge_features: Optional[torch.Tensor] = None   # (n, n, d_e)
    num_nodes: int = 0
    metadata: dict = None

    def __post_init__(self):
        if self.num_nodes == 0:
            self.num_nodes = self.adjacency.size(0)
        if self.metadata is None:
            self.metadata = {}

    @property
    def num_edges(self) -> int:
        if self.adjacency.dim() == 2:
            return int(self.adjacency.sum().item()) // 2
        return 0

    @property
    def density(self) -> float:
        n = self.num_nodes
        max_edges = n * (n - 1) / 2
        return self.num_edges / max_edges if max_edges > 0 else 0.0


class GraphGenerator(ABC, nn.Module):
    """Abstract base class for graph generators."""

    def __init__(self, max_nodes: int, node_feature_dim: int = 0):
        super().__init__()
        self.max_nodes = max_nodes
        self.node_feature_dim = node_feature_dim

    @abstractmethod
    def forward(self, batch) -> torch.Tensor:
        """Training forward pass returning loss."""
        ...

    @abstractmethod
    @torch.no_grad()
    def generate(self, num_graphs: int = 1, **kwargs) -> list[GeneratedGraph]:
        """Generate new graphs."""
        ...

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def adjacency_to_edge_index(adj: torch.Tensor) -> torch.Tensor:
    """Convert adjacency matrix to PyG edge_index format."""
    row, col = torch.where(adj > 0)
    return torch.stack([row, col], dim=0)


def edge_index_to_adjacency(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Convert PyG edge_index to adjacency matrix."""
    adj = torch.zeros(num_nodes, num_nodes)
    adj[edge_index[0], edge_index[1]] = 1.0
    return adj


if __name__ == "__main__":
    # Demonstrate GeneratedGraph usage
    n = 10
    adj = torch.zeros(n, n)
    # Create a random graph with ~30% density
    mask = torch.rand(n, n) < 0.3
    adj = (mask | mask.t()).float()
    adj.fill_diagonal_(0)

    graph = GeneratedGraph(
        adjacency=adj,
        node_features=torch.randn(n, 4),
        metadata={"generator": "random", "density_target": 0.3},
    )
    print(f"Nodes: {graph.num_nodes}, Edges: {graph.num_edges}")
    print(f"Density: {graph.density:.3f}")
    print(f"Edge index shape: {adjacency_to_edge_index(adj).shape}")
```
