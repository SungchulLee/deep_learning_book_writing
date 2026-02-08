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
