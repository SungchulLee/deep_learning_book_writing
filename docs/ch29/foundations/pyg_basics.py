"""
Chapter 29.1.6: PyTorch Geometric Basics
Introduction to PyG data structures, batching, transforms, and utilities.
"""

import torch
import torch.nn as nn
import numpy as np
import networkx as nx
from typing import List, Tuple, Optional, Dict

# Note: In a real environment, you would import from torch_geometric:
# from torch_geometric.data import Data, DataLoader
# from torch_geometric.nn import MessagePassing, GCNConv
# from torch_geometric.utils import to_networkx, from_networkx, degree
# from torch_geometric.datasets import Planetoid
# from torch_geometric.transforms import NormalizeFeatures

# For demonstration, we implement minimal equivalents.


# ============================================================
# 1. Data Object (PyG-compatible)
# ============================================================

class GraphData:
    """
    Minimal PyG-compatible Data object.
    Represents a single graph with node features, edges, and labels.
    """

    def __init__(self, x=None, edge_index=None, edge_attr=None,
                 y=None, **kwargs):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.y = y
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def num_nodes(self):
        if self.x is not None:
            return self.x.shape[0]
        if self.edge_index is not None:
            return int(self.edge_index.max()) + 1
        return 0

    @property
    def num_edges(self):
        if self.edge_index is not None:
            return self.edge_index.shape[1]
        return 0

    @property
    def num_node_features(self):
        if self.x is not None:
            return self.x.shape[1]
        return 0

    @property
    def num_edge_features(self):
        if self.edge_attr is not None:
            return self.edge_attr.shape[1]
        return 0

    def __repr__(self):
        info = [f"x={list(self.x.shape)}" if self.x is not None else None,
                f"edge_index={list(self.edge_index.shape)}" if self.edge_index is not None else None,
                f"edge_attr={list(self.edge_attr.shape)}" if self.edge_attr is not None else None,
                f"y={list(self.y.shape)}" if self.y is not None else None]
        info = [i for i in info if i]
        return f"GraphData({', '.join(info)})"


def demo_data_object():
    """Demonstrate creating and using a Data object."""
    print("=" * 60)
    print("PyG Data Object")
    print("=" * 60)

    # Create a simple graph: 5 nodes, 7 undirected edges
    #   0 -- 1
    #   |  / |
    #   2    3
    #   |    |
    #   4 -- 4 (self-loop removed, 4 connects to 3)

    edge_index = torch.tensor([
        [0, 1, 0, 2, 1, 2, 1, 3, 2, 4, 3, 4],
        [1, 0, 2, 0, 2, 1, 3, 1, 4, 2, 4, 3]
    ], dtype=torch.long)

    # Node features (5 nodes, 3 features each)
    x = torch.tensor([
        [1.0, 0.0, 0.5],
        [0.0, 1.0, 0.3],
        [1.0, 1.0, 0.8],
        [0.5, 0.5, 0.1],
        [0.0, 0.0, 0.9],
    ], dtype=torch.float)

    # Node labels
    y = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)

    data = GraphData(x=x, edge_index=edge_index, y=y)

    print(f"Data: {data}")
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.num_edges}")
    print(f"Number of node features: {data.num_node_features}")
    print(f"Has self-loops: {has_self_loops(data.edge_index)}")
    print(f"Is undirected: {is_undirected(data.edge_index)}")

    return data


# ============================================================
# 2. Utility Functions
# ============================================================

def has_self_loops(edge_index: torch.Tensor) -> bool:
    """Check if edge_index contains self-loops."""
    return bool((edge_index[0] == edge_index[1]).any())


def is_undirected(edge_index: torch.Tensor) -> bool:
    """Check if the graph is undirected."""
    edge_set = set()
    for i in range(edge_index.shape[1]):
        edge_set.add((edge_index[0, i].item(), edge_index[1, i].item()))
    for i in range(edge_index.shape[1]):
        if (edge_index[1, i].item(), edge_index[0, i].item()) not in edge_set:
            return False
    return True


def add_self_loops(edge_index: torch.Tensor,
                    num_nodes: int) -> torch.Tensor:
    """Add self-loops to edge_index."""
    loop_index = torch.arange(num_nodes, dtype=torch.long)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)
    return torch.cat([edge_index, loop_index], dim=1)


def remove_self_loops(edge_index: torch.Tensor) -> torch.Tensor:
    """Remove self-loops from edge_index."""
    mask = edge_index[0] != edge_index[1]
    return edge_index[:, mask]


def compute_degree(edge_index: torch.Tensor,
                    num_nodes: int) -> torch.Tensor:
    """Compute node degrees from edge_index."""
    degrees = torch.zeros(num_nodes, dtype=torch.long)
    for i in range(edge_index.shape[1]):
        degrees[edge_index[1, i]] += 1
    return degrees


def to_dense_adjacency(edge_index: torch.Tensor,
                        num_nodes: int,
                        edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Convert edge_index to dense adjacency matrix."""
    if edge_attr is not None:
        adj = torch.zeros(num_nodes, num_nodes, edge_attr.shape[1])
        for i in range(edge_index.shape[1]):
            adj[edge_index[0, i], edge_index[1, i]] = edge_attr[i]
    else:
        adj = torch.zeros(num_nodes, num_nodes)
        for i in range(edge_index.shape[1]):
            adj[edge_index[0, i], edge_index[1, i]] = 1.0
    return adj


def from_networkx_graph(G: nx.Graph) -> GraphData:
    """Convert NetworkX graph to GraphData."""
    # Edge index
    edges = list(G.edges())
    if len(edges) > 0:
        src = [e[0] for e in edges] + [e[1] for e in edges]
        dst = [e[1] for e in edges] + [e[0] for e in edges]
        edge_index = torch.tensor([src, dst], dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    num_nodes = G.number_of_nodes()
    x = torch.eye(num_nodes)  # One-hot as default features

    return GraphData(x=x, edge_index=edge_index)


def to_networkx_graph(data: GraphData) -> nx.Graph:
    """Convert GraphData to NetworkX graph."""
    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))
    edge_index = data.edge_index.numpy()
    for i in range(edge_index.shape[1]):
        u, v = edge_index[0, i], edge_index[1, i]
        if u < v:  # Avoid duplicate edges
            G.add_edge(u, v)
    return G


def demo_utilities():
    """Demonstrate utility functions."""
    print("\n" + "=" * 60)
    print("Graph Utility Functions")
    print("=" * 60)

    edge_index = torch.tensor([
        [0, 1, 0, 2, 1, 2, 1, 3, 2, 4, 3, 4],
        [1, 0, 2, 0, 2, 1, 3, 1, 4, 2, 4, 3]
    ], dtype=torch.long)
    num_nodes = 5

    # Degree
    degrees = compute_degree(edge_index, num_nodes)
    print(f"Degrees: {degrees.tolist()}")

    # Add self-loops
    ei_with_loops = add_self_loops(edge_index, num_nodes)
    print(f"Edges before self-loops: {edge_index.shape[1]}")
    print(f"Edges after self-loops: {ei_with_loops.shape[1]}")

    # Dense adjacency
    adj = to_dense_adjacency(edge_index, num_nodes)
    print(f"\nDense adjacency:\n{adj}")

    # NetworkX conversion
    data = GraphData(x=torch.randn(num_nodes, 3), edge_index=edge_index)
    G = to_networkx_graph(data)
    print(f"\nNetworkX graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    data_back = from_networkx_graph(G)
    print(f"Converted back: {data_back}")


# ============================================================
# 3. Batching Multiple Graphs
# ============================================================

class Batch:
    """Minimal batch implementation for multiple graphs."""

    @staticmethod
    def from_data_list(data_list: List[GraphData]) -> GraphData:
        """Batch multiple graphs into one disconnected graph."""
        xs, edge_indices, ys, batches = [], [], [], []
        edge_attrs = []
        node_offset = 0

        for idx, data in enumerate(data_list):
            n = data.num_nodes
            xs.append(data.x)

            # Offset edge indices
            ei = data.edge_index + node_offset
            edge_indices.append(ei)

            if data.edge_attr is not None:
                edge_attrs.append(data.edge_attr)

            if data.y is not None:
                ys.append(data.y)

            # Batch vector
            batches.append(torch.full((n,), idx, dtype=torch.long))

            node_offset += n

        batch_data = GraphData(
            x=torch.cat(xs, dim=0),
            edge_index=torch.cat(edge_indices, dim=1),
            y=torch.stack(ys) if ys else None,
            batch=torch.cat(batches, dim=0),
        )

        if edge_attrs:
            batch_data.edge_attr = torch.cat(edge_attrs, dim=0)

        return batch_data


def demo_batching():
    """Demonstrate graph batching."""
    print("\n" + "=" * 60)
    print("Graph Batching")
    print("=" * 60)

    # Create 3 small graphs
    graphs = []

    # Graph 1: Triangle
    g1 = GraphData(
        x=torch.randn(3, 4),
        edge_index=torch.tensor([[0, 1, 1, 2, 0, 2],
                                  [1, 0, 2, 1, 2, 0]], dtype=torch.long),
        y=torch.tensor([0])
    )
    graphs.append(g1)

    # Graph 2: Path of 4 nodes
    g2 = GraphData(
        x=torch.randn(4, 4),
        edge_index=torch.tensor([[0, 1, 1, 2, 2, 3],
                                  [1, 0, 2, 1, 3, 2]], dtype=torch.long),
        y=torch.tensor([1])
    )
    graphs.append(g2)

    # Graph 3: Star with 5 nodes
    g3 = GraphData(
        x=torch.randn(5, 4),
        edge_index=torch.tensor([[0, 1, 0, 2, 0, 3, 0, 4, 1, 2, 3, 4],
                                  [1, 0, 2, 0, 3, 0, 4, 0, 0, 0, 0, 0]], dtype=torch.long),
        y=torch.tensor([0])
    )
    graphs.append(g3)

    print(f"Individual graphs:")
    for i, g in enumerate(graphs):
        print(f"  Graph {i}: {g.num_nodes} nodes, {g.num_edges} edges")

    # Batch
    batch = Batch.from_data_list(graphs)
    print(f"\nBatched graph: {batch}")
    print(f"Total nodes: {batch.num_nodes}")
    print(f"Total edges: {batch.num_edges}")
    print(f"Batch vector: {batch.batch.tolist()}")
    print(f"Labels: {batch.y.tolist()}")

    # Global mean pooling per graph
    unique_graphs = batch.batch.unique()
    for g_idx in unique_graphs:
        mask = batch.batch == g_idx
        pooled = batch.x[mask].mean(dim=0)
        print(f"  Graph {g_idx.item()} pooled shape: {pooled.shape}")

    return batch


# ============================================================
# 4. Transforms
# ============================================================

class NormalizeFeatures:
    """Row-normalize node features."""

    def __call__(self, data: GraphData) -> GraphData:
        if data.x is not None:
            row_sum = data.x.sum(dim=1, keepdim=True).clamp(min=1e-10)
            data.x = data.x / row_sum
        return data


class AddSelfLoops:
    """Add self-loops to graph."""

    def __call__(self, data: GraphData) -> GraphData:
        data.edge_index = add_self_loops(data.edge_index, data.num_nodes)
        return data


class GCNNorm:
    """Apply GCN normalization: D^{-1/2} A D^{-1/2}."""

    def __call__(self, data: GraphData) -> GraphData:
        edge_index = data.edge_index
        num_nodes = data.num_nodes

        # Add self-loops
        ei = add_self_loops(edge_index, num_nodes)

        # Compute degree
        deg = compute_degree(ei, num_nodes).float()
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        # Compute normalized edge weights
        src, dst = ei[0], ei[1]
        edge_weight = deg_inv_sqrt[src] * deg_inv_sqrt[dst]

        data.edge_index = ei
        data.edge_weight = edge_weight
        return data


class Compose:
    """Chain multiple transforms."""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, data: GraphData) -> GraphData:
        for t in self.transforms:
            data = t(data)
        return data


def demo_transforms():
    """Demonstrate transforms."""
    print("\n" + "=" * 60)
    print("Graph Transforms")
    print("=" * 60)

    data = GraphData(
        x=torch.tensor([[1.0, 2.0, 3.0],
                         [4.0, 5.0, 6.0],
                         [7.0, 8.0, 9.0]], dtype=torch.float),
        edge_index=torch.tensor([[0, 1, 1, 2],
                                  [1, 0, 2, 1]], dtype=torch.long),
    )

    print(f"Original: {data}")
    print(f"Original x:\n{data.x}")

    # Normalize features
    norm = NormalizeFeatures()
    data_norm = norm(GraphData(x=data.x.clone(), edge_index=data.edge_index.clone()))
    print(f"\nAfter NormalizeFeatures:")
    print(f"x:\n{data_norm.x}")
    print(f"Row sums: {data_norm.x.sum(dim=1)}")

    # Add self-loops
    data_loops = AddSelfLoops()(GraphData(x=data.x.clone(), edge_index=data.edge_index.clone()))
    print(f"\nAfter AddSelfLoops:")
    print(f"Edges: {data.num_edges} -> {data_loops.num_edges}")

    # GCN normalization
    data_gcn = GCNNorm()(GraphData(x=data.x.clone(), edge_index=data.edge_index.clone()))
    print(f"\nAfter GCNNorm:")
    print(f"Edge weights: {data_gcn.edge_weight}")

    # Compose
    transform = Compose([NormalizeFeatures(), AddSelfLoops()])
    data_composed = transform(GraphData(x=data.x.clone(), edge_index=data.edge_index.clone()))
    print(f"\nAfter Compose(Normalize, AddSelfLoops):")
    print(f"x row sums: {data_composed.x.sum(dim=1)}")
    print(f"Edges: {data_composed.num_edges}")


# ============================================================
# 5. Simple GNN Layer (Message Passing)
# ============================================================

class SimpleMessagePassing(nn.Module):
    """
    Minimal message passing layer.
    h_v = UPDATE(h_v, AGG({MESSAGE(h_v, h_u) : u in N(v)}))
    """

    def __init__(self, in_channels: int, out_channels: int, aggr: str = 'mean'):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels)
        self.aggr = aggr

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.shape[0]
        src, dst = edge_index[0], edge_index[1]

        # Message: transform source node features
        messages = self.lin(x[src])  # [num_edges, out_channels]

        # Aggregate messages for each target node
        out = torch.zeros(num_nodes, messages.shape[1],
                          device=x.device, dtype=x.dtype)

        if self.aggr == 'sum':
            out.scatter_add_(0, dst.unsqueeze(1).expand_as(messages), messages)
        elif self.aggr == 'mean':
            out.scatter_add_(0, dst.unsqueeze(1).expand_as(messages), messages)
            deg = compute_degree(edge_index, num_nodes).float().clamp(min=1)
            out = out / deg.unsqueeze(1)
        elif self.aggr == 'max':
            out.fill_(float('-inf'))
            out.scatter_reduce_(0, dst.unsqueeze(1).expand_as(messages),
                                messages, reduce='amax')
            out[out == float('-inf')] = 0

        return out


def demo_message_passing():
    """Demonstrate basic message passing."""
    print("\n" + "=" * 60)
    print("Simple Message Passing Layer")
    print("=" * 60)

    torch.manual_seed(42)

    # Create graph
    edge_index = torch.tensor([
        [0, 1, 0, 2, 1, 2, 1, 3, 2, 4, 3, 4],
        [1, 0, 2, 0, 2, 1, 3, 1, 4, 2, 4, 3]
    ], dtype=torch.long)
    x = torch.randn(5, 8)

    # Apply message passing
    layer = SimpleMessagePassing(8, 16, aggr='mean')
    out = layer(x, edge_index)

    print(f"Input: x shape = {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output (first 2 nodes):\n{out[:2]}")

    # Stack multiple layers
    layer1 = SimpleMessagePassing(8, 16, aggr='mean')
    layer2 = SimpleMessagePassing(16, 8, aggr='mean')

    h = torch.relu(layer1(x, edge_index))
    h = layer2(h, edge_index)
    print(f"\n2-layer GNN output shape: {h.shape}")

    return out


# ============================================================
# 6. Complete Example: Node Classification Pipeline
# ============================================================

class SimpleGNN(nn.Module):
    """Simple 2-layer GNN for node classification."""

    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SimpleMessagePassing(in_channels, hidden_channels, aggr='mean')
        self.conv2 = SimpleMessagePassing(hidden_channels, out_channels, aggr='mean')

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x


def demo_node_classification():
    """End-to-end node classification example."""
    print("\n" + "=" * 60)
    print("Node Classification Pipeline")
    print("=" * 60)

    torch.manual_seed(42)

    # Create a graph with 2 communities
    G = nx.karate_club_graph()
    n = G.number_of_nodes()

    # Convert to edge_index
    edges = list(G.edges())
    src = [e[0] for e in edges] + [e[1] for e in edges]
    dst = [e[1] for e in edges] + [e[0] for e in edges]
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    # Features: one-hot encoding of node id
    x = torch.eye(n, dtype=torch.float)

    # Labels: community membership
    labels = [G.nodes[i].get('club', 'Mr. Hi') for i in range(n)]
    y = torch.tensor([0 if l == 'Mr. Hi' else 1 for l in labels], dtype=torch.long)

    # Train/test split
    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[:20] = True
    test_mask = ~train_mask

    # Model
    model = SimpleGNN(n, 16, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # Training
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = criterion(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                pred = model(x, edge_index).argmax(dim=1)
                train_acc = (pred[train_mask] == y[train_mask]).float().mean()
                test_acc = (pred[test_mask] == y[test_mask]).float().mean()
            print(f"Epoch {epoch+1:3d}: Loss={loss.item():.4f}, "
                  f"Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")
            model.train()

    print("\nNode classification complete!")


# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":
    demo_data_object()
    demo_utilities()
    demo_batching()
    demo_transforms()
    demo_message_passing()
    demo_node_classification()
