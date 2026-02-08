"""
Chapter 29.2.1: Message Passing Framework
Generic message passing implementation and demonstrations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
from typing import Optional


# ============================================================
# 1. Generic Message Passing Layer
# ============================================================

class MessagePassingLayer(nn.Module):
    """
    Generic message passing layer with configurable message,
    aggregation, and update functions.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 aggr: str = 'mean', msg_type: str = 'linear',
                 update_type: str = 'replace'):
        super().__init__()
        self.aggr = aggr
        self.msg_type = msg_type
        self.update_type = update_type

        if msg_type == 'linear':
            self.msg_lin = nn.Linear(in_channels, out_channels)
        elif msg_type == 'mlp':
            self.msg_mlp = nn.Sequential(
                nn.Linear(2 * in_channels, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels))

        if update_type == 'mlp':
            self.update_mlp = nn.Sequential(
                nn.Linear(in_channels + out_channels, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels))
        elif update_type == 'residual':
            self.res_lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_attr=None):
        src, dst = edge_index[0], edge_index[1]
        num_nodes = x.shape[0]

        # Messages
        if self.msg_type == 'linear':
            messages = self.msg_lin(x[src])
        elif self.msg_type == 'mlp':
            messages = self.msg_mlp(torch.cat([x[src], x[dst]], dim=-1))

        # Aggregation
        out = torch.zeros(num_nodes, messages.shape[1], device=x.device)
        if self.aggr == 'sum':
            out.scatter_add_(0, dst.unsqueeze(1).expand_as(messages), messages)
        elif self.aggr == 'mean':
            out.scatter_add_(0, dst.unsqueeze(1).expand_as(messages), messages)
            deg = torch.zeros(num_nodes, device=x.device)
            deg.scatter_add_(0, dst, torch.ones(dst.shape[0], device=x.device))
            out = out / deg.clamp(min=1).unsqueeze(1)
        elif self.aggr == 'max':
            out.fill_(float('-inf'))
            out.scatter_reduce_(0, dst.unsqueeze(1).expand_as(messages),
                                messages, reduce='amax')
            out[out == float('-inf')] = 0

        # Update
        if self.update_type == 'replace':
            return out
        elif self.update_type == 'mlp':
            return self.update_mlp(torch.cat([x, out], dim=-1))
        elif self.update_type == 'residual':
            return self.res_lin(x) + out


def demo_message_passing_variants():
    """Demonstrate different message passing configurations."""
    print("=" * 60)
    print("Message Passing Variants")
    print("=" * 60)
    torch.manual_seed(42)

    edge_index = torch.tensor([
        [0, 1, 0, 2, 1, 2, 1, 3, 2, 4, 3, 4],
        [1, 0, 2, 0, 2, 1, 3, 1, 4, 2, 4, 3]], dtype=torch.long)
    x = torch.randn(5, 8)

    configs = [
        {'msg_type': 'linear', 'aggr': 'sum', 'update_type': 'replace'},
        {'msg_type': 'linear', 'aggr': 'mean', 'update_type': 'replace'},
        {'msg_type': 'linear', 'aggr': 'max', 'update_type': 'replace'},
        {'msg_type': 'mlp', 'aggr': 'mean', 'update_type': 'mlp'},
        {'msg_type': 'linear', 'aggr': 'mean', 'update_type': 'residual'},
    ]
    for cfg in configs:
        layer = MessagePassingLayer(8, 16, **cfg)
        out = layer(x, edge_index)
        print(f"  MSG={cfg['msg_type']:8s} AGG={cfg['aggr']:5s} "
              f"UPD={cfg['update_type']:10s} -> {out.shape}")


# ============================================================
# 2. Receptive Field Analysis
# ============================================================

def analyze_receptive_field(edge_index, num_nodes, target_node, num_layers):
    """Analyze the receptive field of a node after K layers."""
    adj = {i: set() for i in range(num_nodes)}
    for i in range(edge_index.shape[1]):
        adj[edge_index[1, i].item()].add(edge_index[0, i].item())

    current = {target_node}
    all_reached = {target_node}
    for k in range(1, num_layers + 1):
        next_nodes = set()
        for node in current:
            next_nodes.update(adj[node])
        new = next_nodes - all_reached
        all_reached.update(next_nodes)
        current = new
        print(f"  Layer {k}: +{len(new)} nodes, "
              f"total={len(all_reached)}/{num_nodes} "
              f"({len(all_reached)/num_nodes:.0%})")
    return all_reached


def demo_receptive_field():
    """Demonstrate receptive field growth."""
    print("\n" + "=" * 60)
    print("Receptive Field Analysis")
    print("=" * 60)

    graphs = {
        'Path(20)': nx.path_graph(20),
        'Star(20)': nx.star_graph(19),
        'BA(20)': nx.barabasi_albert_graph(20, 2, seed=42),
    }
    for name, G in graphs.items():
        G = nx.convert_node_labels_to_integers(G)
        edges = list(G.edges())
        src = [e[0] for e in edges] + [e[1] for e in edges]
        dst = [e[1] for e in edges] + [e[0] for e in edges]
        ei = torch.tensor([src, dst], dtype=torch.long)
        print(f"\n{name}:")
        analyze_receptive_field(ei, G.number_of_nodes(), 0, 4)


# ============================================================
# 3. Multi-Layer Message Passing Network
# ============================================================

class MessagePassingNetwork(nn.Module):
    """Multi-layer message passing network."""

    def __init__(self, in_ch, hidden_ch, out_ch, num_layers=3,
                 aggr='mean', dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = dropout

        dims = [in_ch] + [hidden_ch] * (num_layers - 1) + [out_ch]
        for i in range(num_layers):
            self.layers.append(MessagePassingLayer(dims[i], dims[i + 1], aggr=aggr))
            if i < num_layers - 1:
                self.norms.append(nn.LayerNorm(dims[i + 1]))

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, edge_index)
            x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.layers[-1](x, edge_index)


# ============================================================
# 4. Visualize Information Flow
# ============================================================

def visualize_information_flow():
    """Track how information propagates through layers."""
    print("\n" + "=" * 60)
    print("Information Flow Tracking")
    print("=" * 60)

    torch.manual_seed(42)

    # Create a path graph: 0-1-2-3-4
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4],
        [1, 0, 2, 1, 3, 2, 4, 3]], dtype=torch.long)
    num_nodes = 5

    # One-hot features: node 0 has a unique signal
    x = torch.zeros(num_nodes, num_nodes)
    x[0, 0] = 1.0  # Only node 0 has a signal

    print("Initial features (node 0 has signal):")
    print(f"  {x[:, 0].tolist()}")

    # Track propagation with mean aggregation
    for layer_idx in range(4):
        # Simple mean message passing (no learned weights)
        src, dst = edge_index[0], edge_index[1]
        messages = x[src]
        out = torch.zeros_like(x)
        out.scatter_add_(0, dst.unsqueeze(1).expand_as(messages), messages)
        deg = torch.zeros(num_nodes)
        deg.scatter_add_(0, dst, torch.ones(dst.shape[0]))
        out = out / deg.clamp(min=1).unsqueeze(1)

        # Add self-connection (like GCN with self-loops)
        x = 0.5 * x + 0.5 * out

        print(f"\nAfter layer {layer_idx + 1} (signal at node 0):")
        print(f"  {x[:, 0].round(decimals=4).tolist()}")


# ============================================================
# 5. Over-Smoothing Demonstration
# ============================================================

def demo_over_smoothing():
    """Show how node representations converge with many layers."""
    print("\n" + "=" * 60)
    print("Over-Smoothing Demonstration")
    print("=" * 60)

    torch.manual_seed(42)
    G = nx.karate_club_graph()
    edges = list(G.edges())
    src = [e[0] for e in edges] + [e[1] for e in edges]
    dst = [e[1] for e in edges] + [e[0] for e in edges]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    num_nodes = G.number_of_nodes()

    x = torch.randn(num_nodes, 16)

    print(f"Tracking representation diversity (std across nodes):")
    for n_layers in [1, 2, 4, 8, 16, 32]:
        model = MessagePassingNetwork(16, 16, 16, num_layers=n_layers, aggr='mean')
        model.eval()
        with torch.no_grad():
            out = model(x, edge_index)
        # Measure how similar node representations are
        mean_repr = out.mean(dim=0)
        distances = torch.norm(out - mean_repr, dim=1)
        print(f"  {n_layers:2d} layers: mean_dist={distances.mean():.4f}, "
              f"std_dist={distances.std():.4f}")


# ============================================================
# 6. Financial Network Message Passing
# ============================================================

def demo_financial_message_passing():
    """Message passing on a financial correlation network."""
    print("\n" + "=" * 60)
    print("Financial Network Message Passing")
    print("=" * 60)

    torch.manual_seed(42)
    np.random.seed(42)

    # Create financial graph
    n_assets = 6
    names = ["AAPL", "MSFT", "GOOGL", "JPM", "GS", "BAC"]

    # Features: [return, volatility, momentum]
    features = torch.tensor([
        [0.15, 0.20, 0.8],   # AAPL
        [0.12, 0.18, 0.7],   # MSFT
        [0.10, 0.22, 0.6],   # GOOGL
        [0.08, 0.15, 0.3],   # JPM
        [0.06, 0.25, 0.2],   # GS
        [0.09, 0.20, 0.4],   # BAC
    ], dtype=torch.float)

    # Correlation-based edges (tech cluster, finance cluster, cross)
    edge_index = torch.tensor([
        [0, 1, 0, 2, 1, 2, 3, 4, 3, 5, 4, 5, 0, 3,
         1, 0, 2, 0, 2, 1, 4, 3, 5, 3, 5, 4, 3, 0],
        [1, 0, 2, 0, 2, 1, 4, 3, 5, 3, 5, 4, 3, 0,
         0, 1, 0, 2, 1, 2, 3, 4, 3, 5, 4, 5, 0, 3]], dtype=torch.long)

    print("Input features:")
    for i, name in enumerate(names):
        print(f"  {name}: return={features[i, 0]:.2f}, "
              f"vol={features[i, 1]:.2f}, mom={features[i, 2]:.2f}")

    # Apply message passing
    layer = MessagePassingLayer(3, 8, aggr='mean', msg_type='linear',
                                 update_type='replace')
    layer.eval()
    with torch.no_grad():
        out = layer(features, edge_index)

    print(f"\nAfter 1 layer of message passing (output dim=8):")
    for i, name in enumerate(names):
        print(f"  {name}: {out[i, :4].tolist()}")

    # Check if similar assets get similar representations
    print(f"\nPairwise distances after message passing:")
    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            dist = torch.norm(out[i] - out[j]).item()
            print(f"  {names[i]}-{names[j]}: {dist:.4f}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    demo_message_passing_variants()
    demo_receptive_field()
    visualize_information_flow()
    demo_over_smoothing()
    demo_financial_message_passing()
