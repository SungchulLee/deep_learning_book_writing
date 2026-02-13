"""
Chapter 29.3.4: Graph Convolutional Network (GCN)
Kipf & Welling (2017) implementation from scratch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx


class GCNConvManual(nn.Module):
    """GCN convolution layer implemented from scratch."""

    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(self, x, edge_index, num_nodes=None):
        if num_nodes is None:
            num_nodes = x.shape[0]

        # Add self-loops
        loop = torch.arange(num_nodes, device=edge_index.device)
        loop = loop.unsqueeze(0).repeat(2, 1)
        ei = torch.cat([edge_index, loop], dim=1)

        # Compute degree
        src, dst = ei[0], ei[1]
        deg = torch.zeros(num_nodes, device=x.device)
        deg.scatter_add_(0, dst, torch.ones(dst.shape[0], device=x.device))

        # Symmetric normalization: D^{-1/2} A D^{-1/2}
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[src] * deg_inv_sqrt[dst]

        # Transform features
        x_transformed = x @ self.weight

        # Message passing with normalization
        messages = x_transformed[src] * norm.unsqueeze(1)
        out = torch.zeros(num_nodes, x_transformed.shape[1], device=x.device)
        out.scatter_add_(0, dst.unsqueeze(1).expand_as(messages), messages)

        if self.bias is not None:
            out = out + self.bias
        return out


class GCN(nn.Module):
    """Multi-layer GCN for node classification."""

    def __init__(self, in_ch, hidden_ch, out_ch, num_layers=2, dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConvManual(in_ch, hidden_ch))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConvManual(hidden_ch, hidden_ch))
        self.convs.append(GCNConvManual(hidden_ch, out_ch))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)


def demo_gcn_layer():
    """Demonstrate single GCN layer computation."""
    print("=" * 60)
    print("GCN Layer Computation")
    print("=" * 60)

    torch.manual_seed(42)
    # Simple 4-node graph
    edge_index = torch.tensor([[0,1,1,2,2,3,0,2],[1,0,2,1,3,2,2,0]], dtype=torch.long)
    x = torch.tensor([[1,0],[0,1],[1,1],[0,0]], dtype=torch.float)

    layer = GCNConvManual(2, 4)
    out = layer(x, edge_index)
    print(f"Input: {x.shape} -> Output: {out.shape}")
    print(f"Output:\n{out.detach()}")


def demo_gcn_karate():
    """GCN on Karate Club for node classification."""
    print("\n" + "=" * 60)
    print("GCN: Karate Club Node Classification")
    print("=" * 60)

    torch.manual_seed(42)
    G = nx.karate_club_graph()
    n = G.number_of_nodes()
    edges = list(G.edges())
    src = [e[0] for e in edges] + [e[1] for e in edges]
    dst = [e[1] for e in edges] + [e[0] for e in edges]
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    x = torch.eye(n)
    y = torch.tensor([0 if G.nodes[i].get('club','')=='Mr. Hi' else 1
                       for i in range(n)], dtype=torch.long)
    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[[0, 1, 2, 3, 33, 32, 31, 30]] = True

    model = GCN(n, 16, 2, num_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = F.cross_entropy(out[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        pred = model(x, edge_index).argmax(dim=1)
        acc = (pred == y).float().mean()
        test_acc = (pred[~train_mask] == y[~train_mask]).float().mean()
    print(f"Overall accuracy: {acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")


def demo_gcn_normalization_effect():
    """Show the effect of symmetric normalization."""
    print("\n" + "=" * 60)
    print("Normalization Effect")
    print("=" * 60)

    # Star graph: center node 0 has degree 9, others have degree 1
    G = nx.star_graph(9)
    A = nx.adjacency_matrix(G).toarray().astype(float)
    n = 10

    # Add self-loops
    A_tilde = A + np.eye(n)
    D_tilde = np.diag(A_tilde.sum(axis=1))

    # No normalization
    signal = np.zeros(n)
    signal[1] = 1.0  # Signal at a leaf

    aggregated_raw = A_tilde @ signal
    print(f"Raw aggregation (center node 0): {aggregated_raw[0]:.4f}")
    print(f"Raw aggregation (leaf node 1): {aggregated_raw[1]:.4f}")

    # Symmetric normalization
    D_inv_sqrt = np.diag(np.diag(D_tilde) ** -0.5)
    A_hat = D_inv_sqrt @ A_tilde @ D_inv_sqrt
    aggregated_norm = A_hat @ signal
    print(f"\nNormalized aggregation (center): {aggregated_norm[0]:.4f}")
    print(f"Normalized aggregation (leaf 1): {aggregated_norm[1]:.4f}")
    print("Normalization prevents the high-degree center from dominating")


def demo_gcn_financial():
    """GCN on a financial network for sector prediction."""
    print("\n" + "=" * 60)
    print("GCN: Financial Sector Prediction")
    print("=" * 60)

    torch.manual_seed(42)
    np.random.seed(42)
    n = 20
    n_sectors = 3
    sectors = np.random.randint(0, n_sectors, n)

    # Create features
    x = torch.randn(n, 8)
    for i in range(n):
        x[i, :3] += torch.tensor([sectors[i] * 0.5, 0, 0])

    # Correlation edges (same sector more likely)
    src, dst = [], []
    for i in range(n):
        for j in range(i+1, n):
            p = 0.6 if sectors[i] == sectors[j] else 0.15
            if np.random.random() < p:
                src.extend([i, j]); dst.extend([j, i])
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    y = torch.tensor(sectors, dtype=torch.long)

    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[:10] = True

    model = GCN(8, 16, n_sectors, num_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x, edge_index)[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        pred = model(x, edge_index).argmax(1)
        test_acc = (pred[~train_mask] == y[~train_mask]).float().mean()
    print(f"Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    demo_gcn_layer()
    demo_gcn_karate()
    demo_gcn_normalization_effect()
    demo_gcn_financial()
