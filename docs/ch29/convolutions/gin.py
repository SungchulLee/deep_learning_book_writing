"""
Chapter 29.3.7: Graph Isomorphism Network (GIN)
Maximally expressive message passing GNN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx


class GINConvManual(nn.Module):
    """GIN convolution: MLP((1 + eps) * h_v + sum(h_u))."""

    def __init__(self, in_ch, out_ch, eps=0.0, train_eps=True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_ch, out_ch),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Linear(out_ch, out_ch))
        if train_eps:
            self.eps = nn.Parameter(torch.tensor(eps))
        else:
            self.register_buffer('eps', torch.tensor(eps))

    def forward(self, x, edge_index):
        src, dst = edge_index[0], edge_index[1]
        n = x.shape[0]
        agg = torch.zeros(n, x.shape[1], device=x.device)
        agg.scatter_add_(0, dst.unsqueeze(1).expand(-1, x.shape[1]), x[src])
        out = (1 + self.eps) * x + agg
        return self.mlp(out)


class GIN(nn.Module):
    """GIN for graph classification with JK readout."""

    def __init__(self, in_ch, hidden_ch, out_ch, num_layers=5, dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(GINConvManual(in_ch, hidden_ch))
        self.bns.append(nn.BatchNorm1d(hidden_ch))
        for _ in range(num_layers - 1):
            self.convs.append(GINConvManual(hidden_ch, hidden_ch))
            self.bns.append(nn.BatchNorm1d(hidden_ch))
        # JK readout: concatenate all layers
        self.classifier = nn.Linear(hidden_ch * num_layers, out_ch)
        self.dropout = dropout

    def forward(self, x, edge_index, batch=None):
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        num_graphs = batch.max().item() + 1

        layer_readouts = []
        h = x
        for conv, bn in zip(self.convs, self.bns):
            h = conv(h, edge_index)
            h = bn(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            # Graph-level sum pooling
            pooled = torch.zeros(num_graphs, h.shape[1], device=h.device)
            pooled.scatter_add_(0, batch.unsqueeze(1).expand_as(h), h)
            layer_readouts.append(pooled)

        # Concatenate all layer readouts
        graph_repr = torch.cat(layer_readouts, dim=-1)
        return self.classifier(graph_repr)


def demo_gin_expressiveness():
    """Show GIN's superior expressiveness vs GCN/mean aggregation."""
    print("=" * 60)
    print("GIN Expressiveness: Multiset Distinguishing")
    print("=" * 60)

    # Case: Two star graphs with different leaf features
    # Star 1: center + 3 leaves with feature [1]
    # Star 2: center + 1 leaf with feature [1]
    # Mean aggregation: both centers get [1] -> indistinguishable
    # Sum aggregation: center1 gets [3], center2 gets [1] -> distinguishable

    x1 = torch.tensor([[0.0], [1.0], [1.0], [1.0]])
    ei1 = torch.tensor([[1,2,3,0,0,0],[0,0,0,1,2,3]], dtype=torch.long)

    x2 = torch.tensor([[0.0], [1.0]])
    ei2 = torch.tensor([[1,0],[0,1]], dtype=torch.long)

    # Sum aggregation
    sum1 = torch.zeros(4, 1)
    sum1.scatter_add_(0, ei1[1].unsqueeze(1), x1[ei1[0]])
    sum2 = torch.zeros(2, 1)
    sum2.scatter_add_(0, ei2[1].unsqueeze(1), x2[ei2[0]])

    # Mean aggregation
    mean1 = sum1.clone()
    deg1 = torch.zeros(4); deg1.scatter_add_(0, ei1[1], torch.ones(6))
    mean1 = mean1 / deg1.clamp(min=1).unsqueeze(1)
    mean2 = sum2.clone()
    deg2 = torch.zeros(2); deg2.scatter_add_(0, ei2[1], torch.ones(2))
    mean2 = mean2 / deg2.clamp(min=1).unsqueeze(1)

    print(f"Graph 1 center (3 leaves): sum={sum1[0].item():.1f}, mean={mean1[0].item():.3f}")
    print(f"Graph 2 center (1 leaf):   sum={sum2[0].item():.1f}, mean={mean2[0].item():.3f}")
    print(f"Sum distinguishes: {sum1[0].item() != sum2[0].item()}")
    print(f"Mean distinguishes: {mean1[0].item() != mean2[0].item()}")


def create_synthetic_graph_dataset(n_graphs=300):
    """Create synthetic graph classification dataset."""
    graphs = []
    for i in range(n_graphs):
        label = i % 3
        if label == 0:
            G = nx.cycle_graph(np.random.randint(5, 10))
        elif label == 1:
            G = nx.star_graph(np.random.randint(4, 8))
        else:
            n = np.random.randint(5, 10)
            G = nx.path_graph(n)
            for _ in range(n // 2):
                u, v = np.random.choice(n, 2, replace=False)
                G.add_edge(u, v)

        n = G.number_of_nodes()
        x = torch.ones(n, 1)  # Simple features
        edges = list(G.edges())
        if edges:
            src = [e[0] for e in edges] + [e[1] for e in edges]
            dst = [e[1] for e in edges] + [e[0] for e in edges]
        else:
            src, dst = [0], [0]
        ei = torch.tensor([src, dst], dtype=torch.long)
        graphs.append({'x': x, 'edge_index': ei, 'y': label, 'n': n})
    return graphs


def demo_gin_graph_classification():
    """GIN for graph classification."""
    print("\n" + "=" * 60)
    print("GIN: Graph Classification")
    print("=" * 60)

    torch.manual_seed(42)
    np.random.seed(42)
    graphs = create_synthetic_graph_dataset(300)
    train_g = graphs[:240]
    test_g = graphs[240:]

    model = GIN(1, 16, 3, num_layers=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for epoch in range(50):
        np.random.shuffle(train_g)
        total_loss, correct = 0, 0
        for g in train_g:
            optimizer.zero_grad()
            out = model(g['x'], g['edge_index'])
            loss = F.cross_entropy(out, torch.tensor([g['y']]))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (out.argmax(1).item() == g['y'])

        if (epoch + 1) % 10 == 0:
            model.eval()
            test_correct = 0
            with torch.no_grad():
                for g in test_g:
                    pred = model(g['x'], g['edge_index']).argmax(1).item()
                    test_correct += (pred == g['y'])
            print(f"Epoch {epoch+1}: Train Acc={correct/len(train_g):.3f}, "
                  f"Test Acc={test_correct/len(test_g):.3f}")
            model.train()


if __name__ == "__main__":
    demo_gin_expressiveness()
    demo_gin_graph_classification()
