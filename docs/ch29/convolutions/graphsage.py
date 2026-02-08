"""
Chapter 29.3.5: GraphSAGE
Inductive graph learning with neighbor sampling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx


class SAGEConvManual(nn.Module):
    """GraphSAGE convolution layer."""

    def __init__(self, in_ch, out_ch, aggr='mean', normalize=True):
        super().__init__()
        self.lin = nn.Linear(2 * in_ch, out_ch)
        self.aggr = aggr
        self.normalize = normalize
        if aggr == 'pool':
            self.pool_lin = nn.Linear(in_ch, in_ch)

    def forward(self, x, edge_index):
        src, dst = edge_index[0], edge_index[1]
        n = x.shape[0]

        if self.aggr == 'mean':
            neigh = torch.zeros(n, x.shape[1], device=x.device)
            neigh.scatter_add_(0, dst.unsqueeze(1).expand(-1, x.shape[1]), x[src])
            deg = torch.zeros(n, device=x.device)
            deg.scatter_add_(0, dst, torch.ones(dst.shape[0], device=x.device))
            neigh = neigh / deg.clamp(min=1).unsqueeze(1)
        elif self.aggr == 'max':
            transformed = F.relu(self.pool_lin(x[src]))
            neigh = torch.full((n, x.shape[1]), float('-inf'), device=x.device)
            neigh.scatter_reduce_(0, dst.unsqueeze(1).expand_as(transformed),
                                   transformed, reduce='amax')
            neigh[neigh == float('-inf')] = 0
        elif self.aggr == 'pool':
            transformed = F.relu(self.pool_lin(x[src]))
            neigh = torch.full((n, x.shape[1]), float('-inf'), device=x.device)
            neigh.scatter_reduce_(0, dst.unsqueeze(1).expand_as(transformed),
                                   transformed, reduce='amax')
            neigh[neigh == float('-inf')] = 0

        out = self.lin(torch.cat([x, neigh], dim=-1))
        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)
        return out


class GraphSAGE(nn.Module):
    """Multi-layer GraphSAGE."""

    def __init__(self, in_ch, hidden_ch, out_ch, num_layers=2,
                 aggr='mean', dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConvManual(in_ch, hidden_ch, aggr))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConvManual(hidden_ch, hidden_ch, aggr))
        self.convs.append(SAGEConvManual(hidden_ch, out_ch, aggr, normalize=False))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)


def neighbor_sampling(edge_index, num_nodes, target_nodes, num_samples=10):
    """Sample fixed number of neighbors per node."""
    adj = {i: [] for i in range(num_nodes)}
    for i in range(edge_index.shape[1]):
        adj[edge_index[1, i].item()].append(edge_index[0, i].item())

    sampled_src, sampled_dst = [], []
    for node in target_nodes:
        neighbors = adj[node]
        if len(neighbors) == 0:
            continue
        if len(neighbors) <= num_samples:
            selected = neighbors
        else:
            selected = list(np.random.choice(neighbors, num_samples, replace=False))
        for s in selected:
            sampled_src.append(s)
            sampled_dst.append(node)

    return torch.tensor([sampled_src, sampled_dst], dtype=torch.long)


def demo_graphsage():
    """GraphSAGE on Karate Club."""
    print("=" * 60)
    print("GraphSAGE: Karate Club")
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
    train_mask[[0,1,2,3,33,32,31,30]] = True

    for aggr in ['mean', 'pool']:
        torch.manual_seed(42)
        model = GraphSAGE(n, 16, 2, aggr=aggr)
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
        print(f"  Aggregator={aggr:5s}: Test Acc = {test_acc:.4f}")


def demo_neighbor_sampling():
    """Demonstrate neighbor sampling strategy."""
    print("\n" + "=" * 60)
    print("Neighbor Sampling")
    print("=" * 60)

    G = nx.barabasi_albert_graph(50, 3, seed=42)
    edges = list(G.edges())
    src = [e[0] for e in edges] + [e[1] for e in edges]
    dst = [e[1] for e in edges] + [e[0] for e in edges]
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    target = list(range(5))
    for num_samples in [3, 5, 10]:
        sampled = neighbor_sampling(edge_index, 50, target, num_samples)
        print(f"  samples_per_node={num_samples}: total_edges={sampled.shape[1]}")
        for t in target[:3]:
            count = (sampled[1] == t).sum().item()
            actual = G.degree(t)
            print(f"    Node {t}: sampled={count}, actual_degree={actual}")


if __name__ == "__main__":
    demo_graphsage()
    demo_neighbor_sampling()
