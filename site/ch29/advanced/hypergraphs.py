"""
Chapter 29.4.7: Hypergraph Neural Networks
Two-stage message passing on hypergraphs.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np

class HypergraphConv(nn.Module):
    """Hypergraph convolution via two-stage message passing."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.node_to_edge = nn.Linear(in_ch, out_ch)
        self.edge_to_node = nn.Linear(out_ch, out_ch)

    def forward(self, x, incidence):
        """x: [n, d], incidence: [n, m] binary."""
        n, m = incidence.shape
        # Stage 1: Node -> Hyperedge (mean aggregation)
        node_deg = incidence.sum(dim=1, keepdim=True).clamp(min=1)
        edge_deg = incidence.sum(dim=0, keepdim=True).clamp(min=1)
        h_nodes = self.node_to_edge(x)
        h_edges = (incidence.T @ h_nodes) / edge_deg.T  # [m, d]
        # Stage 2: Hyperedge -> Node
        h_agg = (incidence @ h_edges) / node_deg  # [n, d]
        return F.relu(self.edge_to_node(h_agg))

class HypergraphNN(nn.Module):
    def __init__(self, in_ch, hidden_ch, out_ch, num_layers=2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(HypergraphConv(in_ch, hidden_ch))
        for _ in range(num_layers-2):
            self.convs.append(HypergraphConv(hidden_ch, hidden_ch))
        self.convs.append(HypergraphConv(hidden_ch, out_ch))

    def forward(self, x, incidence):
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, incidence))
        return self.convs[-1](x, incidence)

def demo_hypergraph():
    print("=" * 60); print("Hypergraph Neural Network"); print("=" * 60)
    torch.manual_seed(42)
    n_nodes, n_edges = 10, 5
    # Random incidence: each hyperedge contains 2-4 nodes
    incidence = torch.zeros(n_nodes, n_edges)
    for e in range(n_edges):
        size = np.random.randint(2, 5)
        nodes = np.random.choice(n_nodes, size, replace=False)
        incidence[nodes, e] = 1.0

    x = torch.randn(n_nodes, 8)
    y = torch.randint(0, 3, (n_nodes,))

    print(f"Nodes: {n_nodes}, Hyperedges: {n_edges}")
    print(f"Incidence matrix:\n{incidence.int()}")

    model = HypergraphNN(8, 16, 3)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    for epoch in range(100):
        opt.zero_grad()
        loss = F.cross_entropy(model(x, incidence), y)
        loss.backward(); opt.step()
        if (epoch+1) % 25 == 0:
            acc = (model(x, incidence).argmax(1) == y).float().mean()
            print(f"  Epoch {epoch+1}: Loss={loss.item():.4f}, Acc={acc:.4f}")

    # Financial example: portfolios as hyperedges
    print("\nFinancial Hypergraph: Portfolios")
    n_stocks, n_funds = 8, 3
    holdings = torch.zeros(n_stocks, n_funds)
    holdings[:4, 0] = 1  # Fund 0 holds stocks 0-3
    holdings[2:6, 1] = 1  # Fund 1 holds stocks 2-5
    holdings[5:, 2] = 1   # Fund 2 holds stocks 5-7
    stock_features = torch.randn(n_stocks, 4)
    conv = HypergraphConv(4, 4)
    out = conv(stock_features, holdings)
    print(f"  Stock features: {stock_features.shape} -> Enriched: {out.shape}")

if __name__ == "__main__":
    demo_hypergraph()
