"""
Chapter 29.4.1: Deep GNNs - Residual connections, DropEdge, normalization.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, networkx as nx

class GCNLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.lin = nn.Linear(in_ch, out_ch)
    def forward(self, x, edge_index):
        n = x.shape[0]
        src, dst = edge_index[0], edge_index[1]
        loop = torch.arange(n, device=x.device)
        src_all = torch.cat([src, loop]); dst_all = torch.cat([dst, loop])
        deg = torch.zeros(n, device=x.device)
        deg.scatter_add_(0, dst_all, torch.ones(dst_all.shape[0], device=x.device))
        norm = (deg[src_all] * deg[dst_all]).pow(-0.5)
        norm[norm == float('inf')] = 0
        h = self.lin(x)
        msg = h[src_all] * norm.unsqueeze(1)
        out = torch.zeros(n, h.shape[1], device=x.device)
        out.scatter_add_(0, dst_all.unsqueeze(1).expand_as(msg), msg)
        return out

class ResGCN(nn.Module):
    def __init__(self, in_ch, hidden_ch, out_ch, num_layers=8, dropout=0.5):
        super().__init__()
        self.input_lin = nn.Linear(in_ch, hidden_ch)
        self.convs = nn.ModuleList([GCNLayer(hidden_ch, hidden_ch) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_ch) for _ in range(num_layers)])
        self.output_lin = nn.Linear(hidden_ch, out_ch)
        self.dropout = dropout
    def forward(self, x, edge_index):
        x = F.relu(self.input_lin(x))
        for conv, norm in zip(self.convs, self.norms):
            res = x
            x = conv(x, edge_index); x = norm(x); x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + res
        return self.output_lin(x)

def drop_edge(edge_index, p=0.2):
    mask = torch.rand(edge_index.shape[1]) > p
    return edge_index[:, mask]

def demo_deep_gcn():
    print("=" * 60); print("Deep GCN with Residual Connections"); print("=" * 60)
    torch.manual_seed(42)
    G = nx.karate_club_graph(); n = G.number_of_nodes()
    edges = list(G.edges())
    src = [e[0] for e in edges]+[e[1] for e in edges]
    dst = [e[1] for e in edges]+[e[0] for e in edges]
    ei = torch.tensor([src, dst], dtype=torch.long)
    x = torch.eye(n)
    y = torch.tensor([0 if G.nodes[i].get('club','')=='Mr. Hi' else 1 for i in range(n)])
    tm = torch.zeros(n, dtype=torch.bool); tm[::2] = True
    for nl in [2, 4, 8, 16]:
        torch.manual_seed(42)
        model = ResGCN(n, 16, 2, num_layers=nl); opt = torch.optim.Adam(model.parameters(), lr=0.01)
        model.train()
        for _ in range(200):
            opt.zero_grad(); F.cross_entropy(model(x, ei)[tm], y[tm]).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            acc = (model(x, ei).argmax(1)[~tm] == y[~tm]).float().mean()
        print(f"  {nl:2d} layers: Test Acc = {acc:.4f}")

if __name__ == "__main__":
    demo_deep_gcn()
