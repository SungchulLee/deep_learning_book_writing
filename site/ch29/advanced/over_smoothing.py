"""
Chapter 29.4.2: Over-Smoothing - Measurement and mitigation.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, networkx as nx

def measure_smoothness(H, edge_index):
    """Compute MAD and Dirichlet energy."""
    n = H.shape[0]
    # MAD
    diffs = H.unsqueeze(0) - H.unsqueeze(1)
    mad = diffs.norm(dim=-1).mean().item()
    # Dirichlet energy
    src, dst = edge_index[0], edge_index[1]
    energy = ((H[src] - H[dst])**2).sum().item()
    return mad, energy

class SimpleGCN(nn.Module):
    def __init__(self, in_ch, hidden_ch, num_layers):
        super().__init__()
        self.input = nn.Linear(in_ch, hidden_ch)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(nn.Linear(hidden_ch, hidden_ch))
    def forward(self, x, edge_index):
        n = x.shape[0]; x = F.relu(self.input(x))
        src, dst = edge_index[0], edge_index[1]
        loop = torch.arange(n, device=x.device)
        src_a = torch.cat([src, loop]); dst_a = torch.cat([dst, loop])
        deg = torch.zeros(n, device=x.device)
        deg.scatter_add_(0, dst_a, torch.ones(dst_a.shape[0], device=x.device))
        norm = (deg[src_a] * deg[dst_a]).pow(-0.5)
        norm[norm==float('inf')] = 0
        for lin in self.convs:
            h = lin(x); msg = h[src_a] * norm.unsqueeze(1)
            out = torch.zeros(n, h.shape[1], device=x.device)
            out.scatter_add_(0, dst_a.unsqueeze(1).expand_as(msg), msg)
            x = F.relu(out)
        return x

def demo_over_smoothing():
    print("=" * 60); print("Over-Smoothing Analysis"); print("=" * 60)
    torch.manual_seed(42)
    G = nx.karate_club_graph(); n = G.number_of_nodes()
    edges = list(G.edges())
    src = [e[0] for e in edges]+[e[1] for e in edges]
    dst = [e[1] for e in edges]+[e[0] for e in edges]
    ei = torch.tensor([src, dst], dtype=torch.long)
    x = torch.eye(n)
    print(f"{'Layers':>8} {'MAD':>10} {'Dirichlet':>12}")
    for nl in [1, 2, 4, 8, 16, 32]:
        model = SimpleGCN(n, 16, nl); model.eval()
        with torch.no_grad():
            H = model(x, ei)
            mad, energy = measure_smoothness(H, ei)
        print(f"{nl:>8} {mad:>10.4f} {energy:>12.4f}")

if __name__ == "__main__":
    demo_over_smoothing()
