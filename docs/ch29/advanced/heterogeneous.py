"""
Chapter 29.4.5: Heterogeneous Graph Neural Networks
RGCN and type-specific message passing.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np

class RGCNConv(nn.Module):
    """Relational GCN layer with per-relation weights."""
    def __init__(self, in_ch, out_ch, num_relations):
        super().__init__()
        self.weights = nn.ParameterList([nn.Parameter(torch.randn(in_ch, out_ch)*0.01) for _ in range(num_relations)])
        self.self_weight = nn.Parameter(torch.randn(in_ch, out_ch)*0.01)
        self.bias = nn.Parameter(torch.zeros(out_ch))

    def forward(self, x, edge_indices_per_relation):
        n = x.shape[0]; out = x @ self.self_weight
        for r, (ei, W) in enumerate(zip(edge_indices_per_relation, self.weights)):
            if ei.shape[1] == 0: continue
            src, dst = ei[0], ei[1]
            msg = (x[src] @ W)
            agg = torch.zeros(n, W.shape[1], device=x.device)
            agg.scatter_add_(0, dst.unsqueeze(1).expand_as(msg), msg)
            deg = torch.zeros(n, device=x.device)
            deg.scatter_add_(0, dst, torch.ones(dst.shape[0], device=x.device))
            out = out + agg / deg.clamp(min=1).unsqueeze(1)
        return out + self.bias

class HeteroGNN(nn.Module):
    def __init__(self, in_ch, hidden_ch, out_ch, num_relations, num_layers=2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(RGCNConv(in_ch, hidden_ch, num_relations))
        for _ in range(num_layers - 2):
            self.convs.append(RGCNConv(hidden_ch, hidden_ch, num_relations))
        self.convs.append(RGCNConv(hidden_ch, out_ch, num_relations))

    def forward(self, x, edge_indices_per_relation):
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_indices_per_relation))
        return self.convs[-1](x, edge_indices_per_relation)

def demo_heterogeneous():
    print("=" * 60); print("Heterogeneous GNN (RGCN)"); print("=" * 60)
    torch.manual_seed(42)
    n = 15  # 5 companies, 5 banks, 5 investors
    x = torch.randn(n, 8)
    # Relations: 0=company-company, 1=bank-lends-company, 2=investor-holds-company
    edges_r0 = torch.tensor([[0,1,2],[1,2,3]], dtype=torch.long)
    edges_r1 = torch.tensor([[5,6,7,8],[0,1,2,3]], dtype=torch.long)
    edges_r2 = torch.tensor([[10,11,12],[0,1,2]], dtype=torch.long)
    edge_list = [edges_r0, edges_r1, edges_r2]
    y = torch.randint(0, 3, (n,))
    model = HeteroGNN(8, 16, 3, num_relations=3)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    for _ in range(100):
        opt.zero_grad()
        loss = F.cross_entropy(model(x, edge_list), y)
        loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(x, edge_list).argmax(1)
        acc = (pred == y).float().mean()
    print(f"  Training Acc: {acc:.4f}")

if __name__ == "__main__":
    demo_heterogeneous()
