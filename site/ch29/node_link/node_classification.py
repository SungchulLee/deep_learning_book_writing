"""Chapter 29.6.1: Node Classification - Semi-supervised with GCN."""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, networkx as nx

class GCN(nn.Module):
    def __init__(self, in_ch, hidden_ch, out_ch, dropout=0.5):
        super().__init__()
        self.lin1 = nn.Linear(in_ch, hidden_ch)
        self.lin2 = nn.Linear(hidden_ch, out_ch)
        self.dropout = dropout
    def gcn_layer(self, x, ei):
        n = x.shape[0]; src, dst = ei[0], ei[1]
        loop = torch.arange(n, device=x.device)
        sa = torch.cat([src, loop]); da = torch.cat([dst, loop])
        deg = torch.zeros(n, device=x.device)
        deg.scatter_add_(0, da, torch.ones(da.shape[0], device=x.device))
        norm = (deg[sa]*deg[da]).pow(-0.5); norm[norm==float('inf')]=0
        msg = x[sa]*norm.unsqueeze(1)
        out = torch.zeros_like(x)
        out.scatter_add_(0, da.unsqueeze(1).expand_as(msg), msg)
        return out
    def forward(self, x, ei):
        x = self.gcn_layer(self.lin1(x), ei)
        x = F.relu(x); x = F.dropout(x, p=self.dropout, training=self.training)
        return self.gcn_layer(self.lin2(x), ei)

def demo():
    print("=" * 60); print("Node Classification (Semi-supervised)"); print("=" * 60)
    torch.manual_seed(42)
    # Stochastic block model
    sizes = [30, 30, 30]; probs = [[0.3,0.02,0.02],[0.02,0.3,0.02],[0.02,0.02,0.3]]
    G = nx.stochastic_block_model(sizes, probs, seed=42)
    n = G.number_of_nodes()
    edges = list(G.edges())
    src = [e[0] for e in edges]+[e[1] for e in edges]
    dst = [e[1] for e in edges]+[e[0] for e in edges]
    ei = torch.tensor([src, dst], dtype=torch.long)
    x = torch.randn(n, 16)
    y = torch.tensor([0]*30+[1]*30+[2]*30)
    # Add community signal to features
    for i in range(n): x[i, :3] += y[i].float() * 0.5
    # Semi-supervised: 5% labels
    train_mask = torch.zeros(n, dtype=torch.bool)
    for c in range(3): train_mask[c*30:c*30+3] = True
    model = GCN(16, 32, 3); opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    model.train()
    for epoch in range(200):
        opt.zero_grad()
        F.cross_entropy(model(x, ei)[train_mask], y[train_mask]).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(x, ei).argmax(1)
        print(f"  Train acc: {(pred[train_mask]==y[train_mask]).float().mean():.4f}")
        print(f"  Test acc:  {(pred[~train_mask]==y[~train_mask]).float().mean():.4f}")
        print(f"  Overall:   {(pred==y).float().mean():.4f}")

if __name__ == "__main__":
    demo()
