"""Chapter 29.7.3: Social Network Analysis with GNNs."""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, networkx as nx

def demo():
    print("=" * 60); print("Social Network: Bot Detection"); print("=" * 60)
    torch.manual_seed(42); np.random.seed(42)
    # Simulate social network: 80 real users + 20 bots
    G = nx.stochastic_block_model([80, 20], [[0.05, 0.01],[0.01, 0.15]], seed=42)
    n = G.number_of_nodes()
    # Features: [activity, follower_ratio, posting_freq, account_age]
    x = torch.randn(n, 4)
    x[80:, 0] += 2.0  # Bots: higher activity
    x[80:, 1] -= 1.0  # Bots: lower follower ratio
    y = torch.tensor([0]*80 + [1]*20)
    edges = list(G.edges())
    src = [e[0] for e in edges]+[e[1] for e in edges]
    dst = [e[1] for e in edges]+[e[0] for e in edges]
    ei = torch.tensor([src, dst], dtype=torch.long)
    # Simple GCN
    lin1 = nn.Linear(4, 16); lin2 = nn.Linear(16, 2)
    opt = torch.optim.Adam(list(lin1.parameters())+list(lin2.parameters()), lr=0.01)
    tm = torch.zeros(n, dtype=torch.bool); tm[::3] = True
    for epoch in range(200):
        # GCN forward
        sa = torch.cat([ei[0], torch.arange(n)]); da = torch.cat([ei[1], torch.arange(n)])
        deg = torch.zeros(n); deg.scatter_add_(0, da, torch.ones(da.shape[0]))
        norm = (deg[sa]*deg[da]).pow(-0.5); norm[norm==float('inf')]=0
        h = lin1(x); msg = h[sa]*norm.unsqueeze(1)
        out = torch.zeros(n, 16); out.scatter_add_(0, da.unsqueeze(1).expand_as(msg), msg)
        h2 = lin2(F.relu(out))
        opt.zero_grad(); F.cross_entropy(h2[tm], y[tm]).backward(); opt.step()
    pred = h2.argmax(1).detach()
    print(f"  Overall acc: {(pred==y).float().mean():.4f}")
    print(f"  Bot recall: {(pred[80:]==1).float().mean():.4f}")

if __name__ == "__main__":
    demo()
