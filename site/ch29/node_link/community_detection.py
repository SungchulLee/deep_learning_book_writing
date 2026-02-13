"""Chapter 29.6.4: Community Detection with GNNs."""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, networkx as nx
from sklearn.cluster import KMeans

class GNNCommunity(nn.Module):
    """GNN encoder for community detection."""
    def __init__(self, in_ch, hidden_ch, embed_ch):
        super().__init__()
        self.lin1 = nn.Linear(in_ch, hidden_ch)
        self.lin2 = nn.Linear(hidden_ch, embed_ch)
    def forward(self, x, ei):
        n = x.shape[0]; src, dst = ei[0], ei[1]
        loop = torch.arange(n, device=x.device)
        sa = torch.cat([src, loop]); da = torch.cat([dst, loop])
        deg = torch.zeros(n, device=x.device)
        deg.scatter_add_(0, da, torch.ones(da.shape[0], device=x.device))
        norm = (deg[sa]*deg[da]).pow(-0.5); norm[norm==float('inf')]=0
        for lin in [self.lin1, self.lin2]:
            h = lin(x); msg = h[sa]*norm.unsqueeze(1)
            out = torch.zeros(n, h.shape[1], device=x.device)
            out.scatter_add_(0, da.unsqueeze(1).expand_as(msg), msg)
            x = F.relu(out)
        return x

def demo():
    print("=" * 60); print("Community Detection"); print("=" * 60)
    torch.manual_seed(42); np.random.seed(42)
    sizes = [25, 25, 25]; probs = [[0.3,0.02,0.02],[0.02,0.3,0.02],[0.02,0.02,0.3]]
    G = nx.stochastic_block_model(sizes, probs, seed=42)
    n = G.number_of_nodes(); true_labels = [0]*25+[1]*25+[2]*25
    edges = list(G.edges())
    src = [e[0] for e in edges]+[e[1] for e in edges]
    dst = [e[1] for e in edges]+[e[0] for e in edges]
    ei = torch.tensor([src, dst], dtype=torch.long)
    x = torch.eye(n)

    # Train with link prediction loss
    model = GNNCommunity(n, 32, 16)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    for epoch in range(100):
        opt.zero_grad(); z = model(x, ei)
        pos_ei = torch.tensor([[e[0] for e in edges[:50]],[e[1] for e in edges[:50]]], dtype=torch.long)
        pos_s = (z[pos_ei[0]] * z[pos_ei[1]]).sum(dim=-1)
        neg_ei = torch.stack([torch.randint(0,n,(50,)), torch.randint(0,n,(50,))])
        neg_s = (z[neg_ei[0]] * z[neg_ei[1]]).sum(dim=-1)
        loss = -F.logsigmoid(pos_s).mean() - F.logsigmoid(-neg_s).mean()
        loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        z = model(x, ei).numpy()
    km = KMeans(n_clusters=3, random_state=42, n_init=10).fit(z)
    pred = km.labels_
    # NMI-like accuracy (best permutation)
    from itertools import permutations
    best_acc = 0
    for perm in permutations(range(3)):
        mapped = [perm[p] for p in pred]
        acc = sum(1 for a,b in zip(mapped, true_labels) if a==b) / n
        best_acc = max(best_acc, acc)
    print(f"  GNN + KMeans accuracy: {best_acc:.4f}")

    # Compare with spectral clustering
    A = nx.adjacency_matrix(G).toarray().astype(float)
    D = np.diag(A.sum(1)); L = D - A
    _, U = np.linalg.eigh(L)
    km2 = KMeans(n_clusters=3, random_state=42, n_init=10).fit(U[:, 1:4])
    best_acc2 = 0
    for perm in permutations(range(3)):
        mapped = [perm[p] for p in km2.labels_]
        acc = sum(1 for a,b in zip(mapped, true_labels) if a==b) / n
        best_acc2 = max(best_acc2, acc)
    print(f"  Spectral clustering accuracy: {best_acc2:.4f}")

if __name__ == "__main__":
    demo()
