"""Chapter 29.6.2: Link Prediction with GNNs."""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, networkx as nx

class LinkPredictor(nn.Module):
    def __init__(self, in_ch, hidden_ch):
        super().__init__()
        self.lin1 = nn.Linear(in_ch, hidden_ch)
        self.lin2 = nn.Linear(hidden_ch, hidden_ch)
    def encode(self, x, ei):
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
    def decode(self, z, edge_index):
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

def demo():
    print("=" * 60); print("Link Prediction"); print("=" * 60)
    torch.manual_seed(42); np.random.seed(42)
    G = nx.barabasi_albert_graph(100, 3, seed=42)
    edges = list(G.edges()); np.random.shuffle(edges)
    split = int(0.8 * len(edges))
    train_e, test_e = edges[:split], edges[split:]
    n = G.number_of_nodes()
    src = [e[0] for e in train_e]+[e[1] for e in train_e]
    dst = [e[1] for e in train_e]+[e[0] for e in train_e]
    train_ei = torch.tensor([src, dst], dtype=torch.long)
    x = torch.eye(n)
    pos_train = torch.tensor([[e[0] for e in train_e],[e[1] for e in train_e]], dtype=torch.long)
    pos_test = torch.tensor([[e[0] for e in test_e],[e[1] for e in test_e]], dtype=torch.long)
    model = LinkPredictor(n, 32); opt = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    for epoch in range(100):
        opt.zero_grad(); z = model.encode(x, train_ei)
        pos_s = model.decode(z, pos_train)
        neg_ei = torch.stack([torch.randint(0,n,(pos_train.shape[1],)), torch.randint(0,n,(pos_train.shape[1],))])
        neg_s = model.decode(z, neg_ei)
        labels = torch.cat([torch.ones_like(pos_s), torch.zeros_like(neg_s)])
        loss = F.binary_cross_entropy_with_logits(torch.cat([pos_s, neg_s]), labels)
        loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        z = model.encode(x, train_ei)
        ps = torch.sigmoid(model.decode(z, pos_test))
        neg_test = torch.stack([torch.randint(0,n,(pos_test.shape[1],)), torch.randint(0,n,(pos_test.shape[1],))])
        ns = torch.sigmoid(model.decode(z, neg_test))
        all_s = torch.cat([ps, ns]); all_l = torch.cat([torch.ones_like(ps), torch.zeros_like(ns)])
        preds = (all_s > 0.5).float()
        acc = (preds == all_l).float().mean()
        print(f"  Test accuracy: {acc:.4f}")
        print(f"  Pos scores mean: {ps.mean():.4f}, Neg scores mean: {ns.mean():.4f}")

if __name__ == "__main__":
    demo()
