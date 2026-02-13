"""Chapter 29.7.1: Molecular Property Prediction with GNNs."""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np

class MoleculeGNN(nn.Module):
    def __init__(self, atom_features=10, hidden_ch=64, out_ch=1, num_layers=3):
        super().__init__()
        self.atom_enc = nn.Linear(atom_features, hidden_ch)
        self.convs = nn.ModuleList([nn.Linear(hidden_ch, hidden_ch) for _ in range(num_layers)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_ch) for _ in range(num_layers)])
        self.predictor = nn.Sequential(nn.Linear(hidden_ch, hidden_ch), nn.ReLU(), nn.Linear(hidden_ch, out_ch))

    def forward(self, x, edge_index, batch=None):
        n = x.shape[0]
        if batch is None: batch = torch.zeros(n, dtype=torch.long, device=x.device)
        x = self.atom_enc(x); src, dst = edge_index[0], edge_index[1]
        loop = torch.arange(n, device=x.device)
        sa = torch.cat([src, loop]); da = torch.cat([dst, loop])
        deg = torch.zeros(n, device=x.device)
        deg.scatter_add_(0, da, torch.ones(da.shape[0], device=x.device))
        norm = (deg[sa]*deg[da]).pow(-0.5); norm[norm==float('inf')]=0
        for lin, bn in zip(self.convs, self.bns):
            h = lin(x); msg = h[sa]*norm.unsqueeze(1)
            out = torch.zeros(n, h.shape[1], device=x.device)
            out.scatter_add_(0, da.unsqueeze(1).expand_as(msg), msg)
            x = F.relu(bn(out))
        ng = batch.max().item()+1
        pool = torch.zeros(ng, x.shape[1], device=x.device)
        pool.scatter_add_(0, batch.unsqueeze(1).expand_as(x), x)
        return self.predictor(pool).squeeze(-1)

def create_molecule(n_atoms, label_fn):
    x = torch.randn(n_atoms, 10)
    edges = [(i,i+1) for i in range(n_atoms-1)]
    for _ in range(n_atoms//3):
        u,v = np.random.choice(n_atoms, 2, replace=False); edges.append((u,v))
    src = [e[0] for e in edges]+[e[1] for e in edges]
    dst = [e[1] for e in edges]+[e[0] for e in edges]
    ei = torch.tensor([src,dst], dtype=torch.long)
    y = label_fn(x, len(edges))
    return x, ei, y

def demo():
    print("=" * 60); print("Molecular Property Prediction"); print("=" * 60)
    torch.manual_seed(42); np.random.seed(42)
    label_fn = lambda x, ne: x[:, 0].mean().item() + ne * 0.01 + np.random.randn()*0.05
    mols = [create_molecule(np.random.randint(5,15), label_fn) for _ in range(200)]
    model = MoleculeGNN(); opt = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(50):
        total_loss = 0
        for x, ei, y in mols[:160]:
            opt.zero_grad(); pred = model(x, ei)
            loss = F.mse_loss(pred, torch.tensor([y])); loss.backward(); opt.step()
            total_loss += loss.item()
        if (epoch+1) % 10 == 0:
            model.eval()
            tl = sum(F.mse_loss(model(x,ei), torch.tensor([y])).item() for x,ei,y in mols[160:])
            print(f"  Epoch {epoch+1}: Train={total_loss/160:.4f}, Test={tl/40:.4f}")
            model.train()

if __name__ == "__main__":
    demo()
