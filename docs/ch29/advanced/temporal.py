"""
Chapter 29.4.6: Temporal Graph Neural Networks
Snapshot-based and temporal message passing approaches.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np

class SnapshotGNN(nn.Module):
    """GNN + GRU for temporal graph sequences."""
    def __init__(self, in_ch, hidden_ch, out_ch):
        super().__init__()
        self.gcn = nn.Linear(in_ch, hidden_ch)
        self.gru = nn.GRUCell(hidden_ch, hidden_ch)
        self.classifier = nn.Linear(hidden_ch, out_ch)

    def gcn_forward(self, x, edge_index):
        n = x.shape[0]
        src, dst = edge_index[0], edge_index[1]
        loop = torch.arange(n, device=x.device)
        src_a = torch.cat([src, loop]); dst_a = torch.cat([dst, loop])
        h = self.gcn(x)
        out = torch.zeros(n, h.shape[1], device=x.device)
        out.scatter_add_(0, dst_a.unsqueeze(1).expand(-1, h.shape[1]), h[src_a])
        deg = torch.zeros(n, device=x.device)
        deg.scatter_add_(0, dst_a, torch.ones(dst_a.shape[0], device=x.device))
        return F.relu(out / deg.clamp(min=1).unsqueeze(1))

    def forward(self, snapshots):
        h = None
        for x, ei in snapshots:
            z = self.gcn_forward(x, ei)
            z_mean = z.mean(dim=0, keepdim=True)
            if h is None:
                h = torch.zeros_like(z_mean)
            h = self.gru(z_mean, h)
        return self.classifier(h)

class TemporalEncoding(nn.Module):
    """Time-aware positional encoding."""
    def __init__(self, dim):
        super().__init__()
        self.omega = nn.Parameter(torch.randn(dim))
        self.phi = nn.Parameter(torch.zeros(dim))
    def forward(self, t):
        return torch.cos(self.omega * t + self.phi)

def demo_temporal_gnn():
    print("=" * 60); print("Temporal GNN (Snapshot-based)"); print("=" * 60)
    torch.manual_seed(42); np.random.seed(42)
    n = 10; T = 20
    snapshots = []
    for t in range(T):
        x = torch.randn(n, 4) + t * 0.01
        edges_src, edges_dst = [], []
        for i in range(n):
            for j in range(i+1, n):
                if np.random.random() < 0.3:
                    edges_src.extend([i,j]); edges_dst.extend([j,i])
        if not edges_src: edges_src, edges_dst = [0,1], [1,0]
        ei = torch.tensor([edges_src, edges_dst], dtype=torch.long)
        snapshots.append((x, ei))

    model = SnapshotGNN(4, 16, 2)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    labels = torch.tensor([0] * 10 + [1] * 10)
    model.train()
    for epoch in range(50):
        total_loss = 0
        for i in range(0, T-5, 1):
            opt.zero_grad()
            out = model(snapshots[i:i+5])
            label = torch.tensor([i % 2])
            loss = F.cross_entropy(out, label)
            loss.backward(); opt.step()
            total_loss += loss.item()
        if (epoch+1) % 10 == 0:
            print(f"  Epoch {epoch+1}: Avg Loss = {total_loss/(T-5):.4f}")

    # Temporal encoding demo
    te = TemporalEncoding(8)
    times = torch.tensor([0.0, 1.0, 5.0, 10.0])
    for t in times:
        enc = te(t)
        print(f"  t={t:.0f}: encoding norm = {enc.norm():.4f}")

if __name__ == "__main__":
    demo_temporal_gnn()
