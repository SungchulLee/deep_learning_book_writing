"""Chapter 29.5.2: Graph Regression"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np

class GraphRegressor(nn.Module):
    def __init__(self, in_ch, hidden_ch, num_layers=3):
        super().__init__()
        self.convs = nn.ModuleList([nn.Linear(in_ch if i==0 else hidden_ch, hidden_ch) for i in range(num_layers)])
        self.predictor = nn.Sequential(nn.Linear(hidden_ch, hidden_ch), nn.ReLU(), nn.Linear(hidden_ch, 1))
    def forward(self, x, edge_index, batch=None):
        n = x.shape[0]
        if batch is None: batch = torch.zeros(n, dtype=torch.long, device=x.device)
        src, dst = edge_index[0], edge_index[1]
        loop = torch.arange(n, device=x.device)
        src_a = torch.cat([src, loop]); dst_a = torch.cat([dst, loop])
        deg = torch.zeros(n, device=x.device)
        deg.scatter_add_(0, dst_a, torch.ones(dst_a.shape[0], device=x.device))
        norm = (deg[src_a]*deg[dst_a]).pow(-0.5); norm[norm==float('inf')]=0
        for lin in self.convs:
            h = lin(x); msg = h[src_a]*norm.unsqueeze(1)
            out = torch.zeros(n, h.shape[1], device=x.device)
            out.scatter_add_(0, dst_a.unsqueeze(1).expand_as(msg), msg)
            x = F.relu(out)
        ng = batch.max().item()+1
        pool = torch.zeros(ng, x.shape[1], device=x.device)
        pool.scatter_add_(0, batch.unsqueeze(1).expand_as(x), x)
        cnt = torch.zeros(ng, device=x.device)
        cnt.scatter_add_(0, batch, torch.ones(n, device=x.device))
        pool = pool / cnt.clamp(min=1).unsqueeze(1)
        return self.predictor(pool).squeeze(-1)

def demo():
    print("=" * 60); print("Graph Regression"); print("=" * 60)
    torch.manual_seed(42); np.random.seed(42)
    graphs = []
    for i in range(200):
        n = np.random.randint(5, 15); x = torch.randn(n, 4)
        edges = [(j, j+1) for j in range(n-1)]
        for _ in range(n//3):
            u,v = np.random.choice(n, 2, replace=False); edges.append((u,v))
        src = [e[0] for e in edges]+[e[1] for e in edges]
        dst = [e[1] for e in edges]+[e[0] for e in edges]
        ei = torch.tensor([src, dst], dtype=torch.long)
        target = x.sum().item() * 0.1 + len(edges) * 0.05 + np.random.randn() * 0.1
        graphs.append((x, ei, target))

    model = GraphRegressor(4, 32); opt = torch.optim.Adam(model.parameters(), lr=0.005)
    model.train()
    for epoch in range(50):
        total_loss = 0
        for x, ei, y in graphs[:160]:
            opt.zero_grad()
            pred = model(x, ei)
            loss = F.mse_loss(pred, torch.tensor([y]))
            loss.backward(); opt.step(); total_loss += loss.item()
        if (epoch+1) % 10 == 0:
            model.eval(); test_loss = sum(F.mse_loss(model(x,ei), torch.tensor([y])).item() for x,ei,y in graphs[160:])
            print(f"  Epoch {epoch+1}: Train={total_loss/160:.4f}, Test={test_loss/40:.4f}")
            model.train()

if __name__ == "__main__":
    demo()
