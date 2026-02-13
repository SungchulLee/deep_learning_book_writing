"""
Chapter 29.7.6: Financial Networks
GNN applications: portfolio optimization, fraud detection, stock prediction.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, networkx as nx

# ============================================================
# 1. GNN-based Portfolio Optimization
# ============================================================
class PortfolioGNN(nn.Module):
    def __init__(self, n_features, hidden_ch=32):
        super().__init__()
        self.conv1 = nn.Linear(n_features, hidden_ch)
        self.conv2 = nn.Linear(hidden_ch, hidden_ch)
        self.weight_head = nn.Linear(hidden_ch, 1)

    def gcn_forward(self, x, ei):
        n = x.shape[0]; src, dst = ei[0], ei[1]
        loop = torch.arange(n); sa = torch.cat([src, loop]); da = torch.cat([dst, loop])
        deg = torch.zeros(n); deg.scatter_add_(0, da, torch.ones(da.shape[0]))
        norm = (deg[sa]*deg[da]).pow(-0.5); norm[norm==float('inf')]=0
        h = x; msg = h[sa]*norm.unsqueeze(1)
        out = torch.zeros(n, h.shape[1]); out.scatter_add_(0, da.unsqueeze(1).expand_as(msg), msg)
        return out

    def forward(self, x, ei):
        h = F.relu(self.gcn_forward(self.conv1(x), ei))
        h = F.relu(self.gcn_forward(self.conv2(h), ei))
        weights = torch.softmax(self.weight_head(h).squeeze(-1), dim=0)
        return weights

def demo_portfolio():
    print("=" * 60); print("GNN Portfolio Optimization"); print("=" * 60)
    torch.manual_seed(42); np.random.seed(42)
    n_assets = 10; n_days = 252
    returns = np.random.randn(n_days, n_assets) * 0.01 + 0.0003
    corr = np.corrcoef(returns.T)
    # Build graph
    threshold = 0.2
    src, dst = [], []
    for i in range(n_assets):
        for j in range(i+1, n_assets):
            if abs(corr[i,j]) > threshold:
                src.extend([i,j]); dst.extend([j,i])
    ei = torch.tensor([src, dst], dtype=torch.long)
    # Features: [mean_ret, vol, sharpe, skew]
    x = torch.tensor(np.column_stack([
        returns.mean(0), returns.std(0),
        returns.mean(0)/returns.std(0),
        np.array([np.mean(((r-r.mean())/r.std())**3) for r in returns.T])
    ]), dtype=torch.float32)
    model = PortfolioGNN(4)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    returns_t = torch.tensor(returns, dtype=torch.float32)
    for epoch in range(100):
        opt.zero_grad()
        w = model(x, ei)
        port_returns = (returns_t * w.unsqueeze(0)).sum(dim=1)
        sharpe = port_returns.mean() / port_returns.std()
        loss = -sharpe
        loss.backward(); opt.step()
        if (epoch+1) % 25 == 0:
            print(f"  Epoch {epoch+1}: Sharpe={sharpe.item():.4f}, "
                  f"Top weight={w.max().item():.3f}")
    print(f"  Final weights: {w.detach().numpy().round(3)}")

# ============================================================
# 2. Fraud Detection
# ============================================================
def demo_fraud():
    print("\n" + "=" * 60); print("Fraud Detection on Transaction Graph"); print("=" * 60)
    torch.manual_seed(42); np.random.seed(42)
    n = 100  # 90 normal + 10 fraud
    x = torch.randn(n, 6)
    x[90:, 0] += 2.0  # Fraud: higher transaction volume
    x[90:, 3] -= 1.5  # Fraud: newer accounts
    y = torch.tensor([0]*90 + [1]*10)
    # Build transaction graph
    G = nx.stochastic_block_model([90, 10], [[0.03, 0.01],[0.01, 0.2]], seed=42)
    edges = list(G.edges())
    src = [e[0] for e in edges]+[e[1] for e in edges]
    dst = [e[1] for e in edges]+[e[0] for e in edges]
    ei = torch.tensor([src, dst], dtype=torch.long)
    # GCN
    lin1 = nn.Linear(6, 16); lin2 = nn.Linear(16, 2)
    opt = torch.optim.Adam(list(lin1.parameters())+list(lin2.parameters()), lr=0.01)
    tm = torch.zeros(n, dtype=torch.bool); tm[:30] = True; tm[90:95] = True
    for epoch in range(200):
        sa = torch.cat([ei[0], torch.arange(n)]); da = torch.cat([ei[1], torch.arange(n)])
        deg = torch.zeros(n); deg.scatter_add_(0, da, torch.ones(da.shape[0]))
        norm = (deg[sa]*deg[da]).pow(-0.5); norm[norm==float('inf')]=0
        h = lin1(x); msg = h[sa]*norm.unsqueeze(1)
        out = torch.zeros(n, 16); out.scatter_add_(0, da.unsqueeze(1).expand_as(msg), msg)
        h2 = lin2(F.relu(out))
        opt.zero_grad(); F.cross_entropy(h2[tm], y[tm]).backward(); opt.step()
    pred = h2.argmax(1).detach()
    fraud_recall = (pred[90:]==1).float().mean()
    precision = (y[pred==1]==1).float().mean() if (pred==1).any() else torch.tensor(0.0)
    print(f"  Fraud recall: {fraud_recall:.4f}")
    print(f"  Precision: {precision:.4f}")

# ============================================================
# 3. Stock Return Prediction
# ============================================================
def demo_stock_prediction():
    print("\n" + "=" * 60); print("Stock Return Prediction with GNN"); print("=" * 60)
    torch.manual_seed(42); np.random.seed(42)
    n_stocks = 20; n_days = 100
    returns = np.random.randn(n_days, n_stocks) * 0.02
    # Add sector structure
    for i in range(0, 10): returns[:, i] += np.random.randn(n_days) * 0.005
    for i in range(10, 20): returns[:, i] += np.random.randn(n_days) * 0.005
    # Build graph from correlation
    corr = np.corrcoef(returns[:50].T)
    src, dst = [], []
    for i in range(n_stocks):
        for j in range(i+1, n_stocks):
            if abs(corr[i,j]) > 0.2:
                src.extend([i,j]); dst.extend([j,i])
    ei = torch.tensor([src, dst], dtype=torch.long)
    # Predict next-day returns from past features
    lin1 = nn.Linear(5, 16); lin2 = nn.Linear(16, 1)
    opt = torch.optim.Adam(list(lin1.parameters())+list(lin2.parameters()), lr=0.005)
    for day in range(55, 95):
        # Features: [5-day return, 10-day vol, momentum, mean-reversion signal, volume proxy]
        feat = torch.tensor(np.column_stack([
            returns[day-5:day].mean(0), returns[day-10:day].std(0),
            returns[day-20:day].sum(0), -returns[day-1],
            np.random.randn(n_stocks)*0.01
        ]), dtype=torch.float32)
        target = torch.tensor(returns[day], dtype=torch.float32)
        sa = torch.cat([ei[0], torch.arange(n_stocks)])
        da = torch.cat([ei[1], torch.arange(n_stocks)])
        deg = torch.zeros(n_stocks); deg.scatter_add_(0, da, torch.ones(da.shape[0]))
        norm = (deg[sa]*deg[da]).pow(-0.5); norm[norm==float('inf')]=0
        h = lin1(feat); msg = h[sa]*norm.unsqueeze(1)
        out = torch.zeros(n_stocks, 16); out.scatter_add_(0, da.unsqueeze(1).expand_as(msg), msg)
        pred = lin2(F.relu(out)).squeeze(-1)
        opt.zero_grad(); F.mse_loss(pred, target).backward(); opt.step()
    # Evaluate directional accuracy
    correct = 0; total = 0
    with torch.no_grad():
        for day in range(95, 99):
            feat = torch.tensor(np.column_stack([
                returns[day-5:day].mean(0), returns[day-10:day].std(0),
                returns[day-20:day].sum(0), -returns[day-1],
                np.random.randn(n_stocks)*0.01
            ]), dtype=torch.float32)
            h = lin1(feat); msg = h[sa]*norm.unsqueeze(1)
            out = torch.zeros(n_stocks, 16); out.scatter_add_(0, da.unsqueeze(1).expand_as(msg), msg)
            pred = lin2(F.relu(out)).squeeze(-1)
            actual = returns[day]
            correct += ((pred.numpy() > 0) == (actual > 0)).sum()
            total += n_stocks
    print(f"  Directional accuracy: {correct/total:.4f}")

if __name__ == "__main__":
    demo_portfolio()
    demo_fraud()
    demo_stock_prediction()
