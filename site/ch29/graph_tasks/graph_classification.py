"""
Chapter 29.5.1: Graph Classification
End-to-end graph classification pipeline.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np

class GCNLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.lin = nn.Linear(in_ch, out_ch)
    def forward(self, x, edge_index):
        n = x.shape[0]; src, dst = edge_index[0], edge_index[1]
        loop = torch.arange(n, device=x.device)
        src_a = torch.cat([src, loop]); dst_a = torch.cat([dst, loop])
        deg = torch.zeros(n, device=x.device)
        deg.scatter_add_(0, dst_a, torch.ones(dst_a.shape[0], device=x.device))
        norm = (deg[src_a]*deg[dst_a]).pow(-0.5); norm[norm==float('inf')]=0
        h = self.lin(x); msg = h[src_a]*norm.unsqueeze(1)
        out = torch.zeros(n, h.shape[1], device=x.device)
        out.scatter_add_(0, dst_a.unsqueeze(1).expand_as(msg), msg)
        return out

class GraphClassifier(nn.Module):
    def __init__(self, in_ch, hidden_ch, out_ch, num_layers=3, readout='sum'):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(GCNLayer(in_ch, hidden_ch))
        self.bns.append(nn.BatchNorm1d(hidden_ch))
        for _ in range(num_layers-1):
            self.convs.append(GCNLayer(hidden_ch, hidden_ch))
            self.bns.append(nn.BatchNorm1d(hidden_ch))
        self.classifier = nn.Sequential(nn.Linear(hidden_ch, hidden_ch), nn.ReLU(), nn.Linear(hidden_ch, out_ch))
        self.readout = readout

    def forward(self, x, edge_index, batch=None):
        if batch is None: batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, edge_index)))
        ng = batch.max().item()+1
        pool = torch.zeros(ng, x.shape[1], device=x.device)
        pool.scatter_add_(0, batch.unsqueeze(1).expand_as(x), x)
        if self.readout == 'mean':
            cnt = torch.zeros(ng, device=x.device)
            cnt.scatter_add_(0, batch, torch.ones(batch.shape[0], device=x.device))
            pool = pool / cnt.clamp(min=1).unsqueeze(1)
        return self.classifier(pool)

def create_dataset(n_graphs=300):
    np.random.seed(42); graphs = []
    for i in range(n_graphs):
        label = i % 3
        if label == 0: n = np.random.randint(5,10); edges = [(j,j+1) for j in range(n-1)]
        elif label == 1: n = np.random.randint(5,10); edges = [(0,j) for j in range(1,n)]
        else: n = np.random.randint(5,10); edges = [(j,(j+1)%n) for j in range(n)]
        src = [e[0] for e in edges]+[e[1] for e in edges]
        dst = [e[1] for e in edges]+[e[0] for e in edges]
        graphs.append({'x': torch.ones(n,1), 'ei': torch.tensor([src,dst],dtype=torch.long), 'y': label, 'n': n})
    return graphs

def demo_graph_classification():
    print("=" * 60); print("Graph Classification"); print("=" * 60)
    torch.manual_seed(42)
    graphs = create_dataset(300); train_g, test_g = graphs[:240], graphs[240:]
    model = GraphClassifier(1, 32, 3, num_layers=3, readout='sum')
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    for epoch in range(50):
        np.random.shuffle(train_g); correct = 0
        for g in train_g:
            opt.zero_grad()
            out = model(g['x'], g['ei'])
            loss = F.cross_entropy(out, torch.tensor([g['y']]))
            loss.backward(); opt.step()
            correct += (out.argmax(1).item() == g['y'])
        if (epoch+1) % 10 == 0:
            model.eval(); tc = sum(1 for g in test_g if model(g['x'], g['ei']).argmax(1).item() == g['y'])
            print(f"  Epoch {epoch+1}: Train={correct/len(train_g):.3f}, Test={tc/len(test_g):.3f}")
            model.train()

if __name__ == "__main__":
    demo_graph_classification()
