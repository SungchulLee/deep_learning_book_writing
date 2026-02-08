"""Chapter 29.5.3: Graph Pooling Strategies"""
import torch, torch.nn as nn, torch.nn.functional as F

class GlobalPooling(nn.Module):
    def __init__(self, method='sum'):
        super().__init__()
        self.method = method
    def forward(self, x, batch):
        ng = batch.max().item()+1
        out = torch.zeros(ng, x.shape[1], device=x.device)
        out.scatter_add_(0, batch.unsqueeze(1).expand_as(x), x)
        if self.method == 'mean':
            cnt = torch.zeros(ng, device=x.device)
            cnt.scatter_add_(0, batch, torch.ones(batch.shape[0], device=x.device))
            out = out / cnt.clamp(min=1).unsqueeze(1)
        elif self.method == 'max':
            out.fill_(float('-inf'))
            out.scatter_reduce_(0, batch.unsqueeze(1).expand_as(x), x, reduce='amax')
            out[out==float('-inf')] = 0
        return out

class AttentionPooling(nn.Module):
    def __init__(self, hidden_ch):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(hidden_ch, 1))
    def forward(self, x, batch):
        scores = self.gate(x).squeeze(-1)
        # Softmax per graph
        ng = batch.max().item()+1
        s_max = torch.full((ng,), float('-inf'), device=x.device)
        s_max.scatter_reduce_(0, batch, scores, reduce='amax')
        alpha = torch.exp(scores - s_max[batch])
        a_sum = torch.zeros(ng, device=x.device)
        a_sum.scatter_add_(0, batch, alpha)
        alpha = alpha / a_sum[batch].clamp(min=1e-10)
        out = torch.zeros(ng, x.shape[1], device=x.device)
        out.scatter_add_(0, batch.unsqueeze(1).expand_as(x), x * alpha.unsqueeze(1))
        return out

def demo():
    print("=" * 60); print("Graph Pooling Comparison"); print("=" * 60)
    torch.manual_seed(42)
    x = torch.randn(10, 8)
    batch = torch.tensor([0,0,0,0,1,1,1,2,2,2])
    for method in ['sum', 'mean', 'max']:
        pool = GlobalPooling(method)
        out = pool(x, batch)
        print(f"  {method:5s}: {out.shape}, norms={out.norm(dim=1).tolist()}")
    attn = AttentionPooling(8)
    out = attn(x, batch)
    print(f"  attn : {out.shape}, norms={out.norm(dim=1).tolist()}")

if __name__ == "__main__":
    demo()
