"""Chapter 29.5.5: Set2Set Readout"""
import torch, torch.nn as nn, torch.nn.functional as F

class Set2Set(nn.Module):
    def __init__(self, in_ch, processing_steps=3):
        super().__init__()
        self.steps = processing_steps
        self.lstm = nn.LSTMCell(2 * in_ch, in_ch)
        self.in_ch = in_ch
    def forward(self, x, batch):
        ng = batch.max().item() + 1
        q = torch.zeros(ng, self.in_ch, device=x.device)
        c = torch.zeros(ng, self.in_ch, device=x.device)
        r = torch.zeros(ng, self.in_ch, device=x.device)
        for _ in range(self.steps):
            q, c = self.lstm(torch.cat([q, r], dim=-1), (q, c))
            e = (x * q[batch]).sum(dim=-1)
            e_max = torch.full((ng,), float('-inf'), device=x.device)
            e_max.scatter_reduce_(0, batch, e, reduce='amax')
            a = torch.exp(e - e_max[batch])
            a_sum = torch.zeros(ng, device=x.device)
            a_sum.scatter_add_(0, batch, a)
            a = a / a_sum[batch].clamp(min=1e-10)
            r = torch.zeros(ng, self.in_ch, device=x.device)
            r.scatter_add_(0, batch.unsqueeze(1).expand_as(x), x * a.unsqueeze(1))
        return torch.cat([q, r], dim=-1)

def demo():
    print("=" * 60); print("Set2Set Readout"); print("=" * 60)
    torch.manual_seed(42)
    x = torch.randn(10, 8)
    batch = torch.tensor([0,0,0,0,1,1,1,2,2,2])
    s2s = Set2Set(8, processing_steps=3)
    out = s2s(x, batch)
    print(f"Input: {x.shape}, Output: {out.shape}")
    print(f"Output dim = 2 * input_dim = {2*8}")
    # Compare with simple pooling
    pool = torch.zeros(3, 8)
    pool.scatter_add_(0, batch.unsqueeze(1).expand_as(x), x)
    print(f"Sum pool output: {pool.shape}")
    print(f"Set2Set output (richer): {out.shape}")

if __name__ == "__main__":
    demo()
