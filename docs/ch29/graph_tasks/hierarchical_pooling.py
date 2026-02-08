"""Chapter 29.5.4: Hierarchical Pooling - TopK and DiffPool"""
import torch, torch.nn as nn, torch.nn.functional as F

class TopKPool(nn.Module):
    """Select top-k nodes by learned score."""
    def __init__(self, in_ch, ratio=0.5):
        super().__init__()
        self.score = nn.Linear(in_ch, 1)
        self.ratio = ratio
    def forward(self, x, edge_index):
        scores = self.score(x).squeeze(-1)
        k = max(1, int(self.ratio * x.shape[0]))
        _, idx = scores.topk(k)
        idx = idx.sort()[0]
        x_pool = x[idx] * torch.sigmoid(scores[idx]).unsqueeze(-1)
        # Filter edges
        mask = torch.zeros(x.shape[0], dtype=torch.bool)
        mask[idx] = True
        node_map = torch.full((x.shape[0],), -1, dtype=torch.long)
        node_map[idx] = torch.arange(k)
        src, dst = edge_index[0], edge_index[1]
        edge_mask = mask[src] & mask[dst]
        new_src = node_map[src[edge_mask]]; new_dst = node_map[dst[edge_mask]]
        return x_pool, torch.stack([new_src, new_dst])

class SimpleDiffPool(nn.Module):
    """Simplified DiffPool: soft clustering."""
    def __init__(self, in_ch, n_clusters):
        super().__init__()
        self.assign = nn.Sequential(nn.Linear(in_ch, n_clusters), nn.Softmax(dim=-1))
    def forward(self, x, adj):
        S = self.assign(x)  # [n, k]
        x_pool = S.T @ x     # [k, d]
        adj_pool = S.T @ adj @ S  # [k, k]
        return x_pool, adj_pool, S

def demo():
    print("=" * 60); print("Hierarchical Pooling"); print("=" * 60)
    torch.manual_seed(42)
    x = torch.randn(10, 8)
    ei = torch.tensor([[0,1,1,2,2,3,3,4,5,6,6,7,7,8,8,9],[1,0,2,1,3,2,4,3,6,5,7,6,8,7,9,8]], dtype=torch.long)
    # TopK
    topk = TopKPool(8, ratio=0.5)
    x_new, ei_new = topk(x, ei)
    print(f"TopK: {x.shape[0]} -> {x_new.shape[0]} nodes, {ei.shape[1]} -> {ei_new.shape[1]} edges")
    # DiffPool
    adj = torch.zeros(10, 10); adj[ei[0], ei[1]] = 1
    dp = SimpleDiffPool(8, 3)
    x_p, adj_p, S = dp(x, adj)
    print(f"DiffPool: {x.shape[0]} -> {x_p.shape[0]} clusters")
    print(f"  Assignment (first 3 nodes): {S[:3].detach().round(decimals=2)}")

if __name__ == "__main__":
    demo()
