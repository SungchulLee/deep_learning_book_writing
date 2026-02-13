"""Chapter 29.7.4: GNN-based Recommendation System (LightGCN-style)."""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np

class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, embed_dim=32, num_layers=3):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, embed_dim)
        self.item_emb = nn.Embedding(n_items, embed_dim)
        self.num_layers = num_layers
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)

    def forward(self, edge_index, n_users, n_items):
        n = n_users + n_items
        x = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        all_embs = [x]
        src, dst = edge_index[0], edge_index[1]
        deg = torch.zeros(n, device=x.device)
        deg.scatter_add_(0, dst, torch.ones(dst.shape[0], device=x.device))
        norm = (deg[src]*deg[dst]).pow(-0.5); norm[norm==float('inf')]=0
        for _ in range(self.num_layers):
            msg = x[src]*norm.unsqueeze(1)
            out = torch.zeros(n, x.shape[1], device=x.device)
            out.scatter_add_(0, dst.unsqueeze(1).expand_as(msg), msg)
            x = out; all_embs.append(x)
        x = torch.stack(all_embs, dim=0).mean(dim=0)
        return x[:n_users], x[n_users:]

    def bpr_loss(self, user_emb, item_emb, users, pos_items, neg_items):
        u = user_emb[users]; pi = item_emb[pos_items]; ni = item_emb[neg_items]
        pos_s = (u*pi).sum(dim=-1); neg_s = (u*ni).sum(dim=-1)
        return -F.logsigmoid(pos_s - neg_s).mean()

def demo():
    print("=" * 60); print("Recommendation (LightGCN)"); print("=" * 60)
    torch.manual_seed(42); np.random.seed(42)
    n_users, n_items = 50, 30
    # Generate interactions
    interactions = []
    for u in range(n_users):
        n_int = np.random.randint(3, 8)
        items = np.random.choice(n_items, n_int, replace=False)
        for i in items: interactions.append((u, i))
    # Build bipartite edge_index (offset items by n_users)
    src = [u for u,i in interactions]+[i+n_users for u,i in interactions]
    dst = [i+n_users for u,i in interactions]+[u for u,i in interactions]
    ei = torch.tensor([src, dst], dtype=torch.long)

    model = LightGCN(n_users, n_items, embed_dim=16, num_layers=2)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    user_items = {}
    for u,i in interactions: user_items.setdefault(u, []).append(i)

    model.train()
    for epoch in range(50):
        opt.zero_grad()
        u_emb, i_emb = model(ei, n_users, n_items)
        users, pos, neg = [], [], []
        for u, items in user_items.items():
            for i in items:
                users.append(u); pos.append(i)
                ni = np.random.randint(n_items)
                while ni in items: ni = np.random.randint(n_items)
                neg.append(ni)
        loss = model.bpr_loss(u_emb, i_emb, torch.tensor(users), torch.tensor(pos), torch.tensor(neg))
        loss.backward(); opt.step()
        if (epoch+1) % 10 == 0: print(f"  Epoch {epoch+1}: BPR Loss = {loss.item():.4f}")

if __name__ == "__main__":
    demo()
