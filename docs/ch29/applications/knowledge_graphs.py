"""Chapter 29.7.5: Knowledge Graph Completion."""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np

class TransE(nn.Module):
    def __init__(self, n_entities, n_relations, dim=32):
        super().__init__()
        self.ent_emb = nn.Embedding(n_entities, dim)
        self.rel_emb = nn.Embedding(n_relations, dim)
        nn.init.xavier_uniform_(self.ent_emb.weight)
        nn.init.xavier_uniform_(self.rel_emb.weight)
    def score(self, h, r, t):
        return -torch.norm(self.ent_emb(h) + self.rel_emb(r) - self.ent_emb(t), dim=-1)
    def forward(self, triples, neg_tails):
        h, r, t = triples[:, 0], triples[:, 1], triples[:, 2]
        pos_score = self.score(h, r, t)
        neg_score = self.score(h.unsqueeze(1).expand_as(neg_tails),
                                r.unsqueeze(1).expand_as(neg_tails), neg_tails)
        return -F.logsigmoid(pos_score).mean() - F.logsigmoid(-neg_score).mean()

def demo():
    print("=" * 60); print("Knowledge Graph Completion (TransE)"); print("=" * 60)
    torch.manual_seed(42); np.random.seed(42)
    n_ent, n_rel = 50, 5
    # Generate triples
    triples = []
    for _ in range(500):
        h = np.random.randint(n_ent); r = np.random.randint(n_rel); t = np.random.randint(n_ent)
        triples.append([h, r, t])
    triples = torch.tensor(triples)
    train, test = triples[:400], triples[400:]
    model = TransE(n_ent, n_rel, dim=32)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(100):
        opt.zero_grad()
        neg_t = torch.randint(0, n_ent, (train.shape[0], 5))
        loss = model(train, neg_t)
        loss.backward(); opt.step()
        if (epoch+1) % 20 == 0:
            with torch.no_grad():
                h, r, t = test[:,0], test[:,1], test[:,2]
                pos_s = model.score(h, r, t)
                all_scores = torch.stack([model.score(h, r, torch.full_like(t, e)) for e in range(n_ent)], dim=1)
                ranks = (all_scores >= pos_s.unsqueeze(1)).sum(dim=1).float()
                mrr = (1.0/ranks).mean()
                hits10 = (ranks <= 10).float().mean()
            print(f"  Epoch {epoch+1}: Loss={loss.item():.4f}, MRR={mrr:.4f}, Hits@10={hits10:.4f}")

if __name__ == "__main__":
    demo()
