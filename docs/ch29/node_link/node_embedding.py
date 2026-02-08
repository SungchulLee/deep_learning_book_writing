"""Chapter 29.6.3: Node Embedding - DeepWalk and Node2Vec."""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, networkx as nx
from collections import defaultdict

def random_walk(G, start, length):
    walk = [start]
    for _ in range(length - 1):
        neighbors = list(G.neighbors(walk[-1]))
        if not neighbors: break
        walk.append(np.random.choice(neighbors))
    return walk

def node2vec_walk(G, start, length, p=1.0, q=1.0):
    walk = [start]
    if len(list(G.neighbors(start))) == 0: return walk
    walk.append(np.random.choice(list(G.neighbors(start))))
    for _ in range(length - 2):
        cur = walk[-1]; prev = walk[-2]
        neighbors = list(G.neighbors(cur))
        if not neighbors: break
        probs = []
        for n in neighbors:
            if n == prev: probs.append(1.0/p)
            elif G.has_edge(n, prev): probs.append(1.0)
            else: probs.append(1.0/q)
        probs = np.array(probs); probs /= probs.sum()
        walk.append(np.random.choice(neighbors, p=probs))
    return walk

class SkipGramEmbedding(nn.Module):
    def __init__(self, n_nodes, embed_dim):
        super().__init__()
        self.embed = nn.Embedding(n_nodes, embed_dim)
        self.context = nn.Embedding(n_nodes, embed_dim)
    def forward(self, center, context, neg):
        c = self.embed(center); ctx = self.context(context); n = self.context(neg)
        pos_score = (c * ctx).sum(dim=-1)
        neg_score = (c.unsqueeze(1) * n).sum(dim=-1)
        return -F.logsigmoid(pos_score).mean() - F.logsigmoid(-neg_score).mean()

def demo():
    print("=" * 60); print("Node Embedding"); print("=" * 60)
    np.random.seed(42); torch.manual_seed(42)
    G = nx.karate_club_graph(); n = G.number_of_nodes()
    # Generate walks
    walks = []
    for _ in range(10):
        for node in range(n):
            walks.append(node2vec_walk(G, node, length=10, p=1, q=0.5))
    # Build training pairs
    window = 3; centers, contexts = [], []
    for walk in walks:
        for i, c in enumerate(walk):
            for j in range(max(0,i-window), min(len(walk),i+window+1)):
                if i != j: centers.append(c); contexts.append(walk[j])
    model = SkipGramEmbedding(n, 16)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    centers_t = torch.tensor(centers); contexts_t = torch.tensor(contexts)
    for epoch in range(50):
        idx = torch.randperm(len(centers))[:512]
        c, ctx = centers_t[idx], contexts_t[idx]
        neg = torch.randint(0, n, (512, 5))
        loss = model(c, ctx, neg)
        opt.zero_grad(); loss.backward(); opt.step()
        if (epoch+1) % 10 == 0: print(f"  Epoch {epoch+1}: Loss = {loss.item():.4f}")
    # Check embeddings
    emb = model.embed.weight.detach()
    y = [0 if G.nodes[i].get('club','')=='Mr. Hi' else 1 for i in range(n)]
    c0 = emb[torch.tensor([i for i,l in enumerate(y) if l==0])].mean(0)
    c1 = emb[torch.tensor([i for i,l in enumerate(y) if l==1])].mean(0)
    print(f"  Cluster distance: {(c0-c1).norm():.4f}")

if __name__ == "__main__":
    demo()
