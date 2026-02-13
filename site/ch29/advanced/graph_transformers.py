"""
Chapter 29.4.4: Graph Transformers
Self-attention on graphs with structural encodings.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, networkx as nx, math

class GraphTransformerLayer(nn.Module):
    """Single graph transformer layer with multi-head attention."""
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.Wq = nn.Linear(hidden_dim, hidden_dim)
        self.Wk = nn.Linear(hidden_dim, hidden_dim)
        self.Wv = nn.Linear(hidden_dim, hidden_dim)
        self.Wo = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim*4), nn.GELU(), nn.Linear(hidden_dim*4, hidden_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, spatial_bias=None):
        n, d = h.shape
        res = h; h = self.norm1(h)
        Q = self.Wq(h).view(n, self.num_heads, self.head_dim)
        K = self.Wk(h).view(n, self.num_heads, self.head_dim)
        V = self.Wv(h).view(n, self.num_heads, self.head_dim)
        attn = torch.einsum('ihd,jhd->ijh', Q, K) / math.sqrt(self.head_dim)
        if spatial_bias is not None:
            attn = attn + spatial_bias.unsqueeze(-1)
        attn = F.softmax(attn, dim=1)
        attn = self.dropout(attn)
        out = torch.einsum('ijh,jhd->ihd', attn, V).reshape(n, -1)
        h = res + self.dropout(self.Wo(out))
        h = h + self.dropout(self.ffn(self.norm2(h)))
        return h

class SimpleGraphTransformer(nn.Module):
    def __init__(self, in_ch, hidden_ch, out_ch, num_layers=3, num_heads=4):
        super().__init__()
        self.node_enc = nn.Linear(in_ch, hidden_ch)
        self.layers = nn.ModuleList([GraphTransformerLayer(hidden_ch, num_heads) for _ in range(num_layers)])
        self.classifier = nn.Linear(hidden_ch, out_ch)

    def forward(self, x, spatial_bias=None):
        h = self.node_enc(x)
        for layer in self.layers:
            h = layer(h, spatial_bias)
        return self.classifier(h)

def compute_spatial_encoding(G):
    """Compute shortest-path distance matrix as spatial bias."""
    n = G.number_of_nodes()
    dist = torch.zeros(n, n)
    lengths = dict(nx.all_pairs_shortest_path_length(G))
    for i in range(n):
        for j, d in lengths[i].items():
            dist[i, j] = -d * 0.5  # Negative distance as bias
    return dist

def demo_graph_transformer():
    print("=" * 60); print("Graph Transformer"); print("=" * 60)
    torch.manual_seed(42)
    G = nx.karate_club_graph(); n = G.number_of_nodes()
    x = torch.eye(n)
    y = torch.tensor([0 if G.nodes[i].get('club','')=='Mr. Hi' else 1 for i in range(n)])
    spatial_bias = compute_spatial_encoding(G)
    tm = torch.zeros(n, dtype=torch.bool); tm[::2] = True
    model = SimpleGraphTransformer(n, 32, 2, num_layers=3, num_heads=4)
    opt = torch.optim.Adam(model.parameters(), lr=0.005)
    model.train()
    for epoch in range(300):
        opt.zero_grad()
        loss = F.cross_entropy(model(x, spatial_bias)[tm], y[tm])
        loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        acc = (model(x, spatial_bias).argmax(1)[~tm] == y[~tm]).float().mean()
    print(f"  Test Accuracy: {acc:.4f}")

if __name__ == "__main__":
    demo_graph_transformer()
