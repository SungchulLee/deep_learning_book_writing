"""
GraphVAE: Variational Autoencoder for one-shot graph generation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import numpy as np


class GraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.conv_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

    def forward(self, adj, x, node_mask):
        h = self.input_proj(x)
        for conv, norm in zip(self.conv_layers, self.norms):
            h = norm(F.relu(conv(torch.bmm(adj, h))) + h)
        mask = node_mask.unsqueeze(-1)
        h_graph = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.mu_head(h_graph), self.logvar_head(h_graph)


class GraphDecoder(nn.Module):
    def __init__(self, latent_dim, max_nodes, node_feature_dim, hidden_dim=256):
        super().__init__()
        self.max_nodes = max_nodes
        self.node_feature_dim = node_feature_dim
        n_edges = max_nodes * (max_nodes - 1) // 2
        self.backbone = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.adj_head = nn.Linear(hidden_dim, n_edges)
        self.node_head = nn.Linear(hidden_dim, max_nodes)
        self.feat_head = nn.Linear(hidden_dim, max_nodes * node_feature_dim)

    def forward(self, z):
        B, n = z.size(0), self.max_nodes
        h = self.backbone(z)
        adj = torch.zeros(B, n, n, device=z.device)
        idx = torch.triu_indices(n, n, offset=1)
        adj[:, idx[0], idx[1]] = torch.sigmoid(self.adj_head(h))
        adj = adj + adj.transpose(1, 2)
        return adj, torch.sigmoid(self.node_head(h)), self.feat_head(h).view(B, n, self.node_feature_dim)


class GraphVAE(nn.Module):
    def __init__(self, max_nodes, node_feature_dim=1, hidden_dim=128, latent_dim=32, beta=1.0):
        super().__init__()
        self.max_nodes, self.latent_dim, self.beta = max_nodes, latent_dim, beta
        self.encoder = GraphEncoder(node_feature_dim, hidden_dim, latent_dim)
        self.decoder = GraphDecoder(latent_dim, max_nodes, node_feature_dim, hidden_dim)

    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def forward(self, adj, x, node_mask):
        B = adj.size(0)
        mu, logvar = self.encoder(adj, x, node_mask)
        z = self.reparameterize(mu, logvar)
        adj_pred, node_pred, feat_pred = self.decoder(z)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        recon_loss = torch.tensor(0.0, device=adj.device)
        for b in range(B):
            n_actual = int(node_mask[b].sum().item())
            with torch.no_grad():
                cost = torch.zeros(self.max_nodes, self.max_nodes)
                for i in range(self.max_nodes):
                    for j in range(n_actual):
                        cost[i, j] = -F.mse_loss(adj_pred[b, i], adj[b, j], reduction="sum") + node_pred[b, i]
                _, col_ind = linear_sum_assignment((-cost).numpy())
            perm = torch.tensor(col_ind, dtype=torch.long)
            adj_aligned = adj[b][perm][:, perm]
            mask_aligned = node_mask[b][perm]
            idx = torch.triu_indices(self.max_nodes, self.max_nodes, offset=1)
            recon_loss = recon_loss + F.binary_cross_entropy(adj_pred[b][idx[0], idx[1]].clamp(1e-6, 1-1e-6), adj_aligned[idx[0], idx[1]], reduction="mean")
            recon_loss = recon_loss + F.binary_cross_entropy(node_pred[b].clamp(1e-6, 1-1e-6), mask_aligned, reduction="mean")

        recon_loss = recon_loss / B
        total_loss = recon_loss + self.beta * kl_loss
        return {"total_loss": total_loss, "recon_loss": recon_loss, "kl_loss": kl_loss}

    @torch.no_grad()
    def generate(self, num_graphs=1, temperature=1.0):
        self.eval()
        z = torch.randn(num_graphs, self.latent_dim) * temperature
        adj_pred, node_pred, _ = self.decoder(z)
        graphs = []
        for b in range(num_graphs):
            active = node_pred[b] > 0.5
            if active.sum() < 2:
                active[:2] = True
            adj_b = (adj_pred[b] > 0.5).float() * active.unsqueeze(0).float() * active.unsqueeze(1).float()
            adj_b.fill_diagonal_(0)
            graphs.append(adj_b[torch.where(active)[0]][:, torch.where(active)[0]])
        return graphs

    @torch.no_grad()
    def interpolate(self, adj1, x1, mask1, adj2, x2, mask2, steps=5):
        self.eval()
        mu1, _ = self.encoder(adj1.unsqueeze(0), x1.unsqueeze(0), mask1.unsqueeze(0))
        mu2, _ = self.encoder(adj2.unsqueeze(0), x2.unsqueeze(0), mask2.unsqueeze(0))
        graphs = []
        for alpha in torch.linspace(0, 1, steps):
            z = (1 - alpha) * mu1 + alpha * mu2
            adj_pred, node_pred, _ = self.decoder(z)
            active = node_pred[0] > 0.5
            adj_b = (adj_pred[0] > 0.5).float() * active.unsqueeze(0).float() * active.unsqueeze(1).float()
            adj_b.fill_diagonal_(0)
            idx = torch.where(active)[0]
            graphs.append(adj_b[idx][:, idx] if len(idx) >= 2 else adj_b[:2, :2])
        return graphs


if __name__ == "__main__":
    torch.manual_seed(42)
    max_n, feat_dim = 12, 4

    print("=== GraphVAE Demo ===\n")
    adjs, feats, masks = [], [], []
    for _ in range(150):
        n = torch.randint(4, max_n, (1,)).item()
        adj = torch.zeros(max_n, max_n)
        for i in range(n):
            for j in range(i + 1, n):
                if torch.rand(1) < 0.3:
                    adj[i, j] = adj[j, i] = 1
        mask = torch.zeros(max_n)
        mask[:n] = 1.0
        adjs.append(adj); feats.append(torch.randn(max_n, feat_dim)); masks.append(mask)

    adjs, feats, masks = torch.stack(adjs), torch.stack(feats), torch.stack(masks)

    model = GraphVAE(max_n, feat_dim, hidden_dim=64, latent_dim=16, beta=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(30):
        model.train()
        idx = torch.randperm(150)[:32]
        result = model(adjs[idx], feats[idx], masks[idx])
        optimizer.zero_grad()
        result["total_loss"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: loss={result['total_loss'].item():.4f} (recon={result['recon_loss'].item():.4f}, kl={result['kl_loss'].item():.4f})")

    print("\n=== Generation ===")
    for i, g in enumerate(model.generate(10)):
        n, e = g.size(0), int(g.sum().item()) // 2
        print(f"Graph {i}: {n} nodes, {e} edges")

    print("\n=== Latent Interpolation ===")
    for i, g in enumerate(model.interpolate(adjs[0], feats[0], masks[0], adjs[1], feats[1], masks[1], 5)):
        n, e = g.size(0), int(g.sum().item()) // 2
        print(f"Step {i}: {n} nodes, {e} edges, density={2*e/(n*(n-1)) if n>1 else 0:.3f}")
