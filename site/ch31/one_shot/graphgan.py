"""
GraphGAN: adversarial graph generation with WGAN-GP training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad as torch_grad


class GraphGenerator(nn.Module):
    """Generator: noise -> graph adjacency matrix."""

    def __init__(self, latent_dim, max_nodes, hidden_dim=256, temperature=0.5):
        super().__init__()
        self.max_nodes = max_nodes
        self.temperature = temperature
        n_edges = max_nodes * (max_nodes - 1) // 2
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.LeakyReLU(0.2), nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.2), nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, n_edges),
        )

    def forward(self, z, hard=False):
        B, n = z.size(0), self.max_nodes
        logits = self.net(z)
        if self.training:
            u = torch.rand_like(logits).clamp(1e-8, 1 - 1e-8)
            gumbel = -torch.log(-torch.log(u))
            edge_probs = torch.sigmoid((logits + gumbel) / self.temperature)
        else:
            edge_probs = torch.sigmoid(logits)
        if hard:
            edge_hard = (edge_probs > 0.5).float()
            edge_probs = edge_hard - edge_probs.detach() + edge_probs
        adj = torch.zeros(B, n, n, device=z.device)
        idx = torch.triu_indices(n, n, offset=1)
        adj[:, idx[0], idx[1]] = edge_probs
        return adj + adj.transpose(1, 2)


class GraphDiscriminator(nn.Module):
    """GNN-based discriminator for graphs."""

    def __init__(self, max_nodes, hidden_dim=128, num_layers=3):
        super().__init__()
        self.input_proj = nn.Linear(max_nodes, hidden_dim)
        self.conv_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.2))
            for _ in range(num_layers)
        ])
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.2), nn.Linear(hidden_dim, 1)
        )

    def forward(self, adj):
        h = self.input_proj(adj)
        for conv in self.conv_layers:
            h = conv(torch.bmm(adj, h) / (adj.sum(-1, keepdim=True) + 1)) + h
        return self.readout(h.sum(dim=1)).squeeze(-1)


def gradient_penalty(discriminator, real, fake, lambda_gp=10.0):
    B = real.size(0)
    alpha = torch.rand(B, 1, 1, device=real.device)
    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interpolated = discriminator(interpolated)
    gradients = torch_grad(outputs=d_interpolated, inputs=interpolated,
                           grad_outputs=torch.ones_like(d_interpolated),
                           create_graph=True, retain_graph=True)[0]
    return lambda_gp * ((gradients.view(B, -1).norm(2, dim=1) - 1) ** 2).mean()


class GraphGAN:
    """WGAN-GP training wrapper for graph generation."""

    def __init__(self, max_nodes, latent_dim=64, hidden_dim=128,
                 lr_g=1e-4, lr_d=1e-4, n_critic=5, lambda_gp=10.0):
        self.latent_dim, self.n_critic, self.lambda_gp = latent_dim, n_critic, lambda_gp
        self.generator = GraphGenerator(latent_dim, max_nodes, hidden_dim)
        self.discriminator = GraphDiscriminator(max_nodes, hidden_dim)
        self.opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr_g, betas=(0.5, 0.9))
        self.opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr_d, betas=(0.5, 0.9))

    def train_step(self, real_adj):
        B, device = real_adj.size(0), real_adj.device
        d_losses = []
        for _ in range(self.n_critic):
            fake_adj = self.generator(torch.randn(B, self.latent_dim, device=device)).detach()
            d_real, d_fake = self.discriminator(real_adj).mean(), self.discriminator(fake_adj).mean()
            gp = gradient_penalty(self.discriminator, real_adj, fake_adj, self.lambda_gp)
            d_loss = d_fake - d_real + gp
            self.opt_d.zero_grad(); d_loss.backward(); self.opt_d.step()
            d_losses.append(d_loss.item())

        fake_adj = self.generator(torch.randn(B, self.latent_dim, device=device))
        g_loss = -self.discriminator(fake_adj).mean()
        self.opt_g.zero_grad(); g_loss.backward(); self.opt_g.step()

        return {"d_loss": sum(d_losses)/len(d_losses), "g_loss": g_loss.item(),
                "wasserstein": (d_real - d_fake).item()}

    @torch.no_grad()
    def generate(self, num_graphs=1):
        self.generator.eval()
        adj = self.generator(torch.randn(num_graphs, self.latent_dim), hard=True)
        graphs = []
        for b in range(num_graphs):
            active = adj[b].sum(1) > 0
            if active.sum() < 2: active[:2] = True
            idx = torch.where(active)[0]
            graphs.append(adj[b][idx][:, idx])
        self.generator.train()
        return graphs


if __name__ == "__main__":
    torch.manual_seed(42)
    max_n = 12

    print("=== GraphGAN Demo ===\n")
    train_adjs = []
    for _ in range(200):
        n = torch.randint(6, max_n + 1, (1,)).item()
        adj = torch.zeros(max_n, max_n)
        mid = n // 2
        for i in range(mid):
            for j in range(i + 1, mid):
                if torch.rand(1) < 0.5: adj[i, j] = adj[j, i] = 1
        for i in range(mid, n):
            for j in range(i + 1, n):
                if torch.rand(1) < 0.5: adj[i, j] = adj[j, i] = 1
        for i in range(mid):
            for j in range(mid, n):
                if torch.rand(1) < 0.1: adj[i, j] = adj[j, i] = 1
        train_adjs.append(adj)
    train_adjs = torch.stack(train_adjs)

    print("Training GraphGAN (WGAN-GP)...")
    gan = GraphGAN(max_n, latent_dim=32, hidden_dim=64, n_critic=3)
    print(f"G params: {sum(p.numel() for p in gan.generator.parameters()):,}, "
          f"D params: {sum(p.numel() for p in gan.discriminator.parameters()):,}")

    for epoch in range(100):
        idx = torch.randperm(len(train_adjs))[:32]
        metrics = gan.train_step(train_adjs[idx])
        if (epoch + 1) % 25 == 0:
            print(f"Epoch {epoch+1}: d={metrics['d_loss']:.4f} g={metrics['g_loss']:.4f} W={metrics['wasserstein']:.4f}")

    print("\n=== Generation ===")
    for i, g in enumerate(gan.generate(10)):
        n, e = g.size(0), int(g.sum().item()) // 2
        print(f"Graph {i}: {n} nodes, {e} edges, density={2*e/(n*(n-1)) if n>1 else 0:.3f}")
