"""
GDSS: Score-based graph generation via stochastic differential equations.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class VPSDE:
    """Variance Preserving SDE for graph diffusion."""

    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0, T: float = 1.0):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.T = T

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        log_alpha = -0.5 * (self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t ** 2)
        return torch.exp(log_alpha)

    def marginal_params(self, t: torch.Tensor):
        """Return mean coefficient and std for q(x_t | x_0)."""
        ab = self.alpha_bar(t)
        mean_coeff = torch.sqrt(ab)
        std = torch.sqrt(1 - ab)
        return mean_coeff, std

    def sample_marginal(self, x_0, t, noise=None):
        """Sample x_t ~ q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x_0)
        mean_coeff, std = self.marginal_params(t)
        # Broadcast time dims
        while mean_coeff.dim() < x_0.dim():
            mean_coeff = mean_coeff.unsqueeze(-1)
            std = std.unsqueeze(-1)
        return mean_coeff * x_0 + std * noise, noise


class ScoreNetworkGNN(nn.Module):
    """
    GNN-based score network for joint node/edge score estimation.
    """

    def __init__(
        self,
        max_nodes: int,
        node_feat_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
    ):
        super().__init__()
        self.max_nodes = max_nodes
        self.node_feat_dim = node_feat_dim

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Node feature projection
        self.node_proj = nn.Linear(node_feat_dim, hidden_dim)
        # Adjacency projection (treat rows as features)
        self.adj_proj = nn.Linear(max_nodes, hidden_dim)

        # GNN layers
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gnn_layers.append(nn.ModuleDict({
                "node_msg": nn.Linear(hidden_dim, hidden_dim),
                "node_update": nn.Linear(hidden_dim * 3, hidden_dim),
                "edge_update": nn.Linear(hidden_dim * 3, hidden_dim),
                "norm_n": nn.LayerNorm(hidden_dim),
                "norm_e": nn.LayerNorm(hidden_dim),
            }))

        # Output heads
        self.score_x = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_feat_dim),
        )
        self.score_a = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x_t: torch.Tensor,
        a_t: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate joint score.
        
        Args:
            x_t: (B, n, d) noisy node features
            a_t: (B, n, n) noisy adjacency
            t: (B,) time in [0, 1]
        Returns:
            score_x: (B, n, d) node score
            score_a: (B, n, n) adjacency score
        """
        B, n, d = x_t.shape

        t_emb = self.time_mlp(t.unsqueeze(-1))  # (B, hidden)
        h_node = self.node_proj(x_t)  # (B, n, hidden)
        h_edge = self.adj_proj(a_t)  # (B, n, hidden) -- each node gets its adj row

        for layer in self.gnn_layers:
            # Message passing using noisy adjacency as weights
            a_norm = a_t / (a_t.abs().sum(-1, keepdim=True).clamp(min=1))
            msg = torch.bmm(a_norm, layer["node_msg"](h_node))

            # Node update with time conditioning
            h_node_new = layer["node_update"](
                torch.cat([h_node, msg, t_emb.unsqueeze(1).expand(-1, n, -1)], dim=-1)
            )
            h_node = layer["norm_n"](h_node + h_node_new)

            # Edge update
            h_i = h_node.unsqueeze(2).expand(-1, -1, n, -1)
            h_j = h_node.unsqueeze(1).expand(-1, n, -1, -1)
            h_edge_expanded = h_edge.unsqueeze(2).expand(-1, -1, n, -1)
            e_input = torch.cat([h_i, h_j, h_edge_expanded], dim=-1)
            h_edge_new = layer["edge_update"](e_input).mean(dim=2)
            h_edge = layer["norm_e"](h_edge + h_edge_new)

        # Output scores
        score_x = self.score_x(h_node)  # (B, n, d)

        # Edge score from pairwise node features
        h_i = h_node.unsqueeze(2).expand(-1, -1, n, -1)
        h_j = h_node.unsqueeze(1).expand(-1, n, -1, -1)
        h_pair = (h_i + h_j) / 2  # Symmetric combination
        score_a = self.score_a(h_pair).squeeze(-1)  # (B, n, n)
        # Enforce symmetry
        score_a = (score_a + score_a.transpose(1, 2)) / 2
        score_a.diagonal(dim1=1, dim2=2).zero_()

        return score_x, score_a


class GDSS(nn.Module):
    """
    Complete GDSS model: SDE-based graph generation.
    """

    def __init__(
        self,
        max_nodes: int,
        node_feat_dim: int = 1,
        hidden_dim: int = 128,
        num_layers: int = 4,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
    ):
        super().__init__()
        self.max_nodes = max_nodes
        self.node_feat_dim = node_feat_dim
        self.sde = VPSDE(beta_min, beta_max)
        self.score_net = ScoreNetworkGNN(
            max_nodes, node_feat_dim, hidden_dim, num_layers
        )

    def forward(
        self,
        x_0: torch.Tensor,
        a_0: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Training step: score matching loss.
        
        Args:
            x_0: (B, n, d) clean node features
            a_0: (B, n, n) clean adjacency
        """
        B = x_0.size(0)
        device = x_0.device

        # Sample time uniformly
        t = torch.rand(B, device=device) * 0.999 + 0.001  # avoid t=0

        # Sample noisy data
        noise_x = torch.randn_like(x_0)
        noise_a = torch.randn_like(a_0)
        noise_a = (noise_a + noise_a.transpose(1, 2)) / 2
        noise_a.diagonal(dim1=1, dim2=2).zero_()

        x_t, _ = self.sde.sample_marginal(x_0, t, noise_x)
        a_t, _ = self.sde.sample_marginal(a_0, t, noise_a)

        # Predict scores
        score_x, score_a = self.score_net(x_t, a_t, t)

        # Target: -noise / std
        _, std = self.sde.marginal_params(t)
        while std.dim() < noise_x.dim():
            std = std.unsqueeze(-1)

        target_x = -noise_x / std
        std_2d = std.squeeze(-1) if std.dim() > noise_a.dim() else std
        while std_2d.dim() < noise_a.dim():
            std_2d = std_2d.unsqueeze(-1)
        target_a = -noise_a / std_2d

        # Weighted MSE loss
        beta_t = self.sde.beta(t)
        weight = beta_t
        while weight.dim() < score_x.dim():
            weight = weight.unsqueeze(-1)

        loss_x = (weight * (score_x - target_x) ** 2).mean()

        weight_2d = beta_t
        while weight_2d.dim() < score_a.dim():
            weight_2d = weight_2d.unsqueeze(-1)
        n = self.max_nodes
        idx = torch.triu_indices(n, n, offset=1)
        loss_a = (weight_2d * (score_a - target_a) ** 2)[:, idx[0], idx[1]].mean()

        return {"loss": loss_x + loss_a, "loss_x": loss_x, "loss_a": loss_a}

    @torch.no_grad()
    def generate(
        self,
        num_graphs: int = 1,
        num_steps: int = 100,
        corrector_steps: int = 1,
        snr: float = 0.1,
        device: str = "cpu",
    ) -> list[torch.Tensor]:
        """Predictor-corrector sampling."""
        self.eval()
        n = self.max_nodes

        # Start from noise
        x_t = torch.randn(num_graphs, n, self.node_feat_dim, device=device)
        a_t = torch.randn(num_graphs, n, n, device=device)
        a_t = (a_t + a_t.transpose(1, 2)) / 2
        a_t.diagonal(dim1=1, dim2=2).zero_()

        dt = 1.0 / num_steps
        times = torch.linspace(1.0, 0.001, num_steps, device=device)

        for i, t_val in enumerate(times):
            t = torch.full((num_graphs,), t_val, device=device)
            score_x, score_a = self.score_net(x_t, a_t, t)

            beta = self.sde.beta(t)

            # --- Corrector (Langevin) ---
            for _ in range(corrector_steps):
                noise_x = torch.randn_like(x_t) * snr
                noise_a = torch.randn_like(a_t) * snr
                noise_a = (noise_a + noise_a.transpose(1, 2)) / 2
                noise_a.diagonal(dim1=1, dim2=2).zero_()

                step_size = snr ** 2
                s_x, s_a = self.score_net(x_t, a_t, t)
                x_t = x_t + 0.5 * step_size * s_x + math.sqrt(step_size) * noise_x
                a_t = a_t + 0.5 * step_size * s_a + math.sqrt(step_size) * noise_a
                a_t = (a_t + a_t.transpose(1, 2)) / 2
                a_t.diagonal(dim1=1, dim2=2).zero_()

            # --- Predictor (reverse SDE) ---
            drift_x = 0.5 * beta.view(-1, 1, 1) * (x_t + score_x)
            drift_a = 0.5 * beta.view(-1, 1, 1) * (a_t + score_a)

            noise_x = torch.randn_like(x_t)
            noise_a = torch.randn_like(a_t)
            noise_a = (noise_a + noise_a.transpose(1, 2)) / 2
            noise_a.diagonal(dim1=1, dim2=2).zero_()

            diffusion = torch.sqrt(beta * dt).view(-1, 1, 1)
            x_t = x_t - drift_x * dt + diffusion * noise_x
            a_t = a_t - drift_a * dt + diffusion * noise_a
            a_t = (a_t + a_t.transpose(1, 2)) / 2
            a_t.diagonal(dim1=1, dim2=2).zero_()

        # Threshold adjacency
        graphs = []
        for b in range(num_graphs):
            adj = (a_t[b] > 0).float()
            adj = torch.triu(adj, diagonal=1)
            adj = adj + adj.t()
            active = adj.sum(1) > 0
            if active.sum() < 2:
                active[:2] = True
            idx = torch.where(active)[0]
            graphs.append(adj[idx][:, idx].cpu())

        return graphs


if __name__ == "__main__":
    torch.manual_seed(42)
    max_n, feat_dim = 8, 2

    print("=== GDSS Demo ===\n")

    # Training data
    train_x, train_a = [], []
    for _ in range(100):
        n = torch.randint(4, max_n + 1, (1,)).item()
        x = torch.randn(max_n, feat_dim)
        x[n:] = 0
        adj = torch.zeros(max_n, max_n)
        for i in range(n):
            for j in range(i + 1, n):
                if torch.rand(1) < 0.3:
                    adj[i, j] = adj[j, i] = 1
        train_x.append(x)
        train_a.append(adj)

    train_x = torch.stack(train_x)
    train_a = torch.stack(train_a)

    # Train
    model = GDSS(max_n, feat_dim, hidden_dim=64, num_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(50):
        model.train()
        idx = torch.randperm(100)[:32]
        result = model(train_x[idx], train_a[idx])
        optimizer.zero_grad()
        result["loss"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: loss={result['loss'].item():.4f} "
                  f"(x={result['loss_x'].item():.4f}, a={result['loss_a'].item():.4f})")

    # Generate
    print("\n=== Generation ===")
    generated = model.generate(num_graphs=8, num_steps=50, corrector_steps=1)
    for i, g in enumerate(generated):
        n, e = g.size(0), int(g.sum().item()) // 2
        print(f"Graph {i}: {n} nodes, {e} edges, density={2*e/(n*(n-1)) if n>1 else 0:.3f}")
