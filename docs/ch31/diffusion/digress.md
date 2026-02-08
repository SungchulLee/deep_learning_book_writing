# DiGress: Discrete Denoising Diffusion for Graph Generation

## Overview

DiGress (Vignac et al., 2023) is a discrete diffusion model that generates graphs by denoising categorical node and edge types simultaneously. Unlike continuous relaxation approaches, DiGress operates natively on discrete graph structures using a categorical diffusion process. The model employs a graph transformer as the denoising network and introduces a noise model specifically designed for graphs — incorporating absorbing states and marginal distributions that improve generation quality. DiGress achieves state-of-the-art results on molecular and generic graph generation benchmarks.

## Discrete Diffusion on Graphs

A graph in DiGress is represented as $\mathcal{G} = (\mathbf{X}, \mathbf{E})$ where $\mathbf{X} \in \{0, \ldots, a\}^n$ are categorical node types (with $a$ classes plus one "absent" class) and $\mathbf{E} \in \{0, \ldots, b\}^{n \times n}$ are categorical edge types (with $b$ classes plus "no edge").

### Forward Process

Each node type $X_i$ and edge type $E_{ij}$ independently transitions according to a categorical Markov chain:

$$
q(X_i^{(t)} \mid X_i^{(t-1)}) = \text{Cat}(X_i^{(t)}; \mathbf{Q}_t^X \cdot \text{onehot}(X_i^{(t-1)}))
$$

$$
q(E_{ij}^{(t)} \mid E_{ij}^{(t-1)}) = \text{Cat}(E_{ij}^{(t)}; \mathbf{Q}_t^E \cdot \text{onehot}(E_{ij}^{(t-1)}))
$$

where $\mathbf{Q}_t^X \in \mathbb{R}^{(a+1) \times (a+1)}$ and $\mathbf{Q}_t^E \in \mathbb{R}^{(b+1) \times (b+1)}$ are transition matrices.

The $t$-step marginal is:

$$
q(\mathcal{G}_t \mid \mathcal{G}_0) = \prod_{i} \text{Cat}(X_i^{(t)}; \bar{\mathbf{Q}}_t^X \cdot \text{onehot}(X_i^{(0)})) \prod_{i<j} \text{Cat}(E_{ij}^{(t)}; \bar{\mathbf{Q}}_t^E \cdot \text{onehot}(E_{ij}^{(0)}))
$$

where $\bar{\mathbf{Q}}_t = \prod_{s=1}^{t} \mathbf{Q}_s$.

### Noise Models

DiGress proposes two noise models:

**Uniform noise.** Transitions uniformly corrupt categories toward a uniform distribution:

$$
\mathbf{Q}_t = (1 - \beta_t) \mathbf{I} + \beta_t \frac{\mathbf{1}\mathbf{1}^\top}{K}
$$

where $K$ is the number of categories. At $t = T$, all categories are equally likely.

**Marginal noise.** Transitions toward the marginal distribution of the training data:

$$
\mathbf{Q}_t = (1 - \beta_t) \mathbf{I} + \beta_t \mathbf{1} \mathbf{m}^\top
$$

where $\mathbf{m}$ is the empirical marginal probability over categories. This concentrates the prior at $t = T$ on the data marginals rather than the uniform distribution, making denoising easier since the noisy distribution at large $t$ already reflects the correct class frequencies.

### Reverse Process

The reverse step uses Bayes' rule to compute the posterior:

$$
q(\mathcal{G}_{t-1} \mid \mathcal{G}_t, \mathcal{G}_0) \propto q(\mathcal{G}_t \mid \mathcal{G}_{t-1}) \cdot q(\mathcal{G}_{t-1} \mid \mathcal{G}_0)
$$

The denoising network $\phi_\theta(\mathcal{G}_t, t)$ predicts a distribution over clean graphs $\hat{p}_\theta(\mathcal{G}_0 \mid \mathcal{G}_t)$, and the reverse transition is:

$$
p_\theta(\mathcal{G}_{t-1} \mid \mathcal{G}_t) = \sum_{\hat{\mathcal{G}}_0} q(\mathcal{G}_{t-1} \mid \mathcal{G}_t, \hat{\mathcal{G}}_0) \cdot \hat{p}_\theta(\hat{\mathcal{G}}_0 \mid \mathcal{G}_t)
$$

## Graph Transformer Denoiser

The denoising network is a graph transformer that processes the noisy graph and timestep to predict the clean graph. Each layer updates both node and edge representations:

**Node update:**
$$
\mathbf{h}_i^{(\ell+1)} = \mathbf{h}_i^{(\ell)} + \text{MHA}^{(\ell)}(\mathbf{h}_i^{(\ell)}, \{\mathbf{h}_j^{(\ell)}\}_{j \neq i}, \{\mathbf{e}_{ij}^{(\ell)}\}_{j \neq i})
$$

**Edge update:**
$$
\mathbf{e}_{ij}^{(\ell+1)} = \mathbf{e}_{ij}^{(\ell)} + \text{MLP}^{(\ell)}([\mathbf{h}_i^{(\ell+1)} \| \mathbf{h}_j^{(\ell+1)} \| \mathbf{e}_{ij}^{(\ell)}])
$$

The attention mechanism incorporates edge features into the attention weights:

$$
\alpha_{ij}^{(\ell)} = \frac{(\mathbf{W}_Q \mathbf{h}_i)^\top (\mathbf{W}_K \mathbf{h}_j) + (\mathbf{W}_E \mathbf{e}_{ij})^\top \mathbf{w}_a}{\sqrt{d_k}}
$$

The output heads predict logits over node and edge categories:

$$
\hat{p}(X_i^{(0)} \mid \mathcal{G}_t) = \text{softmax}(\text{MLP}_X(\mathbf{h}_i^{(L)}))
$$
$$
\hat{p}(E_{ij}^{(0)} \mid \mathcal{G}_t) = \text{softmax}(\text{MLP}_E(\mathbf{e}_{ij}^{(L)}))
$$

## Training Loss

The variational lower bound for discrete diffusion decomposes into per-step KL divergences:

$$
\mathcal{L}_{\text{VLB}} = \sum_{t=1}^{T} \mathbb{E}_{q(\mathcal{G}_t \mid \mathcal{G}_0)} \left[ D_{\text{KL}}(q(\mathcal{G}_{t-1} \mid \mathcal{G}_t, \mathcal{G}_0) \| p_\theta(\mathcal{G}_{t-1} \mid \mathcal{G}_t)) \right]
$$

In practice, DiGress uses a simpler cross-entropy loss between the predicted and true clean categories:

$$
\mathcal{L}_{\text{CE}} = \mathbb{E}_{t, \mathcal{G}_0, \mathcal{G}_t} \left[ -\sum_i \log \hat{p}_\theta(X_i^{(0)} \mid \mathcal{G}_t) - \sum_{i<j} \log \hat{p}_\theta(E_{ij}^{(0)} \mid \mathcal{G}_t) \right]
$$

## Implementation: DiGress Core Components

```python
"""
DiGress: Discrete denoising diffusion for graph generation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CategoricalDiffusion:
    """Categorical diffusion process for nodes and edges."""

    def __init__(
        self,
        num_node_classes: int,
        num_edge_classes: int,
        num_timesteps: int = 500,
        noise_type: str = "marginal",
        node_marginals: torch.Tensor = None,
        edge_marginals: torch.Tensor = None,
    ):
        self.T = num_timesteps
        self.num_node_cls = num_node_classes
        self.num_edge_cls = num_edge_classes
        self.noise_type = noise_type

        # Noise schedule (cosine)
        s = 0.008
        steps = torch.arange(num_timesteps + 1, dtype=torch.float64)
        alpha_bar = torch.cos((steps / num_timesteps + s) / (1 + s) * math.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = (1 - alpha_bar[1:] / alpha_bar[:-1]).clamp(max=0.999).float()

        # Build transition matrices
        self.Q_bar_X = []  # Cumulative transition for nodes
        self.Q_bar_E = []  # Cumulative transition for edges

        Q_bar_x = torch.eye(num_node_classes)
        Q_bar_e = torch.eye(num_edge_classes)

        if node_marginals is None:
            node_marginals = torch.ones(num_node_classes) / num_node_classes
        if edge_marginals is None:
            edge_marginals = torch.ones(num_edge_classes) / num_edge_classes

        for t in range(num_timesteps):
            beta = betas[t].item()
            if noise_type == "uniform":
                Qt_x = (1 - beta) * torch.eye(num_node_classes) + beta / num_node_classes
                Qt_e = (1 - beta) * torch.eye(num_edge_classes) + beta / num_edge_classes
            else:  # marginal
                Qt_x = (1 - beta) * torch.eye(num_node_classes) + beta * node_marginals.unsqueeze(0).expand(num_node_classes, -1)
                Qt_e = (1 - beta) * torch.eye(num_edge_classes) + beta * edge_marginals.unsqueeze(0).expand(num_edge_classes, -1)

            Q_bar_x = Q_bar_x @ Qt_x
            Q_bar_e = Q_bar_e @ Qt_e
            self.Q_bar_X.append(Q_bar_x.clone())
            self.Q_bar_E.append(Q_bar_e.clone())

    def q_sample(
        self,
        x_0: torch.Tensor,
        e_0: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample noisy graph at timestep t.
        
        Args:
            x_0: (B, n) integer node types
            e_0: (B, n, n) integer edge types
            t: (B,) timesteps
        """
        B, n = x_0.shape
        device = x_0.device

        x_t = torch.zeros_like(x_0)
        e_t = torch.zeros_like(e_0)

        for b in range(B):
            tb = t[b].item()
            Q_x = self.Q_bar_X[tb].to(device)
            Q_e = self.Q_bar_E[tb].to(device)

            # Sample noisy node types
            x_onehot = F.one_hot(x_0[b], self.num_node_cls).float()
            x_probs = x_onehot @ Q_x  # (n, num_node_cls)
            x_t[b] = torch.multinomial(x_probs, 1).squeeze(-1)

            # Sample noisy edge types
            e_onehot = F.one_hot(e_0[b], self.num_edge_cls).float()
            e_flat = e_onehot.view(-1, self.num_edge_cls)
            e_probs = e_flat @ Q_e
            e_t[b] = torch.multinomial(e_probs, 1).squeeze(-1).view(n, n)

        return x_t, e_t

    def posterior(
        self,
        x_t: torch.Tensor,
        x_0_pred: torch.Tensor,
        t: int,
    ) -> torch.Tensor:
        """
        Compute posterior q(x_{t-1} | x_t, x_0) for a single variable.
        
        Args:
            x_t: one-hot (K,)
            x_0_pred: predicted distribution (K,)
            t: current timestep
        """
        if t == 0:
            return x_0_pred

        Q_t = self.Q_bar_X[t] if t < len(self.Q_bar_X) else self.Q_bar_X[-1]
        Q_tm1 = self.Q_bar_X[t - 1] if t > 0 else torch.eye(self.num_node_cls)

        # q(x_t | x_0) and q(x_{t-1} | x_0) for each possible x_0
        prob_xt_given_x0 = x_0_pred @ Q_t  # (K,)
        prob_xtm1_given_x0 = x_0_pred @ Q_tm1  # (K,)

        # Bayes: q(x_{t-1} | x_t, x_0) ∝ q(x_t | x_{t-1}) q(x_{t-1} | x_0)
        posterior_unnorm = prob_xtm1_given_x0 * (Q_t[:, x_t.argmax()] + 1e-8)
        return posterior_unnorm / (posterior_unnorm.sum() + 1e-8)


class DiGressTransformerLayer(nn.Module):
    """Transformer layer that updates both node and edge representations."""

    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        head_dim = hidden_dim // num_heads

        self.W_Q = nn.Linear(hidden_dim, hidden_dim)
        self.W_K = nn.Linear(hidden_dim, hidden_dim)
        self.W_V = nn.Linear(hidden_dim, hidden_dim)
        self.W_E = nn.Linear(hidden_dim, num_heads)

        self.node_out = nn.Linear(hidden_dim, hidden_dim)
        self.edge_out = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.norm_n = nn.LayerNorm(hidden_dim)
        self.norm_e = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm_ffn = nn.LayerNorm(hidden_dim)

    def forward(self, h, e, t_emb):
        """
        Args:
            h: (B, n, d) node features
            e: (B, n, n, d) edge features
            t_emb: (B, d) time embedding
        """
        B, n, d = h.shape
        heads = self.num_heads
        hd = d // heads

        Q = self.W_Q(h).view(B, n, heads, hd)
        K = self.W_K(h).view(B, n, heads, hd)
        V = self.W_V(h).view(B, n, heads, hd)

        # Attention with edge bias
        attn = torch.einsum("bihd,bjhd->bijh", Q, K) / math.sqrt(hd)
        edge_bias = self.W_E(e)  # (B, n, n, heads)
        attn = attn + edge_bias
        attn = F.softmax(attn, dim=2)

        out = torch.einsum("bijh,bjhd->bihd", attn, V).reshape(B, n, d)
        h = self.norm_n(h + self.node_out(out) + t_emb.unsqueeze(1))
        h = self.norm_ffn(h + self.ffn(h))

        # Edge update
        e_input = torch.cat([
            h.unsqueeze(2).expand(-1, -1, n, -1),
            h.unsqueeze(1).expand(-1, n, -1, -1),
            e,
        ], dim=-1)
        e = self.norm_e(e + self.edge_out(e_input))

        return h, e


class DiGressDenoiser(nn.Module):
    """Full DiGress denoising network."""

    def __init__(
        self,
        num_node_classes: int,
        num_edge_classes: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        num_timesteps: int = 500,
    ):
        super().__init__()
        self.node_embed = nn.Embedding(num_node_classes, hidden_dim)
        self.edge_embed = nn.Embedding(num_edge_classes, hidden_dim)
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim),
        )

        self.layers = nn.ModuleList([
            DiGressTransformerLayer(hidden_dim, num_heads)
            for _ in range(num_layers)
        ])

        self.node_out = nn.Linear(hidden_dim, num_node_classes)
        self.edge_out = nn.Linear(hidden_dim, num_edge_classes)
        self.T = num_timesteps

    def forward(self, x_t, e_t, t):
        """
        Predict clean graph from noisy graph.
        
        Args:
            x_t: (B, n) noisy node types
            e_t: (B, n, n) noisy edge types
            t: (B,) timesteps
            
        Returns:
            x_logits: (B, n, num_node_cls)
            e_logits: (B, n, n, num_edge_cls)
        """
        h = self.node_embed(x_t)  # (B, n, d)
        e = self.edge_embed(e_t)  # (B, n, n, d)
        t_emb = self.time_embed(t.float().unsqueeze(-1) / self.T)

        for layer in self.layers:
            h, e = layer(h, e, t_emb)

        x_logits = self.node_out(h)
        e_logits = self.edge_out(e)
        # Symmetrize edge logits
        e_logits = (e_logits + e_logits.transpose(1, 2)) / 2

        return x_logits, e_logits


if __name__ == "__main__":
    torch.manual_seed(42)

    n_node_cls = 3  # e.g., C, N, O for molecules
    n_edge_cls = 3  # no-edge, single, double
    max_n = 8
    T = 100

    print("=== DiGress Demo ===\n")

    # Create synthetic training data
    B = 50
    x_0 = torch.randint(0, n_node_cls, (B, max_n))
    e_0 = torch.randint(0, n_edge_cls, (B, max_n, max_n))
    e_0 = torch.triu(e_0, diagonal=1)
    e_0 = e_0 + e_0.transpose(1, 2)

    # Categorical diffusion
    node_marg = torch.tensor([0.6, 0.3, 0.1])
    edge_marg = torch.tensor([0.7, 0.2, 0.1])
    diffusion = CategoricalDiffusion(
        n_node_cls, n_edge_cls, T,
        noise_type="marginal",
        node_marginals=node_marg,
        edge_marginals=edge_marg,
    )

    # Test forward process
    t = torch.randint(0, T, (B,))
    x_t, e_t = diffusion.q_sample(x_0, e_0, t)
    print(f"Clean nodes unique: {x_0[0].unique().tolist()}")
    print(f"Noisy nodes (t={t[0].item()}): {x_t[0].unique().tolist()}")

    # Test denoiser
    denoiser = DiGressDenoiser(
        n_node_cls, n_edge_cls,
        hidden_dim=64, num_layers=2, num_heads=4, num_timesteps=T,
    )
    params = sum(p.numel() for p in denoiser.parameters())
    print(f"\nDenoiser parameters: {params:,}")

    x_logits, e_logits = denoiser(x_t, e_t, t)
    print(f"Node logits: {x_logits.shape}")
    print(f"Edge logits: {e_logits.shape}")

    # Training loop
    print("\n=== Training ===")
    optimizer = torch.optim.Adam(denoiser.parameters(), lr=1e-3)

    for epoch in range(40):
        denoiser.train()
        t = torch.randint(0, T, (B,))
        x_t, e_t = diffusion.q_sample(x_0, e_0, t)

        x_logits, e_logits = denoiser(x_t, e_t, t)

        # Cross-entropy loss
        loss_x = F.cross_entropy(x_logits.view(-1, n_node_cls), x_0.view(-1))
        idx = torch.triu_indices(max_n, max_n, offset=1)
        loss_e = F.cross_entropy(
            e_logits[:, idx[0], idx[1]].reshape(-1, n_edge_cls),
            e_0[:, idx[0], idx[1]].reshape(-1),
        )
        loss = loss_x + loss_e

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: loss={loss.item():.4f} (node={loss_x.item():.4f}, edge={loss_e.item():.4f})")

    print("\nDone. DiGress training framework operational.")
```
