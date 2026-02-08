# Graph Diffusion Models

## Overview

Diffusion models for graphs extend the denoising diffusion framework to generate graph-structured data. The core idea is to define a forward noising process that gradually corrupts a graph into random noise, then train a neural network to reverse this process step by step. Unlike autoregressive or one-shot methods, diffusion models generate graphs through iterative refinement — starting from pure noise and progressively denoising into a valid graph structure. This paradigm naturally handles permutation invariance and produces high-quality samples, establishing state-of-the-art results on molecular and generic graph generation benchmarks.

## Challenges of Diffusion on Graphs

Adapting diffusion to graphs introduces unique challenges beyond those encountered in continuous domains (images, audio):

**Discrete structure.** Adjacency matrices are binary — edges either exist or don't. Standard Gaussian diffusion assumes continuous data, requiring either discrete diffusion formulations or continuous relaxations of the graph.

**Joint generation.** A graph consists of multiple coupled components — node types $\mathbf{X}$, edge types $\mathbf{E}$, and possibly edge weights — that must be generated coherently. The diffusion process must corrupt and denoise these components jointly.

**Permutation equivariance.** The denoising network $\epsilon_\theta(\mathcal{G}_t, t)$ must be equivariant to node permutations: permuting the noisy input must produce a correspondingly permuted prediction. GNN-based denoisers naturally satisfy this requirement.

## Continuous Relaxation Approach

One strategy relaxes the discrete adjacency matrix to continuous values and applies standard Gaussian diffusion.

**Forward process.** Starting from a graph $\mathcal{G}_0 = (\mathbf{A}_0, \mathbf{X}_0)$, add Gaussian noise over $T$ steps:

$$
q(\mathcal{G}_t \mid \mathcal{G}_{t-1}) = \mathcal{N}(\mathcal{G}_t; \sqrt{1 - \beta_t} \mathcal{G}_{t-1}, \beta_t \mathbf{I})
$$

where $\beta_t$ is the noise schedule. The closed-form marginal is:

$$
q(\mathcal{G}_t \mid \mathcal{G}_0) = \mathcal{N}(\mathcal{G}_t; \sqrt{\bar{\alpha}_t} \mathcal{G}_0, (1 - \bar{\alpha}_t)\mathbf{I})
$$

with $\bar{\alpha}_t = \prod_{s=1}^{t} (1 - \beta_s)$.

**Reverse process.** A neural network predicts the clean graph from the noisy version:

$$
p_\theta(\mathcal{G}_{t-1} \mid \mathcal{G}_t) = \mathcal{N}(\mathcal{G}_{t-1}; \boldsymbol{\mu}_\theta(\mathcal{G}_t, t), \sigma_t^2 \mathbf{I})
$$

**Post-processing.** After denoising to $\mathcal{G}_0$, threshold the continuous adjacency values to obtain a discrete graph: $A_{ij} = \mathbf{1}[\hat{A}_{ij} > 0.5]$.

## Discrete Diffusion Approach

An alternative operates directly on discrete graph structures using categorical diffusion.

**Forward process.** Each discrete variable (edge presence, node type) transitions according to a Markov chain. For a binary edge variable:

$$
q(A_{ij}^{(t)} \mid A_{ij}^{(t-1)}) = \text{Cat}(A_{ij}^{(t)}; \mathbf{Q}_t \cdot \text{onehot}(A_{ij}^{(t-1)}))
$$

where $\mathbf{Q}_t \in \mathbb{R}^{2 \times 2}$ is the transition matrix. A common choice:

$$
\mathbf{Q}_t = \begin{pmatrix} 1 - \beta_t & \beta_t \\ \beta_t & 1 - \beta_t \end{pmatrix}
$$

This uniformly corrupts edge states toward a uniform distribution over $\{0, 1\}$.

The $t$-step marginal is:

$$
q(A_{ij}^{(t)} \mid A_{ij}^{(0)}) = \text{Cat}(A_{ij}^{(t)}; \bar{\mathbf{Q}}_t \cdot \text{onehot}(A_{ij}^{(0)}))
$$

where $\bar{\mathbf{Q}}_t = \prod_{s=1}^{t} \mathbf{Q}_s$.

**Reverse process.** The denoising network predicts the clean graph $\hat{\mathcal{G}}_0$ from the noisy graph $\mathcal{G}_t$:

$$
p_\theta(\mathcal{G}_{t-1} \mid \mathcal{G}_t) = \sum_{\hat{\mathcal{G}}_0} q(\mathcal{G}_{t-1} \mid \mathcal{G}_t, \hat{\mathcal{G}}_0) \cdot p_\theta(\hat{\mathcal{G}}_0 \mid \mathcal{G}_t)
$$

The posterior $q(\mathcal{G}_{t-1} \mid \mathcal{G}_t, \mathcal{G}_0)$ is computed in closed form using Bayes' rule.

## Denoising Network Architecture

The denoiser must process a noisy graph $\mathcal{G}_t$ and timestep $t$ to predict either the noise $\boldsymbol{\epsilon}$, the clean graph $\mathcal{G}_0$, or the score $\nabla_{\mathcal{G}_t} \log q(\mathcal{G}_t)$. A graph transformer architecture is the standard choice:

$$
\mathbf{H}^{(\ell+1)} = \text{GraphTransformerBlock}^{(\ell)}(\mathbf{H}^{(\ell)}, \mathbf{E}^{(\ell)}, t)
$$

Key design choices:
- **Node features**: Concatenate noisy node types with sinusoidal time embeddings
- **Edge features**: Noisy edge types processed as edge attributes in attention
- **Global conditioning**: Time embedding added to all layers via FiLM conditioning or additive bias
- **Output heads**: Separate MLPs predict denoised node types and edge types

## Noise Schedules for Graphs

The noise schedule $\beta_1, \ldots, \beta_T$ controls the rate of corruption. For graphs:

- **Linear schedule**: $\beta_t$ increases linearly from $\beta_1 = 10^{-4}$ to $\beta_T = 0.02$
- **Cosine schedule**: $\bar{\alpha}_t = \cos^2\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)$ for smoother transitions
- **Polynomial schedule**: $\bar{\alpha}_t = (1 - t/T)^s$ with exponent $s$ tuned per dataset

For discrete diffusion, the schedule controls how quickly categorical variables approach the uniform distribution. Faster schedules improve training efficiency but may make the denoising task too difficult.

## Finance Application: Diffusion-Based Network Scenarios

Diffusion models can generate diverse, high-quality financial network scenarios by: (1) training on historical network snapshots, (2) sampling from the learned distribution to produce plausible alternative configurations, and (3) interpolating between existing networks via the diffusion latent space. The iterative refinement process naturally produces globally coherent structures, avoiding the local inconsistencies common in autoregressive and one-shot methods.

## Implementation: Graph Diffusion Framework

```python
"""
Graph diffusion model: continuous and discrete forward/reverse processes.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class ContinuousGraphDiffusion(nn.Module):
    """
    Gaussian diffusion on continuous-relaxed adjacency matrices.
    """

    def __init__(
        self,
        max_nodes: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_timesteps: int = 100,
        schedule: str = "cosine",
    ):
        super().__init__()
        self.max_nodes = max_nodes
        self.T = num_timesteps

        # Noise schedule
        if schedule == "cosine":
            s = 0.008
            steps = torch.arange(num_timesteps + 1, dtype=torch.float64)
            alpha_bar = torch.cos((steps / num_timesteps + s) / (1 + s) * math.pi / 2) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]
            betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
            betas = betas.clamp(max=0.999).float()
        else:
            betas = torch.linspace(1e-4, 0.02, num_timesteps)

        alphas = 1 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1 - alpha_bar))

        # Denoiser: simple GNN
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.input_proj = nn.Linear(max_nodes, hidden_dim)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                "msg": nn.Linear(hidden_dim, hidden_dim),
                "update": nn.Linear(hidden_dim * 2, hidden_dim),
                "norm": nn.LayerNorm(hidden_dim),
                "time_proj": nn.Linear(hidden_dim, hidden_dim),
            }))

        # Output: predict noise on upper-triangular adjacency
        n_edges = max_nodes * (max_nodes - 1) // 2
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_edges),
        )

    def q_sample(
        self,
        adj_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward process: add noise to clean adjacency."""
        if noise is None:
            noise = torch.randn_like(adj_0)

        sqrt_ab = self.sqrt_alpha_bar[t].view(-1, 1, 1)
        sqrt_omab = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1)

        adj_t = sqrt_ab * adj_0 + sqrt_omab * noise
        return adj_t, noise

    def _denoise(
        self,
        adj_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Predict noise from noisy adjacency."""
        B = adj_t.size(0)

        # Time embedding
        t_emb = self.time_embed(t.float().unsqueeze(-1) / self.T)

        # Use adjacency rows as node features
        h = self.input_proj(adj_t)  # (B, n, hidden)

        for layer in self.layers:
            # Message passing with adjacency
            adj_norm = adj_t / (adj_t.abs().sum(-1, keepdim=True) + 1)
            msg = torch.bmm(adj_norm, layer["msg"](h))

            # Update with time conditioning
            t_bias = layer["time_proj"](t_emb).unsqueeze(1)
            h_new = layer["update"](torch.cat([h, msg], dim=-1)) + t_bias
            h = layer["norm"](h_new + h)

        # Predict noise for upper-triangular edges
        h_graph = h.sum(dim=1)  # (B, hidden)
        noise_pred = self.output_proj(h_graph)  # (B, n_edges)

        # Reconstruct full noise matrix
        n = self.max_nodes
        noise_matrix = torch.zeros(B, n, n, device=adj_t.device)
        idx = torch.triu_indices(n, n, offset=1)
        noise_matrix[:, idx[0], idx[1]] = noise_pred
        noise_matrix = noise_matrix + noise_matrix.transpose(1, 2)

        return noise_matrix

    def forward(self, adj_0: torch.Tensor) -> dict[str, torch.Tensor]:
        """Training step: sample t, add noise, predict noise."""
        B = adj_0.size(0)
        device = adj_0.device

        # Sample random timesteps
        t = torch.randint(0, self.T, (B,), device=device)

        # Add noise
        noise = torch.randn_like(adj_0)
        adj_t, _ = self.q_sample(adj_0, t, noise)

        # Predict noise
        noise_pred = self._denoise(adj_t, t)

        # MSE loss on upper triangle
        n = self.max_nodes
        idx = torch.triu_indices(n, n, offset=1)
        loss = F.mse_loss(
            noise_pred[:, idx[0], idx[1]],
            noise[:, idx[0], idx[1]],
        )

        return {"loss": loss}

    @torch.no_grad()
    def generate(
        self,
        num_graphs: int = 1,
        device: str = "cpu",
    ) -> list[torch.Tensor]:
        """Generate graphs via reverse diffusion."""
        self.eval()
        n = self.max_nodes

        # Start from pure noise
        adj_t = torch.randn(num_graphs, n, n, device=device)
        adj_t = (adj_t + adj_t.transpose(1, 2)) / 2  # Symmetrize
        adj_t.diagonal(dim1=1, dim2=2).zero_()

        for t_idx in reversed(range(self.T)):
            t = torch.full((num_graphs,), t_idx, device=device, dtype=torch.long)

            # Predict noise
            noise_pred = self._denoise(adj_t, t)

            # DDPM update
            alpha = self.alphas[t_idx]
            alpha_bar = self.alpha_bar[t_idx]
            beta = self.betas[t_idx]

            mean = (1 / alpha.sqrt()) * (
                adj_t - (beta / (1 - alpha_bar).sqrt()) * noise_pred
            )

            if t_idx > 0:
                noise = torch.randn_like(adj_t)
                noise = (noise + noise.transpose(1, 2)) / 2
                noise.diagonal(dim1=1, dim2=2).zero_()
                adj_t = mean + beta.sqrt() * noise
            else:
                adj_t = mean

        # Threshold to binary
        graphs = []
        for b in range(num_graphs):
            adj_b = (adj_t[b] > 0).float()
            adj_b = torch.triu(adj_b, diagonal=1)
            adj_b = adj_b + adj_b.t()
            # Remove isolated nodes
            active = adj_b.sum(1) > 0
            if active.sum() < 2:
                active[:2] = True
            idx = torch.where(active)[0]
            graphs.append(adj_b[idx][:, idx].cpu())

        return graphs


if __name__ == "__main__":
    torch.manual_seed(42)
    max_n = 10
    T = 50

    # Training data
    print("=== Graph Diffusion Demo ===\n")
    train_adjs = []
    for _ in range(200):
        n = torch.randint(5, max_n + 1, (1,)).item()
        adj = torch.zeros(max_n, max_n)
        for i in range(n):
            for j in range(i + 1, n):
                if torch.rand(1) < 0.25:
                    adj[i, j] = adj[j, i] = 1
        train_adjs.append(adj)
    train_adjs = torch.stack(train_adjs)

    # Train
    model = ContinuousGraphDiffusion(
        max_nodes=max_n, hidden_dim=64,
        num_layers=3, num_timesteps=T,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(60):
        model.train()
        idx = torch.randperm(len(train_adjs))[:32]
        result = model(train_adjs[idx])
        optimizer.zero_grad()
        result["loss"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}: loss={result['loss'].item():.4f}")

    # Generate
    print("\n=== Generation ===")
    generated = model.generate(num_graphs=8)
    for i, g in enumerate(generated):
        n, e = g.size(0), int(g.sum().item()) // 2
        print(f"Graph {i}: {n} nodes, {e} edges, density={2*e/(n*(n-1)) if n>1 else 0:.3f}")
```
