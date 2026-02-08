# GraphGAN: Adversarial Graph Generation

## Overview

GraphGAN applies the generative adversarial network framework to graph generation, training a generator to produce realistic graph structures while a discriminator learns to distinguish generated graphs from real ones. Unlike GraphVAE which optimizes a reconstruction objective, GraphGAN uses an implicit density model — the generator never explicitly computes $p_\theta(\mathcal{G})$, instead learning to transform noise into graph-like structures through adversarial feedback. This enables sharper, more realistic outputs but introduces the well-known challenges of GAN training: mode collapse, training instability, and difficulty evaluating convergence.

## Architecture

### Generator

The generator maps a latent vector $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ to a graph $\hat{\mathcal{G}} = (\hat{\mathbf{A}}, \hat{\mathbf{X}})$:

$$
\hat{\mathbf{A}}, \hat{\mathbf{X}} = G_\theta(\mathbf{z})
$$

The generator typically uses an MLP or factored architecture to produce continuous edge probabilities $\hat{\mathbf{A}} \in [0,1]^{n \times n}$. To obtain discrete graphs for the discriminator, the continuous output must be discretized. Three approaches are used:

**Hard thresholding.** $A_{ij} = \mathbf{1}[\hat{A}_{ij} > 0.5]$. Simple but non-differentiable — gradients cannot flow through the threshold.

**Straight-through estimator.** Use hard thresholding in the forward pass but pass gradients through as if the operation were the identity:

$$
A_{ij}^{\text{forward}} = \mathbf{1}[\hat{A}_{ij} > 0.5], \quad \frac{\partial \mathcal{L}}{\partial \hat{A}_{ij}} \approx \frac{\partial \mathcal{L}}{\partial A_{ij}^{\text{forward}}}
$$

**Gumbel-Softmax relaxation.** Replace discrete sampling with a continuous approximation using the Gumbel-Softmax trick:

$$
A_{ij} = \sigma\left(\frac{\log \hat{A}_{ij} - \log(1 - \hat{A}_{ij}) + g_1 - g_0}{\tau}\right)
$$

where $g_0, g_1 \sim \text{Gumbel}(0,1)$ and $\tau > 0$ is a temperature parameter. As $\tau \to 0$, the relaxation approaches discrete sampling.

### Discriminator

The discriminator $D_\phi(\mathcal{G})$ takes a graph and outputs a scalar probability of it being real. A GNN-based discriminator processes the graph through message-passing layers and aggregates to a graph-level score:

$$
D_\phi(\mathcal{G}) = \sigma\left(\text{MLP}\left(\text{READOUT}(\text{GNN}_\phi(\mathbf{A}, \mathbf{X}))\right)\right)
$$

The discriminator must be permutation-invariant — it should assign the same score to isomorphic graphs regardless of node ordering. Using a GNN with a permutation-invariant readout (sum, mean, or attention-based pooling) naturally ensures this.

## Training Objective

The standard GAN objective adapted for graphs:

$$
\min_\theta \max_\phi \; \mathbb{E}_{\mathcal{G} \sim p_{\text{data}}} [\log D_\phi(\mathcal{G})] + \mathbb{E}_{\mathbf{z} \sim p(\mathbf{z})} [\log(1 - D_\phi(G_\theta(\mathbf{z})))]
$$

In practice, the Wasserstein GAN (WGAN) objective with gradient penalty is preferred for stability:

$$
\min_\theta \max_\phi \; \mathbb{E}_{\mathcal{G} \sim p_{\text{data}}} [D_\phi(\mathcal{G})] - \mathbb{E}_{\mathbf{z} \sim p(\mathbf{z})} [D_\phi(G_\theta(\mathbf{z}))] - \lambda \mathbb{E}_{\hat{\mathcal{G}}} \left[(\|\nabla_{\hat{\mathcal{G}}} D_\phi(\hat{\mathcal{G}})\|_2 - 1)^2\right]
$$

where $\hat{\mathcal{G}}$ is interpolated between real and generated graphs and $\lambda$ controls the gradient penalty strength.

## Conditional Generation

GraphGAN can be conditioned on desired graph properties $\mathbf{c}$ (e.g., number of nodes, target degree distribution, domain-specific constraints):

$$
\hat{\mathcal{G}} = G_\theta(\mathbf{z}, \mathbf{c})
$$

The discriminator also receives the condition:

$$
D_\phi(\mathcal{G}, \mathbf{c}) \to [0,1]
$$

This enables targeted generation — for instance, generating financial networks with specified density or degree distribution.

## Challenges Specific to Graph GANs

**Discrete structure.** The fundamental challenge is that graphs are discrete objects, making gradient-based optimization of the generator difficult. The Gumbel-Softmax relaxation helps but introduces temperature-dependent approximation error.

**Permutation invariance.** The discriminator must be invariant to node ordering, which limits architectural choices. Using GNN-based discriminators is the standard solution.

**Mode collapse.** Graph GANs are prone to generating only a few graph types. This is especially problematic when the target distribution contains multiple distinct graph families (e.g., different community structures).

**Evaluation difficulty.** Unlike image GANs where visual inspection provides quick feedback, evaluating graph GAN quality requires computing statistical metrics (MMD, FGD) over large sample sets.

## Comparison with GraphVAE

| Aspect | GraphVAE | GraphGAN |
|--------|----------|----------|
| Training signal | Reconstruction + KL | Adversarial |
| Density | Explicit (tractable ELBO) | Implicit |
| Output quality | Blurry (averaged) | Sharper |
| Training stability | Stable | Requires careful tuning |
| Mode coverage | Good (KL prevents collapse) | Prone to mode collapse |
| Latent space | Structured (interpolable) | Unstructured |

## Finance Application: Adversarial Stress Testing

GraphGAN's adversarial framework can be repurposed for stress testing: train a generator to produce financial network configurations that are both plausible (pass the discriminator) and maximally stressful for a given risk model. This adversarial stress testing generates worst-case but realistic scenarios, providing stronger robustness guarantees than standard Monte Carlo stress tests.

## Implementation: GraphGAN with WGAN-GP

```python
"""
GraphGAN: adversarial graph generation with WGAN-GP training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad as torch_grad


class GraphGenerator(nn.Module):
    """Generator: noise -> graph adjacency matrix."""

    def __init__(
        self,
        latent_dim: int,
        max_nodes: int,
        hidden_dim: int = 256,
        temperature: float = 0.5,
    ):
        super().__init__()
        self.max_nodes = max_nodes
        self.temperature = temperature
        n_edges = max_nodes * (max_nodes - 1) // 2

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, n_edges),
        )

    def forward(self, z: torch.Tensor, hard: bool = False) -> torch.Tensor:
        """
        Args:
            z: (B, latent_dim) noise
            hard: if True, use Gumbel-Softmax with hard=True
        Returns:
            adj: (B, n, n) adjacency (soft or hard)
        """
        B = z.size(0)
        n = self.max_nodes
        logits = self.net(z)  # (B, n_edges)

        # Gumbel-Softmax relaxation for each edge
        if self.training:
            # Sample Gumbel noise
            u = torch.rand_like(logits).clamp(1e-8, 1 - 1e-8)
            gumbel = -torch.log(-torch.log(u))
            edge_probs = torch.sigmoid(
                (logits + gumbel) / self.temperature
            )
        else:
            edge_probs = torch.sigmoid(logits)

        if hard:
            edge_hard = (edge_probs > 0.5).float()
            edge_probs = edge_hard - edge_probs.detach() + edge_probs

        # Build symmetric adjacency
        adj = torch.zeros(B, n, n, device=z.device)
        idx = torch.triu_indices(n, n, offset=1)
        adj[:, idx[0], idx[1]] = edge_probs
        adj = adj + adj.transpose(1, 2)

        return adj


class GraphDiscriminator(nn.Module):
    """GNN-based discriminator for graphs."""

    def __init__(
        self,
        max_nodes: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
    ):
        super().__init__()
        self.max_nodes = max_nodes

        # Input: adjacency row as node feature
        self.input_proj = nn.Linear(max_nodes, hidden_dim)

        self.conv_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LeakyReLU(0.2),
                )
            )

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            adj: (B, n, n) adjacency matrix
        Returns:
            score: (B,) real/fake score
        """
        B = adj.size(0)

        # Use adjacency rows as initial node features
        h = self.input_proj(adj)  # (B, n, hidden)

        # Message passing
        for conv in self.conv_layers:
            h_msg = torch.bmm(adj, h) / (adj.sum(-1, keepdim=True) + 1)
            h = conv(h_msg) + h

        # Sum pooling (permutation invariant)
        h_graph = h.sum(dim=1)  # (B, hidden)
        score = self.readout(h_graph).squeeze(-1)  # (B,)

        return score


def gradient_penalty(
    discriminator: nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    lambda_gp: float = 10.0,
) -> torch.Tensor:
    """Compute WGAN gradient penalty."""
    B = real.size(0)
    alpha = torch.rand(B, 1, 1, device=real.device)
    interpolated = alpha * real + (1 - alpha) * fake
    interpolated.requires_grad_(True)

    d_interpolated = discriminator(interpolated)
    gradients = torch_grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradients = gradients.view(B, -1)
    penalty = lambda_gp * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return penalty


class GraphGAN:
    """WGAN-GP training wrapper for graph generation."""

    def __init__(
        self,
        max_nodes: int,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        lr_g: float = 1e-4,
        lr_d: float = 1e-4,
        n_critic: int = 5,
        lambda_gp: float = 10.0,
    ):
        self.latent_dim = latent_dim
        self.n_critic = n_critic
        self.lambda_gp = lambda_gp

        self.generator = GraphGenerator(latent_dim, max_nodes, hidden_dim)
        self.discriminator = GraphDiscriminator(max_nodes, hidden_dim)

        self.opt_g = torch.optim.Adam(
            self.generator.parameters(), lr=lr_g, betas=(0.5, 0.9)
        )
        self.opt_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr_d, betas=(0.5, 0.9)
        )

    def train_step(
        self, real_adj: torch.Tensor
    ) -> dict[str, float]:
        """Single training step (multiple critic updates + one generator update)."""
        B = real_adj.size(0)
        device = real_adj.device

        # --- Train Discriminator ---
        d_losses = []
        for _ in range(self.n_critic):
            z = torch.randn(B, self.latent_dim, device=device)
            fake_adj = self.generator(z).detach()

            d_real = self.discriminator(real_adj).mean()
            d_fake = self.discriminator(fake_adj).mean()
            gp = gradient_penalty(
                self.discriminator, real_adj, fake_adj, self.lambda_gp
            )

            d_loss = d_fake - d_real + gp

            self.opt_d.zero_grad()
            d_loss.backward()
            self.opt_d.step()
            d_losses.append(d_loss.item())

        # --- Train Generator ---
        z = torch.randn(B, self.latent_dim, device=device)
        fake_adj = self.generator(z)
        g_loss = -self.discriminator(fake_adj).mean()

        self.opt_g.zero_grad()
        g_loss.backward()
        self.opt_g.step()

        return {
            "d_loss": sum(d_losses) / len(d_losses),
            "g_loss": g_loss.item(),
            "wasserstein": (d_real - d_fake).item(),
        }

    @torch.no_grad()
    def generate(self, num_graphs: int = 1) -> list[torch.Tensor]:
        self.generator.eval()
        z = torch.randn(num_graphs, self.latent_dim)
        adj = self.generator(z, hard=True)

        graphs = []
        for b in range(num_graphs):
            # Remove isolated nodes
            degrees = adj[b].sum(dim=1)
            active = degrees > 0
            if active.sum() < 2:
                active[:2] = True
            active_idx = torch.where(active)[0]
            graphs.append(adj[b][active_idx][:, active_idx])

        self.generator.train()
        return graphs


if __name__ == "__main__":
    torch.manual_seed(42)

    max_n = 12

    # Create training data
    print("=== GraphGAN Demo ===\n")
    train_adjs = []
    for _ in range(200):
        n = torch.randint(6, max_n + 1, (1,)).item()
        adj = torch.zeros(max_n, max_n)
        # Community graph
        mid = n // 2
        for i in range(mid):
            for j in range(i + 1, mid):
                if torch.rand(1) < 0.5:
                    adj[i, j] = adj[j, i] = 1
        for i in range(mid, n):
            for j in range(i + 1, n):
                if torch.rand(1) < 0.5:
                    adj[i, j] = adj[j, i] = 1
        for i in range(mid):
            for j in range(mid, n):
                if torch.rand(1) < 0.1:
                    adj[i, j] = adj[j, i] = 1
        train_adjs.append(adj)

    train_adjs = torch.stack(train_adjs)

    # Train
    print("Training GraphGAN (WGAN-GP)...")
    gan = GraphGAN(
        max_nodes=max_n,
        latent_dim=32,
        hidden_dim=64,
        n_critic=3,
    )
    g_params = sum(p.numel() for p in gan.generator.parameters())
    d_params = sum(p.numel() for p in gan.discriminator.parameters())
    print(f"Generator params: {g_params:,}, Discriminator params: {d_params:,}")

    batch_size = 32
    for epoch in range(100):
        idx = torch.randperm(len(train_adjs))[:batch_size]
        metrics = gan.train_step(train_adjs[idx])

        if (epoch + 1) % 25 == 0:
            print(f"Epoch {epoch+1}: d_loss={metrics['d_loss']:.4f}, "
                  f"g_loss={metrics['g_loss']:.4f}, "
                  f"wasserstein={metrics['wasserstein']:.4f}")

    # Generate
    print("\n=== Generation ===")
    generated = gan.generate(num_graphs=10)
    for i, g in enumerate(generated):
        n = g.size(0)
        e = int(g.sum().item()) // 2
        density = 2 * e / (n * (n - 1)) if n > 1 else 0
        print(f"Graph {i}: {n} nodes, {e} edges, density={density:.3f}")

    # Compare statistics
    ref_densities = []
    for a in train_adjs:
        active = a.sum(1) > 0
        n = active.sum().item()
        e = a.sum().item() / 2
        ref_densities.append(2 * e / (n * (n - 1)) if n > 1 else 0)
    gen_densities = [2 * int(g.sum().item()) // 2 / (g.size(0) * (g.size(0) - 1))
                     if g.size(0) > 1 else 0 for g in generated]
    print(f"\nRef avg density: {sum(ref_densities)/len(ref_densities):.3f}")
    print(f"Gen avg density: {sum(gen_densities)/len(gen_densities):.3f}")
```
