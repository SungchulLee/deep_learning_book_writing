# Score Matching

Score matching (Hyvärinen, 2005) trains an unnormalised model by matching the gradient of its log-density to the gradient of the data log-density, without ever computing the intractable partition function. This section derives the **explicit score matching (ESM)** objective, develops the practical loss via integration by parts, discusses its computational cost, introduces **sliced score matching** as a scalable alternative, and compares score matching with maximum likelihood estimation.

## The Fisher Divergence

The score matching objective minimises the **Fisher divergence** between the data distribution and the model:

$$D_F(p_{\text{data}} \| p_\theta) = \frac{1}{2} \, \mathbb{E}_{p_{\text{data}}}\!\left[\|\mathbf{s}_\theta(\mathbf{x}) - \mathbf{s}_{\text{data}}(\mathbf{x})\|^2\right]$$

where $\mathbf{s}_\theta = \nabla_{\mathbf{x}} \log p_\theta$ is the model score and $\mathbf{s}_{\text{data}} = \nabla_{\mathbf{x}} \log p_{\text{data}}$ is the data score.

The Fisher divergence is zero if and only if $\mathbf{s}_\theta = \mathbf{s}_{\text{data}}$ almost everywhere under $p_{\text{data}}$, which (under regularity conditions) implies $p_\theta = p_{\text{data}}$ up to normalisation.

### The Problem

We do not have access to $\mathbf{s}_{\text{data}}$—computing the gradient of the true data log-density is just as intractable as computing the density itself.

## Hyvärinen's Trick: Integration by Parts

Expanding the squared norm inside the Fisher divergence:

$$D_F = \frac{1}{2}\mathbb{E}_{p_{\text{data}}}\!\left[\|\mathbf{s}_\theta\|^2\right] - \mathbb{E}_{p_{\text{data}}}\!\left[\mathbf{s}_\theta^\top \mathbf{s}_{\text{data}}\right] + \frac{1}{2}\mathbb{E}_{p_{\text{data}}}\!\left[\|\mathbf{s}_{\text{data}}\|^2\right]$$

The third term is a constant with respect to $\theta$. The key insight is that the cross-term can be rewritten using integration by parts (equivalently, Stein's identity). Under mild boundary conditions:

$$\mathbb{E}_{p_{\text{data}}}\!\left[\mathbf{s}_\theta^\top \mathbf{s}_{\text{data}}\right] = -\mathbb{E}_{p_{\text{data}}}\!\left[\text{tr}\!\left(\nabla_{\mathbf{x}} \mathbf{s}_\theta(\mathbf{x})\right)\right]$$

Substituting back and dropping constants:

$$J_{\text{ESM}}(\theta) = \mathbb{E}_{p_{\text{data}}}\!\left[\frac{1}{2}\|\mathbf{s}_\theta(\mathbf{x})\|^2 + \text{tr}\!\left(\nabla_{\mathbf{x}} \mathbf{s}_\theta(\mathbf{x})\right)\right]$$

The trace of the Jacobian $\nabla_{\mathbf{x}} \mathbf{s}_\theta$ is the **Laplacian** of the log-density (equivalently, the trace of the Hessian of the energy). This objective depends only on $\mathbf{s}_\theta$ and its derivatives—$\mathbf{s}_{\text{data}}$ has been eliminated entirely.

### Regularity Conditions

The integration-by-parts step requires that $p_{\text{data}}(\mathbf{x}) \, \mathbf{s}_\theta(\mathbf{x}) \to \mathbf{0}$ as $\|\mathbf{x}\| \to \infty$ (the boundary term vanishes). This holds for distributions with sufficiently fast-decaying tails and smooth score networks.

### Consistency

Under the regularity conditions, minimising $J_{\text{ESM}}$ is equivalent to minimising the Fisher divergence. The global minimiser satisfies $\mathbf{s}_\theta = \mathbf{s}_{\text{data}}$, so the model learns the correct score.

## For Energy-Based Models

When the model is parameterised as $p_\theta(\mathbf{x}) \propto \exp(-E_\theta(\mathbf{x}))$, the score is $\mathbf{s}_\theta = -\nabla_{\mathbf{x}} E_\theta$ and the loss becomes:

$$J_{\text{ESM}}(\theta) = \mathbb{E}_{p_{\text{data}}}\!\left[\frac{1}{2}\|\nabla_{\mathbf{x}} E_\theta(\mathbf{x})\|^2 + \nabla_{\mathbf{x}}^2 E_\theta(\mathbf{x})\right]$$

where $\nabla_{\mathbf{x}}^2 E = \text{tr}(\mathbf{H}_E)$ is the Laplacian. No samples from the model are needed (unlike contrastive divergence), and no partition function appears.

## Computational Cost

The Laplacian $\text{tr}(\nabla_{\mathbf{x}} \mathbf{s}_\theta)$ requires computing the diagonal of the Jacobian. For an input of dimension $D$, this costs $D$ backward passes through the network (one per diagonal entry). For low-dimensional problems ($D \lesssim 50$) this is feasible; for images ($D > 10^4$) it is prohibitively expensive.

## Sliced Score Matching

Song et al. (2020) introduced **sliced score matching** to reduce the cost from $O(D)$ backward passes to $O(1)$. The idea is to project the score onto random directions $\mathbf{v}$:

$$J_{\text{SSM}}(\theta) = \mathbb{E}_{\mathbf{v}}\mathbb{E}_{p_{\text{data}}}\!\left[\frac{1}{2}(\mathbf{v}^\top \mathbf{s}_\theta)^2 + \mathbf{v}^\top \nabla_{\mathbf{x}} \mathbf{s}_\theta \, \mathbf{v}\right]$$

where $\mathbf{v}$ is drawn from a distribution over unit vectors (typically Rademacher or standard normal). The directional derivative $\mathbf{v}^\top \nabla_{\mathbf{x}} \mathbf{s}_\theta \, \mathbf{v}$ requires only a single Hessian-vector product, computable via one forward-over-backward pass. This makes the cost independent of $D$.

## Comparison with Maximum Likelihood

| Aspect | Maximum Likelihood | Score Matching |
|--------|-------------------|----------------|
| **Objective** | $\mathbb{E}[\log p_\theta(\mathbf{x})]$ | $\mathbb{E}[\frac{1}{2}\|\mathbf{s}_\theta\|^2 + \text{tr}(\nabla \mathbf{s}_\theta)]$ |
| **Partition function** | Required | Not required |
| **Model samples** | Often needed (CD, PCD) | Not needed |
| **Consistency** | Yes | Yes (under regularity) |
| **Efficiency** | $O(1)$ per sample (but $Z$ cost) | $O(D)$ per sample (ESM) or $O(1)$ (SSM) |
| **High dimensions** | Intractable $Z$ | ESM intractable; SSM or DSM needed |

## PyTorch Implementation

### Explicit Score Matching Loss

```python
import torch
import torch.nn as nn


class EnergyNetwork(nn.Module):
    """MLP energy function E(x) for score matching."""

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def score_matching_loss(energy_net: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Explicit score matching loss: ½ E[‖∇E‖²] + E[∇²E].

    Args:
        energy_net: Scalar energy network E(x).
        x: Data samples [batch, dim].

    Returns:
        Scalar loss.
    """
    x = x.requires_grad_(True)
    energy = energy_net(x)

    # Score = ∇_x E(x)
    score = torch.autograd.grad(
        energy.sum(), x, create_graph=True, retain_graph=True
    )[0]

    # ½ ‖∇E‖²
    score_norm_sq = (score**2).sum(dim=1)

    # Laplacian = tr(H_E) via D backward passes
    laplacian = torch.zeros(x.shape[0], device=x.device)
    for i in range(x.shape[1]):
        grad_i = torch.autograd.grad(
            score[:, i].sum(), x, create_graph=True, retain_graph=True
        )[0]
        laplacian += grad_i[:, i]

    return 0.5 * score_norm_sq.mean() + laplacian.mean()
```

### Sliced Score Matching Loss

```python
def sliced_score_matching_loss(
    energy_net: nn.Module, x: torch.Tensor, n_projections: int = 1
) -> torch.Tensor:
    """Sliced score matching loss (O(1) in dimension).

    Args:
        energy_net: Scalar energy network E(x).
        x: Data samples [batch, dim].
        n_projections: Number of random projection vectors.

    Returns:
        Scalar loss.
    """
    x = x.requires_grad_(True)
    energy = energy_net(x)

    score = torch.autograd.grad(
        energy.sum(), x, create_graph=True, retain_graph=True
    )[0]

    loss = torch.tensor(0.0, device=x.device)
    for _ in range(n_projections):
        # Random Rademacher direction
        v = torch.randint(0, 2, x.shape, device=x.device).float() * 2 - 1

        # v^T s_theta
        vs = (v * score).sum(dim=1)

        # v^T (∇_x s_theta) v  via Hessian-vector product
        hvp = torch.autograd.grad(
            vs.sum(), x, create_graph=True, retain_graph=True
        )[0]
        vhv = (v * hvp).sum(dim=1)

        loss = loss + (0.5 * vs**2 + vhv).mean()

    return loss / n_projections
```

### Training Example

```python
def train_esm(data: torch.Tensor, n_epochs: int = 500,
              batch_size: int = 128, lr: float = 1e-3):
    """Train an EBM with explicit score matching."""
    energy_net = EnergyNetwork(input_dim=data.shape[1], hidden_dim=64)
    optimizer = torch.optim.Adam(energy_net.parameters(), lr=lr)

    for epoch in range(n_epochs):
        perm = torch.randperm(len(data))
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(data), batch_size):
            batch = data[perm[i:i + batch_size]]
            optimizer.zero_grad()
            loss = score_matching_loss(energy_net, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}: loss = {epoch_loss / n_batches:.4f}")

    return energy_net
```

## Limitations and the Path to Denoising Score Matching

ESM is elegant but has two critical practical limitations. First, the $O(D)$ Hessian-trace cost (or the variance of sliced estimates) makes it expensive for high-dimensional data. Second, score estimates are unreliable in low-density regions where few training samples exist.

Both problems are solved by **Denoising Score Matching (DSM)**, which replaces the Hessian trace with a simple regression loss against a known target and fills low-density regions via noise perturbation. See [Denoising Score Matching](denoising_score_matching.md) for the full treatment.

## Exercises

1. **Integration by parts.** Starting from the Fisher divergence, carry out the integration-by-parts derivation step by step, verifying that the boundary term vanishes for Gaussian data.

2. **ESM vs SSM.** Compare explicit and sliced score matching on a 2-D Gaussian mixture. Measure loss convergence and wall-clock time.

3. **Dimension scaling.** Time the ESM loss for input dimensions $D \in \{2, 10, 50, 200\}$. Verify that cost scales linearly in $D$ and compare with SSM.

## References

1. Hyvärinen, A. (2005). "Estimation of Non-Normalized Statistical Models by Score Matching." *JMLR*.
2. Song, Y., et al. (2020). "Sliced Score Matching: A Scalable Approach to Density and Score Estimation." *UAI*.
3. Vincent, P. (2011). "A Connection Between Score Matching and Denoising Autoencoders." *Neural Computation*.
