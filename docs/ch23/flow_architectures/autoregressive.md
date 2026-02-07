# Autoregressive Flows

Autoregressive flows exploit the chain rule of probability to build normalizing flows with **triangular Jacobians**.  By making each output dimension depend only on previous dimensions, they achieve maximal per-layer expressiveness among flows with $O(d)$ Jacobian computation.  The two flagship architectures are **MAF** (fast density evaluation, slow sampling) and **IAF** (fast sampling, slow density evaluation), connected by a duality that determines which direction is parallelisable.

## The Autoregressive Factorisation

The chain rule decomposes any joint density exactly:

$$p(x) = \prod_{d=1}^{D} p(x_d \mid x_1, \ldots, x_{d-1})$$

An autoregressive model parameterises each conditional with a neural network.  For Gaussian conditionals:

$$p(x_d \mid x_{<d}) = \mathcal{N}\!\bigl(x_d;\;\mu_d(x_{<d}),\;\sigma_d^2(x_{<d})\bigr)$$

This is exact—no variational approximation—and naturally yields a triangular Jacobian because $x_d$ depends only on earlier dimensions.

## MADE: Efficient Autoregressive Networks

### The Problem

A naïve autoregressive model needs $D$ separate forward passes (one per dimension) to compute all conditionals.  This is impractical for large $D$.

### MADE's Solution

MADE (Germain et al., 2015) computes **all** conditionals in a single forward pass by masking the weight matrices of a standard MLP so that output $d$ cannot depend on inputs $d, d+1, \ldots, D$.

**Masking rule.** Assign each hidden unit a random integer $m \in \{1, \ldots, D-1\}$.  Connect input $j$ to hidden unit $k$ only if $m_k \ge j$, and connect hidden unit $k$ to output $d$ only if $d > m_k$.  This ensures the autoregressive property while allowing all outputs to be computed in parallel.

```python
import torch
import torch.nn as nn
import numpy as np


class MaskedLinear(nn.Module):
    """Linear layer with a binary mask applied to weights."""

    def __init__(self, in_features, out_features, mask):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.register_buffer("mask", mask)

    def forward(self, x):
        return nn.functional.linear(
            x, self.linear.weight * self.mask, self.linear.bias
        )


class MADE(nn.Module):
    """Masked Autoencoder for Distribution Estimation."""

    def __init__(self, dim, hidden_dims=(256, 256), n_params=2):
        super().__init__()
        self.dim = dim
        self.n_params = n_params     # e.g. 2 for (mu, log_sigma)

        # Assign ordering
        ms = [torch.arange(dim)]
        for h in hidden_dims:
            ms.append(torch.randint(1, dim, (h,)))
        ms.append(torch.arange(dim).repeat(n_params))

        # Build masked layers
        layers = []
        for i in range(len(ms) - 1):
            if i < len(ms) - 2:
                mask = (ms[i + 1].unsqueeze(1) >= ms[i].unsqueeze(0)).float()
            else:
                # Output: strict inequality for autoregressive property
                out_m = ms[-1][:dim].unsqueeze(1)  # only first copy
                mask = (out_m > ms[i].unsqueeze(0)).float()
                mask = mask.repeat(n_params, 1)
            layers.append(MaskedLinear(ms[i].shape[0], ms[i + 1].shape[0], mask))
            if i < len(ms) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """Return all conditional parameters in one pass."""
        out = self.net(x)
        return out.view(x.shape[0], self.n_params, self.dim)
```

## Masked Autoregressive Flow (MAF)

### From MADE to MAF

MAF (Papamakarios et al., 2017) uses MADE not as a density estimator but as a **flow layer**.  Given data $x$, MADE predicts per-dimension means $\mu_d(x_{<d})$ and scales $\sigma_d(x_{<d})$, and MAF applies the standardisation:

$$z_d = \frac{x_d - \mu_d(x_{<d})}{\sigma_d(x_{<d})}$$

### Properties

**Density evaluation (inverse direction).**  Given data $x$, all $(\mu_d, \sigma_d)$ are computed in one MADE forward pass, then each $z_d$ follows in parallel.  This is $O(1)$ in depth (fully parallelisable).

**Sampling (forward direction).**  To generate $x$ from $z$, we must compute sequentially: $x_d = \mu_d(x_{<d}) + \sigma_d(x_{<d}) \cdot z_d$.  Each step depends on the previous, so sampling is $O(D)$ sequential.

### Log-Determinant

The Jacobian is lower triangular with diagonal entries $1/\sigma_d$:

$$\log|\det J| = -\sum_{d=1}^{D}\log \sigma_d(x_{<d})$$

### Implementation

```python
class MAFLayer(nn.Module):
    """Single MAF layer using a MADE network."""

    def __init__(self, dim, hidden_dims=(256, 256)):
        super().__init__()
        self.dim = dim
        self.made = MADE(dim, hidden_dims, n_params=2)

    def forward(self, z):
        """z -> x (SLOW: sequential)."""
        x = torch.zeros_like(z)
        for d in range(self.dim):
            params = self.made(x)
            mu, log_sigma = params[:, 0, d], params[:, 1, d]
            x[:, d] = mu + z[:, d] * log_sigma.exp()
        log_det = log_sigma.sum(dim=-1)  # approximate; see note
        return x, log_det

    def inverse(self, x):
        """x -> z (FAST: parallel)."""
        params = self.made(x)            # single forward pass
        mu = params[:, 0]
        log_sigma = params[:, 1]
        z = (x - mu) / log_sigma.exp()
        log_det = -log_sigma.sum(dim=-1)
        return z, log_det
```

### Stacking MAF Layers

A single MAF layer has a fixed autoregressive ordering.  Stacking multiple layers with **permuted orderings** increases expressiveness:

```python
class MAF(nn.Module):
    def __init__(self, dim, n_layers=5, hidden_dims=(256, 256)):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(MAFLayer(dim, hidden_dims))
            # Reverse ordering between layers
            self.layers.append(Permutation(dim, mode="reverse"))

    def log_prob(self, x):
        log_det = torch.zeros(x.shape[0], device=x.device)
        z = x
        for layer in self.layers:
            z, ld = layer.inverse(z)
            log_det += ld
        log_pz = -0.5 * (z ** 2 + np.log(2 * np.pi)).sum(-1)
        return log_pz + log_det
```

## Inverse Autoregressive Flow (IAF)

### The Duality

IAF (Kingma et al., 2016) is the "transpose" of MAF.  Instead of standardising data, it transforms latent samples:

$$x_d = \mu_d(z_{<d}) + \sigma_d(z_{<d}) \cdot z_d$$

**Sampling** is now parallel (all $z_d$ are available), while **density evaluation** requires sequential computation.

| | MAF | IAF |
|---|---|---|
| Fast density evaluation | ✓ | ✗ (sequential) |
| Fast sampling | ✗ (sequential) | ✓ |
| Typical use | Density estimation | VAE posterior |

### When to Use Which

Use **MAF** when the primary task is density estimation, likelihood evaluation, or anomaly detection—all of which require fast density computation.  Use **IAF** when the primary task is sampling from an approximate posterior (e.g., in a VAE), where fast generation matters more.

### Implementation

```python
class IAFLayer(nn.Module):
    """Single IAF layer."""

    def __init__(self, dim, hidden_dims=(256, 256)):
        super().__init__()
        self.dim = dim
        self.made = MADE(dim, hidden_dims, n_params=2)

    def forward(self, z):
        """z -> x (FAST: parallel)."""
        params = self.made(z)
        mu = params[:, 0]
        log_sigma = params[:, 1]
        x = mu + z * log_sigma.exp()
        log_det = log_sigma.sum(dim=-1)
        return x, log_det

    def inverse(self, x):
        """x -> z (SLOW: sequential)."""
        z = torch.zeros_like(x)
        for d in range(self.dim):
            params = self.made(z)
            mu, log_sigma = params[:, 0, d], params[:, 1, d]
            z[:, d] = (x[:, d] - mu) / log_sigma.exp()
        log_det = -log_sigma.sum(dim=-1)
        return z, log_det
```

## Autoregressive Spline Flows

The affine transform ($x_d = \mu_d + \sigma_d z_d$) can be replaced with a monotonic spline, exactly as in Neural Spline Flows.  This combines the full triangular Jacobian of autoregressive models with the per-dimension nonlinear expressiveness of splines, yielding state-of-the-art density estimation on many benchmarks.

## Comparison: Coupling vs. Autoregressive

| Property | Coupling | Autoregressive |
|---|---|---|
| Forward | Parallel | MAF: sequential / IAF: parallel |
| Inverse | Parallel | MAF: parallel / IAF: sequential |
| Jacobian | Block triangular | Fully triangular |
| Per-layer expressiveness | Medium | High |
| Layers needed | More | Fewer |
| Conditioner complexity | Can be large | MADE (masked MLP) |

In practice, coupling flows (RealNVP, Glow) tend to be preferred for tasks that need both fast sampling *and* fast density evaluation, while MAF is preferred for pure density estimation.

## Finance Applications

For financial density estimation—modelling return distributions for risk measurement, scenario generation, or likelihood-based model selection—MAF is a natural choice because density evaluation is the bottleneck.  The autoregressive structure also maps well onto temporal financial data where sequential dependence is the norm.

## Key References

1. Germain, M., Gregor, K., Murray, I. & Larochelle, H. (2015). MADE: Masked Autoencoder for Distribution Estimation. *ICML*.
2. Papamakarios, G., Pavlakou, T. & Murray, I. (2017). Masked Autoregressive Flow for Density Estimation. *NeurIPS*.
3. Kingma, D. P., et al. (2016). Improved Variational Inference with Inverse Autoregressive Flow. *NeurIPS*.
4. Durkan, C., et al. (2019). Neural Spline Flows. *NeurIPS*.
