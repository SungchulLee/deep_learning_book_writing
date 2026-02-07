# Time Series Generation with Diffusion Models

Diffusion models can generate realistic financial time series that preserve the statistical properties—fat tails, volatility clustering, cross-asset correlations, and regime-dependent dynamics—that characterise real market data.

## Motivation

Traditional parametric models (GARCH, Heston, etc.) impose specific distributional assumptions that rarely capture the full complexity of financial returns. Diffusion models learn the data distribution nonparametrically, generating samples that match arbitrary marginals, temporal dependencies, and cross-sectional structures without specifying a closed-form model.

## Architecture Adaptations

### Temporal U-Net

For time series of length $L$ with $D$ features, the input is $x_0 \in \mathbb{R}^{L \times D}$. The standard 2D U-Net is replaced with a 1D temporal U-Net:

- **1D convolutions** along the time axis (instead of 2D spatial convolutions)
- **Temporal attention** captures long-range dependencies
- **Feature-wise conditioning** on market regime or macro variables

### Transformer-Based Architecture

Alternatively, treat the time series as a sequence of $L$ tokens of dimension $D$:

- Standard transformer encoder with causal or bidirectional attention
- Timestep conditioning via adaLN (as in DiT)
- Positional encoding preserves temporal ordering

## Training on Financial Data

### Data Preprocessing

1. **Normalisation**: Standardise returns to zero mean, unit variance per asset
2. **Windowing**: Extract overlapping windows of length $L$ from historical data
3. **Augmentation**: Time reversal, random cropping, noise injection

### Forward Process

The standard Gaussian forward process applies directly:

$$x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\, \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I)$$

where $x_0 \in \mathbb{R}^{L \times D}$ is a window of normalised returns.

### Conditional Generation

Condition on observable features for scenario-specific generation:

$$\epsilon_\theta(x_t, t, c) \qquad c \in \{\text{VIX level, yield curve slope, regime label, partial path}\}$$

This enables generating return paths conditional on specific market environments.

## PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalResBlock(nn.Module):
    """1D residual block with time conditioning."""

    def __init__(self, channels: int, time_dim: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.time_proj = nn.Linear(time_dim, channels)

    def forward(
        self, x: torch.Tensor, t_emb: torch.Tensor
    ) -> torch.Tensor:
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = h + self.time_proj(t_emb).unsqueeze(-1)
        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        return x + h


class TimeSeriesDiffusion(nn.Module):
    """Simplified 1D U-Net for time series diffusion."""

    def __init__(
        self,
        n_features: int,
        seq_len: int,
        hidden_channels: int = 64,
        time_dim: int = 128,
        n_cond: int = 0,
    ):
        super().__init__()
        self.input_proj = nn.Conv1d(n_features, hidden_channels, 1)

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Optional conditioning projection
        self.cond_proj = (
            nn.Linear(n_cond, time_dim) if n_cond > 0 else None
        )

        # Encoder
        self.enc1 = TemporalResBlock(hidden_channels, time_dim)
        self.enc2 = TemporalResBlock(hidden_channels, time_dim)
        self.down = nn.Conv1d(hidden_channels, hidden_channels, 2, stride=2)

        # Bottleneck
        self.mid = TemporalResBlock(hidden_channels, time_dim)

        # Decoder
        self.up = nn.ConvTranspose1d(hidden_channels, hidden_channels, 2, stride=2)
        self.dec1 = TemporalResBlock(hidden_channels * 2, time_dim)
        self.dec2 = TemporalResBlock(hidden_channels * 2, time_dim)

        self.output_proj = nn.Conv1d(hidden_channels * 2, n_features, 1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Noisy time series [B, D, L] (channels-first).
            t: Timestep indices [B].
            cond: Optional conditioning [B, n_cond].

        Returns:
            Predicted noise [B, D, L].
        """
        # Time embedding
        t_emb = sinusoidal_embedding(t, self.time_mlp[0].in_features)
        t_emb = self.time_mlp(t_emb)

        if cond is not None and self.cond_proj is not None:
            t_emb = t_emb + self.cond_proj(cond)

        # Encode
        h = self.input_proj(x)
        h1 = self.enc1(h, t_emb)
        h2 = self.enc2(h1, t_emb)
        h_down = self.down(h2)

        # Bottleneck
        h_mid = self.mid(h_down, t_emb)

        # Decode
        h_up = self.up(h_mid)
        # Handle size mismatch from odd sequence lengths
        if h_up.shape[-1] != h2.shape[-1]:
            h_up = F.pad(h_up, (0, h2.shape[-1] - h_up.shape[-1]))
        h = self.dec1(torch.cat([h_up, h2], dim=1), t_emb)
        h = self.dec2(torch.cat([h[:, : h1.shape[1]], h1], dim=1), t_emb)

        return self.output_proj(h)


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    import numpy as np
    half = dim // 2
    freqs = torch.exp(
        -np.log(10000) * torch.arange(half, device=t.device) / half
    )
    args = t.float().unsqueeze(-1) * freqs
    return torch.cat([args.cos(), args.sin()], dim=-1)
```

## Evaluation Metrics for Financial Time Series

Beyond FID (which applies to any distribution), financial time series have domain-specific metrics:

| Metric | What it measures |
|--------|-----------------|
| Marginal distribution (KS test) | Match of return distributions |
| Autocorrelation of returns | Absence of linear predictability |
| Autocorrelation of $|r|$ and $r^2$ | Volatility clustering |
| Cross-asset correlation matrix | Correlation structure |
| Maximum drawdown distribution | Tail risk properties |
| Hurst exponent | Long-range dependence |

## Applications

**Risk model backtesting.** Generate thousands of synthetic return paths to stress-test VaR and CVaR models under diverse market conditions.

**Portfolio optimisation.** Use generated scenarios for robust portfolio construction that accounts for fat tails and regime changes.

**Regulatory stress testing.** Produce adverse scenarios conditioned on specific macroeconomic states.

## References

1. Coletta, A., et al. (2023). "On the Constrained Time-Series Generation Problem." *NeurIPS*.
2. Lim, H., et al. (2023). "Regular Time-Series Generation using SGM." *arXiv*.
