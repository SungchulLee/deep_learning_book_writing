# Scenario Generation with Diffusion Models

**Scenario generation** produces plausible future market states for risk management, stress testing, and portfolio optimisation. Diffusion models excel here because they can learn complex, multimodal distributions and generate diverse scenarios that respect empirical correlations and tail dependencies.

## Problem Formulation

Given historical observations $\{x_i\}_{i=1}^N$ of a $D$-dimensional market state (returns, volatilities, spreads, macro indicators), generate new samples $\hat{x} \sim p_{\text{data}}$ that are statistically indistinguishable from real market states.

For **conditional scenario generation**, produce samples from $p(x | c)$ where $c$ encodes a specific condition: a market regime, a stress event, a partial observation, or a macro-economic state.

## Why Diffusion Models for Scenarios?

| Requirement | Diffusion model capability |
|------------|---------------------------|
| Fat tails | Learned nonparametrically; no Gaussian assumption |
| Multimodality | Full distribution coverage (no mode collapse) |
| Cross-asset dependencies | Captured in the joint distribution |
| Conditional generation | Classifier-free guidance on regime/macro variables |
| Diversity | Stochastic sampling ensures varied scenarios |
| Density evaluation | Probability flow ODE provides log-likelihoods |

## Architecture for Market Scenarios

### Single-Step Scenarios

For generating cross-sectional snapshots (e.g., daily returns across 100 assets), a simple MLP-based denoising network suffices:

```python
import torch
import torch.nn as nn


class ScenarioGenerator(nn.Module):
    """MLP-based diffusion model for market scenario generation."""

    def __init__(
        self,
        market_dim: int,
        hidden_dim: int = 256,
        time_dim: int = 64,
        cond_dim: int = 0,
    ):
        super().__init__()
        input_dim = market_dim + time_dim + cond_dim

        self.time_embed = nn.Sequential(
            nn.Linear(time_dim, time_dim), nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.cond_proj = nn.Linear(cond_dim, cond_dim) if cond_dim > 0 else None

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, market_dim),
        )

    def forward(
        self, x: torch.Tensor, t: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        import numpy as np
        half = 32
        freqs = torch.exp(-np.log(10000) * torch.arange(half, device=t.device) / half)
        t_emb = torch.cat([(t.unsqueeze(-1) * freqs).cos(),
                           (t.unsqueeze(-1) * freqs).sin()], dim=-1)
        t_emb = self.time_embed(t_emb)

        parts = [x, t_emb]
        if cond is not None and self.cond_proj is not None:
            parts.append(self.cond_proj(cond))

        return self.net(torch.cat(parts, dim=-1))
```

### Conditional Scenarios via Classifier-Free Guidance

Train with random conditioning dropout (10–20% null conditioning), then at inference:

$$\hat{\epsilon} = \epsilon_\theta(x_t, t, \varnothing) + w \cdot \bigl(\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \varnothing)\bigr)$$

where $c$ encodes the target scenario condition (e.g., "recession", "rate hike", "oil shock").

## Applications

### Stress Testing

Generate scenarios conditioned on extreme states:

1. Define stress condition $c$ (e.g., VIX > 40, 10Y yield spike > 100bp)
2. Generate 10,000 conditional scenarios from $p(x|c)$
3. Evaluate portfolio P&L under each scenario
4. Report tail statistics (VaR, CVaR, maximum drawdown)

### Monte Carlo Simulation

Replace parametric assumptions in Monte Carlo engines:

**Traditional**: Assume returns $\sim$ multivariate normal or Student-$t$
**Diffusion-based**: Sample from the learned joint distribution, preserving nonlinear dependencies, regime effects, and realistic tail behaviour.

### Copula-Free Dependence Modelling

Diffusion models learn the full joint distribution directly, avoiding the two-step approach of fitting marginals then coupling them via a copula. This captures asymmetric tail dependence, time-varying correlations, and higher-order dependencies that copulas may miss.

## Validation Framework

| Test | Purpose | Method |
|------|---------|--------|
| Marginal distributions | Per-asset return quality | KS test, QQ plots |
| Correlation matrix | Cross-asset structure | Frobenius norm distance |
| Tail dependence | Joint extreme behaviour | Lower/upper tail dependence coefficients |
| Stylised facts | Volatility clustering, leverage effect | ACF of $|r|$, return-volatility correlation |
| Coverage test | VaR model accuracy | Kupiec and Christoffersen tests |

## References

1. Wiese, M., et al. (2020). "Quant GANs: Deep Generation of Financial Time Series." *Quantitative Finance*.
2. Cont, R. (2001). "Empirical Properties of Asset Returns: Stylized Facts and Statistical Issues." *Quantitative Finance*.
