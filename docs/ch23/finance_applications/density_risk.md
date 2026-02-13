# Density Estimation for Risk

## Overview

Normalizing flows provide exact density estimation for high-dimensional financial data, enabling accurate risk assessment through full distributional modeling rather than point estimates.

## Portfolio Return Distributions

Flows can model the joint distribution of asset returns, capturing non-Gaussian features:

- **Fat tails**: extreme returns are more frequent than Gaussian models predict
- **Skewness**: asymmetric return distributions
- **Non-linear dependence**: tail dependence, asymmetric correlations

```python
class PortfolioFlow(nn.Module):
    def __init__(self, n_assets, n_layers=8):
        super().__init__()
        self.flow = RealNVP(dim=n_assets, n_layers=n_layers)
    
    def log_prob(self, returns):
        z, log_det = self.flow.inverse(returns)
        log_p = torch.distributions.Normal(0, 1).log_prob(z).sum(-1)
        return log_p + log_det
    
    def sample(self, n_samples):
        z = torch.randn(n_samples, self.flow.dim)
        return self.flow.forward(z)
```

## VaR and CVaR Estimation

Given the learned density, compute risk measures via sampling:

$$\text{VaR}_\alpha = F^{-1}(\alpha), \quad \text{CVaR}_\alpha = \mathbb{E}[L \mid L \geq \text{VaR}_\alpha]$$

Flows enable exact density evaluation for importance sampling-based estimation of tail risk measures.

## Advantages Over Parametric Models

Traditional risk models assume specific parametric forms (Gaussian, Student-t, Clayton copula). Flows learn the distributional form from data, automatically capturing complex dependence structures without specifying the parametric family.

## Challenges

- **Non-stationarity**: financial distributions change over time; the flow must be retrained or adapted
- **Tail estimation**: flows trained on limited historical data may not capture extreme tail behavior beyond the training range
- **Curse of dimensionality**: modeling hundreds of assets jointly requires large datasets
