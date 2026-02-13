# Time Series Forecasting

## Overview

Autoregressive models are natural fits for financial time series forecasting, as returns and prices unfold sequentially in time. The autoregressive framework provides probabilistic forecasts with exact density evaluation.

## Autoregressive Return Modeling

Model the conditional distribution of future returns given past returns:

$$p(r_{t+1} \mid r_t, r_{t-1}, \ldots, r_{t-L})$$

Unlike ARMA/GARCH models which assume parametric forms, neural AR models learn the conditional distribution non-parametrically.

```python
class FinancialARModel(nn.Module):
    def __init__(self, context_len=252, d_model=128, n_heads=4, n_layers=4):
        super().__init__()
        self.embedding = nn.Linear(1, d_model)
        self.pos_enc = nn.Embedding(context_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, 
                                                     dim_feedforward=512)
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        self.output = nn.Linear(d_model, 1)  # predict next return
    
    def forward(self, returns):
        # returns: (batch, seq_len, 1)
        x = self.embedding(returns)
        positions = torch.arange(returns.shape[1], device=returns.device)
        x = x + self.pos_enc(positions)
        
        mask = nn.Transformer.generate_square_subsequent_mask(x.shape[1])
        x = self.transformer(x.transpose(0, 1), mask=mask).transpose(0, 1)
        return self.output(x)
```

## Multi-Asset Modeling

Model joint return distributions across assets:

$$p(r_t^1, r_t^2, \ldots, r_t^N \mid r_{<t}) = \prod_{i=1}^{N} p(r_t^i \mid r_t^{<i}, r_{<t})$$

This captures both temporal dynamics and cross-sectional dependence.

## Probabilistic Forecasting

AR models naturally produce probabilistic forecasts by sampling multiple futures:

```python
def forecast_scenarios(model, history, n_scenarios=1000, horizon=21):
    scenarios = []
    for _ in range(n_scenarios):
        trajectory = history.clone()
        for t in range(horizon):
            pred_dist = model(trajectory)
            next_return = pred_dist.sample()
            trajectory = torch.cat([trajectory, next_return], dim=1)
        scenarios.append(trajectory[:, -horizon:])
    return torch.stack(scenarios)
```

## Advantages Over Traditional Models

- No assumption of specific distribution (Gaussian, Student-t)
- Automatically learns regime-dependent dynamics
- Captures non-linear dependencies
- Produces full conditional distributions, not just point forecasts
