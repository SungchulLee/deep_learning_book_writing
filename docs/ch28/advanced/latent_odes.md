# Latent ODEs for Time Series

## Overview

Latent ODEs (Rubanova et al., 2019) combine the Neural ODE framework with variational autoencoders for irregularly-sampled time series. They learn a latent trajectory that evolves continuously, allowing prediction at arbitrary time points.

## Architecture

```
Observations → RNN Encoder → z₀ ~ q(z₀|x) → ODE Solver → z(t) → Decoder → x̂(t)
```

1. **Encoder**: an RNN (run backward in time) maps observations to a distribution over initial latent states $q(z_0 \mid x_{1:T})$
2. **ODE dynamics**: $dz/dt = f_\theta(z(t))$ evolves the latent state continuously
3. **Decoder**: maps latent state to observation space at query times

## Key Advantage: Irregular Sampling

Unlike RNNs and standard time series models that require fixed time steps, Latent ODEs naturally handle irregular sampling:

- Observations can arrive at arbitrary times
- Missing data requires no imputation
- Predictions can be made at any continuous time point

This is critical for financial data where trading activity varies (weekends, holidays, variable-frequency tick data).

## Training

Trained with the ELBO:

$$\mathcal{L} = \mathbb{E}_{q(z_0|x)}\left[\sum_t \log p(x_t \mid z(t))\right] - D_{\text{KL}}(q(z_0 \mid x) \| p(z_0))$$

## ODE-RNN Variant

A simpler approach: use an ODE to model dynamics between observations, and an RNN update at each observation:

$$z_{t^-} = \text{ODESolve}(z_{t_{\text{prev}}}, t_{\text{prev}}, t)$$
$$z_t = \text{RNNCell}(z_{t^-}, x_t)$$

This hybrid approach is more computationally efficient and often performs comparably.

## Applications in Finance

- Modeling irregularly-spaced trade data
- Forecasting with missing observations (holidays, halts)
- Multi-frequency data fusion (daily + intraday)
- Continuous-time portfolio dynamics
