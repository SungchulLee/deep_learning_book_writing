# Volatility Modeling

## Overview

Autoregressive models can learn the dynamics of financial volatility without assuming a specific parametric form (GARCH, stochastic volatility). This data-driven approach captures complex volatility patterns including clustering, leverage effects, and regime dependence.

## Volatility as Conditional Variance

The conditional variance of returns is:

$$\sigma_t^2 = \text{Var}(r_t \mid r_{<t}) = \mathbb{E}[r_t^2 \mid r_{<t}] - (\mathbb{E}[r_t \mid r_{<t}])^2$$

An autoregressive model that outputs a full conditional distribution (e.g., mixture of Gaussians) implicitly models time-varying volatility.

## Mixture Density AR Model

Instead of predicting a single value, output parameters of a mixture distribution:

$$p(r_t \mid r_{<t}) = \sum_{k=1}^{K} \pi_k(r_{<t}) \cdot \mathcal{N}(r_t; \mu_k(r_{<t}), \sigma_k^2(r_{<t}))$$

The mixture components can capture different regimes (calm markets, volatile markets) with the mixing weights adapting to current conditions.

## Comparison with GARCH

| Feature | GARCH | Neural AR |
|---------|-------|-----------|
| Parametric form | Fixed (e.g., GARCH(1,1)) | Learned |
| Leverage effect | Requires extension (EGARCH) | Learned automatically |
| Regime dependence | Requires MS-GARCH | Learned automatically |
| Multi-asset | DCC, BEKK | Natural extension |
| Interpretability | High | Low |
| Data requirements | Low | Higher |

## Realized Volatility Forecasting

Alternatively, directly model realized volatility (computed from intraday data) as a time series:

$$\text{RV}_t = \sum_{i=1}^{M} r_{t,i}^2$$

Autoregressive models can predict $\text{RV}_{t+1}$ from past realized volatility, capturing the long-memory property of volatility.
