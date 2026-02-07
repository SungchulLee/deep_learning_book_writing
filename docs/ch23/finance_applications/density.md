# Finance Applications of Normalizing Flows

Normalizing flows are particularly well-suited to quantitative finance, where exact density evaluation, non-parametric distributional modelling, and fast scenario generation are all critical.  This section develops five core applications: return distribution modelling, risk measurement, option pricing, portfolio optimisation, and scenario generation.

## Return Distribution Modelling

### Why Flows for Financial Distributions?

Parametric models (Gaussian, Student-t, GARCH) assume a fixed distributional form and often misspecify tails and asymmetries.  Historical simulation is limited to observed scenarios.  Kernel density estimation suffers from the curse of dimensionality.

Normalizing flows learn **flexible, non-parametric** return distributions directly from data while providing exact density evaluation and fast sampling.

| Capability | Flows | Parametric | Historical | KDE |
|---|---|---|---|---|
| Exact density | ✓ | ✓ | ✗ | ≈ |
| Multivariate | ✓ | Limited | ✓ | Poor |
| Tail modelling | ✓ | If specified | Limited | Poor |
| Generative | ✓ | ✓ | ✗ | ✗ |

### Univariate Return Modelling

```python
import torch
import torch.nn as nn
import numpy as np


class ReturnFlowModel:
    """Flow-based return distribution model."""

    def __init__(self, flow_model):
        self.flow = flow_model

    def fit(self, returns, epochs=200, lr=1e-3):
        """Train on historical returns."""
        data = torch.tensor(returns, dtype=torch.float32).unsqueeze(1)
        optimiser = torch.optim.Adam(self.flow.parameters(), lr=lr)
        for epoch in range(epochs):
            loss = -self.flow.log_prob(data).mean()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

    def pdf(self, x):
        """Evaluate learned density at points x."""
        x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
        with torch.no_grad():
            return self.flow.log_prob(x_t).exp().numpy()

    def sample(self, n):
        """Generate synthetic returns."""
        with torch.no_grad():
            return self.flow.sample(n).squeeze(1).numpy()
```

### Multivariate Joint Distributions

For multi-asset modelling, flows learn the full joint distribution $p(r_1, r_2, \ldots, r_d)$, capturing non-linear dependencies and tail correlations that Gaussian copulas miss.  This is critical for portfolio risk: correlations between assets tend to increase during market stress, a phenomenon that flows can learn from data.

### Conditional Distributions

By conditioning the flow on regime indicators, macroeconomic variables, or time features, one obtains **conditional density estimation**:

$$p(r_t \mid \text{VIX}_t, \text{regime}_t) = p_Z(z)\;\left|\det J\right|^{-1}$$

where the conditioning information enters through the conditioner networks in coupling or autoregressive layers.

## Risk Measurement

### Value at Risk (VaR)

VaR at confidence level $\alpha$ is the loss threshold exceeded with probability $1 - \alpha$:

$$\text{VaR}_\alpha = -F^{-1}(1 - \alpha)$$

With a trained flow, VaR is estimated by sampling:

```python
class FlowVaR:
    def __init__(self, flow, n_samples=100_000):
        self.flow = flow
        self.n_samples = n_samples

    def compute(self, alpha=0.99):
        with torch.no_grad():
            samples = self.flow.sample(self.n_samples).squeeze().numpy()
        return -np.percentile(samples, (1 - alpha) * 100)
```

### Conditional Value at Risk (CVaR / Expected Shortfall)

CVaR is the expected loss given that VaR is exceeded:

$$\text{CVaR}_\alpha = -\mathbb{E}\bigl[R \mid R \le -\text{VaR}_\alpha\bigr]$$

```python
    def compute_cvar(self, alpha=0.99):
        with torch.no_grad():
            samples = self.flow.sample(self.n_samples).squeeze().numpy()
        var = -np.percentile(samples, (1 - alpha) * 100)
        tail = samples[samples <= -var]
        return -tail.mean() if len(tail) > 0 else var
```

### Tail Risk Analysis

Because flows learn the full density, tail probabilities can be computed directly:

$$P(R < r) = \int_{-\infty}^{r} p_\theta(x)\,dx \approx \frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[x_i < r]$$

The flow's ability to extrapolate beyond observed data (via the learned transformation of Gaussian tails) provides estimates of extreme-event probabilities even when historical data is sparse.

### Comparison with Parametric Methods

| Metric | Gaussian | Student-t | Flow |
|---|---|---|---|
| Tail risk capture | Poor | Better | Best |
| Distributional assumptions | Strong | Moderate | None |
| Multi-asset dependence | Linear | Linear | Arbitrary |
| Backtesting performance | Frequent violations | Moderate | Low violation rate |

## Option Pricing

### Risk-Neutral Density Estimation

Under the risk-neutral measure $\mathbb{Q}$, option prices are discounted expectations:

$$C_0 = e^{-rT}\,\mathbb{E}^{\mathbb{Q}}\!\bigl[\max(S_T - K, 0)\bigr]$$

A flow can learn the risk-neutral distribution of $S_T$ from observed option prices across strikes and maturities, without assuming Black-Scholes, Heston, or any specific stochastic model.

### Approach

1. **Extract implied information** from market option prices across strikes.
2. **Train a flow** to fit the risk-neutral density that is consistent with observed prices.
3. **Price exotic options** by Monte Carlo under the learned density.

```python
class FlowOptionPricer:
    def __init__(self, flow, r, T):
        self.flow = flow
        self.discount = np.exp(-r * T)

    def price_european_call(self, K, n_samples=100_000):
        with torch.no_grad():
            S_T = self.flow.sample(n_samples).squeeze().numpy()
        payoffs = np.maximum(S_T - K, 0)
        return self.discount * payoffs.mean()

    def price_european_put(self, K, n_samples=100_000):
        with torch.no_grad():
            S_T = self.flow.sample(n_samples).squeeze().numpy()
        payoffs = np.maximum(K - S_T, 0)
        return self.discount * payoffs.mean()
```

### Advantages Over Parametric Models

Flows capture the volatility smile and skew directly from data, automatically adapting to market conditions.  They can model multi-asset options by learning the joint risk-neutral distribution, capturing cross-asset dependencies that copula-based methods may miss.

### Arbitrage-Free Constraints

The learned density must satisfy the no-arbitrage martingale condition $\mathbb{E}^{\mathbb{Q}}[S_T] = S_0 e^{rT}$.  This can be enforced as a penalty during training or by post-hoc adjustment.

## Portfolio Optimisation

### Beyond Mean-Variance

Markowitz mean-variance optimisation assumes Gaussian returns and measures risk by variance alone.  Real returns exhibit heavy tails, skewness, and tail dependence—all of which a flow captures.

### Distribution-Aware Optimisation

With a flow that models the joint return distribution $p(r_1, \ldots, r_d)$, we can optimise for any risk measure:

$$\min_w \; \text{CVaR}_\alpha\!\left(\sum_i w_i r_i\right) \quad \text{s.t.} \quad \mathbb{E}\!\left[\sum_i w_i r_i\right] \ge \mu_{\text{target}}, \;\; \sum_i w_i = 1$$

The expectation and CVaR are estimated by sampling from the flow:

```python
def portfolio_cvar(weights, flow, alpha=0.95, n_samples=50_000):
    """Estimate portfolio CVaR using flow samples."""
    with torch.no_grad():
        returns = flow.sample(n_samples).numpy()     # (n_samples, n_assets)
    portfolio_returns = returns @ weights
    var = -np.percentile(portfolio_returns, (1 - alpha) * 100)
    tail = portfolio_returns[portfolio_returns <= -var]
    return -tail.mean() if len(tail) > 0 else var
```

### Capturing Tail Dependence

During market crises, asset correlations spike—a phenomenon called *tail dependence*.  Gaussian-based optimisation misses this entirely.  Flows learn the joint distribution including tail behaviour, leading to more conservative and realistic allocations in stressed conditions.

### Conditional Allocation

By training conditional flows $p(r \mid \text{regime})$, portfolio weights can be adapted to market conditions (bull vs. bear, high vs. low volatility), providing regime-aware dynamic allocation strategies.

## Scenario Generation

### Motivation

Stress testing, regulatory compliance, and simulation-based risk management all require realistic multi-asset scenarios that preserve the statistical properties of historical data.

### Unconditional Scenario Generation

Sample directly from the trained flow:

```python
def generate_scenarios(flow, n_scenarios, device="cpu"):
    with torch.no_grad():
        z = torch.randn(n_scenarios, flow.dim, device=device)
        scenarios, _ = flow.inverse(z)
    return scenarios.numpy()
```

### Stress Scenarios via Latent-Space Manipulation

Because flows provide a bijective mapping between data space and latent space, extreme scenarios can be generated by sampling from the tails of the base distribution:

```python
def generate_stress_scenarios(flow, n_scenarios, n_sigma=3.0):
    """Generate scenarios from the tail of the base distribution."""
    z = torch.randn(n_scenarios, flow.dim)
    # Keep only samples beyond n_sigma standard deviations
    norms = z.norm(dim=-1)
    z = z[norms > n_sigma]
    with torch.no_grad():
        scenarios, _ = flow.inverse(z)
    return scenarios.numpy()
```

### Conditional Scenario Generation

Generate scenarios conditioned on specific market conditions (e.g., "VIX > 30 and 10Y yield rises 50bp") by training a conditional flow and sampling from it with the desired conditioning values.

### Importance-Weighted Scenarios

The exact density from flows enables importance sampling: generate scenarios from a proposal distribution and reweight using the flow's density to estimate expectations under the data distribution:

$$\mathbb{E}_{p_\theta}[g(x)] = \mathbb{E}_{q}\!\left[\frac{p_\theta(x)}{q(x)}\,g(x)\right]$$

This is useful for pricing rare events or computing tail risk measures more efficiently.

### Advantages Over Traditional Methods

| Method | Preserves Dependence | Generates Unseen | Fast | Density Available |
|---|---|---|---|---|
| Historical bootstrap | ✓ | ✗ | ✓ | ✗ |
| Gaussian simulation | Partial | ✓ | ✓ | ✓ |
| Copula + margins | Partial | ✓ | ✓ | ✓ |
| **Normalizing flows** | **✓** | **✓** | **✓** | **✓** |

## Comprehensive Example: End-to-End Workflow

```python
# 1. Prepare data
returns = load_multi_asset_returns()           # (T, d) array
train, test = returns[:int(0.8*len(returns))], returns[int(0.8*len(returns)):]
train_t = torch.tensor(train, dtype=torch.float32)

# 2. Build and train flow
from flow_models import RealNVP                 # or MAF, NSF
model = RealNVP(dim=train.shape[1], n_layers=12)
train_flow(model, train_t, epochs=300)

# 3. Evaluate density fit
test_ll = test_log_likelihood(model, test_t)
print(f"Test log-likelihood: {test_ll:.2f}")

# 4. Risk measurement
var_99 = FlowVaR(model).compute(alpha=0.99)
cvar_99 = FlowVaR(model).compute_cvar(alpha=0.99)

# 5. Generate scenarios for stress testing
stress = generate_stress_scenarios(model, n_scenarios=10000, n_sigma=2.5)

# 6. Portfolio optimisation under true distribution
from scipy.optimize import minimize
result = minimize(
    lambda w: portfolio_cvar(w, model, alpha=0.95),
    x0=np.ones(train.shape[1]) / train.shape[1],
    constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
    bounds=[(0, 1)] * train.shape[1],
)
optimal_weights = result.x
```

## Summary

Normalizing flows bring three capabilities to quantitative finance that traditional methods lack: exact non-parametric density evaluation, faithful capture of multi-asset dependence including tail behaviour, and fast generative sampling.  These capabilities directly improve risk measurement, option pricing, portfolio construction, and scenario generation—the core tasks of quantitative finance.

## References

1. Papamakarios, G., et al. (2021). Normalizing Flows for Probabilistic Modeling and Inference. *JMLR*.
2. Wiese, M., et al. (2020). Quant GANs: Deep Generation of Financial Time Series. *Quantitative Finance*.
3. Kondratyev, A. & Schwarz, C. (2019). The Market Generator. *SSRN*.
4. Durkan, C., et al. (2019). Neural Spline Flows. *NeurIPS*.
