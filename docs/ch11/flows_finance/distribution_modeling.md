# Probability Distribution Modeling with Normalizing Flows

## Introduction

Normalizing flows excel at modeling complex probability distributions encountered in quantitative finance. Unlike parametric models that assume specific distributional forms, flows learn flexible, arbitrage-free distributions directly from data while providing exact density evaluation—a critical capability for risk management, pricing, and portfolio optimization.

## Why Flows for Financial Distributions?

### Limitations of Traditional Approaches

**Parametric models** (Normal, Student-t, etc.):
- Assume fixed distributional shape
- Cannot capture complex dependencies
- Often misspecify tails and skewness

**Historical simulation**:
- Limited by available data
- Cannot extrapolate beyond observed scenarios
- Non-smooth density estimates

**Kernel density estimation**:
- Curse of dimensionality
- No generative capability
- Bandwidth selection challenges

### Flow Advantages

| Capability | Flows | Parametric | Historical | KDE |
|------------|-------|------------|------------|-----|
| Exact density | ✅ | ✅ | ❌ | ≈ |
| Flexible shape | ✅ | ❌ | ✅ | ✅ |
| High dimensions | ✅ | ✅ | ❌ | ❌ |
| Sampling | ✅ | ✅ | ✅ | ❌ |
| Tail modeling | ✅ | Limited | Limited | ❌ |

## Modeling Univariate Distributions

### Return Distribution Modeling

```python
import torch
import torch.nn as nn
import numpy as np

class UnivariateReturnFlow(nn.Module):
    """
    Normalizing flow for modeling univariate return distributions.
    Captures skewness, kurtosis, and fat tails.
    """
    
    def __init__(self, n_layers=8, num_bins=8, tail_bound=4.0):
        super().__init__()
        
        self.transforms = nn.ModuleList()
        
        for _ in range(n_layers):
            self.transforms.append(
                RationalQuadraticSplineTransform(
                    num_bins=num_bins,
                    tail_bound=tail_bound
                )
            )
    
    def forward(self, x):
        """Transform data to base distribution."""
        log_det = torch.zeros(x.shape[0], device=x.device)
        z = x
        
        for transform in self.transforms:
            z, ld = transform(z)
            log_det += ld
        
        return z, log_det
    
    def inverse(self, z):
        """Transform base distribution to data."""
        log_det = torch.zeros(z.shape[0], device=z.device)
        x = z
        
        for transform in reversed(self.transforms):
            x, ld = transform.inverse(x)
            log_det += ld
        
        return x, log_det
    
    def log_prob(self, x):
        """Compute log-probability."""
        z, log_det = self.forward(x)
        log_pz = -0.5 * (z ** 2 + np.log(2 * np.pi))
        return log_pz + log_det
    
    def sample(self, n_samples, device='cpu'):
        """Generate samples."""
        z = torch.randn(n_samples, device=device)
        x, _ = self.inverse(z)
        return x
    
    def cdf(self, x):
        """Cumulative distribution function."""
        z, _ = self.forward(x)
        return torch.distributions.Normal(0, 1).cdf(z)
    
    def quantile(self, p):
        """Quantile function (inverse CDF)."""
        z = torch.distributions.Normal(0, 1).icdf(p)
        x, _ = self.inverse(z)
        return x
```

### Fitting to Historical Returns

```python
def fit_return_distribution(returns, n_epochs=500, lr=1e-3):
    """
    Fit flow to historical returns.
    
    Args:
        returns: Historical return series (numpy array or tensor)
        n_epochs: Training epochs
        lr: Learning rate
    
    Returns:
        Trained flow model
    """
    # Prepare data
    if isinstance(returns, np.ndarray):
        returns = torch.tensor(returns, dtype=torch.float32)
    
    # Standardize for numerical stability
    mean = returns.mean()
    std = returns.std()
    returns_std = (returns - mean) / std
    
    # Create model
    model = UnivariateReturnFlow(n_layers=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training
    batch_size = min(256, len(returns))
    
    for epoch in range(n_epochs):
        # Sample batch
        idx = torch.randint(0, len(returns_std), (batch_size,))
        batch = returns_std[idx]
        
        optimizer.zero_grad()
        loss = -model.log_prob(batch).mean()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}, NLL: {loss.item():.4f}")
    
    # Wrap model to handle standardization
    return StandardizedFlow(model, mean, std)


class StandardizedFlow(nn.Module):
    """Wrapper that handles standardization."""
    
    def __init__(self, flow, mean, std):
        super().__init__()
        self.flow = flow
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
    
    def log_prob(self, x):
        x_std = (x - self.mean) / self.std
        return self.flow.log_prob(x_std) - torch.log(self.std)
    
    def sample(self, n_samples):
        x_std = self.flow.sample(n_samples)
        return x_std * self.std + self.mean
    
    def cdf(self, x):
        x_std = (x - self.mean) / self.std
        return self.flow.cdf(x_std)
    
    def quantile(self, p):
        x_std = self.flow.quantile(p)
        return x_std * self.std + self.mean
```

## Modeling Multivariate Distributions

### Joint Return Distribution

```python
class MultivariateReturnFlow(nn.Module):
    """
    Flow for joint distribution of multiple asset returns.
    Captures correlations, tail dependencies, and asymmetries.
    """
    
    def __init__(self, dim, n_layers=8, hidden_dims=[128, 128]):
        super().__init__()
        
        self.dim = dim
        self.layers = nn.ModuleList()
        
        for i in range(n_layers):
            # Alternating masks for coupling layers
            mask = torch.zeros(dim)
            if i % 2 == 0:
                mask[:dim // 2] = 1
            else:
                mask[dim // 2:] = 1
            
            self.layers.append(
                AffineCouplingLayer(dim, mask, hidden_dims)
            )
    
    def forward(self, x):
        log_det = torch.zeros(x.shape[0], device=x.device)
        z = x
        
        for layer in self.layers:
            z, ld = layer(z)
            log_det += ld
        
        return z, log_det
    
    def inverse(self, z):
        log_det = torch.zeros(z.shape[0], device=z.device)
        x = z
        
        for layer in reversed(self.layers):
            x, ld = layer.inverse(x)
            log_det += ld
        
        return x, log_det
    
    def log_prob(self, x):
        z, log_det = self.forward(x)
        log_pz = -0.5 * (z ** 2 + np.log(2 * np.pi)).sum(dim=-1)
        return log_pz + log_det
    
    def sample(self, n_samples, device='cpu'):
        z = torch.randn(n_samples, self.dim, device=device)
        x, _ = self.inverse(z)
        return x
    
    def conditional_sample(self, x_known, known_dims, n_samples=1000):
        """
        Sample from conditional distribution p(x_unknown | x_known).
        Uses rejection sampling approximation.
        """
        unknown_dims = [i for i in range(self.dim) if i not in known_dims]
        
        # Generate many samples
        samples = self.sample(n_samples * 10)
        
        # Find samples close to known values
        distances = ((samples[:, known_dims] - x_known) ** 2).sum(dim=-1)
        threshold = torch.quantile(distances, 0.1)
        
        mask = distances < threshold
        conditional_samples = samples[mask][:, unknown_dims]
        
        return conditional_samples[:n_samples]
```

### Copula-Flow Hybrid

Separate marginals from dependence structure:

```python
class CopulaFlow(nn.Module):
    """
    Copula-based flow: separate marginal and dependence modeling.
    
    1. Transform each marginal to uniform via univariate flows
    2. Model dependence in copula space with multivariate flow
    """
    
    def __init__(self, dim, marginal_layers=4, copula_layers=6):
        super().__init__()
        
        self.dim = dim
        
        # Univariate flows for each marginal
        self.marginal_flows = nn.ModuleList([
            UnivariateReturnFlow(n_layers=marginal_layers)
            for _ in range(dim)
        ])
        
        # Multivariate flow for copula
        self.copula_flow = MultivariateReturnFlow(
            dim, n_layers=copula_layers
        )
    
    def forward(self, x):
        """Transform to base distribution."""
        batch_size = x.shape[0]
        log_det = torch.zeros(batch_size, device=x.device)
        
        # Transform each marginal to standard normal
        u = torch.zeros_like(x)
        for i, flow in enumerate(self.marginal_flows):
            u[:, i], ld = flow.forward(x[:, i])
            log_det += ld
        
        # Transform through copula flow
        z, ld = self.copula_flow.forward(u)
        log_det += ld
        
        return z, log_det
    
    def inverse(self, z):
        """Transform base to data distribution."""
        batch_size = z.shape[0]
        log_det = torch.zeros(batch_size, device=z.device)
        
        # Inverse copula
        u, ld = self.copula_flow.inverse(z)
        log_det += ld
        
        # Inverse marginals
        x = torch.zeros_like(u)
        for i, flow in enumerate(self.marginal_flows):
            x[:, i], ld = flow.inverse(u[:, i])
            log_det += ld
        
        return x, log_det
    
    def log_prob(self, x):
        z, log_det = self.forward(x)
        log_pz = -0.5 * (z ** 2 + np.log(2 * np.pi)).sum(dim=-1)
        return log_pz + log_det
    
    def sample(self, n_samples, device='cpu'):
        z = torch.randn(n_samples, self.dim, device=device)
        x, _ = self.inverse(z)
        return x
    
    def marginal_cdf(self, x, dim):
        """CDF of marginal distribution."""
        z, _ = self.marginal_flows[dim].forward(x)
        return torch.distributions.Normal(0, 1).cdf(z)
    
    def marginal_quantile(self, p, dim):
        """Quantile of marginal distribution."""
        z = torch.distributions.Normal(0, 1).icdf(p)
        x, _ = self.marginal_flows[dim].inverse(z)
        return x
```

## Time-Varying Distributions

### Conditional Flow for Regime-Dependent Returns

```python
class ConditionalReturnFlow(nn.Module):
    """
    Return distribution conditioned on market regime/state.
    p(r_t | state_t)
    """
    
    def __init__(self, return_dim, state_dim, n_layers=8, hidden_dims=[128, 128]):
        super().__init__()
        
        self.return_dim = return_dim
        self.state_dim = state_dim
        
        self.layers = nn.ModuleList()
        
        for i in range(n_layers):
            mask = torch.zeros(return_dim)
            mask[:return_dim // 2] = 1 if i % 2 == 0 else 0
            mask[return_dim // 2:] = 0 if i % 2 == 0 else 1
            
            self.layers.append(
                ConditionalAffineCoupling(
                    return_dim, state_dim, mask, hidden_dims
                )
            )
    
    def forward(self, x, state):
        """Transform returns given state."""
        log_det = torch.zeros(x.shape[0], device=x.device)
        z = x
        
        for layer in self.layers:
            z, ld = layer(z, state)
            log_det += ld
        
        return z, log_det
    
    def log_prob(self, x, state):
        """Conditional log-probability."""
        z, log_det = self.forward(x, state)
        log_pz = -0.5 * (z ** 2 + np.log(2 * np.pi)).sum(dim=-1)
        return log_pz + log_det
    
    def sample(self, state, n_samples=1):
        """Sample returns given state."""
        z = torch.randn(n_samples, self.return_dim, device=state.device)
        x = z
        
        # Expand state if needed
        if state.dim() == 1:
            state = state.unsqueeze(0).expand(n_samples, -1)
        
        for layer in reversed(self.layers):
            x, _ = layer.inverse(x, state)
        
        return x


class ConditionalAffineCoupling(nn.Module):
    """Affine coupling conditioned on external state."""
    
    def __init__(self, dim, state_dim, mask, hidden_dims):
        super().__init__()
        
        self.register_buffer('mask', mask)
        self.d_cond = int(mask.sum().item())
        self.d_trans = dim - self.d_cond
        
        # Network takes [x_masked, state] and outputs [scale, shift]
        input_dim = self.d_cond + state_dim
        output_dim = self.d_trans * 2
        
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
    
    def forward(self, x, state):
        x_cond = x[:, self.mask.bool()]
        x_trans = x[:, ~self.mask.bool()]
        
        # Condition on both x_cond and state
        context = torch.cat([x_cond, state], dim=-1)
        params = self.net(context)
        s, t = params.chunk(2, dim=-1)
        s = torch.tanh(s) * 2
        
        y_trans = x_trans * torch.exp(s) + t
        
        y = torch.zeros_like(x)
        y[:, self.mask.bool()] = x_cond
        y[:, ~self.mask.bool()] = y_trans
        
        return y, s.sum(dim=-1)
    
    def inverse(self, y, state):
        y_cond = y[:, self.mask.bool()]
        y_trans = y[:, ~self.mask.bool()]
        
        context = torch.cat([y_cond, state], dim=-1)
        params = self.net(context)
        s, t = params.chunk(2, dim=-1)
        s = torch.tanh(s) * 2
        
        x_trans = (y_trans - t) * torch.exp(-s)
        
        x = torch.zeros_like(y)
        x[:, self.mask.bool()] = y_cond
        x[:, ~self.mask.bool()] = x_trans
        
        return x, -s.sum(dim=-1)
```

## Distribution Diagnostics

### Comparing Flow to Empirical Distribution

```python
class DistributionDiagnostics:
    """Tools for evaluating flow distribution quality."""
    
    def __init__(self, flow, data):
        self.flow = flow
        self.data = data
    
    def qq_plot_data(self, n_quantiles=100):
        """Generate Q-Q plot data."""
        # Empirical quantiles
        sorted_data = torch.sort(self.data)[0]
        n = len(sorted_data)
        empirical_quantiles = sorted_data[
            torch.linspace(0, n-1, n_quantiles).long()
        ]
        
        # Theoretical quantiles from flow
        probs = torch.linspace(0.01, 0.99, n_quantiles)
        with torch.no_grad():
            flow_quantiles = self.flow.quantile(probs)
        
        return empirical_quantiles, flow_quantiles
    
    def ks_statistic(self, n_samples=10000):
        """Kolmogorov-Smirnov statistic."""
        # Sample from flow
        with torch.no_grad():
            flow_samples = self.flow.sample(n_samples)
        
        # Compute CDFs
        all_points = torch.cat([self.data, flow_samples])
        all_points = torch.sort(all_points)[0]
        
        # Empirical CDF of data
        data_cdf = (self.data.unsqueeze(1) <= all_points.unsqueeze(0)).float().mean(dim=0)
        
        # CDF from flow
        with torch.no_grad():
            flow_cdf = self.flow.cdf(all_points)
        
        ks_stat = (data_cdf - flow_cdf).abs().max()
        
        return ks_stat.item()
    
    def moment_comparison(self):
        """Compare moments of data and flow."""
        with torch.no_grad():
            samples = self.flow.sample(len(self.data) * 10)
        
        moments = {}
        for name, tensor in [('data', self.data), ('flow', samples)]:
            moments[f'{name}_mean'] = tensor.mean().item()
            moments[f'{name}_std'] = tensor.std().item()
            moments[f'{name}_skew'] = ((tensor - tensor.mean()) ** 3).mean().item() / tensor.std().item() ** 3
            moments[f'{name}_kurtosis'] = ((tensor - tensor.mean()) ** 4).mean().item() / tensor.std().item() ** 4
        
        return moments
    
    def tail_comparison(self, percentiles=[1, 5, 95, 99]):
        """Compare tail behavior."""
        results = {}
        
        for p in percentiles:
            # Empirical
            emp_q = torch.quantile(self.data, p / 100)
            
            # Flow
            with torch.no_grad():
                flow_q = self.flow.quantile(torch.tensor(p / 100))
            
            results[f'p{p}_empirical'] = emp_q.item()
            results[f'p{p}_flow'] = flow_q.item()
        
        return results
```

## Applications

### Density-Based Anomaly Detection

```python
def detect_anomalies(flow, returns, threshold_percentile=1):
    """
    Detect anomalous returns using flow density.
    
    Low probability under the learned distribution = anomaly.
    """
    with torch.no_grad():
        log_probs = flow.log_prob(returns)
    
    threshold = torch.quantile(log_probs, threshold_percentile / 100)
    anomalies = log_probs < threshold
    
    return anomalies, log_probs


def rolling_anomaly_detection(flow, returns, window=252):
    """Rolling window anomaly detection."""
    anomaly_scores = []
    
    for i in range(window, len(returns)):
        # Refit on rolling window
        window_returns = returns[i-window:i]
        # ... refit flow ...
        
        # Score current return
        current = returns[i:i+1]
        with torch.no_grad():
            score = flow.log_prob(current)
        anomaly_scores.append(score.item())
    
    return torch.tensor(anomaly_scores)
```

### Likelihood-Based Model Comparison

```python
def compare_models(models, test_data):
    """
    Compare distribution models using log-likelihood.
    """
    results = {}
    
    for name, model in models.items():
        with torch.no_grad():
            log_probs = model.log_prob(test_data)
        
        results[name] = {
            'mean_ll': log_probs.mean().item(),
            'std_ll': log_probs.std().item(),
            'total_ll': log_probs.sum().item()
        }
    
    # Compute Bayes factors (approximate)
    best_ll = max(r['mean_ll'] for r in results.values())
    for name in results:
        results[name]['bayes_factor'] = np.exp(results[name]['mean_ll'] - best_ll)
    
    return results
```

## Summary

Normalizing flows for distribution modeling provide:

1. **Flexible density estimation** without parametric assumptions
2. **Exact likelihood** for model comparison and anomaly detection
3. **Efficient sampling** for Monte Carlo applications
4. **Multivariate support** with dependency modeling
5. **Conditional distributions** for regime-dependent analysis
6. **CDF/quantile functions** for risk metrics

Key applications in finance:
- Return distribution modeling
- Tail risk estimation
- Copula construction
- Anomaly detection
- Model validation

Flows bridge the gap between fully parametric models (too restrictive) and non-parametric methods (too data-hungry), providing a principled approach to distribution modeling in quantitative finance.

## References

1. Papamakarios, G., et al. (2021). Normalizing Flows for Probabilistic Modeling and Inference. *JMLR*.
2. Kobyzev, I., et al. (2020). Normalizing Flows: An Introduction and Review of Current Methods. *IEEE TPAMI*.
3. Wiese, M., et al. (2020). Quant GANs: Deep Generation of Financial Time Series. *Quantitative Finance*.
4. Cont, R. (2001). Empirical Properties of Asset Returns: Stylized Facts and Statistical Issues. *Quantitative Finance*.
