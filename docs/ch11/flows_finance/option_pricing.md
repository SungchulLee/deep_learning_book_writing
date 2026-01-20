# Option Pricing with Normalizing Flows

## Introduction

Normalizing flows provide a powerful framework for option pricing by learning complex, arbitrage-free risk-neutral distributions from market data. Unlike parametric models (Black-Scholes, Heston), flows can capture arbitrary distributional shapes including skew, kurtosis, and multi-modality, while maintaining tractable density evaluation for pricing and hedging.

## The Option Pricing Problem

### Risk-Neutral Pricing

Under risk-neutral measure $\mathbb{Q}$, option prices are discounted expectations:

$$C_0 = e^{-rT} \mathbb{E}^{\mathbb{Q}}[\max(S_T - K, 0)]$$

where:
- $C_0$: Call option price today
- $r$: Risk-free rate
- $T$: Time to maturity
- $S_T$: Asset price at expiry
- $K$: Strike price

### The Distribution Problem

Traditional approach: assume $S_T$ follows a specific distribution (log-normal, Heston, etc.)

**Limitation**: Real market distributions exhibit:
- Volatility smile/skew
- Fat tails
- Time-varying shapes
- Complex dependencies (for multi-asset)

**Flow solution**: Learn the risk-neutral distribution directly from option prices.

## Flow-Based Risk-Neutral Density

### Model Setup

Learn the distribution of log-returns $x = \log(S_T/S_0)$:

$$x \sim p_\theta(x)$$

where $p_\theta$ is parameterized by a normalizing flow.

### From Returns to Prices

Given flow density $p_\theta(x)$:

$$S_T = S_0 \exp(x)$$

Option price:
$$C_0 = e^{-rT} \int_{-\infty}^{\infty} \max(S_0 e^x - K, 0) \cdot p_\theta(x) \, dx$$

$$= e^{-rT} \int_{\log(K/S_0)}^{\infty} (S_0 e^x - K) \cdot p_\theta(x) \, dx$$

## Implementation

### Flow for Returns Distribution

```python
import torch
import torch.nn as nn
import numpy as np

class OptionPricingFlow(nn.Module):
    """
    Normalizing flow for modeling risk-neutral return distributions.
    """
    
    def __init__(self, n_layers=8, hidden_dims=[64, 64]):
        super().__init__()
        
        # 1D flow for log-returns
        self.flow = self._build_flow(n_layers, hidden_dims)
        
        # Learnable base distribution parameters (for flexibility)
        self.base_loc = nn.Parameter(torch.tensor(0.0))
        self.base_log_scale = nn.Parameter(torch.tensor(0.0))
    
    def _build_flow(self, n_layers, hidden_dims):
        """Build 1D spline flow."""
        from nflows.flows import Flow
        from nflows.distributions import StandardNormal
        from nflows.transforms import CompositeTransform, RationalQuadraticSplineTransform
        
        transforms = []
        for _ in range(n_layers):
            transforms.append(
                RationalQuadraticSplineTransform(
                    num_bins=8,
                    tail_bound=4.0
                )
            )
        
        return Flow(
            transform=CompositeTransform(transforms),
            distribution=StandardNormal([1])
        )
    
    def log_prob(self, x):
        """Log-probability of returns."""
        # Standardize
        x_std = (x - self.base_loc) / torch.exp(self.base_log_scale)
        
        # Flow log-prob + Jacobian of standardization
        log_prob = self.flow.log_prob(x_std.unsqueeze(-1)).squeeze(-1)
        log_prob = log_prob - self.base_log_scale
        
        return log_prob
    
    def sample(self, n_samples):
        """Sample returns."""
        z = self.flow.sample(n_samples).squeeze(-1)
        x = z * torch.exp(self.base_log_scale) + self.base_loc
        return x
    
    def density(self, x):
        """Probability density at x."""
        return torch.exp(self.log_prob(x))
```

### Option Pricing via Monte Carlo

```python
class FlowOptionPricer:
    """Price options using flow-based risk-neutral density."""
    
    def __init__(self, flow, S0, r, T):
        """
        Args:
            flow: Trained normalizing flow for returns
            S0: Current spot price
            r: Risk-free rate
            T: Time to maturity
        """
        self.flow = flow
        self.S0 = S0
        self.r = r
        self.T = T
        self.discount = np.exp(-r * T)
    
    def price_call(self, K, n_samples=100000):
        """Price European call option via Monte Carlo."""
        with torch.no_grad():
            # Sample returns from flow
            x = self.flow.sample(n_samples)
            
            # Convert to prices
            ST = self.S0 * torch.exp(x)
            
            # Payoff
            payoff = torch.clamp(ST - K, min=0)
            
            # Discounted expectation
            price = self.discount * payoff.mean()
            std_err = self.discount * payoff.std() / np.sqrt(n_samples)
        
        return price.item(), std_err.item()
    
    def price_put(self, K, n_samples=100000):
        """Price European put option via Monte Carlo."""
        with torch.no_grad():
            x = self.flow.sample(n_samples)
            ST = self.S0 * torch.exp(x)
            payoff = torch.clamp(K - ST, min=0)
            price = self.discount * payoff.mean()
            std_err = self.discount * payoff.std() / np.sqrt(n_samples)
        
        return price.item(), std_err.item()
    
    def price_call_numerical(self, K, n_points=1000):
        """Price call via numerical integration (more accurate)."""
        # Integration bounds
        x_min, x_max = -5.0, 5.0
        x = torch.linspace(x_min, x_max, n_points)
        
        # Density
        with torch.no_grad():
            density = self.flow.density(x)
        
        # Payoff
        ST = self.S0 * torch.exp(x)
        payoff = torch.clamp(ST - K, min=0)
        
        # Numerical integration (trapezoidal)
        dx = (x_max - x_min) / (n_points - 1)
        integrand = payoff * density
        integral = torch.trapz(integrand, x)
        
        return (self.discount * integral).item()
```

### Greeks via Automatic Differentiation

```python
class FlowGreeks:
    """Compute option Greeks using automatic differentiation."""
    
    def __init__(self, flow, S0, r, T):
        self.flow = flow
        self.S0 = S0
        self.r = r
        self.T = T
    
    def delta(self, K, n_samples=50000):
        """Compute Delta = dC/dS."""
        S0 = torch.tensor(self.S0, requires_grad=True)
        
        # Sample and compute price
        x = self.flow.sample(n_samples)
        ST = S0 * torch.exp(x)
        payoff = torch.clamp(ST - K, min=0)
        price = np.exp(-self.r * self.T) * payoff.mean()
        
        # Differentiate
        delta = torch.autograd.grad(price, S0)[0]
        
        return delta.item()
    
    def gamma(self, K, n_samples=50000):
        """Compute Gamma = d²C/dS²."""
        S0 = torch.tensor(self.S0, requires_grad=True)
        
        x = self.flow.sample(n_samples)
        ST = S0 * torch.exp(x)
        payoff = torch.clamp(ST - K, min=0)
        price = np.exp(-self.r * self.T) * payoff.mean()
        
        # First derivative
        delta = torch.autograd.grad(price, S0, create_graph=True)[0]
        
        # Second derivative
        gamma = torch.autograd.grad(delta, S0)[0]
        
        return gamma.item()
    
    def vega_flow(self, K, n_samples=50000):
        """
        Compute sensitivity to flow parameters.
        Useful for hedging model risk.
        """
        # Enable gradients for flow parameters
        for param in self.flow.parameters():
            param.requires_grad_(True)
        
        x = self.flow.sample(n_samples)
        ST = self.S0 * torch.exp(x)
        payoff = torch.clamp(ST - K, min=0)
        price = np.exp(-self.r * self.T) * payoff.mean()
        
        # Gradient w.r.t. all flow parameters
        price.backward()
        
        vegas = {
            name: param.grad.clone() 
            for name, param in self.flow.named_parameters()
        }
        
        return vegas
```

## Calibration to Market Data

### Objective: Match Market Prices

Given market option prices $\{C_i^{\text{mkt}}\}$ for strikes $\{K_i\}$:

$$\mathcal{L}(\theta) = \sum_i \left(C_i^{\text{flow}}(\theta) - C_i^{\text{mkt}}\right)^2$$

### Calibration with Arbitrage Constraints

```python
class FlowCalibrator:
    """Calibrate flow to market option prices."""
    
    def __init__(self, S0, r, T):
        self.S0 = S0
        self.r = r
        self.T = T
        self.discount = np.exp(-r * T)
    
    def calibrate(self, flow, strikes, market_prices, n_epochs=1000, lr=1e-3):
        """
        Calibrate flow to match market prices.
        
        Args:
            flow: OptionPricingFlow model
            strikes: Array of strike prices
            market_prices: Array of market call prices
        """
        optimizer = torch.optim.Adam(flow.parameters(), lr=lr)
        
        strikes = torch.tensor(strikes, dtype=torch.float32)
        market_prices = torch.tensor(market_prices, dtype=torch.float32)
        
        losses = []
        
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            
            # Price options with current flow
            model_prices = self._price_calls_batch(flow, strikes)
            
            # MSE loss
            loss = ((model_prices - market_prices) ** 2).mean()
            
            # Regularization: encourage smooth density
            reg_loss = self._smoothness_regularization(flow)
            
            total_loss = loss + 0.01 * reg_loss
            
            total_loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")
        
        return losses
    
    def _price_calls_batch(self, flow, strikes, n_samples=10000):
        """Price multiple calls efficiently."""
        # Sample once
        x = flow.sample(n_samples)
        ST = self.S0 * torch.exp(x)
        
        # Price for each strike
        prices = []
        for K in strikes:
            payoff = torch.clamp(ST - K, min=0)
            price = self.discount * payoff.mean()
            prices.append(price)
        
        return torch.stack(prices)
    
    def _smoothness_regularization(self, flow, n_points=100):
        """Encourage smooth density."""
        x = torch.linspace(-3, 3, n_points, requires_grad=True)
        log_p = flow.log_prob(x)
        
        # Second derivative of log-density (penalize curvature)
        grad_log_p = torch.autograd.grad(
            log_p.sum(), x, create_graph=True
        )[0]
        grad2_log_p = torch.autograd.grad(
            grad_log_p.sum(), x, create_graph=True
        )[0]
        
        return (grad2_log_p ** 2).mean()
```

### Enforcing Arbitrage-Free Constraints

```python
class ArbitrageFreeFlow(nn.Module):
    """
    Flow with built-in arbitrage-free constraints.
    """
    
    def __init__(self, base_flow, S0, r, T):
        super().__init__()
        self.flow = base_flow
        self.S0 = S0
        self.r = r
        self.T = T
    
    def forward_price_constraint(self):
        """
        Martingale constraint: E[S_T] = S_0 * exp(r*T)
        """
        x = self.flow.sample(10000)
        ST = self.S0 * torch.exp(x)
        expected_ST = ST.mean()
        target = self.S0 * np.exp(self.r * self.T)
        
        return (expected_ST - target) ** 2
    
    def positive_density_constraint(self, x_grid):
        """Ensure density is positive everywhere."""
        log_p = self.flow.log_prob(x_grid)
        # Penalize very negative log-probs (near-zero density)
        return torch.clamp(-log_p - 10, min=0).mean()
    
    def calibrate_with_constraints(self, strikes, market_prices, 
                                    n_epochs=1000, martingale_weight=1.0):
        """Calibrate with arbitrage constraints."""
        optimizer = torch.optim.Adam(self.flow.parameters(), lr=1e-3)
        
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            
            # Pricing loss
            model_prices = self._price_calls(strikes)
            price_loss = ((model_prices - market_prices) ** 2).mean()
            
            # Martingale constraint
            martingale_loss = self.forward_price_constraint()
            
            # Total loss
            loss = price_loss + martingale_weight * martingale_loss
            
            loss.backward()
            optimizer.step()
```

## Implied Volatility Surface

### Extracting Implied Volatility

```python
from scipy.optimize import brentq
from scipy.stats import norm

def black_scholes_call(S, K, r, T, sigma):
    """Black-Scholes call price."""
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

def implied_volatility(price, S, K, r, T):
    """Extract implied volatility from price."""
    def objective(sigma):
        return black_scholes_call(S, K, r, T, sigma) - price
    
    try:
        return brentq(objective, 0.001, 5.0)
    except:
        return np.nan

class ImpliedVolSurface:
    """Generate implied volatility surface from flow."""
    
    def __init__(self, flow_pricer):
        self.pricer = flow_pricer
    
    def compute_surface(self, strikes, maturities):
        """Compute IV surface."""
        surface = np.zeros((len(maturities), len(strikes)))
        
        for i, T in enumerate(maturities):
            self.pricer.T = T
            self.pricer.discount = np.exp(-self.pricer.r * T)
            
            for j, K in enumerate(strikes):
                price, _ = self.pricer.price_call(K)
                iv = implied_volatility(
                    price, self.pricer.S0, K, self.pricer.r, T
                )
                surface[i, j] = iv
        
        return surface
```

## Multi-Asset Options

### Joint Distribution with Flows

```python
class MultiAssetFlow(nn.Module):
    """Flow for joint distribution of multiple assets."""
    
    def __init__(self, n_assets, n_layers=8, hidden_dims=[128, 128]):
        super().__init__()
        self.n_assets = n_assets
        
        # Multi-dimensional coupling flow
        self.flow = self._build_coupling_flow(n_assets, n_layers, hidden_dims)
    
    def _build_coupling_flow(self, dim, n_layers, hidden_dims):
        layers = []
        for i in range(n_layers):
            # Alternating mask
            mask = torch.zeros(dim)
            mask[:dim//2] = 1 if i % 2 == 0 else 0
            mask[dim//2:] = 0 if i % 2 == 0 else 1
            
            layers.append(AffineCouplingLayer(dim, mask, hidden_dims))
        
        return nn.ModuleList(layers)
    
    def log_prob(self, x):
        """Joint log-probability."""
        log_det = 0
        z = x
        for layer in self.flow:
            z, ld = layer(z)
            log_det += ld
        
        log_pz = -0.5 * (z ** 2 + np.log(2 * np.pi)).sum(dim=-1)
        return log_pz + log_det
    
    def sample(self, n_samples):
        """Sample joint returns."""
        z = torch.randn(n_samples, self.n_assets)
        x = z
        for layer in reversed(self.flow):
            x, _ = layer.inverse(x)
        return x


class BasketOptionPricer:
    """Price basket options using multi-asset flow."""
    
    def __init__(self, flow, S0_vec, r, T, weights):
        self.flow = flow
        self.S0 = torch.tensor(S0_vec)
        self.r = r
        self.T = T
        self.weights = torch.tensor(weights)
        self.discount = np.exp(-r * T)
    
    def price_basket_call(self, K, n_samples=100000):
        """Price basket call option."""
        with torch.no_grad():
            # Sample joint returns
            x = self.flow.sample(n_samples)
            
            # Asset prices at maturity
            ST = self.S0 * torch.exp(x)
            
            # Basket value
            basket = (ST * self.weights).sum(dim=-1)
            
            # Payoff
            payoff = torch.clamp(basket - K, min=0)
            
            price = self.discount * payoff.mean()
        
        return price.item()
```

## Summary

Normalizing flows for option pricing provide:

1. **Flexible distributions**: Capture skew, kurtosis, multi-modality
2. **Exact density**: Tractable pricing and calibration
3. **Automatic differentiation**: Greeks computation
4. **Multi-asset modeling**: Joint distributions for exotic options
5. **Market consistency**: Calibrate to observed prices

Key advantages over parametric models:
- No distributional assumptions
- Better fit to market smiles
- Model-free hedging via flow gradients

## References

1. Wiese, M., et al. (2019). Deep Hedging: Learning Risk-Neutral Implied Volatility Dynamics. *SSRN*.
2. Buehler, H., et al. (2019). Deep Hedging. *Quantitative Finance*.
3. Horvath, B., et al. (2021). Deep Learning Volatility. *Quantitative Finance*.
4. Cohen, S., et al. (2021). Detecting and Repairing Arbitrage in Traded Option Prices. *Applied Mathematical Finance*.
