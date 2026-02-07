# Scenario Generation

Using VAEs to generate market scenarios for stress testing, risk management, and portfolio analysis.

---

## Learning Objectives

By the end of this section, you will be able to:

- Generate diverse, realistic market scenarios using trained VAEs
- Apply Conditional VAEs for regime-specific scenario generation
- Use latent space manipulation for targeted stress scenarios
- Integrate VAE scenarios into risk management workflows

---

## Why Scenario Generation?

### Limitations of Historical Simulation

Historical simulation for VaR and stress testing is limited to scenarios that have actually occurred. It cannot generate novel crisis scenarios, underestimates tail risks in short sample periods, and cannot condition on specific market conditions.

### VAE Advantages

VAEs learn the underlying distribution of market data, enabling generation of plausible scenarios beyond historical experience, targeted generation conditioned on specific factors, smooth interpolation between known market regimes, and uncertainty-aware scenario construction.

---

## Basic Scenario Generation

### Unconditional Generation

```python
def generate_scenarios(model, num_scenarios, device='cpu'):
    """
    Generate market scenarios by sampling from the prior.
    
    Returns:
        scenarios: [num_scenarios, num_assets] return vectors
    """
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_scenarios, model.latent_dim, device=device)
        scenarios = model.decode(z)
    return scenarios.cpu().numpy()

# Generate 10,000 scenarios for VaR estimation
scenarios = generate_scenarios(trained_model, num_scenarios=10000)
```

### Tail Scenario Generation

To focus on extreme scenarios, sample from the tails of the prior:

```python
def generate_tail_scenarios(model, num_scenarios, tail_threshold=2.0, device='cpu'):
    """
    Generate extreme scenarios by sampling from prior tails.
    
    Only keep latent samples where ||z|| > tail_threshold.
    """
    model.eval()
    scenarios = []
    
    with torch.no_grad():
        while len(scenarios) < num_scenarios:
            z = torch.randn(num_scenarios * 5, model.latent_dim, device=device)
            norms = z.norm(dim=1)
            tail_mask = norms > tail_threshold
            
            if tail_mask.any():
                tail_z = z[tail_mask]
                tail_scenarios = model.decode(tail_z)
                scenarios.append(tail_scenarios.cpu())
    
    scenarios = torch.cat(scenarios)[:num_scenarios]
    return scenarios.numpy()
```

---

## Conditional Scenario Generation

### Regime-Conditioned Scenarios

Using a Conditional VAE, generate scenarios specific to market regimes:

```python
def generate_regime_scenarios(cvae_model, regime_label, num_scenarios, device='cpu'):
    """
    Generate scenarios conditioned on market regime.
    
    Args:
        regime_label: Integer label (e.g., 0=bull, 1=bear, 2=crisis)
    """
    return cvae_model.generate(y=regime_label, num_samples=num_scenarios, device=device)

# Generate crisis scenarios
crisis_scenarios = generate_regime_scenarios(model, regime_label=2, num_scenarios=5000)
```

### Factor-Conditioned Generation

If the VAE's latent dimensions correspond to market factors, manipulate specific dimensions to create targeted scenarios:

```python
def generate_factor_scenario(model, factor_dim, factor_value, num_scenarios, device='cpu'):
    """
    Generate scenarios with a specific factor fixed.
    
    E.g., fix the 'volatility' latent dimension to a high value.
    """
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_scenarios, model.latent_dim, device=device)
        z[:, factor_dim] = factor_value  # Fix one factor
        scenarios = model.decode(z)
    return scenarios.cpu().numpy()

# High-volatility scenarios
vol_scenarios = generate_factor_scenario(model, factor_dim=0, factor_value=3.0, num_scenarios=5000)
```

---

## Risk Management Applications

### Value at Risk (VaR)

```python
import numpy as np

def compute_var_from_scenarios(portfolio_weights, scenarios, confidence=0.99):
    """
    Compute VaR using VAE-generated scenarios.
    
    Args:
        portfolio_weights: [num_assets] array of portfolio weights
        scenarios: [num_scenarios, num_assets] return scenarios
        confidence: VaR confidence level
    """
    # Portfolio returns under each scenario
    portfolio_returns = scenarios @ portfolio_weights
    
    # VaR at given confidence
    var = np.percentile(portfolio_returns, (1 - confidence) * 100)
    
    # Expected Shortfall (CVaR)
    cvar = portfolio_returns[portfolio_returns <= var].mean()
    
    return {'VaR': var, 'CVaR': cvar}
```

### Stress Testing

```python
def stress_test(model, portfolio_weights, stress_conditions, num_scenarios=10000):
    """
    Stress test portfolio under various conditions.
    
    Args:
        stress_conditions: dict mapping factor dimensions to values
    """
    z = torch.randn(num_scenarios, model.latent_dim)
    
    # Apply stress conditions
    for dim, value in stress_conditions.items():
        z[:, dim] = value
    
    with torch.no_grad():
        scenarios = model.decode(z).numpy()
    
    portfolio_pnl = scenarios @ portfolio_weights
    
    return {
        'mean_loss': portfolio_pnl.mean(),
        'worst_case': portfolio_pnl.min(),
        'loss_probability': (portfolio_pnl < 0).mean()
    }
```

---

## Interpolation Between Regimes

VAEs enable smooth interpolation between market regimes in latent space:

```python
def interpolate_regimes(model, z_bull, z_bear, num_steps=20):
    """
    Generate scenarios along the path from bull to bear market.
    """
    model.eval()
    regime_scenarios = []
    
    with torch.no_grad():
        for t in torch.linspace(0, 1, num_steps):
            z = (1 - t) * z_bull + t * z_bear
            scenario = model.decode(z)
            regime_scenarios.append(scenario.cpu())
    
    return torch.stack(regime_scenarios)
```

This generates a continuum of scenarios from benign to extreme, useful for understanding how portfolio risk evolves across market conditions.

---

## Validation

### Statistical Properties

Generated scenarios should match historical data in terms of return distributions (marginal statistics), cross-asset correlations, and tail behavior (kurtosis, extreme quantiles).

### Plausibility Assessment

Domain experts should review generated scenarios for economic coherence. A scenario where equities crash but credit spreads tighten would be implausible, and such failures indicate the VAE hasn't fully captured market structure.

---

## Summary

| Application | Method | Benefit |
|-------------|--------|---------|
| **VaR estimation** | Unconditional generation | More scenarios than historical |
| **Stress testing** | Latent space manipulation | Targeted extreme scenarios |
| **Regime analysis** | CVAE conditional generation | Regime-specific risk assessment |
| **Sensitivity analysis** | Factor interpolation | Continuous risk surface |

---

## Exercises

### Exercise 1

Train a VAE on multi-asset return data. Generate 10,000 scenarios and compute 99% VaR. Compare with historical VaR.

### Exercise 2

Use a β-VAE to learn disentangled market factors. Identify which latent dimensions correspond to market-wide risk, sector rotation, and volatility.

### Exercise 3

Implement regime-conditional scenario generation. Generate 1,000 scenarios each for bull, bear, and crisis regimes. Compare return distributions and correlation structures across regimes.

---

## Chapter Conclusion

This chapter has covered VAEs from probabilistic foundations through modern variants and practical financial applications. The key takeaway is that VAEs provide a principled framework for generative modeling that naturally handles uncertainty — a critical requirement in quantitative finance. By combining the mathematical rigor of variational inference with the expressiveness of deep neural networks, VAEs enable synthetic data generation, missing data imputation, and scenario analysis that go far beyond traditional statistical methods.
