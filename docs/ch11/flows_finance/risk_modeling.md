# Risk Modeling with Normalizing Flows

## Introduction

Risk measurement is fundamental to quantitative finance. Traditional methods rely on parametric assumptions (Gaussian returns) that fail during market stress. Normalizing flows provide **non-parametric risk estimates** with exact likelihood computation, enabling more accurate Value-at-Risk (VaR), Conditional Value-at-Risk (CVaR), and tail risk assessment.

## Value at Risk (VaR)

### Definition

VaR at confidence level $\alpha$ (typically 95% or 99%) is the loss threshold that will not be exceeded with probability $\alpha$:

$$\text{VaR}_\alpha = -\inf\{x : F(x) \geq 1 - \alpha\} = -F^{-1}(1 - \alpha)$$

Equivalently, VaR is the negative of the $(1-\alpha)$ quantile of the return distribution.

### VaR Estimation with Flows

```python
import torch
import numpy as np
from typing import Tuple, Dict

class FlowVaR:
    """Value-at-Risk estimation using normalizing flows."""
    
    def __init__(self, flow_model, n_samples: int = 100000):
        """
        Args:
            flow_model: Trained normalizing flow
            n_samples: Number of Monte Carlo samples for estimation
        """
        self.flow = flow_model
        self.n_samples = n_samples
    
    def estimate_var(
        self, 
        alpha: float = 0.95,
        portfolio_value: float = 1.0,
        holding_period: int = 1
    ) -> float:
        """
        Estimate VaR at given confidence level.
        
        Args:
            alpha: Confidence level (e.g., 0.95 for 95% VaR)
            portfolio_value: Current portfolio value
            holding_period: Time horizon in days (for scaling)
        
        Returns:
            VaR estimate (positive number representing potential loss)
        """
        self.flow.eval()
        
        with torch.no_grad():
            # Generate return samples
            samples = self.flow.sample(self.n_samples)
            returns = samples.numpy().flatten()
        
        # Quantile for VaR
        quantile = 1 - alpha  # e.g., 0.05 for 95% VaR
        var_return = np.quantile(returns, quantile)
        
        # Scale for holding period (assuming i.i.d., scale by sqrt(T))
        var_return_scaled = var_return * np.sqrt(holding_period)
        
        # Convert to dollar loss
        var_dollar = -var_return_scaled * portfolio_value
        
        return var_dollar
    
    def var_confidence_interval(
        self,
        alpha: float = 0.95,
        bootstrap_samples: int = 1000,
        ci_level: float = 0.95
    ) -> Tuple[float, float, float]:
        """
        Estimate VaR with bootstrap confidence interval.
        
        Returns:
            (var_estimate, ci_lower, ci_upper)
        """
        self.flow.eval()
        
        with torch.no_grad():
            samples = self.flow.sample(self.n_samples).numpy().flatten()
        
        # Bootstrap
        var_estimates = []
        for _ in range(bootstrap_samples):
            boot_sample = np.random.choice(samples, size=len(samples), replace=True)
            var_estimates.append(-np.quantile(boot_sample, 1 - alpha))
        
        var_estimates = np.array(var_estimates)
        
        var_mean = var_estimates.mean()
        ci_lower = np.quantile(var_estimates, (1 - ci_level) / 2)
        ci_upper = np.quantile(var_estimates, 1 - (1 - ci_level) / 2)
        
        return var_mean, ci_lower, ci_upper
```

### Comparing VaR Methods

```python
def compare_var_methods(
    returns: np.ndarray,
    flow_model,
    alpha: float = 0.95
) -> pd.DataFrame:
    """Compare VaR estimates from different methods."""
    from scipy import stats
    
    results = []
    
    # 1. Historical VaR
    hist_var = -np.quantile(returns, 1 - alpha)
    results.append({'Method': 'Historical', 'VaR': hist_var})
    
    # 2. Parametric (Normal) VaR
    mu, sigma = returns.mean(), returns.std()
    normal_var = -(mu + stats.norm.ppf(1 - alpha) * sigma)
    results.append({'Method': 'Normal', 'VaR': normal_var})
    
    # 3. Student-t VaR
    t_params = stats.t.fit(returns)
    t_var = -(t_params[1] + stats.t.ppf(1 - alpha, t_params[0]) * t_params[2])
    results.append({'Method': 'Student-t', 'VaR': t_var})
    
    # 4. Flow VaR
    flow_var_estimator = FlowVaR(flow_model)
    flow_var = flow_var_estimator.estimate_var(alpha)
    results.append({'Method': 'Flow', 'VaR': flow_var})
    
    return pd.DataFrame(results)
```

## Conditional Value at Risk (CVaR / Expected Shortfall)

### Definition

CVaR (also called Expected Shortfall) is the expected loss given that loss exceeds VaR:

$$\text{CVaR}_\alpha = -\mathbb{E}[X | X \leq -\text{VaR}_\alpha] = -\frac{1}{1-\alpha} \int_0^{1-\alpha} F^{-1}(u) \, du$$

CVaR is a **coherent risk measure** (satisfies subadditivity), unlike VaR.

### CVaR Estimation with Flows

```python
class FlowCVaR(FlowVaR):
    """Conditional VaR (Expected Shortfall) using normalizing flows."""
    
    def estimate_cvar(
        self,
        alpha: float = 0.95,
        portfolio_value: float = 1.0,
        holding_period: int = 1
    ) -> float:
        """
        Estimate CVaR at given confidence level.
        
        Args:
            alpha: Confidence level
            portfolio_value: Current portfolio value
            holding_period: Time horizon
        
        Returns:
            CVaR estimate
        """
        self.flow.eval()
        
        with torch.no_grad():
            samples = self.flow.sample(self.n_samples)
            returns = samples.numpy().flatten()
        
        # VaR threshold
        var_threshold = np.quantile(returns, 1 - alpha)
        
        # Average of returns below VaR
        tail_returns = returns[returns <= var_threshold]
        cvar_return = tail_returns.mean()
        
        # Scale and convert
        cvar_return_scaled = cvar_return * np.sqrt(holding_period)
        cvar_dollar = -cvar_return_scaled * portfolio_value
        
        return cvar_dollar
    
    def tail_analysis(
        self,
        alphas: List[float] = [0.90, 0.95, 0.99, 0.995]
    ) -> pd.DataFrame:
        """Comprehensive tail risk analysis."""
        
        self.flow.eval()
        
        with torch.no_grad():
            samples = self.flow.sample(self.n_samples).numpy().flatten()
        
        results = []
        for alpha in alphas:
            var = -np.quantile(samples, 1 - alpha)
            
            tail_returns = samples[samples <= np.quantile(samples, 1 - alpha)]
            cvar = -tail_returns.mean()
            
            # Tail ratio: CVaR / VaR
            tail_ratio = cvar / var if var > 0 else np.nan
            
            results.append({
                'Confidence': f'{alpha:.1%}',
                'VaR': var,
                'CVaR': cvar,
                'Tail Ratio': tail_ratio
            })
        
        return pd.DataFrame(results)
```

## Portfolio Risk with Flows

### Joint Distribution Approach

```python
class PortfolioRiskFlow:
    """Portfolio risk using joint return distribution flow."""
    
    def __init__(self, flow_model, asset_names: List[str]):
        """
        Args:
            flow_model: Flow trained on joint asset returns
            asset_names: Names of assets (must match flow dimensions)
        """
        self.flow = flow_model
        self.asset_names = asset_names
        self.n_assets = len(asset_names)
    
    def portfolio_var(
        self,
        weights: np.ndarray,
        alpha: float = 0.95,
        n_samples: int = 100000
    ) -> float:
        """
        Compute portfolio VaR using Monte Carlo.
        
        Args:
            weights: Portfolio weights (sum to 1)
            alpha: Confidence level
            n_samples: Number of samples
        
        Returns:
            Portfolio VaR
        """
        assert len(weights) == self.n_assets
        
        self.flow.eval()
        
        with torch.no_grad():
            # Sample joint returns
            asset_returns = self.flow.sample(n_samples).numpy()
        
        # Portfolio returns
        portfolio_returns = asset_returns @ weights
        
        # VaR
        var = -np.quantile(portfolio_returns, 1 - alpha)
        
        return var
    
    def marginal_var(
        self,
        weights: np.ndarray,
        alpha: float = 0.95,
        n_samples: int = 100000,
        delta: float = 0.01
    ) -> np.ndarray:
        """
        Compute marginal VaR for each asset.
        
        Marginal VaR = change in portfolio VaR from small weight change
        """
        base_var = self.portfolio_var(weights, alpha, n_samples)
        
        marginal_vars = []
        for i in range(self.n_assets):
            # Perturb weight i
            perturbed_weights = weights.copy()
            perturbed_weights[i] += delta
            perturbed_weights /= perturbed_weights.sum()  # Renormalize
            
            perturbed_var = self.portfolio_var(perturbed_weights, alpha, n_samples)
            
            marginal_var_i = (perturbed_var - base_var) / delta
            marginal_vars.append(marginal_var_i)
        
        return np.array(marginal_vars)
    
    def component_var(
        self,
        weights: np.ndarray,
        alpha: float = 0.95,
        n_samples: int = 100000
    ) -> np.ndarray:
        """
        Component VaR: contribution of each asset to total VaR.
        
        Component VaR_i = weight_i × Marginal VaR_i
        Sum of Component VaRs = Total VaR (approximately)
        """
        marginal_vars = self.marginal_var(weights, alpha, n_samples)
        component_vars = weights * marginal_vars
        
        return component_vars
    
    def risk_decomposition(
        self,
        weights: np.ndarray,
        alpha: float = 0.95,
        n_samples: int = 100000
    ) -> pd.DataFrame:
        """Full risk decomposition analysis."""
        
        total_var = self.portfolio_var(weights, alpha, n_samples)
        marginal_vars = self.marginal_var(weights, alpha, n_samples)
        component_vars = weights * marginal_vars
        
        # Percentage contribution
        pct_contribution = component_vars / total_var * 100
        
        results = pd.DataFrame({
            'Asset': self.asset_names,
            'Weight': weights,
            'Marginal VaR': marginal_vars,
            'Component VaR': component_vars,
            '% Contribution': pct_contribution
        })
        
        # Add total row
        total_row = pd.DataFrame({
            'Asset': ['Total'],
            'Weight': [weights.sum()],
            'Marginal VaR': [np.nan],
            'Component VaR': [component_vars.sum()],
            '% Contribution': [100.0]
        })
        
        results = pd.concat([results, total_row], ignore_index=True)
        
        return results
```

## Stress Testing and Extreme Scenarios

### Conditional Sampling for Stress Tests

```python
class FlowStressTesting:
    """Stress testing using conditional flow sampling."""
    
    def __init__(self, conditional_flow, feature_names: List[str]):
        """
        Args:
            conditional_flow: Flow conditioned on market features
            feature_names: Names of conditioning features
        """
        self.flow = conditional_flow
        self.feature_names = feature_names
    
    def stress_scenario(
        self,
        stress_features: Dict[str, float],
        base_features: torch.Tensor,
        n_samples: int = 10000
    ) -> Dict[str, float]:
        """
        Generate returns under stressed market conditions.
        
        Args:
            stress_features: Dictionary of feature name -> stressed value
            base_features: Normal market features
            n_samples: Number of samples
        
        Returns:
            Risk metrics under stress
        """
        # Create stressed feature vector
        stressed = base_features.clone()
        for name, value in stress_features.items():
            if name in self.feature_names:
                idx = self.feature_names.index(name)
                stressed[:, idx] = value
        
        self.flow.eval()
        
        with torch.no_grad():
            # Sample under stress
            stressed_expanded = stressed.expand(n_samples, -1)
            stressed_returns = self.flow.sample(stressed_expanded).numpy()
        
        # Compute risk metrics
        return {
            'mean_return': stressed_returns.mean(),
            'volatility': stressed_returns.std(),
            'var_95': -np.quantile(stressed_returns, 0.05),
            'var_99': -np.quantile(stressed_returns, 0.01),
            'min_return': stressed_returns.min(),
            'max_return': stressed_returns.max(),
            'skewness': ((stressed_returns - stressed_returns.mean()) ** 3).mean() / stressed_returns.std() ** 3,
            'kurtosis': ((stressed_returns - stressed_returns.mean()) ** 4).mean() / stressed_returns.std() ** 4
        }
    
    def scenario_comparison(
        self,
        scenarios: Dict[str, Dict[str, float]],
        base_features: torch.Tensor,
        n_samples: int = 10000
    ) -> pd.DataFrame:
        """Compare multiple stress scenarios."""
        
        results = []
        
        # Base case
        base_metrics = self.stress_scenario({}, base_features, n_samples)
        base_metrics['Scenario'] = 'Base'
        results.append(base_metrics)
        
        # Stress scenarios
        for name, stress_features in scenarios.items():
            metrics = self.stress_scenario(stress_features, base_features, n_samples)
            metrics['Scenario'] = name
            results.append(metrics)
        
        df = pd.DataFrame(results)
        # Reorder columns
        cols = ['Scenario'] + [c for c in df.columns if c != 'Scenario']
        return df[cols]
```

### Historical Stress Replication

```python
def replicate_historical_stress(
    flow_model,
    crisis_returns: np.ndarray,
    crisis_name: str = "2008 Crisis",
    n_samples: int = 10000
) -> Dict:
    """
    Use flow to generate scenarios similar to historical crisis.
    
    Encode crisis returns to latent space, then sample nearby.
    """
    flow_model.eval()
    
    # Encode crisis returns
    crisis_tensor = torch.tensor(crisis_returns, dtype=torch.float32)
    
    with torch.no_grad():
        z_crisis, _ = flow_model.inverse(crisis_tensor)
        z_mean = z_crisis.mean(dim=0)
        z_std = z_crisis.std(dim=0)
    
    # Sample similar scenarios (around crisis latent distribution)
    z_similar = torch.randn(n_samples, z_mean.shape[0]) * z_std + z_mean
    
    with torch.no_grad():
        similar_returns, _ = flow_model.forward(z_similar)
    
    similar_returns = similar_returns.numpy()
    
    return {
        'crisis_name': crisis_name,
        'original_crisis_stats': {
            'mean': crisis_returns.mean(),
            'std': crisis_returns.std(),
            'min': crisis_returns.min(),
            'max': crisis_returns.max()
        },
        'generated_stats': {
            'mean': similar_returns.mean(),
            'std': similar_returns.std(),
            'min': similar_returns.min(),
            'max': similar_returns.max()
        },
        'generated_returns': similar_returns
    }
```

## Backtesting VaR Models

### Violation Analysis

```python
class VaRBacktester:
    """Backtest VaR models using historical data."""
    
    def __init__(self, flow_model, returns: np.ndarray, features: np.ndarray = None):
        """
        Args:
            flow_model: Trained flow model
            returns: Historical returns for backtesting
            features: Conditioning features (if conditional flow)
        """
        self.flow = flow_model
        self.returns = returns
        self.features = features
    
    def run_backtest(
        self,
        alpha: float = 0.95,
        rolling_window: int = 252,
        n_samples: int = 10000
    ) -> pd.DataFrame:
        """
        Rolling VaR backtest.
        
        Returns:
            DataFrame with dates, returns, VaR forecasts, and violations
        """
        results = []
        
        for t in range(rolling_window, len(self.returns)):
            # Training data: [t-window, t)
            train_returns = self.returns[t-rolling_window:t]
            
            # Retrain or use pre-trained model
            # (For efficiency, often use pre-trained model with sliding features)
            
            # Forecast VaR
            self.flow.eval()
            with torch.no_grad():
                if self.features is not None:
                    context = torch.tensor(
                        self.features[t-1:t], 
                        dtype=torch.float32
                    ).expand(n_samples, -1)
                    samples = self.flow.sample(context)
                else:
                    samples = self.flow.sample(n_samples)
            
            samples = samples.numpy().flatten()
            var_forecast = -np.quantile(samples, 1 - alpha)
            
            # Actual return
            actual_return = self.returns[t]
            
            # Violation
            violation = actual_return < -var_forecast
            
            results.append({
                'date': t,
                'actual_return': actual_return,
                'var_forecast': var_forecast,
                'violation': violation
            })
        
        return pd.DataFrame(results)
    
    def evaluate_backtest(self, backtest_results: pd.DataFrame, alpha: float = 0.95) -> Dict:
        """
        Evaluate backtest results.
        
        Tests:
        1. Unconditional coverage (Kupiec test)
        2. Independence (Christoffersen test)
        3. Conditional coverage
        """
        from scipy import stats
        
        violations = backtest_results['violation'].values
        n = len(violations)
        n_violations = violations.sum()
        
        # Expected violation rate
        expected_rate = 1 - alpha
        actual_rate = n_violations / n
        
        # 1. Kupiec test (unconditional coverage)
        # H0: actual rate = expected rate
        # LR = -2 * (LL_constrained - LL_unconstrained)
        if n_violations > 0 and n_violations < n:
            lr_uc = -2 * (
                n_violations * np.log(expected_rate) + 
                (n - n_violations) * np.log(1 - expected_rate) -
                n_violations * np.log(actual_rate) - 
                (n - n_violations) * np.log(1 - actual_rate)
            )
            p_uc = 1 - stats.chi2.cdf(lr_uc, 1)
        else:
            lr_uc = np.nan
            p_uc = np.nan
        
        # 2. Christoffersen independence test
        # Count transitions
        n_00 = ((violations[:-1] == 0) & (violations[1:] == 0)).sum()
        n_01 = ((violations[:-1] == 0) & (violations[1:] == 1)).sum()
        n_10 = ((violations[:-1] == 1) & (violations[1:] == 0)).sum()
        n_11 = ((violations[:-1] == 1) & (violations[1:] == 1)).sum()
        
        if n_01 + n_11 > 0 and n_00 + n_10 > 0:
            pi_01 = n_01 / (n_00 + n_01) if (n_00 + n_01) > 0 else 0
            pi_11 = n_11 / (n_10 + n_11) if (n_10 + n_11) > 0 else 0
            pi = (n_01 + n_11) / (n - 1)
            
            # Likelihood ratio
            if pi_01 > 0 and pi_01 < 1 and pi_11 > 0 and pi_11 < 1:
                lr_ind = -2 * (
                    (n_00 + n_10) * np.log(1 - pi) + (n_01 + n_11) * np.log(pi) -
                    n_00 * np.log(1 - pi_01) - n_01 * np.log(pi_01) -
                    n_10 * np.log(1 - pi_11) - n_11 * np.log(pi_11)
                )
                p_ind = 1 - stats.chi2.cdf(lr_ind, 1)
            else:
                lr_ind = np.nan
                p_ind = np.nan
        else:
            lr_ind = np.nan
            p_ind = np.nan
        
        return {
            'n_observations': n,
            'n_violations': n_violations,
            'expected_violations': int(n * expected_rate),
            'actual_rate': actual_rate,
            'expected_rate': expected_rate,
            'kupiec_lr': lr_uc,
            'kupiec_p_value': p_uc,
            'independence_lr': lr_ind,
            'independence_p_value': p_ind,
            'pass_kupiec': p_uc > 0.05 if not np.isnan(p_uc) else None,
            'pass_independence': p_ind > 0.05 if not np.isnan(p_ind) else None
        }
```

## Summary

Normalizing flows enable sophisticated risk modeling:

| Metric | Flow Advantage |
|--------|----------------|
| **VaR** | Non-parametric, captures tails |
| **CVaR** | Exact tail expectations |
| **Marginal VaR** | Full dependence structure |
| **Stress Testing** | Conditional sampling |
| **Backtesting** | Exact likelihood comparison |

Key benefits:
1. **No distributional assumptions** - let the data speak
2. **Exact likelihoods** - proper model comparison
3. **Fast sampling** - Monte Carlo risk efficiently
4. **Conditional modeling** - time-varying risk dynamics

## References

1. Artzner, P., et al. (1999). Coherent Measures of Risk. *Mathematical Finance*.
2. Kupiec, P. (1995). Techniques for Verifying the Accuracy of Risk Measurement Models. *Journal of Derivatives*.
3. Christoffersen, P. (1998). Evaluating Interval Forecasts. *International Economic Review*.
4. Durkan, C., et al. (2019). Neural Spline Flows. *NeurIPS*.
