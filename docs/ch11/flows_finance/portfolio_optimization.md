# Portfolio Optimization with Normalizing Flows

## Introduction

Traditional portfolio optimization relies on parametric assumptions (Gaussian returns, linear correlations) that fail to capture the true nature of financial returns. Normalizing flows enable **distribution-aware portfolio optimization** that accounts for heavy tails, skewness, and complex dependencies—leading to more robust allocation decisions.

## Limitations of Traditional Methods

### Mean-Variance Optimization Issues

Markowitz mean-variance optimization assumes:
- Returns are normally distributed
- Risk is fully captured by variance
- Correlations are constant

Reality:
- Heavy tails (extreme losses more likely)
- Asymmetric returns (skewness)
- Tail dependence (correlations spike in crises)
- Time-varying dynamics

### How Flows Address These Issues

| Problem | Flow Solution |
|---------|--------------|
| Heavy tails | Learns true tail distribution |
| Skewness | Captures asymmetry |
| Tail dependence | Models joint extreme events |
| Dynamic | Conditional flows for regimes |

## Flow-Based Risk Measures

### Beyond Variance: Full Distribution Risk

```python
import torch
import numpy as np
from scipy.optimize import minimize
from typing import Tuple, Dict, List

class FlowPortfolioOptimizer:
    """Portfolio optimization using normalizing flow scenarios."""
    
    def __init__(
        self,
        flow_model,
        asset_names: List[str],
        n_scenarios: int = 50000,
        device: str = 'cpu'
    ):
        """
        Args:
            flow_model: Trained joint return distribution flow
            asset_names: Asset names
            n_scenarios: Number of Monte Carlo scenarios
            device: Computation device
        """
        self.flow = flow_model.to(device)
        self.asset_names = asset_names
        self.n_assets = len(asset_names)
        self.n_scenarios = n_scenarios
        self.device = device
        
        # Pre-generate scenarios for optimization
        self._generate_scenarios()
    
    def _generate_scenarios(self):
        """Generate and cache scenarios."""
        self.flow.eval()
        with torch.no_grad():
            self.scenarios = self.flow.sample(
                self.n_scenarios, 
                device=self.device
            ).cpu().numpy()
    
    def portfolio_return(self, weights: np.ndarray) -> np.ndarray:
        """Compute portfolio returns for all scenarios."""
        return self.scenarios @ weights
    
    def expected_return(self, weights: np.ndarray) -> float:
        """Expected portfolio return."""
        return self.portfolio_return(weights).mean()
    
    def variance(self, weights: np.ndarray) -> float:
        """Portfolio variance."""
        return self.portfolio_return(weights).var()
    
    def var(self, weights: np.ndarray, alpha: float = 0.95) -> float:
        """Value at Risk."""
        returns = self.portfolio_return(weights)
        return -np.quantile(returns, 1 - alpha)
    
    def cvar(self, weights: np.ndarray, alpha: float = 0.95) -> float:
        """Conditional Value at Risk (Expected Shortfall)."""
        returns = self.portfolio_return(weights)
        var_threshold = np.quantile(returns, 1 - alpha)
        return -returns[returns <= var_threshold].mean()
    
    def downside_deviation(self, weights: np.ndarray, threshold: float = 0.0) -> float:
        """Downside deviation (semi-deviation below threshold)."""
        returns = self.portfolio_return(weights)
        downside = returns[returns < threshold] - threshold
        return np.sqrt((downside ** 2).mean()) if len(downside) > 0 else 0.0
    
    def sortino_ratio(self, weights: np.ndarray, rf: float = 0.0) -> float:
        """Sortino ratio: excess return / downside deviation."""
        excess_return = self.expected_return(weights) - rf
        dd = self.downside_deviation(weights, rf)
        return excess_return / dd if dd > 0 else np.inf
    
    def omega_ratio(self, weights: np.ndarray, threshold: float = 0.0) -> float:
        """Omega ratio: probability-weighted gains / losses."""
        returns = self.portfolio_return(weights)
        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns <= threshold]
        
        sum_gains = gains.sum() if len(gains) > 0 else 0
        sum_losses = losses.sum() if len(losses) > 0 else 1e-10
        
        return sum_gains / sum_losses
```

## Optimization Objectives

### Minimum CVaR Portfolio

```python
def optimize_min_cvar(
    optimizer: FlowPortfolioOptimizer,
    min_return: float = None,
    alpha: float = 0.95,
    long_only: bool = True
) -> Tuple[np.ndarray, Dict]:
    """
    Find portfolio minimizing CVaR.
    
    Args:
        optimizer: Flow portfolio optimizer
        min_return: Minimum expected return constraint
        alpha: CVaR confidence level
        long_only: Whether to enforce long-only constraint
    
    Returns:
        (optimal_weights, metrics)
    """
    n = optimizer.n_assets
    
    # Objective: minimize CVaR
    def objective(w):
        return optimizer.cvar(w, alpha)
    
    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda w: w.sum() - 1}  # Weights sum to 1
    ]
    
    if min_return is not None:
        constraints.append({
            'type': 'ineq',
            'fun': lambda w: optimizer.expected_return(w) - min_return
        })
    
    # Bounds
    if long_only:
        bounds = [(0, 1) for _ in range(n)]
    else:
        bounds = [(-1, 1) for _ in range(n)]
    
    # Initial guess (equal weight)
    w0 = np.ones(n) / n
    
    # Optimize
    result = minimize(
        objective,
        w0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-8}
    )
    
    optimal_weights = result.x
    
    # Compute metrics at optimum
    metrics = {
        'expected_return': optimizer.expected_return(optimal_weights),
        'volatility': np.sqrt(optimizer.variance(optimal_weights)),
        'var_95': optimizer.var(optimal_weights, 0.95),
        'cvar_95': optimizer.cvar(optimal_weights, 0.95),
        'var_99': optimizer.var(optimal_weights, 0.99),
        'cvar_99': optimizer.cvar(optimal_weights, 0.99),
        'sortino': optimizer.sortino_ratio(optimal_weights),
        'converged': result.success
    }
    
    return optimal_weights, metrics
```

### Maximum Sortino Ratio

```python
def optimize_max_sortino(
    optimizer: FlowPortfolioOptimizer,
    risk_free_rate: float = 0.0,
    max_cvar: float = None,
    long_only: bool = True
) -> Tuple[np.ndarray, Dict]:
    """
    Find portfolio maximizing Sortino ratio.
    
    Args:
        optimizer: Flow portfolio optimizer
        risk_free_rate: Risk-free rate
        max_cvar: Maximum CVaR constraint
        long_only: Whether to enforce long-only
    
    Returns:
        (optimal_weights, metrics)
    """
    n = optimizer.n_assets
    
    # Objective: maximize Sortino = minimize negative Sortino
    def objective(w):
        return -optimizer.sortino_ratio(w, risk_free_rate)
    
    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda w: w.sum() - 1}
    ]
    
    if max_cvar is not None:
        constraints.append({
            'type': 'ineq',
            'fun': lambda w: max_cvar - optimizer.cvar(w, 0.95)
        })
    
    # Bounds
    bounds = [(0, 1) if long_only else (-1, 1) for _ in range(n)]
    
    # Initial guess
    w0 = np.ones(n) / n
    
    # Optimize
    result = minimize(
        objective,
        w0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000}
    )
    
    optimal_weights = result.x
    
    metrics = {
        'expected_return': optimizer.expected_return(optimal_weights),
        'downside_deviation': optimizer.downside_deviation(optimal_weights, risk_free_rate),
        'sortino_ratio': optimizer.sortino_ratio(optimal_weights, risk_free_rate),
        'cvar_95': optimizer.cvar(optimal_weights, 0.95),
        'converged': result.success
    }
    
    return optimal_weights, metrics
```

### Mean-CVaR Efficient Frontier

```python
def compute_cvar_frontier(
    optimizer: FlowPortfolioOptimizer,
    n_points: int = 20,
    alpha: float = 0.95,
    long_only: bool = True
) -> pd.DataFrame:
    """
    Compute the mean-CVaR efficient frontier.
    
    Args:
        optimizer: Flow portfolio optimizer
        n_points: Number of frontier points
        alpha: CVaR confidence level
        long_only: Long-only constraint
    
    Returns:
        DataFrame with frontier portfolios
    """
    # Find return range
    # Min return: minimum CVaR portfolio
    w_min_cvar, _ = optimize_min_cvar(optimizer, min_return=None, alpha=alpha, long_only=long_only)
    min_ret = optimizer.expected_return(w_min_cvar)
    
    # Max return: maximum return portfolio (100% in best asset)
    asset_returns = [optimizer.expected_return(np.eye(optimizer.n_assets)[i]) 
                    for i in range(optimizer.n_assets)]
    max_ret = max(asset_returns) * 0.95  # Slightly below max for feasibility
    
    # Generate frontier
    target_returns = np.linspace(min_ret, max_ret, n_points)
    
    frontier = []
    for target in target_returns:
        try:
            weights, metrics = optimize_min_cvar(
                optimizer, 
                min_return=target, 
                alpha=alpha, 
                long_only=long_only
            )
            
            if metrics['converged']:
                frontier.append({
                    'expected_return': metrics['expected_return'],
                    'cvar': metrics[f'cvar_{int(alpha*100)}'],
                    'volatility': metrics['volatility'],
                    'weights': weights
                })
        except:
            continue
    
    return pd.DataFrame(frontier)
```

## Risk Parity with Flows

### CVaR Contribution Risk Parity

```python
def cvar_risk_parity(
    optimizer: FlowPortfolioOptimizer,
    alpha: float = 0.95,
    tol: float = 1e-6,
    max_iter: int = 100
) -> Tuple[np.ndarray, Dict]:
    """
    Find portfolio where each asset contributes equally to CVaR.
    
    Risk parity: each asset's marginal contribution × weight is equal.
    
    Args:
        optimizer: Flow portfolio optimizer
        alpha: CVaR confidence level
        tol: Convergence tolerance
        max_iter: Maximum iterations
    
    Returns:
        (weights, metrics)
    """
    n = optimizer.n_assets
    
    def marginal_cvar(weights, delta=0.001):
        """Compute marginal CVaR for each asset."""
        base_cvar = optimizer.cvar(weights, alpha)
        marginals = np.zeros(n)
        
        for i in range(n):
            w_perturbed = weights.copy()
            w_perturbed[i] += delta
            w_perturbed /= w_perturbed.sum()  # Renormalize
            
            marginals[i] = (optimizer.cvar(w_perturbed, alpha) - base_cvar) / delta
        
        return marginals
    
    def cvar_contributions(weights):
        """CVaR contribution = weight × marginal CVaR."""
        marginals = marginal_cvar(weights)
        return weights * marginals
    
    # Objective: minimize variance of risk contributions
    def objective(w):
        if np.any(w <= 0):
            return 1e10
        w = w / w.sum()  # Normalize
        contributions = cvar_contributions(w)
        # Minimize variance of contributions
        return np.var(contributions)
    
    # Bounds (positive weights)
    bounds = [(0.001, 1) for _ in range(n)]
    
    # Initial guess
    w0 = np.ones(n) / n
    
    # Optimize
    result = minimize(
        objective,
        w0,
        method='SLSQP',
        bounds=bounds,
        constraints=[{'type': 'eq', 'fun': lambda w: w.sum() - 1}],
        options={'maxiter': max_iter, 'ftol': tol}
    )
    
    optimal_weights = result.x / result.x.sum()
    
    # Compute contributions at optimum
    contributions = cvar_contributions(optimal_weights)
    
    metrics = {
        'expected_return': optimizer.expected_return(optimal_weights),
        'cvar_95': optimizer.cvar(optimal_weights, alpha),
        'contributions': contributions,
        'contribution_std': np.std(contributions),
        'converged': result.success
    }
    
    return optimal_weights, metrics
```

## Robust Optimization

### Worst-Case CVaR

Account for model uncertainty by considering worst case over distribution shifts:

```python
class RobustFlowOptimizer:
    """Robust portfolio optimization with distribution uncertainty."""
    
    def __init__(
        self,
        flow_model,
        asset_names: List[str],
        n_scenarios: int = 50000,
        device: str = 'cpu'
    ):
        self.flow = flow_model.to(device)
        self.asset_names = asset_names
        self.n_assets = len(asset_names)
        self.device = device
        
        # Generate base scenarios
        self.flow.eval()
        with torch.no_grad():
            self.base_scenarios = self.flow.sample(
                n_scenarios, device=device
            ).cpu().numpy()
    
    def worst_case_cvar(
        self,
        weights: np.ndarray,
        alpha: float = 0.95,
        uncertainty_radius: float = 0.1
    ) -> float:
        """
        Compute worst-case CVaR over uncertainty set.
        
        Simple approach: perturb scenarios by adding uncertainty.
        
        Args:
            weights: Portfolio weights
            alpha: CVaR confidence level
            uncertainty_radius: Maximum perturbation (fraction of return)
        
        Returns:
            Worst-case CVaR
        """
        # Perturb scenarios adversarially
        portfolio_returns = self.base_scenarios @ weights
        
        # Worst case: reduce all returns by uncertainty
        worst_returns = portfolio_returns - uncertainty_radius * np.abs(portfolio_returns)
        
        # CVaR on worst-case returns
        var_threshold = np.quantile(worst_returns, 1 - alpha)
        worst_cvar = -worst_returns[worst_returns <= var_threshold].mean()
        
        return worst_cvar
    
    def optimize_robust(
        self,
        min_return: float = None,
        alpha: float = 0.95,
        uncertainty_radius: float = 0.1,
        long_only: bool = True
    ) -> Tuple[np.ndarray, Dict]:
        """
        Robust portfolio minimizing worst-case CVaR.
        """
        n = self.n_assets
        
        def objective(w):
            return self.worst_case_cvar(w, alpha, uncertainty_radius)
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: w.sum() - 1}
        ]
        
        if min_return is not None:
            # Use base scenarios for return constraint
            constraints.append({
                'type': 'ineq',
                'fun': lambda w: (self.base_scenarios @ w).mean() - min_return
            })
        
        bounds = [(0, 1) if long_only else (-1, 1) for _ in range(n)]
        w0 = np.ones(n) / n
        
        result = minimize(
            objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        optimal_weights = result.x
        
        metrics = {
            'expected_return': (self.base_scenarios @ optimal_weights).mean(),
            'base_cvar': self._cvar(optimal_weights, alpha),
            'worst_case_cvar': self.worst_case_cvar(optimal_weights, alpha, uncertainty_radius),
            'converged': result.success
        }
        
        return optimal_weights, metrics
    
    def _cvar(self, weights, alpha):
        returns = self.base_scenarios @ weights
        var = np.quantile(returns, 1 - alpha)
        return -returns[returns <= var].mean()
```

## Dynamic Portfolio Optimization

### Conditional Flow for Regime-Aware Allocation

```python
class DynamicFlowOptimizer:
    """Dynamic portfolio optimization using conditional flows."""
    
    def __init__(
        self,
        conditional_flow,
        asset_names: List[str],
        feature_names: List[str],
        n_scenarios: int = 50000,
        device: str = 'cpu'
    ):
        self.flow = conditional_flow.to(device)
        self.asset_names = asset_names
        self.feature_names = feature_names
        self.n_assets = len(asset_names)
        self.n_scenarios = n_scenarios
        self.device = device
    
    def optimize_for_state(
        self,
        current_features: np.ndarray,
        risk_aversion: float = 1.0,
        alpha: float = 0.95,
        long_only: bool = True
    ) -> Tuple[np.ndarray, Dict]:
        """
        Optimize portfolio given current market state.
        
        Args:
            current_features: Current market features
            risk_aversion: Risk aversion parameter
            alpha: CVaR confidence level
            long_only: Long-only constraint
        
        Returns:
            (optimal_weights, metrics)
        """
        # Generate scenarios conditioned on current state
        self.flow.eval()
        
        features_tensor = torch.tensor(
            current_features.reshape(1, -1), 
            dtype=torch.float32,
            device=self.device
        ).expand(self.n_scenarios, -1)
        
        with torch.no_grad():
            scenarios = self.flow.sample(features_tensor).cpu().numpy()
        
        # Create optimizer for these scenarios
        optimizer = FlowPortfolioOptimizer(
            None,  # We pass scenarios directly
            self.asset_names,
            self.n_scenarios
        )
        optimizer.scenarios = scenarios
        
        # Optimize using mean-CVaR utility
        def objective(w):
            expected_ret = optimizer.expected_return(w)
            cvar = optimizer.cvar(w, alpha)
            # Utility = expected return - risk_aversion × CVaR
            return -(expected_ret - risk_aversion * cvar)
        
        n = self.n_assets
        constraints = [{'type': 'eq', 'fun': lambda w: w.sum() - 1}]
        bounds = [(0, 1) if long_only else (-1, 1) for _ in range(n)]
        w0 = np.ones(n) / n
        
        result = minimize(
            objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        optimal_weights = result.x
        
        metrics = {
            'expected_return': optimizer.expected_return(optimal_weights),
            'volatility': np.sqrt(optimizer.variance(optimal_weights)),
            'cvar_95': optimizer.cvar(optimal_weights, 0.95),
            'features': current_features,
            'converged': result.success
        }
        
        return optimal_weights, metrics
    
    def backtest_dynamic(
        self,
        features_history: np.ndarray,
        returns_history: np.ndarray,
        risk_aversion: float = 1.0,
        rebalance_frequency: int = 20  # Monthly
    ) -> pd.DataFrame:
        """
        Backtest dynamic strategy.
        
        Args:
            features_history: (T, n_features) features over time
            returns_history: (T, n_assets) actual returns
            risk_aversion: Risk aversion parameter
            rebalance_frequency: Days between rebalancing
        
        Returns:
            Backtest results DataFrame
        """
        T = len(returns_history)
        results = []
        
        current_weights = np.ones(self.n_assets) / self.n_assets
        
        for t in range(T):
            # Rebalance
            if t % rebalance_frequency == 0:
                current_weights, _ = self.optimize_for_state(
                    features_history[t],
                    risk_aversion
                )
            
            # Compute return
            portfolio_return = returns_history[t] @ current_weights
            
            results.append({
                'date': t,
                'portfolio_return': portfolio_return,
                'weights': current_weights.copy()
            })
        
        return pd.DataFrame(results)
```

## Comparison with Traditional Methods

```python
def compare_optimization_methods(
    flow_model,
    historical_returns: np.ndarray,
    asset_names: List[str],
    test_returns: np.ndarray
) -> pd.DataFrame:
    """
    Compare flow-based optimization with traditional methods.
    
    Args:
        flow_model: Trained flow
        historical_returns: Training data
        asset_names: Asset names
        test_returns: Out-of-sample test data
    
    Returns:
        Comparison metrics DataFrame
    """
    results = []
    
    # 1. Mean-Variance (traditional)
    mean = historical_returns.mean(axis=0)
    cov = np.cov(historical_returns.T)
    
    def mv_objective(w, gamma=1):
        ret = w @ mean
        risk = w @ cov @ w
        return -(ret - gamma * risk)
    
    n = len(asset_names)
    w_mv = minimize(
        mv_objective,
        np.ones(n) / n,
        constraints=[{'type': 'eq', 'fun': lambda w: w.sum() - 1}],
        bounds=[(0, 1) for _ in range(n)]
    ).x
    
    # 2. Flow-based CVaR optimization
    flow_optimizer = FlowPortfolioOptimizer(flow_model, asset_names)
    w_flow, _ = optimize_min_cvar(flow_optimizer, min_return=mean @ w_mv)
    
    # 3. Equal weight
    w_equal = np.ones(n) / n
    
    # Evaluate on test set
    methods = {
        'Mean-Variance': w_mv,
        'Flow CVaR': w_flow,
        'Equal Weight': w_equal
    }
    
    for name, weights in methods.items():
        test_portfolio = test_returns @ weights
        
        results.append({
            'Method': name,
            'Return (ann.)': test_portfolio.mean() * 252,
            'Volatility (ann.)': test_portfolio.std() * np.sqrt(252),
            'Sharpe': test_portfolio.mean() / test_portfolio.std() * np.sqrt(252),
            'VaR 95%': -np.quantile(test_portfolio, 0.05),
            'CVaR 95%': -test_portfolio[test_portfolio <= np.quantile(test_portfolio, 0.05)].mean(),
            'Max Drawdown': compute_max_drawdown(np.cumprod(1 + test_portfolio)),
            'Sortino': test_portfolio.mean() / test_portfolio[test_portfolio < 0].std() * np.sqrt(252)
        })
    
    return pd.DataFrame(results)


def compute_max_drawdown(prices):
    """Compute maximum drawdown from price series."""
    peak = np.maximum.accumulate(prices)
    drawdown = (prices - peak) / peak
    return drawdown.min()
```

## Summary

Normalizing flows transform portfolio optimization by:

| Traditional | Flow-Based |
|------------|------------|
| Gaussian returns | Learned distribution |
| Variance risk | CVaR, downside measures |
| Linear correlation | Full dependence structure |
| Static | Dynamic, regime-aware |
| Point estimates | Full distribution of outcomes |

Key optimization frameworks:
1. **Minimum CVaR**: Minimize tail risk
2. **Maximum Sortino**: Best risk-adjusted returns
3. **Risk Parity**: Equal risk contribution
4. **Robust**: Worst-case over uncertainty
5. **Dynamic**: Regime-conditional allocation

## References

1. Rockafellar, R. T., & Uryasev, S. (2000). Optimization of Conditional Value-at-Risk. *Journal of Risk*.
2. Maillard, S., et al. (2010). The Properties of Equally Weighted Risk Contribution Portfolios. *Journal of Portfolio Management*.
3. Papamakarios, G., et al. (2021). Normalizing Flows for Probabilistic Modeling and Inference. *JMLR*.
4. Kolm, P. N., et al. (2014). 60 Years of Portfolio Optimization. *European Journal of Operational Research*.
