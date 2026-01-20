# Scenario Generation with Normalizing Flows

## Introduction

Scenario generation is essential for stress testing, portfolio optimization, risk assessment, and derivative pricing. Traditional approaches rely on parametric models (multivariate Gaussian, copulas) that may miss complex dependencies. Normalizing flows learn the **true data distribution** directly, generating realistic scenarios that preserve all statistical properties of historical data.

## Why Flows for Scenario Generation?

### Traditional Methods Limitations

| Method | Limitation |
|--------|------------|
| Historical Simulation | Limited to observed scenarios |
| Parametric (Gaussian) | Misses heavy tails, skewness |
| Copulas | Fixed dependence structure |
| GARCH | Assumes specific dynamics |

### Flow Advantages

1. **Learns arbitrary distributions** from data
2. **Captures complex dependencies** (non-linear, tail)
3. **Fast sampling** (single forward pass)
4. **Exact density** for importance weighting
5. **Conditional generation** for targeted scenarios

## Basic Scenario Generation

### Unconditional Sampling

```python
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

class ScenarioGenerator:
    """Generate financial scenarios using normalizing flows."""
    
    def __init__(
        self, 
        flow_model,
        asset_names: List[str],
        device: str = 'cpu'
    ):
        """
        Args:
            flow_model: Trained normalizing flow
            asset_names: Names of assets (columns)
            device: Computation device
        """
        self.flow = flow_model.to(device)
        self.asset_names = asset_names
        self.device = device
        self.n_assets = len(asset_names)
    
    def generate_scenarios(
        self, 
        n_scenarios: int,
        as_dataframe: bool = True
    ) -> np.ndarray:
        """
        Generate unconditional scenarios.
        
        Args:
            n_scenarios: Number of scenarios to generate
            as_dataframe: Return as DataFrame with asset names
        
        Returns:
            Array of shape (n_scenarios, n_assets) or DataFrame
        """
        self.flow.eval()
        
        with torch.no_grad():
            scenarios = self.flow.sample(n_scenarios, device=self.device)
            scenarios = scenarios.cpu().numpy()
        
        if as_dataframe:
            return pd.DataFrame(scenarios, columns=self.asset_names)
        return scenarios
    
    def generate_paths(
        self,
        n_scenarios: int,
        n_steps: int,
        dt: float = 1/252  # Daily
    ) -> np.ndarray:
        """
        Generate multi-step return paths.
        
        Assumes returns are i.i.d. (simplification).
        
        Args:
            n_scenarios: Number of paths
            n_steps: Number of time steps
            dt: Time step size (fraction of year)
        
        Returns:
            Array of shape (n_scenarios, n_steps, n_assets)
        """
        self.flow.eval()
        
        paths = np.zeros((n_scenarios, n_steps, self.n_assets))
        
        with torch.no_grad():
            for t in range(n_steps):
                returns = self.flow.sample(n_scenarios, device=self.device)
                paths[:, t, :] = returns.cpu().numpy()
        
        return paths
    
    def paths_to_prices(
        self,
        paths: np.ndarray,
        initial_prices: np.ndarray
    ) -> np.ndarray:
        """
        Convert return paths to price paths.
        
        Args:
            paths: Return paths (n_scenarios, n_steps, n_assets)
            initial_prices: Starting prices (n_assets,)
        
        Returns:
            Price paths (n_scenarios, n_steps+1, n_assets)
        """
        n_scenarios, n_steps, n_assets = paths.shape
        
        prices = np.zeros((n_scenarios, n_steps + 1, n_assets))
        prices[:, 0, :] = initial_prices
        
        for t in range(n_steps):
            prices[:, t+1, :] = prices[:, t, :] * np.exp(paths[:, t, :])
        
        return prices
```

### Scenario Statistics

```python
def analyze_scenarios(
    scenarios: pd.DataFrame,
    real_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Compare generated scenarios to real data.
    
    Args:
        scenarios: Generated scenarios DataFrame
        real_data: Historical data DataFrame
    
    Returns:
        Comparison statistics
    """
    stats = []
    
    for col in scenarios.columns:
        gen = scenarios[col]
        real = real_data[col] if col in real_data.columns else None
        
        row = {
            'Asset': col,
            'Gen Mean': gen.mean(),
            'Gen Std': gen.std(),
            'Gen Skew': gen.skew(),
            'Gen Kurt': gen.kurtosis(),
            'Gen Min': gen.min(),
            'Gen Max': gen.max(),
            'Gen 1%': gen.quantile(0.01),
            'Gen 99%': gen.quantile(0.99),
        }
        
        if real is not None:
            row.update({
                'Real Mean': real.mean(),
                'Real Std': real.std(),
                'Real Skew': real.skew(),
                'Real Kurt': real.kurtosis(),
            })
        
        stats.append(row)
    
    return pd.DataFrame(stats)
```

## Conditional Scenario Generation

### Market Regime Conditioning

```python
class ConditionalScenarioGenerator:
    """Generate scenarios conditioned on market state."""
    
    def __init__(
        self,
        conditional_flow,
        asset_names: List[str],
        feature_names: List[str],
        device: str = 'cpu'
    ):
        self.flow = conditional_flow.to(device)
        self.asset_names = asset_names
        self.feature_names = feature_names
        self.device = device
    
    def generate_conditional(
        self,
        features: np.ndarray,
        n_scenarios: int = 1000
    ) -> pd.DataFrame:
        """
        Generate scenarios given market features.
        
        Args:
            features: Market state features (1, n_features)
            n_scenarios: Number of scenarios
        
        Returns:
            DataFrame of conditional scenarios
        """
        self.flow.eval()
        
        features_tensor = torch.tensor(
            features, dtype=torch.float32, device=self.device
        )
        
        # Expand for all scenarios
        if features_tensor.dim() == 1:
            features_tensor = features_tensor.unsqueeze(0)
        features_expanded = features_tensor.expand(n_scenarios, -1)
        
        with torch.no_grad():
            scenarios = self.flow.sample(features_expanded)
            scenarios = scenarios.cpu().numpy()
        
        return pd.DataFrame(scenarios, columns=self.asset_names)
    
    def regime_scenarios(
        self,
        regime_features: Dict[str, np.ndarray],
        n_scenarios_per_regime: int = 1000
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate scenarios for multiple regimes.
        
        Args:
            regime_features: {regime_name: feature_vector}
            n_scenarios_per_regime: Scenarios per regime
        
        Returns:
            {regime_name: scenarios_df}
        """
        results = {}
        
        for regime_name, features in regime_features.items():
            scenarios = self.generate_conditional(features, n_scenarios_per_regime)
            results[regime_name] = scenarios
        
        return results
```

### Stress Scenario Generation

```python
def generate_stress_scenarios(
    generator: ConditionalScenarioGenerator,
    base_features: np.ndarray,
    stress_multipliers: Dict[str, float],
    n_scenarios: int = 10000
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate base and stressed scenarios for comparison.
    
    Args:
        generator: Conditional scenario generator
        base_features: Normal market features
        stress_multipliers: {feature_name: stress_multiplier}
        n_scenarios: Number of scenarios per case
    
    Returns:
        (base_scenarios, stressed_scenarios)
    """
    # Base case
    base_scenarios = generator.generate_conditional(base_features, n_scenarios)
    
    # Stressed case
    stressed_features = base_features.copy()
    for feature_name, multiplier in stress_multipliers.items():
        if feature_name in generator.feature_names:
            idx = generator.feature_names.index(feature_name)
            stressed_features[0, idx] *= multiplier
    
    stressed_scenarios = generator.generate_conditional(stressed_features, n_scenarios)
    
    return base_scenarios, stressed_scenarios
```

## Importance-Weighted Scenarios

### Generating Tail Scenarios

Use importance sampling to oversample extreme scenarios:

```python
class ImportanceWeightedGenerator:
    """Generate scenarios with importance weighting for tail events."""
    
    def __init__(self, flow_model, asset_names: List[str]):
        self.flow = flow_model
        self.asset_names = asset_names
    
    def generate_tail_scenarios(
        self,
        n_scenarios: int,
        tail_probability: float = 0.05,
        n_proposal: int = 100000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate scenarios from the tail with importance weights.
        
        Args:
            n_scenarios: Number of tail scenarios needed
            tail_probability: What constitutes "tail" (e.g., 5%)
            n_proposal: Proposal samples to generate
        
        Returns:
            (tail_scenarios, importance_weights)
        """
        self.flow.eval()
        
        with torch.no_grad():
            # Generate many scenarios
            all_scenarios = self.flow.sample(n_proposal)
            
            # Compute portfolio returns (equal weight for simplicity)
            portfolio_returns = all_scenarios.mean(dim=1)
            
            # Find tail threshold
            threshold = torch.quantile(portfolio_returns, tail_probability)
            
            # Select tail scenarios
            tail_mask = portfolio_returns <= threshold
            tail_scenarios = all_scenarios[tail_mask].numpy()
        
        # Importance weights (for proper expectation estimation)
        # w = p(x) / q(x) where q is uniform over selected samples
        # Here we just use equal weights since we're conditioning
        n_tail = len(tail_scenarios)
        
        if n_tail >= n_scenarios:
            # Subsample if we have enough
            indices = np.random.choice(n_tail, n_scenarios, replace=False)
            return tail_scenarios[indices], np.ones(n_scenarios) / n_scenarios
        else:
            # Use all with adjusted weights
            return tail_scenarios, np.ones(n_tail) / n_tail
    
    def estimate_tail_expectation(
        self,
        tail_scenarios: np.ndarray,
        weights: np.ndarray,
        function: callable
    ) -> float:
        """
        Estimate expectation of function over tail distribution.
        
        E[f(X) | X in tail] ≈ Σ w_i × f(x_i)
        """
        values = np.array([function(s) for s in tail_scenarios])
        return np.sum(weights * values) / np.sum(weights)
```

### Targeted Scenario Generation

```python
def generate_targeted_scenarios(
    flow_model,
    target_condition: callable,
    n_scenarios: int,
    max_attempts: int = 1000000,
    device: str = 'cpu'
) -> np.ndarray:
    """
    Generate scenarios satisfying a specific condition.
    
    Uses rejection sampling.
    
    Args:
        flow_model: Trained flow
        target_condition: Function (scenario) -> bool
        n_scenarios: Number of scenarios needed
        max_attempts: Maximum sampling attempts
        device: Computation device
    
    Returns:
        Array of scenarios satisfying condition
    """
    flow_model.eval()
    
    collected = []
    batch_size = 10000
    attempts = 0
    
    while len(collected) < n_scenarios and attempts < max_attempts:
        with torch.no_grad():
            batch = flow_model.sample(batch_size, device=device).cpu().numpy()
        
        for scenario in batch:
            if target_condition(scenario):
                collected.append(scenario)
                if len(collected) >= n_scenarios:
                    break
        
        attempts += batch_size
    
    if len(collected) < n_scenarios:
        print(f"Warning: Only found {len(collected)} scenarios after {attempts} attempts")
    
    return np.array(collected[:n_scenarios])


# Example: Generate scenarios where SPY drops > 5%
def spy_crash_condition(scenario):
    spy_idx = 0  # Assuming SPY is first asset
    return scenario[spy_idx] < -0.05

crash_scenarios = generate_targeted_scenarios(
    flow_model,
    spy_crash_condition,
    n_scenarios=1000
)
```

## Multi-Period Scenario Trees

### Building Scenario Trees

```python
class ScenarioTree:
    """Multi-period scenario tree for dynamic optimization."""
    
    def __init__(
        self,
        conditional_flow,
        asset_names: List[str],
        feature_extractor: callable
    ):
        """
        Args:
            conditional_flow: Flow conditioned on current state
            asset_names: Asset names
            feature_extractor: Function to extract features from history
        """
        self.flow = conditional_flow
        self.asset_names = asset_names
        self.feature_extractor = feature_extractor
    
    def build_tree(
        self,
        initial_features: np.ndarray,
        n_periods: int,
        branches_per_node: int = 10
    ) -> Dict:
        """
        Build scenario tree recursively.
        
        Args:
            initial_features: Starting market state
            n_periods: Number of time periods
            branches_per_node: Scenarios at each node
        
        Returns:
            Nested dictionary representing tree
        """
        self.flow.eval()
        
        def build_node(features, depth):
            if depth >= n_periods:
                return None
            
            # Generate scenarios from this node
            features_tensor = torch.tensor(
                features.reshape(1, -1), 
                dtype=torch.float32
            ).expand(branches_per_node, -1)
            
            with torch.no_grad():
                returns = self.flow.sample(features_tensor).numpy()
            
            # Build children
            children = []
            for i in range(branches_per_node):
                # Update features based on this return
                new_features = self.feature_extractor(features, returns[i])
                
                child = {
                    'return': returns[i],
                    'features': new_features,
                    'probability': 1.0 / branches_per_node,
                    'children': build_node(new_features, depth + 1)
                }
                children.append(child)
            
            return children
        
        return {
            'features': initial_features,
            'children': build_node(initial_features, 0)
        }
    
    def tree_to_paths(self, tree: Dict) -> List[np.ndarray]:
        """Convert scenario tree to list of paths."""
        paths = []
        
        def traverse(node, current_path):
            if node['children'] is None:
                paths.append(np.array(current_path))
                return
            
            for child in node['children']:
                traverse(child, current_path + [child['return']])
        
        for child in tree['children']:
            traverse(child, [child['return']])
        
        return paths
```

## Practical Applications

### Monte Carlo Simulation

```python
def monte_carlo_valuation(
    generator: ScenarioGenerator,
    payoff_function: callable,
    n_scenarios: int = 100000,
    risk_free_rate: float = 0.05,
    time_to_maturity: float = 1.0
) -> Tuple[float, float]:
    """
    Monte Carlo option/derivative pricing using flow scenarios.
    
    Args:
        generator: Scenario generator
        payoff_function: Function(final_returns) -> payoff
        n_scenarios: Number of Monte Carlo paths
        risk_free_rate: Risk-free rate for discounting
        time_to_maturity: Time to maturity in years
    
    Returns:
        (price_estimate, standard_error)
    """
    # Generate scenarios
    scenarios = generator.generate_scenarios(n_scenarios, as_dataframe=False)
    
    # Compute payoffs
    payoffs = np.array([payoff_function(s) for s in scenarios])
    
    # Discount to present value
    discount_factor = np.exp(-risk_free_rate * time_to_maturity)
    pv_payoffs = discount_factor * payoffs
    
    # Statistics
    price = pv_payoffs.mean()
    std_error = pv_payoffs.std() / np.sqrt(n_scenarios)
    
    return price, std_error


# Example: Basket option
def basket_call_payoff(returns, strike=0.0, weights=None):
    """Payoff of call option on basket of assets."""
    if weights is None:
        weights = np.ones(len(returns)) / len(returns)
    basket_return = np.dot(weights, returns)
    return max(basket_return - strike, 0)
```

### Risk Budget Allocation

```python
def scenario_based_optimization(
    generator: ScenarioGenerator,
    target_return: float,
    risk_budget: float,
    n_scenarios: int = 10000
) -> np.ndarray:
    """
    Portfolio optimization using flow-generated scenarios.
    
    Minimize CVaR subject to return constraint.
    
    Args:
        generator: Scenario generator
        target_return: Minimum required return
        risk_budget: Maximum CVaR
        n_scenarios: Number of scenarios
    
    Returns:
        Optimal weights
    """
    from scipy.optimize import minimize
    
    # Generate scenarios
    scenarios = generator.generate_scenarios(n_scenarios, as_dataframe=False)
    n_assets = scenarios.shape[1]
    
    # CVaR computation
    def compute_cvar(weights, alpha=0.95):
        portfolio_returns = scenarios @ weights
        var = np.quantile(portfolio_returns, 1 - alpha)
        cvar = -portfolio_returns[portfolio_returns <= var].mean()
        return cvar
    
    # Expected return
    def expected_return(weights):
        return (scenarios @ weights).mean()
    
    # Objective: minimize CVaR
    def objective(weights):
        return compute_cvar(weights)
    
    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda w: w.sum() - 1},  # Sum to 1
        {'type': 'ineq', 'fun': lambda w: expected_return(w) - target_return},  # Return constraint
    ]
    
    # Bounds: long-only
    bounds = [(0, 1) for _ in range(n_assets)]
    
    # Initial guess
    x0 = np.ones(n_assets) / n_assets
    
    # Optimize
    result = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    return result.x
```

## Summary

Normalizing flows provide a powerful framework for scenario generation:

| Application | Flow Capability |
|------------|-----------------|
| **Basic Scenarios** | Sample from learned distribution |
| **Conditional** | Generate given market state |
| **Stress Testing** | Targeted extreme scenarios |
| **Importance Sampling** | Tail event oversampling |
| **Multi-Period** | Scenario trees for dynamic problems |

Key benefits:
1. **Realistic scenarios** matching all moments of historical data
2. **Fast sampling** for large-scale Monte Carlo
3. **Exact densities** for importance weighting
4. **Flexible conditioning** for regime-specific scenarios

## References

1. Papamakarios, G., et al. (2021). Normalizing Flows for Probabilistic Modeling and Inference. *JMLR*.
2. Kaut, M., & Wallace, S. W. (2007). Evaluation of Scenario-Generation Methods for Stochastic Programming. *Pacific Journal of Optimization*.
3. Høyland, K., & Wallace, S. W. (2001). Generating Scenario Trees for Multistage Decision Problems. *Management Science*.
