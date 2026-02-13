"""
Chapter 35.1.5: Market Simulation for Financial RL
===================================================
Order execution, transaction costs, market impact, and synthetic data.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


# ============================================================
# Transaction Cost Models
# ============================================================

class ProportionalCostModel:
    """Simple proportional transaction cost."""
    
    def __init__(self, cost_rate: float = 0.001):
        self.cost_rate = cost_rate
    
    def compute(self, trade_values: np.ndarray) -> float:
        return self.cost_rate * np.abs(trade_values).sum()


class SpreadBasedCostModel:
    """Half-spread cost model: buy at ask, sell at bid."""
    
    def __init__(self, base_spread_bps: float = 5.0):
        self.base_spread = base_spread_bps / 10000
    
    def compute(self, trade_values: np.ndarray,
                spreads: Optional[np.ndarray] = None) -> float:
        if spreads is None:
            spreads = np.full(len(trade_values), self.base_spread)
        return (0.5 * spreads * np.abs(trade_values)).sum()


class TieredCommissionModel:
    """Tiered commission structure based on trade value."""
    
    def __init__(self, tiers: Optional[List[Tuple[float, float]]] = None,
                 min_commission: float = 1.0):
        self.tiers = tiers or [
            (10_000, 0.005),     # < $10K: 50 bps
            (100_000, 0.001),    # < $100K: 10 bps
            (float('inf'), 0.0005),  # >= $100K: 5 bps
        ]
        self.min_commission = min_commission
    
    def compute(self, trade_values: np.ndarray) -> float:
        total = 0.0
        for tv in np.abs(trade_values):
            if tv < 1e-6:
                continue
            for threshold, rate in self.tiers:
                if tv <= threshold:
                    total += max(self.min_commission, tv * rate)
                    break
        return total


# ============================================================
# Market Impact Models
# ============================================================

class LinearImpactModel:
    """Linear market impact: Δp = λ * q / V."""
    
    def __init__(self, lambda_coeff: float = 0.1):
        self.lambda_coeff = lambda_coeff
    
    def compute_impact(self, order_sizes: np.ndarray,
                       volumes: np.ndarray) -> np.ndarray:
        participation = order_sizes / (volumes + 1e-10)
        return self.lambda_coeff * participation


class SquareRootImpactModel:
    """
    Square-root impact (Almgren-Chriss model).
    
    Δp = σ * η * sign(q) * sqrt(|q| / V)
    
    Better captures the concave relationship between trade size and impact.
    """
    
    def __init__(self, eta: float = 0.1):
        self.eta = eta
    
    def compute_impact(self, order_sizes: np.ndarray,
                       volumes: np.ndarray,
                       volatilities: np.ndarray) -> np.ndarray:
        participation = np.abs(order_sizes) / (volumes + 1e-10)
        impact = volatilities * self.eta * np.sign(order_sizes) * np.sqrt(participation)
        return impact


class TemporaryPermanentImpact:
    """
    Decompose impact into temporary and permanent components.
    
    p_{t+1} = p_t + γ * q/V (permanent) + η * sign(q) * sqrt(|q|/V) (temporary)
    """
    
    def __init__(self, gamma: float = 0.05, eta: float = 0.1):
        self.gamma = gamma  # Permanent impact
        self.eta = eta      # Temporary impact
    
    def compute(self, order_sizes: np.ndarray,
                volumes: np.ndarray,
                volatilities: np.ndarray) -> Dict[str, np.ndarray]:
        participation = np.abs(order_sizes) / (volumes + 1e-10)
        
        permanent = self.gamma * order_sizes / (volumes + 1e-10)
        temporary = volatilities * self.eta * np.sign(order_sizes) * np.sqrt(participation)
        
        return {
            'permanent': permanent,
            'temporary': temporary,
            'total': permanent + temporary,
        }


# ============================================================
# Slippage Models
# ============================================================

class DeterministicSlippage:
    """Fixed slippage amount."""
    
    def __init__(self, slippage_bps: float = 1.0):
        self.slippage = slippage_bps / 10000
    
    def compute(self, order_signs: np.ndarray,
                rng: np.random.Generator = None) -> np.ndarray:
        return np.abs(order_signs) * self.slippage


class StochasticSlippage:
    """Random slippage drawn from normal distribution."""
    
    def __init__(self, mean_bps: float = 0.5, std_bps: float = 1.0):
        self.mean = mean_bps / 10000
        self.std = std_bps / 10000
    
    def compute(self, order_signs: np.ndarray,
                rng: np.random.Generator = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        n = len(order_signs)
        slippage = np.abs(rng.normal(self.mean, self.std, n))
        return slippage


class VolumeDepSlippage:
    """Volume-dependent slippage: larger orders suffer more."""
    
    def __init__(self, base_bps: float = 0.5, vol_coeff: float = 5.0):
        self.base = base_bps / 10000
        self.vol_coeff = vol_coeff / 10000
    
    def compute(self, order_sizes: np.ndarray,
                volumes: np.ndarray,
                rng: np.random.Generator = None) -> np.ndarray:
        participation = np.abs(order_sizes) / (volumes + 1e-10)
        return self.base + self.vol_coeff * participation


# ============================================================
# Full Market Simulator
# ============================================================

class MarketSimulator:
    """
    Full market simulation with configurable components.
    
    Combines cost model, impact model, and slippage model
    to simulate realistic order execution.
    """
    
    def __init__(self, cost_model=None, impact_model=None,
                 slippage_model=None):
        self.cost_model = cost_model or ProportionalCostModel(0.001)
        self.impact_model = impact_model or SquareRootImpactModel(0.1)
        self.slippage_model = slippage_model or StochasticSlippage(0.5, 1.0)
    
    def execute_trades(self, target_weights: np.ndarray,
                       current_weights: np.ndarray,
                       portfolio_value: float,
                       prices: np.ndarray,
                       volumes: Optional[np.ndarray] = None,
                       volatilities: Optional[np.ndarray] = None,
                       rng: Optional[np.random.Generator] = None) -> Dict:
        """
        Simulate order execution.
        
        Returns:
            Dictionary with execution details.
        """
        if rng is None:
            rng = np.random.default_rng()
        
        trades = target_weights - current_weights
        trade_values = trades * portfolio_value
        order_sizes = trade_values / (prices + 1e-10)
        
        # Default volumes/volatilities if not provided
        n = len(prices)
        if volumes is None:
            volumes = np.full(n, 1e6)
        if volatilities is None:
            volatilities = np.full(n, 0.02)
        
        # Compute slippage
        slippage = self.slippage_model.compute(
            order_sizes if hasattr(self.slippage_model, 'compute') and
            'order_sizes' in self.slippage_model.compute.__code__.co_varnames
            else np.sign(trades),
            rng=rng
        )
        
        # Compute market impact
        if isinstance(self.impact_model, SquareRootImpactModel):
            impact = self.impact_model.compute_impact(order_sizes, volumes, volatilities)
        elif isinstance(self.impact_model, LinearImpactModel):
            impact = self.impact_model.compute_impact(order_sizes, volumes)
        else:
            impact = np.zeros(n)
        
        # Fill prices
        fill_prices = prices * (1 + np.sign(trades) * slippage + impact)
        
        # Transaction costs
        costs = self.cost_model.compute(np.abs(trade_values))
        
        # Actual fill values (accounting for impact and slippage)
        actual_trade_values = order_sizes * fill_prices
        execution_shortfall = np.abs(actual_trade_values - trade_values).sum()
        
        return {
            'target_weights': target_weights,
            'trades': trades,
            'trade_values': trade_values,
            'fill_prices': fill_prices,
            'slippage': slippage,
            'impact': impact,
            'costs': costs,
            'execution_shortfall': execution_shortfall,
            'turnover': np.abs(trades).sum(),
        }


# ============================================================
# Synthetic Data Generation
# ============================================================

class GBMSimulator:
    """
    Geometric Brownian Motion price simulator.
    
    dS = μ * S * dt + σ * S * dW
    """
    
    def __init__(self, num_assets: int = 4, seed: int = 42):
        self.num_assets = num_assets
        self.rng = np.random.default_rng(seed)
    
    def simulate(self, num_steps: int = 1000,
                 mu: Optional[np.ndarray] = None,
                 sigma: Optional[np.ndarray] = None,
                 correlation: Optional[np.ndarray] = None,
                 initial_prices: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate correlated GBM price paths."""
        N = self.num_assets
        
        if mu is None:
            mu = self.rng.uniform(0.0001, 0.0005, N)
        if sigma is None:
            sigma = self.rng.uniform(0.01, 0.03, N)
        if initial_prices is None:
            initial_prices = self.rng.uniform(50, 200, N)
        if correlation is None:
            # Random positive-definite correlation
            A = self.rng.uniform(0.3, 0.7, (N, N))
            correlation = A @ A.T
            d = np.sqrt(np.diag(correlation))
            correlation = correlation / np.outer(d, d)
            np.fill_diagonal(correlation, 1.0)
        
        L = np.linalg.cholesky(correlation)
        
        log_returns = np.zeros((num_steps, N))
        for t in range(num_steps):
            z = self.rng.standard_normal(N)
            corr_z = L @ z
            log_returns[t] = (mu - 0.5 * sigma ** 2) + sigma * corr_z
        
        prices = initial_prices * np.exp(np.cumsum(log_returns, axis=0))
        return prices


class RegimeSwitchingSimulator:
    """
    Regime-switching price simulator with bull/bear/crisis states.
    """
    
    def __init__(self, num_assets: int = 4, seed: int = 42):
        self.num_assets = num_assets
        self.rng = np.random.default_rng(seed)
        
        # Regime parameters
        self.regimes = {
            'bull': {'mu': 0.0005, 'sigma': 0.012, 'correlation': 0.3},
            'bear': {'mu': -0.0003, 'sigma': 0.020, 'correlation': 0.5},
            'crisis': {'mu': -0.002, 'sigma': 0.040, 'correlation': 0.8},
        }
        
        # Transition matrix
        self.transition = np.array([
            [0.98, 0.015, 0.005],  # bull -> bull/bear/crisis
            [0.03, 0.95, 0.02],    # bear -> bull/bear/crisis
            [0.05, 0.10, 0.85],    # crisis -> bull/bear/crisis
        ])
    
    def simulate(self, num_steps: int = 1000,
                 initial_prices: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Generate regime-switching prices. Returns (prices, regimes)."""
        N = self.num_assets
        regime_names = list(self.regimes.keys())
        
        if initial_prices is None:
            initial_prices = self.rng.uniform(50, 200, N)
        
        # Generate regime sequence
        regimes = np.zeros(num_steps, dtype=int)
        regimes[0] = 0  # Start in bull
        for t in range(1, num_steps):
            regimes[t] = self.rng.choice(3, p=self.transition[regimes[t-1]])
        
        # Generate returns per regime
        log_returns = np.zeros((num_steps, N))
        for t in range(num_steps):
            regime = self.regimes[regime_names[regimes[t]]]
            mu = regime['mu'] + self.rng.uniform(-0.0001, 0.0001, N)
            sigma = regime['sigma'] * self.rng.uniform(0.8, 1.2, N)
            
            # Correlation
            corr = np.eye(N)
            rho = regime['correlation']
            for i in range(N):
                for j in range(i+1, N):
                    corr[i, j] = corr[j, i] = rho
            L = np.linalg.cholesky(corr)
            
            z = self.rng.standard_normal(N)
            log_returns[t] = mu + sigma * (L @ z)
        
        prices = initial_prices * np.exp(np.cumsum(log_returns, axis=0))
        return prices, regimes


class BootstrapResampler:
    """Block bootstrap resampling of historical returns."""
    
    def __init__(self, block_size: int = 20, seed: int = 42):
        self.block_size = block_size
        self.rng = np.random.default_rng(seed)
    
    def resample(self, historical_returns: np.ndarray,
                 num_steps: int = None) -> np.ndarray:
        """
        Generate synthetic returns via block bootstrap.
        
        Preserves short-term autocorrelation by resampling blocks.
        """
        T, N = historical_returns.shape
        if num_steps is None:
            num_steps = T
        
        num_blocks = (num_steps + self.block_size - 1) // self.block_size
        
        synthetic = []
        for _ in range(num_blocks):
            start = self.rng.integers(0, T - self.block_size)
            block = historical_returns[start:start + self.block_size]
            synthetic.append(block)
        
        synthetic = np.vstack(synthetic)[:num_steps]
        return synthetic
    
    def resample_prices(self, historical_prices: np.ndarray,
                        num_paths: int = 1,
                        num_steps: int = None) -> np.ndarray:
        """Generate synthetic price paths from historical data."""
        log_returns = np.diff(np.log(historical_prices), axis=0)
        
        paths = []
        for _ in range(num_paths):
            synth_returns = self.resample(log_returns, num_steps)
            synth_prices = historical_prices[0] * np.exp(np.cumsum(synth_returns, axis=0))
            paths.append(synth_prices)
        
        return np.array(paths)  # (num_paths, T, N)


class NoiseAugmenter:
    """Add calibrated noise to historical data for augmentation."""
    
    def __init__(self, noise_scale: float = 0.001, seed: int = 42):
        self.noise_scale = noise_scale
        self.rng = np.random.default_rng(seed)
    
    def augment(self, prices: np.ndarray) -> np.ndarray:
        """Add multiplicative noise to prices."""
        noise = self.rng.normal(0, self.noise_scale, prices.shape)
        return prices * np.exp(noise)
    
    def augment_returns(self, returns: np.ndarray) -> np.ndarray:
        """Add additive noise to returns."""
        noise = self.rng.normal(0, self.noise_scale, returns.shape)
        return returns + noise


# ============================================================
# Domain Randomization
# ============================================================

class DomainRandomizer:
    """Randomize environment parameters for robustness."""
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
    
    def randomize(self) -> Dict:
        """Generate randomized environment parameters."""
        return {
            'transaction_cost': self.rng.uniform(0.0003, 0.002),
            'slippage_mean_bps': self.rng.uniform(0.0, 2.0),
            'slippage_std_bps': self.rng.uniform(0.5, 3.0),
            'impact_eta': self.rng.uniform(0.05, 0.3),
            'spread_bps': self.rng.uniform(1.0, 10.0),
        }


# ============================================================
# Demo
# ============================================================

def demo_market_simulation():
    """Demonstrate market simulation components."""
    print("=" * 60)
    print("Market Simulation Demo")
    print("=" * 60)
    
    np.random.seed(42)
    rng = np.random.default_rng(42)
    
    # 1. Transaction cost models
    print("\n--- Transaction Cost Models ---")
    trade_values = np.array([50_000, 120_000, 5_000, 80_000])
    
    prop = ProportionalCostModel(0.001)
    spread = SpreadBasedCostModel(5.0)
    tiered = TieredCommissionModel()
    
    print(f"Trade values: {trade_values}")
    print(f"Proportional (10 bps): ${prop.compute(trade_values):.2f}")
    print(f"Spread-based (5 bps):  ${spread.compute(trade_values):.2f}")
    print(f"Tiered commission:     ${tiered.compute(trade_values):.2f}")
    
    # 2. Market impact
    print("\n--- Market Impact Models ---")
    order_sizes = np.array([1000, 5000, 200, 3000])
    volumes = np.array([100_000, 50_000, 200_000, 80_000])
    volatilities = np.array([0.02, 0.015, 0.025, 0.018])
    
    linear = LinearImpactModel(lambda_coeff=0.1)
    sqrt_impact = SquareRootImpactModel(eta=0.1)
    temp_perm = TemporaryPermanentImpact(gamma=0.05, eta=0.1)
    
    print(f"Order sizes:    {order_sizes}")
    print(f"Volumes:        {volumes}")
    print(f"Participation:  {(order_sizes / volumes * 100).round(2)}%")
    print(f"\nLinear impact (bps):    {(linear.compute_impact(order_sizes, volumes) * 10000).round(2)}")
    print(f"Sqrt impact (bps):      {(sqrt_impact.compute_impact(order_sizes, volumes, volatilities) * 10000).round(2)}")
    
    tp = temp_perm.compute(order_sizes, volumes, volatilities)
    print(f"Permanent (bps):        {(tp['permanent'] * 10000).round(2)}")
    print(f"Temporary (bps):        {(tp['temporary'] * 10000).round(2)}")
    
    # 3. Full execution simulation
    print("\n--- Full Market Simulator ---")
    
    sim = MarketSimulator(
        cost_model=ProportionalCostModel(0.001),
        impact_model=SquareRootImpactModel(0.1),
        slippage_model=StochasticSlippage(0.5, 1.0),
    )
    
    prices = np.array([150.0, 2800.0, 350.0, 180.0])
    current_weights = np.array([0.25, 0.25, 0.25, 0.25])
    target_weights = np.array([0.40, 0.20, 0.30, 0.10])
    portfolio_value = 1_000_000
    
    result = sim.execute_trades(
        target_weights, current_weights, portfolio_value,
        prices, volumes, volatilities, rng
    )
    
    print(f"Current weights: {current_weights}")
    print(f"Target weights:  {target_weights}")
    print(f"Trades:          {result['trades'].round(4)}")
    print(f"Turnover:        {result['turnover']:.4f}")
    print(f"Total costs:     ${result['costs']:.2f}")
    print(f"Slippage (bps):  {(result['slippage'] * 10000).round(2)}")
    print(f"Impact (bps):    {(result['impact'] * 10000).round(2)}")
    
    # 4. Synthetic data generation
    print("\n--- Synthetic Data Generation ---")
    
    # GBM
    gbm = GBMSimulator(num_assets=4, seed=42)
    gbm_prices = gbm.simulate(num_steps=500)
    print(f"GBM prices shape: {gbm_prices.shape}")
    print(f"  Final prices: {gbm_prices[-1].round(2)}")
    
    # Regime-switching
    regime_sim = RegimeSwitchingSimulator(num_assets=4, seed=42)
    rs_prices, regimes = regime_sim.simulate(num_steps=500)
    regime_names = ['bull', 'bear', 'crisis']
    regime_counts = {name: (regimes == i).sum() for i, name in enumerate(regime_names)}
    print(f"\nRegime-switching prices shape: {rs_prices.shape}")
    print(f"  Regime distribution: {regime_counts}")
    print(f"  Final prices: {rs_prices[-1].round(2)}")
    
    # Bootstrap
    hist_returns = np.diff(np.log(gbm_prices), axis=0)
    bootstrap = BootstrapResampler(block_size=20, seed=42)
    synth_returns = bootstrap.resample(hist_returns, num_steps=500)
    print(f"\nBootstrap resampled returns shape: {synth_returns.shape}")
    print(f"  Original mean: {hist_returns.mean(axis=0).round(6)}")
    print(f"  Resampled mean: {synth_returns.mean(axis=0).round(6)}")
    
    # Noise augmentation
    augmenter = NoiseAugmenter(noise_scale=0.002, seed=42)
    aug_prices = augmenter.augment(gbm_prices)
    print(f"\nNoise augmented:")
    print(f"  Max relative diff: {np.abs(aug_prices / gbm_prices - 1).max():.4%}")
    
    # 5. Domain randomization
    print("\n--- Domain Randomization ---")
    randomizer = DomainRandomizer(seed=42)
    for i in range(3):
        params = randomizer.randomize()
        print(f"  Episode {i}: cost={params['transaction_cost']*10000:.1f}bps, "
              f"spread={params['spread_bps']:.1f}bps, "
              f"impact_η={params['impact_eta']:.3f}")
    
    print("\nMarket simulation demo complete!")


if __name__ == "__main__":
    demo_market_simulation()
