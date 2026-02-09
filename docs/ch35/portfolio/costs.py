"""
Chapter 35.2.4: Transaction Costs
==================================
Realistic transaction cost models including proportional, spread-based,
market impact, and slippage models for RL portfolio environments.
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field


# ============================================================
# Configuration
# ============================================================

@dataclass
class TransactionCostConfig:
    """Configuration for transaction cost models."""
    # Proportional costs
    commission_rate: float = 0.0005    # 5 bps
    exchange_fee: float = 0.0001       # 1 bp
    tax_rate: float = 0.0              # Stamp duty etc.

    # Spread
    default_spread: float = 0.001      # 10 bps default spread

    # Market impact
    impact_coefficient: float = 0.1    # eta
    impact_exponent: float = 0.5       # Square root model

    # Slippage
    slippage_std: float = 0.0005       # Random slippage std

    # Penalty weights for reward
    turnover_penalty: float = 0.0
    smoothing_penalty: float = 0.0


# ============================================================
# Base Transaction Cost Model
# ============================================================

class TransactionCostModel:
    """Base class for transaction cost models."""

    def compute_cost(
        self,
        old_weights: np.ndarray,
        new_weights: np.ndarray,
        portfolio_value: float,
        **kwargs,
    ) -> Dict[str, float]:
        """
        Compute transaction costs.

        Args:
            old_weights: (N,) current weights
            new_weights: (N,) target weights
            portfolio_value: current portfolio value

        Returns:
            dict with 'total_cost', 'cost_rate', and component costs
        """
        raise NotImplementedError


# ============================================================
# Proportional Cost Model
# ============================================================

class ProportionalCostModel(TransactionCostModel):
    """
    Simple proportional cost: TC = c * |delta_w| * V
    Linear in turnover, single rate parameter.
    """

    def __init__(self, config: TransactionCostConfig):
        self.config = config
        self.total_rate = (
            config.commission_rate + config.exchange_fee + config.tax_rate
        )

    def compute_cost(
        self,
        old_weights: np.ndarray,
        new_weights: np.ndarray,
        portfolio_value: float,
        **kwargs,
    ) -> Dict[str, float]:
        turnover = np.sum(np.abs(new_weights - old_weights))
        total_cost = self.total_rate * turnover * portfolio_value
        cost_rate = self.total_rate * turnover

        return {
            "total_cost": float(total_cost),
            "cost_rate": float(cost_rate),
            "turnover": float(turnover),
            "commission": float(self.config.commission_rate * turnover * portfolio_value),
            "exchange_fee": float(self.config.exchange_fee * turnover * portfolio_value),
            "tax": float(self.config.tax_rate * turnover * portfolio_value),
        }


# ============================================================
# Spread-Based Cost Model
# ============================================================

class SpreadCostModel(TransactionCostModel):
    """
    Spread-based cost model.
    Cost per asset = half_spread * |delta_w| * V
    """

    def __init__(self, config: TransactionCostConfig, spreads: Optional[np.ndarray] = None):
        """
        Args:
            config: cost configuration
            spreads: (N,) bid-ask spreads per asset (if None, uses default)
        """
        self.config = config
        self.spreads = spreads

    def compute_cost(
        self,
        old_weights: np.ndarray,
        new_weights: np.ndarray,
        portfolio_value: float,
        spreads: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Dict[str, float]:
        N = len(old_weights)
        s = spreads if spreads is not None else self.spreads
        if s is None:
            s = np.full(N, self.config.default_spread)

        delta_w = np.abs(new_weights - old_weights)
        half_spreads = s / 2.0

        # Per-asset cost
        per_asset_cost = half_spreads * delta_w * portfolio_value
        total_cost = float(np.sum(per_asset_cost))
        turnover = float(np.sum(delta_w))

        return {
            "total_cost": total_cost,
            "cost_rate": total_cost / (portfolio_value + 1e-8),
            "turnover": turnover,
            "per_asset_cost": per_asset_cost.tolist(),
            "avg_half_spread": float(np.mean(half_spreads)),
        }


# ============================================================
# Market Impact Model (Almgren-Chriss)
# ============================================================

class MarketImpactModel(TransactionCostModel):
    """
    Square root market impact model (Almgren-Chriss).

    Impact = eta * sigma * sign(v) * sqrt(|v| / ADV)

    Total cost includes both spread and impact.
    """

    def __init__(
        self,
        config: TransactionCostConfig,
        daily_volumes: Optional[np.ndarray] = None,
        daily_volatilities: Optional[np.ndarray] = None,
    ):
        self.config = config
        self.daily_volumes = daily_volumes
        self.daily_volatilities = daily_volatilities

    def compute_cost(
        self,
        old_weights: np.ndarray,
        new_weights: np.ndarray,
        portfolio_value: float,
        daily_volumes: Optional[np.ndarray] = None,
        daily_volatilities: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Dict[str, float]:
        N = len(old_weights)

        adv = daily_volumes if daily_volumes is not None else self.daily_volumes
        vol = daily_volatilities if daily_volatilities is not None else self.daily_volatilities

        if adv is None:
            adv = np.full(N, 1e8)  # $100M default ADV
        if vol is None:
            vol = np.full(N, 0.02)  # 2% daily vol default

        delta_w = new_weights - old_weights
        trade_value = np.abs(delta_w) * portfolio_value

        # Spread cost
        spread_cost = self.config.default_spread / 2.0 * trade_value

        # Market impact (square root)
        eta = self.config.impact_coefficient
        impact_per_asset = (
            eta * vol * np.sqrt(trade_value / (adv + 1e-8))
            * trade_value
        )

        # Total
        total_spread = float(np.sum(spread_cost))
        total_impact = float(np.sum(impact_per_asset))
        total_cost = total_spread + total_impact

        return {
            "total_cost": total_cost,
            "cost_rate": total_cost / (portfolio_value + 1e-8),
            "turnover": float(np.sum(np.abs(delta_w))),
            "spread_cost": total_spread,
            "impact_cost": total_impact,
            "per_asset_impact": impact_per_asset.tolist(),
            "participation_rate": (trade_value / (adv + 1e-8)).tolist(),
        }

    def estimate_optimal_horizon(
        self,
        trade_value: float,
        adv: float,
        volatility: float,
        risk_aversion: float = 1e-6,
    ) -> float:
        """
        Estimate optimal execution horizon (Almgren-Chriss).

        T* = (trade_value / (volatility * adv))^(2/3) / risk_aversion^(1/3)
        """
        participation = trade_value / (adv + 1e-8)
        t_star = (participation ** (2 / 3)) / (risk_aversion ** (1 / 3) + 1e-8)
        return max(1.0, min(t_star, 20.0))  # Clamp to [1, 20] days


# ============================================================
# Slippage Model
# ============================================================

class SlippageModel:
    """
    Random slippage model.
    Execution price = mid_price + half_spread * sign(trade) + random_slippage
    """

    def __init__(self, config: TransactionCostConfig):
        self.config = config

    def simulate_execution(
        self,
        mid_prices: np.ndarray,
        trade_directions: np.ndarray,
        spreads: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate execution prices with slippage.

        Args:
            mid_prices: (N,) mid prices
            trade_directions: (N,) sign of trade (+1 buy, -1 sell, 0 hold)
            spreads: (N,) bid-ask spreads (optional)

        Returns:
            exec_prices: (N,) execution prices
            slippage: (N,) realized slippage
        """
        N = len(mid_prices)
        if spreads is None:
            spreads = np.full(N, self.config.default_spread)

        # Spread crossing
        spread_cost = spreads / 2.0 * trade_directions

        # Random slippage
        random_slip = np.random.normal(0, self.config.slippage_std, N)
        random_slip *= np.abs(trade_directions)  # Only for actual trades

        exec_prices = mid_prices + spread_cost + random_slip
        total_slippage = spread_cost + random_slip

        return exec_prices, total_slippage


# ============================================================
# Composite Cost Model
# ============================================================

class CompositeCostModel(TransactionCostModel):
    """
    Combines multiple cost components into a single model.
    Used in production environments for realistic simulation.
    """

    def __init__(
        self,
        config: TransactionCostConfig,
        spreads: Optional[np.ndarray] = None,
        daily_volumes: Optional[np.ndarray] = None,
        daily_volatilities: Optional[np.ndarray] = None,
    ):
        self.config = config
        self.proportional = ProportionalCostModel(config)
        self.spread = SpreadCostModel(config, spreads)
        self.impact = MarketImpactModel(config, daily_volumes, daily_volatilities)
        self.slippage = SlippageModel(config)

    def compute_cost(
        self,
        old_weights: np.ndarray,
        new_weights: np.ndarray,
        portfolio_value: float,
        spreads: Optional[np.ndarray] = None,
        daily_volumes: Optional[np.ndarray] = None,
        daily_volatilities: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Dict[str, float]:
        # Commission and fees
        prop = self.proportional.compute_cost(old_weights, new_weights, portfolio_value)

        # Spread cost
        spr = self.spread.compute_cost(
            old_weights, new_weights, portfolio_value, spreads=spreads
        )

        # Market impact
        imp = self.impact.compute_cost(
            old_weights, new_weights, portfolio_value,
            daily_volumes=daily_volumes,
            daily_volatilities=daily_volatilities,
        )

        total = prop["commission"] + spr["total_cost"] + imp["impact_cost"]

        return {
            "total_cost": total,
            "cost_rate": total / (portfolio_value + 1e-8),
            "turnover": prop["turnover"],
            "commission": prop["commission"],
            "spread_cost": spr["total_cost"],
            "impact_cost": imp["impact_cost"],
            "participation_rate": imp.get("participation_rate", []),
        }


# ============================================================
# Cost-Aware Reward Wrapper
# ============================================================

class CostAwareReward:
    """
    Wraps a base reward with transaction cost penalties.
    r_adjusted = r_base - TC - lambda_turn * turnover - lambda_smooth * smoothness
    """

    def __init__(
        self,
        cost_model: TransactionCostModel,
        config: TransactionCostConfig,
    ):
        self.cost_model = cost_model
        self.config = config
        self.prev_weights: Optional[np.ndarray] = None

    def reset(self):
        self.prev_weights = None

    def compute(
        self,
        base_reward: float,
        old_weights: np.ndarray,
        new_weights: np.ndarray,
        portfolio_value: float,
        **kwargs,
    ) -> Tuple[float, Dict]:
        # Transaction cost
        cost_info = self.cost_model.compute_cost(
            old_weights, new_weights, portfolio_value, **kwargs
        )
        tc_penalty = cost_info["cost_rate"]

        # Turnover penalty
        turnover_penalty = self.config.turnover_penalty * cost_info["turnover"]

        # Smoothing penalty
        smooth_penalty = 0.0
        if self.prev_weights is not None:
            smooth_penalty = (
                self.config.smoothing_penalty
                * np.sum((new_weights - self.prev_weights) ** 2)
            )
        self.prev_weights = new_weights.copy()

        adjusted_reward = base_reward - tc_penalty - turnover_penalty - smooth_penalty

        info = {
            **cost_info,
            "base_reward": base_reward,
            "adjusted_reward": adjusted_reward,
            "turnover_penalty": turnover_penalty,
            "smoothing_penalty": smooth_penalty,
        }
        return float(adjusted_reward), info


# ============================================================
# Demonstration
# ============================================================

def demo_transaction_costs():
    """Demonstrate transaction cost models."""
    print("=" * 70)
    print("Transaction Cost Models Demonstration")
    print("=" * 70)

    config = TransactionCostConfig(
        commission_rate=0.0005,
        exchange_fee=0.0001,
        default_spread=0.001,
        impact_coefficient=0.1,
        slippage_std=0.0005,
    )

    N = 5
    old_weights = np.array([0.30, 0.25, 0.20, 0.15, 0.10])
    new_weights = np.array([0.25, 0.20, 0.25, 0.20, 0.10])
    portfolio_value = 1_000_000.0

    print(f"\nPortfolio value: ${portfolio_value:,.0f}")
    print(f"Old weights: {old_weights}")
    print(f"New weights: {new_weights}")
    print(f"Turnover: {np.sum(np.abs(new_weights - old_weights)):.4f}")

    # --- Proportional Cost ---
    print("\n--- Proportional Cost Model ---")
    prop_model = ProportionalCostModel(config)
    result = prop_model.compute_cost(old_weights, new_weights, portfolio_value)
    print(f"Total cost: ${result['total_cost']:,.2f}")
    print(f"Cost rate: {result['cost_rate']*10000:.2f} bps")
    print(f"Commission: ${result['commission']:,.2f}")

    # --- Spread-Based Cost ---
    print("\n--- Spread-Based Cost Model ---")
    spreads = np.array([0.0005, 0.0008, 0.001, 0.002, 0.003])
    spread_model = SpreadCostModel(config, spreads)
    result = spread_model.compute_cost(old_weights, new_weights, portfolio_value)
    print(f"Total cost: ${result['total_cost']:,.2f}")
    print(f"Per-asset cost: {['${:.2f}'.format(c) for c in result['per_asset_cost']]}")

    # --- Market Impact ---
    print("\n--- Market Impact Model ---")
    adv = np.array([5e8, 3e8, 2e8, 1e8, 5e7])  # Average daily volume
    vols = np.array([0.015, 0.018, 0.022, 0.025, 0.030])  # Daily vol
    impact_model = MarketImpactModel(config, adv, vols)
    result = impact_model.compute_cost(
        old_weights, new_weights, portfolio_value,
        daily_volumes=adv, daily_volatilities=vols,
    )
    print(f"Total cost: ${result['total_cost']:,.2f}")
    print(f"Spread cost: ${result['spread_cost']:,.2f}")
    print(f"Impact cost: ${result['impact_cost']:,.2f}")
    print(f"Participation rates: {[f'{r:.4f}' for r in result['participation_rate']]}")

    # --- Impact scaling ---
    print("\n--- Cost Scaling with Trade Size ---")
    print(f"{'Trade Size ($)':>15} {'Prop Cost':>12} {'Impact Cost':>12} {'Total':>12}")
    print("-" * 55)
    for multiplier in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        pv = portfolio_value * multiplier
        p = prop_model.compute_cost(old_weights, new_weights, pv)
        i = impact_model.compute_cost(
            old_weights, new_weights, pv,
            daily_volumes=adv, daily_volatilities=vols,
        )
        print(f"${pv:>14,.0f} ${p['total_cost']:>10,.0f} "
              f"${i['impact_cost']:>10,.0f} ${p['total_cost']+i['impact_cost']:>10,.0f}")

    # --- Slippage ---
    print("\n--- Slippage Simulation ---")
    slippage_model = SlippageModel(config)
    mid_prices = np.array([150.0, 2800.0, 340.0, 180.0, 95.0])
    directions = np.sign(new_weights - old_weights)

    np.random.seed(42)
    exec_prices, slippage = slippage_model.simulate_execution(
        mid_prices, directions, spreads
    )
    print(f"Mid prices:  {mid_prices}")
    print(f"Exec prices: {exec_prices}")
    print(f"Slippage:    {slippage}")
    print(f"Avg slip:    {np.mean(np.abs(slippage[directions != 0])):.6f}")

    # --- Composite Cost ---
    print("\n--- Composite Cost Model ---")
    composite = CompositeCostModel(config, spreads, adv, vols)
    result = composite.compute_cost(
        old_weights, new_weights, portfolio_value,
        spreads=spreads, daily_volumes=adv, daily_volatilities=vols,
    )
    print(f"Commission:  ${result['commission']:,.2f}")
    print(f"Spread:      ${result['spread_cost']:,.2f}")
    print(f"Impact:      ${result['impact_cost']:,.2f}")
    print(f"TOTAL:       ${result['total_cost']:,.2f}")
    print(f"Total bps:   {result['cost_rate']*10000:.2f}")

    # --- Cost-Aware Reward ---
    print("\n--- Cost-Aware Reward ---")
    cost_config = TransactionCostConfig(
        commission_rate=0.0005,
        default_spread=0.001,
        turnover_penalty=0.001,
        smoothing_penalty=0.01,
    )
    cost_reward = CostAwareReward(
        ProportionalCostModel(cost_config), cost_config
    )
    cost_reward.reset()

    base_reward = 0.005  # 50 bps base return
    adj_reward, info = cost_reward.compute(
        base_reward, old_weights, new_weights, portfolio_value
    )
    print(f"Base reward:      {base_reward:.6f}")
    print(f"TC penalty:       {info['cost_rate']:.6f}")
    print(f"Turnover penalty: {info['turnover_penalty']:.6f}")
    print(f"Adjusted reward:  {adj_reward:.6f}")

    # --- Annual cost drag ---
    print("\n--- Annual Cost Drag Analysis ---")
    print(f"{'Turnover/yr':>12} {'Avg Cost':>10} {'Annual Drag':>12}")
    print("-" * 38)
    for annual_to in [1.0, 2.0, 5.0, 10.0, 20.0]:
        for avg_cost in [0.0005, 0.001, 0.002]:
            drag = annual_to * avg_cost * 100
            print(f"{annual_to:>11.0f}x {avg_cost*10000:>8.1f}bps {drag:>10.1f}%")
        print()


if __name__ == "__main__":
    demo_transaction_costs()
