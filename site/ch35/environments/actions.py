"""
Chapter 35.1.3: Action Spaces for Financial RL
===============================================
Action space designs, constraint handling, and action transformations.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


# ============================================================
# Action Transformations
# ============================================================

def softmax_transform(raw_action: np.ndarray) -> np.ndarray:
    """Transform raw output to long-only weights via softmax."""
    exp_a = np.exp(raw_action - np.max(raw_action))
    return exp_a / exp_a.sum()


def simplex_projection(v: np.ndarray, z: float = 1.0) -> np.ndarray:
    """
    Project onto the simplex {w >= 0, sum(w) = z}.
    Efficient O(n log n) algorithm (Duchi et al., 2008).
    """
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    rho = np.nonzero(u * np.arange(1, n + 1) > cssv)[0][-1]
    theta = cssv[rho] / (rho + 1.0)
    return np.maximum(v - theta, 0)


def long_short_transform(raw_action: np.ndarray,
                          max_leverage: float = 1.0) -> np.ndarray:
    """Transform to long-short weights with leverage constraint."""
    weights = np.tanh(raw_action)
    gross = np.abs(weights).sum()
    if gross > max_leverage:
        weights *= max_leverage / gross
    return weights


def discretize_weights(weights: np.ndarray,
                       granularity: float = 0.05) -> np.ndarray:
    """Snap continuous weights to nearest grid point."""
    discretized = np.round(weights / granularity) * granularity
    # Re-normalize
    total = discretized.sum()
    if abs(total) > 1e-8:
        discretized = discretized / total
    return discretized


# ============================================================
# Constraint Handlers
# ============================================================

@dataclass
class PortfolioConstraints:
    """Portfolio constraint specification."""
    max_position: float = 0.25        # Max weight per asset
    min_position: float = 0.0         # Min weight (0 for long-only)
    max_leverage: float = 1.0         # Max gross leverage
    max_turnover: float = 1.0         # Max turnover per step
    min_trade_size: float = 0.001     # Min trade size (filter noise)
    sector_limits: Optional[Dict[str, float]] = None  # Sector max exposure


class ConstraintHandler:
    """
    Enforces portfolio constraints on raw actions.
    
    Pipeline:
    1. Activation (softmax / tanh)
    2. Position limits
    3. Leverage constraint
    4. Sector limits
    5. Turnover constraint
    6. Min trade filter
    """
    
    def __init__(self, constraints: PortfolioConstraints,
                 sector_mapping: Optional[Dict[int, str]] = None):
        self.constraints = constraints
        self.sector_mapping = sector_mapping or {}
    
    def apply(self, raw_action: np.ndarray,
              current_weights: np.ndarray,
              long_only: bool = True) -> np.ndarray:
        """Apply all constraints to produce valid target weights."""
        # Step 1: Activation
        if long_only:
            weights = softmax_transform(raw_action)
        else:
            weights = long_short_transform(raw_action, self.constraints.max_leverage)
        
        # Step 2: Position limits
        weights = np.clip(weights,
                         self.constraints.min_position,
                         self.constraints.max_position)
        
        # Re-normalize after clipping
        if long_only and weights.sum() > 0:
            weights = weights / weights.sum()
        
        # Step 3: Leverage constraint
        gross = np.abs(weights).sum()
        if gross > self.constraints.max_leverage:
            weights *= self.constraints.max_leverage / gross
        
        # Step 4: Sector limits
        if self.constraints.sector_limits and self.sector_mapping:
            weights = self._apply_sector_limits(weights)
        
        # Step 5: Turnover constraint
        trades = weights - current_weights
        turnover = np.abs(trades).sum()
        if turnover > self.constraints.max_turnover:
            scale = self.constraints.max_turnover / turnover
            trades *= scale
            weights = current_weights + trades
        
        # Step 6: Min trade filter
        trades = weights - current_weights
        small_trades = np.abs(trades) < self.constraints.min_trade_size
        weights[small_trades] = current_weights[small_trades]
        
        return weights
    
    def _apply_sector_limits(self, weights: np.ndarray) -> np.ndarray:
        """Enforce sector exposure limits."""
        for sector, limit in self.constraints.sector_limits.items():
            indices = [i for i, s in self.sector_mapping.items() if s == sector]
            sector_weight = np.abs(weights[indices]).sum()
            if sector_weight > limit:
                scale = limit / sector_weight
                weights[indices] *= scale
        return weights


# ============================================================
# Action Space Definitions
# ============================================================

class DiscreteActionSpace:
    """
    Discrete action space for trading.
    
    Maps integer actions to predefined portfolio allocations.
    """
    
    def __init__(self, num_assets: int, num_levels: int = 5):
        self.num_assets = num_assets
        self.num_levels = num_levels
        
        # Create action mapping
        levels = np.linspace(0, 1, num_levels)
        self.action_map = self._build_action_map(levels)
        self.n = len(self.action_map)
    
    def _build_action_map(self, levels: np.ndarray) -> List[np.ndarray]:
        """Build feasible weight combinations."""
        from itertools import product
        
        actions = []
        for combo in product(levels, repeat=self.num_assets):
            weights = np.array(combo)
            total = weights.sum()
            if 0.9 <= total <= 1.1 and total > 0:  # Approximately fully invested
                weights = weights / total
                actions.append(weights)
        
        # Add equal-weight and cash positions
        actions.append(np.ones(self.num_assets) / self.num_assets)
        actions.append(np.zeros(self.num_assets))
        
        return actions
    
    def decode(self, action_idx: int) -> np.ndarray:
        """Convert action index to portfolio weights."""
        return self.action_map[min(action_idx, len(self.action_map) - 1)].copy()
    
    def sample(self, rng: np.random.Generator = None) -> int:
        """Sample random action."""
        if rng is None:
            return np.random.randint(self.n)
        return rng.integers(self.n)


class FactoredDiscreteActionSpace:
    """
    Factored discrete action space.
    
    Each asset has independent discrete actions (buy/hold/sell),
    avoiding combinatorial explosion.
    """
    
    SELL = 0
    HOLD = 1
    BUY = 2
    
    def __init__(self, num_assets: int, trade_fraction: float = 0.1):
        self.num_assets = num_assets
        self.trade_fraction = trade_fraction
        self.n_per_asset = 3  # sell, hold, buy
    
    def decode(self, actions: np.ndarray,
               current_weights: np.ndarray) -> np.ndarray:
        """
        Convert per-asset discrete actions to target weights.
        
        Args:
            actions: (N,) array of {0=sell, 1=hold, 2=buy}
            current_weights: Current portfolio weights
        """
        target = current_weights.copy()
        
        for i, action in enumerate(actions):
            if action == self.SELL:
                target[i] -= self.trade_fraction
            elif action == self.BUY:
                target[i] += self.trade_fraction
            # HOLD: no change
        
        # Clip and normalize
        target = np.clip(target, 0, 1)
        total = target.sum()
        if total > 1:
            target = target / total
        
        return target
    
    def sample(self, rng: np.random.Generator = None) -> np.ndarray:
        if rng is None:
            return np.random.randint(0, 3, self.num_assets)
        return rng.integers(0, 3, self.num_assets)


class ContinuousActionSpace:
    """
    Continuous action space with configurable constraints.
    """
    
    def __init__(self, num_assets: int, long_only: bool = True,
                 constraints: Optional[PortfolioConstraints] = None):
        self.num_assets = num_assets
        self.long_only = long_only
        self.constraints = constraints or PortfolioConstraints()
        self.handler = ConstraintHandler(self.constraints)
    
    def transform(self, raw_action: np.ndarray,
                  current_weights: np.ndarray) -> np.ndarray:
        """Transform raw network output to valid portfolio weights."""
        return self.handler.apply(raw_action, current_weights, self.long_only)
    
    def sample(self, rng: np.random.Generator = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        if self.long_only:
            raw = rng.random(self.num_assets)
            return raw / raw.sum()
        else:
            return rng.uniform(-1, 1, self.num_assets)


# ============================================================
# Action Masking
# ============================================================

class ActionMask:
    """
    Generates action masks based on portfolio state.
    Prevents infeasible actions.
    """
    
    def __init__(self, num_assets: int, allow_short: bool = False):
        self.num_assets = num_assets
        self.allow_short = allow_short
    
    def get_mask(self, current_weights: np.ndarray,
                 cash_ratio: float,
                 halted_assets: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Generate action mask.
        
        Returns:
            Dictionary with 'can_buy' and 'can_sell' boolean arrays.
        """
        can_buy = np.ones(self.num_assets, dtype=bool)
        can_sell = np.ones(self.num_assets, dtype=bool)
        
        # Cannot buy without cash
        if cash_ratio <= 0.01:
            can_buy[:] = False
        
        # Cannot sell without position (if no shorting)
        if not self.allow_short:
            can_sell = current_weights > 0.001
        
        # Cannot trade halted assets
        if halted_assets is not None:
            can_buy[halted_assets] = False
            can_sell[halted_assets] = False
        
        return {
            'can_buy': can_buy,
            'can_sell': can_sell,
            'tradeable': can_buy | can_sell,
        }
    
    def apply_mask_to_discrete(self, action_logits: np.ndarray,
                                mask: Dict[str, np.ndarray],
                                large_negative: float = -1e9) -> np.ndarray:
        """
        Apply mask to discrete action logits.
        
        For factored actions: (N, 3) logits for [sell, hold, buy].
        """
        masked = action_logits.copy()
        
        for i in range(self.num_assets):
            if not mask['can_sell'][i]:
                masked[i, 0] = large_negative  # Mask sell
            if not mask['can_buy'][i]:
                masked[i, 2] = large_negative   # Mask buy
        
        return masked


# ============================================================
# Demo
# ============================================================

def demo_action_spaces():
    """Demonstrate different action space designs."""
    print("=" * 60)
    print("Action Spaces Demo")
    print("=" * 60)
    
    num_assets = 4
    asset_names = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
    
    # 1. Continuous action space
    print("\n--- Continuous Action Space (Long-Only) ---")
    constraints = PortfolioConstraints(
        max_position=0.4,
        max_leverage=1.0,
        max_turnover=0.5,
        min_trade_size=0.01,
    )
    continuous = ContinuousActionSpace(num_assets, long_only=True,
                                        constraints=constraints)
    
    current_weights = np.array([0.25, 0.25, 0.25, 0.25])
    raw_action = np.array([2.0, 0.5, -1.0, 0.3])
    
    target = continuous.transform(raw_action, current_weights)
    trades = target - current_weights
    
    print(f"Raw action:       {raw_action}")
    print(f"Current weights:  {current_weights}")
    print(f"Target weights:   {target.round(4)}")
    print(f"Trades:           {trades.round(4)}")
    print(f"Turnover:         {np.abs(trades).sum():.4f}")
    print(f"Gross leverage:   {np.abs(target).sum():.4f}")
    
    # 2. Long-short
    print("\n--- Continuous Action Space (Long-Short) ---")
    ls_space = ContinuousActionSpace(num_assets, long_only=False,
                                      constraints=PortfolioConstraints(max_leverage=1.5))
    raw_ls = np.array([1.5, -0.8, 0.3, -1.2])
    target_ls = ls_space.transform(raw_ls, np.zeros(num_assets))
    print(f"Raw action:      {raw_ls}")
    print(f"Target weights:  {target_ls.round(4)}")
    print(f"Long exposure:   {target_ls[target_ls > 0].sum():.4f}")
    print(f"Short exposure:  {target_ls[target_ls < 0].sum():.4f}")
    print(f"Net exposure:    {target_ls.sum():.4f}")
    print(f"Gross leverage:  {np.abs(target_ls).sum():.4f}")
    
    # 3. Discrete action space
    print("\n--- Discrete Action Space ---")
    discrete = DiscreteActionSpace(num_assets, num_levels=3)
    print(f"Total actions: {discrete.n}")
    print(f"Sample actions:")
    for i in range(min(5, discrete.n)):
        print(f"  Action {i}: {discrete.decode(i).round(3)}")
    
    # 4. Factored discrete
    print("\n--- Factored Discrete Action Space ---")
    factored = FactoredDiscreteActionSpace(num_assets, trade_fraction=0.05)
    current = np.array([0.3, 0.2, 0.3, 0.2])
    
    actions = np.array([2, 1, 0, 1])  # Buy AAPL, Hold GOOGL, Sell MSFT, Hold AMZN
    action_names = ['BUY', 'HOLD', 'SELL']
    
    print(f"Current: {current}")
    print(f"Actions: {[f'{asset_names[i]}={action_names[a]}' for i, a in enumerate(actions)]}")
    target = factored.decode(actions, current)
    print(f"Target:  {target.round(4)}")
    
    # 5. Action transformations
    print("\n--- Action Transformations ---")
    raw = np.array([1.0, 2.0, 0.5, 1.5])
    
    print(f"Raw action:       {raw}")
    print(f"Softmax:          {softmax_transform(raw).round(4)}")
    print(f"Simplex proj:     {simplex_projection(raw).round(4)}")
    print(f"Long-short (L=1): {long_short_transform(raw, 1.0).round(4)}")
    print(f"Discretized:      {discretize_weights(softmax_transform(raw), 0.1).round(2)}")
    
    # 6. Action masking
    print("\n--- Action Masking ---")
    masker = ActionMask(num_assets, allow_short=False)
    
    weights = np.array([0.0, 0.3, 0.5, 0.0])
    mask = masker.get_mask(weights, cash_ratio=0.2)
    
    print(f"Weights:    {weights}")
    print(f"Can buy:    {mask['can_buy']}")
    print(f"Can sell:   {mask['can_sell']}")
    
    # No cash scenario
    mask_no_cash = masker.get_mask(weights, cash_ratio=0.0)
    print(f"\nNo cash:")
    print(f"Can buy:    {mask_no_cash['can_buy']}")
    print(f"Can sell:   {mask_no_cash['can_sell']}")
    
    # With halted asset
    mask_halt = masker.get_mask(weights, cash_ratio=0.2,
                                 halted_assets=np.array([1]))
    print(f"\nGOOGL halted:")
    print(f"Can buy:    {mask_halt['can_buy']}")
    print(f"Can sell:   {mask_halt['can_sell']}")
    
    print("\nAction spaces demo complete!")


if __name__ == "__main__":
    demo_action_spaces()
