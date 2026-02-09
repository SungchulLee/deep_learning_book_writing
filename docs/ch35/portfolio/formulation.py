"""
Chapter 35.2.1: Portfolio Management Problem Formulation
========================================================
MDP formulation for portfolio management with Gymnasium-compatible environment.
Includes classical Markowitz baseline and RL policy architectures.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any, List
from dataclasses import dataclass, field


# ============================================================
# Configuration
# ============================================================

@dataclass
class PortfolioMDPConfig:
    """Configuration for the Portfolio Management MDP."""
    # Assets
    num_assets: int = 5
    lookback: int = 60
    episode_length: int = 252

    # Portfolio constraints
    initial_capital: float = 1_000_000.0
    allow_short: bool = False
    max_leverage: float = 1.0
    max_position: float = 0.40
    include_cash: bool = True

    # Costs
    transaction_cost: float = 0.001  # 10 bps
    slippage_std: float = 0.0005

    # Reward
    reward_type: str = "log_return"  # log_return, sharpe, sortino, risk_adjusted
    risk_penalty: float = 0.5
    sharpe_window: int = 20
    risk_free_rate: float = 0.02 / 252  # Daily

    # Discount factor
    gamma: float = 0.99


# ============================================================
# State Constructor
# ============================================================

class StateConstructor:
    """Constructs the state representation for portfolio management."""

    def __init__(self, config: PortfolioMDPConfig):
        self.config = config
        self.lookback = config.lookback
        self.num_assets = config.num_assets

    def build_state(
        self,
        prices: np.ndarray,
        volumes: Optional[np.ndarray],
        current_weights: np.ndarray,
        portfolio_value: float,
        step: int,
        episode_length: int,
    ) -> np.ndarray:
        """
        Build the state vector from market and portfolio data.

        Args:
            prices: (lookback, num_assets) price window
            volumes: (lookback, num_assets) volume window or None
            current_weights: (num_assets,) current portfolio weights
            portfolio_value: current portfolio value
            step: current step in episode
            episode_length: total episode length

        Returns:
            state: flattened state vector
        """
        features = []

        # --- Market features ---
        # Log returns
        log_returns = np.diff(np.log(prices + 1e-8), axis=0)  # (lookback-1, N)
        features.append(log_returns.flatten())

        # Normalized prices (relative to first price in window)
        norm_prices = prices / (prices[0:1] + 1e-8) - 1.0
        features.append(norm_prices.flatten())

        # Rolling volatility (last 20 steps)
        if len(log_returns) >= 20:
            vol = np.std(log_returns[-20:], axis=0)
        else:
            vol = np.std(log_returns, axis=0)
        features.append(vol)

        # Rolling correlation (flattened upper triangle)
        if len(log_returns) >= 20:
            corr = np.corrcoef(log_returns[-20:].T)
            upper_tri = corr[np.triu_indices(self.num_assets, k=1)]
            features.append(upper_tri)

        # Volume features (if available)
        if volumes is not None:
            norm_vol = volumes / (np.mean(volumes, axis=0, keepdims=True) + 1e-8)
            features.append(norm_vol[-1])  # Most recent normalized volume

        # --- Portfolio features ---
        features.append(current_weights)

        # Portfolio value normalized
        features.append(np.array([portfolio_value / self.config.initial_capital - 1.0]))

        # --- Context features ---
        # Time features
        time_frac = step / max(episode_length, 1)
        features.append(np.array([time_frac, np.sin(2 * np.pi * time_frac)]))

        return np.concatenate(features).astype(np.float32)

    def get_state_dim(self) -> int:
        """Calculate the dimension of the state vector."""
        N = self.num_assets
        L = self.lookback

        dim = 0
        dim += (L - 1) * N   # log returns
        dim += L * N          # normalized prices
        dim += N              # volatility
        dim += N * (N - 1) // 2  # correlation upper triangle
        dim += N              # current weights
        dim += 1              # portfolio value
        dim += 2              # time features
        return dim


# ============================================================
# Action Processor
# ============================================================

class ActionProcessor:
    """Processes raw actions into valid portfolio weights."""

    def __init__(self, config: PortfolioMDPConfig):
        self.config = config
        self.num_assets = config.num_assets

    def process_action(self, raw_action: np.ndarray) -> np.ndarray:
        """
        Convert raw action to valid portfolio weights.

        For long-only: softmax normalization
        For long-short: tanh + normalization to leverage constraint
        """
        if self.config.allow_short:
            # Tanh maps to [-1, 1], then normalize to leverage constraint
            weights = np.tanh(raw_action)
            total_exposure = np.sum(np.abs(weights))
            if total_exposure > self.config.max_leverage:
                weights = weights * (self.config.max_leverage / total_exposure)
        else:
            # Softmax for long-only
            exp_action = np.exp(raw_action - np.max(raw_action))
            weights = exp_action / (np.sum(exp_action) + 1e-8)

        # Apply position limits
        weights = np.clip(weights, -self.config.max_position, self.config.max_position)

        # Re-normalize
        if not self.config.allow_short:
            weights = weights / (np.sum(weights) + 1e-8)

        return weights

    def compute_turnover(
        self, new_weights: np.ndarray, old_weights: np.ndarray
    ) -> float:
        """Compute portfolio turnover (L1 distance)."""
        return float(np.sum(np.abs(new_weights - old_weights)))


# ============================================================
# Reward Engine
# ============================================================

class PortfolioRewardEngine:
    """Computes rewards for portfolio management."""

    def __init__(self, config: PortfolioMDPConfig):
        self.config = config
        self.return_history: List[float] = []

    def reset(self):
        """Reset return history."""
        self.return_history = []

    def compute_reward(
        self,
        portfolio_return: float,
        turnover: float,
        portfolio_value: float,
        peak_value: float,
    ) -> float:
        """
        Compute the reward based on configured reward type.

        Args:
            portfolio_return: single-step log return
            turnover: L1 turnover of rebalancing
            portfolio_value: current portfolio value
            peak_value: maximum portfolio value so far
        """
        self.return_history.append(portfolio_return)
        tc = self.config.transaction_cost * turnover

        if self.config.reward_type == "log_return":
            reward = portfolio_return - tc

        elif self.config.reward_type == "sharpe":
            reward = self._differential_sharpe(portfolio_return, tc)

        elif self.config.reward_type == "sortino":
            reward = self._sortino_reward(portfolio_return, tc)

        elif self.config.reward_type == "risk_adjusted":
            drawdown = (peak_value - portfolio_value) / (peak_value + 1e-8)
            reward = (portfolio_return - tc
                      - self.config.risk_penalty * drawdown)
        else:
            reward = portfolio_return - tc

        return float(reward)

    def _differential_sharpe(self, ret: float, cost: float) -> float:
        """
        Differential Sharpe ratio (Moody & Saffell, 2001).
        Provides a single-step reward that approximates the gradient of the Sharpe ratio.
        """
        net_ret = ret - cost
        window = self.config.sharpe_window

        if len(self.return_history) < window:
            return net_ret

        recent = np.array(self.return_history[-window:])
        A = np.mean(recent)
        B = np.mean(recent ** 2)
        delta_A = net_ret - A
        delta_B = net_ret ** 2 - B

        denom = (B - A ** 2) ** 1.5
        if abs(denom) < 1e-8:
            return net_ret

        dS = (B * delta_A - 0.5 * A * delta_B) / denom
        return float(dS)

    def _sortino_reward(self, ret: float, cost: float) -> float:
        """Sortino-style reward penalizing only downside deviation."""
        net_ret = ret - cost
        window = self.config.sharpe_window

        if len(self.return_history) < window:
            return net_ret

        recent = np.array(self.return_history[-window:])
        mean_ret = np.mean(recent)
        downside = recent[recent < self.config.risk_free_rate]
        if len(downside) < 2:
            downside_std = 1e-8
        else:
            downside_std = np.std(downside)

        sortino = (mean_ret - self.config.risk_free_rate) / (downside_std + 1e-8)
        return float(net_ret + 0.01 * sortino)


# ============================================================
# Portfolio Management Environment (Gymnasium-compatible)
# ============================================================

class PortfolioManagementEnv:
    """
    Gymnasium-compatible portfolio management environment.

    State: market features + portfolio state + context
    Action: target portfolio weights
    Reward: configurable (log return, Sharpe, Sortino, risk-adjusted)
    """

    def __init__(self, prices: np.ndarray, config: PortfolioMDPConfig,
                 volumes: Optional[np.ndarray] = None):
        """
        Args:
            prices: (T, N) array of asset prices
            config: environment configuration
            volumes: (T, N) array of volumes (optional)
        """
        self.config = config
        self.prices = prices
        self.volumes = volumes
        self.num_steps = len(prices)
        self.num_assets = config.num_assets

        self.state_constructor = StateConstructor(config)
        self.action_processor = ActionProcessor(config)
        self.reward_engine = PortfolioRewardEngine(config)

        self.state_dim = self.state_constructor.get_state_dim()
        self.action_dim = config.num_assets

        # State variables
        self.current_step = 0
        self.current_weights = np.zeros(config.num_assets)
        self.portfolio_value = config.initial_capital
        self.peak_value = config.initial_capital
        self.initial_weights_set = False

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment."""
        if seed is not None:
            np.random.seed(seed)

        lookback = self.config.lookback
        max_start = self.num_steps - self.config.episode_length - lookback
        if max_start > lookback:
            self.start_idx = np.random.randint(lookback, max_start)
        else:
            self.start_idx = lookback

        self.current_step = 0
        self.portfolio_value = self.config.initial_capital
        self.peak_value = self.config.initial_capital
        self.reward_engine.reset()

        # Equal weight initialization
        if self.config.include_cash:
            self.current_weights = np.zeros(self.num_assets)
        else:
            self.current_weights = np.ones(self.num_assets) / self.num_assets

        self.initial_weights_set = False

        state = self._get_state()
        info = self._get_info()
        return state, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step of portfolio management.

        Args:
            action: raw action vector (will be processed to valid weights)

        Returns:
            state, reward, terminated, truncated, info
        """
        # Process action to valid weights
        target_weights = self.action_processor.process_action(action)

        # Compute turnover
        if not self.initial_weights_set:
            turnover = np.sum(np.abs(target_weights))
            self.initial_weights_set = True
        else:
            turnover = self.action_processor.compute_turnover(
                target_weights, self.current_weights
            )

        # Get market returns for this step
        data_idx = self.start_idx + self.config.lookback + self.current_step
        if data_idx + 1 >= self.num_steps:
            return self._get_state(), 0.0, True, False, self._get_info()

        current_prices = self.prices[data_idx]
        next_prices = self.prices[data_idx + 1]
        asset_returns = (next_prices - current_prices) / (current_prices + 1e-8)

        # Portfolio return (weighted sum of asset returns)
        portfolio_return = float(np.dot(target_weights, asset_returns))

        # Transaction cost
        tc = self.config.transaction_cost * turnover
        net_return = portfolio_return - tc

        # Update portfolio value
        self.portfolio_value *= (1.0 + net_return)
        self.peak_value = max(self.peak_value, self.portfolio_value)

        # Drift weights (after market moves, before rebalancing)
        drift_weights = target_weights * (1 + asset_returns)
        drift_weights = drift_weights / (np.sum(drift_weights) + 1e-8)
        self.current_weights = drift_weights

        # Compute reward
        log_return = np.log(1.0 + net_return + 1e-8)
        reward = self.reward_engine.compute_reward(
            portfolio_return=log_return,
            turnover=turnover,
            portfolio_value=self.portfolio_value,
            peak_value=self.peak_value,
        )

        # Advance step
        self.current_step += 1

        # Termination
        terminated = self.portfolio_value <= 0.1 * self.config.initial_capital
        truncated = self.current_step >= self.config.episode_length

        state = self._get_state()
        info = self._get_info()
        info.update({
            "portfolio_return": portfolio_return,
            "turnover": turnover,
            "transaction_cost": tc,
            "target_weights": target_weights.copy(),
        })

        return state, reward, terminated, truncated, info

    def _get_state(self) -> np.ndarray:
        """Construct the current state."""
        data_idx = self.start_idx + self.current_step
        end_idx = data_idx + self.config.lookback
        end_idx = min(end_idx, self.num_steps)
        start_idx = max(0, end_idx - self.config.lookback)

        price_window = self.prices[start_idx:end_idx]
        vol_window = self.volumes[start_idx:end_idx] if self.volumes is not None else None

        return self.state_constructor.build_state(
            prices=price_window,
            volumes=vol_window,
            current_weights=self.current_weights,
            portfolio_value=self.portfolio_value,
            step=self.current_step,
            episode_length=self.config.episode_length,
        )

    def _get_info(self) -> Dict[str, Any]:
        """Get environment info."""
        drawdown = (self.peak_value - self.portfolio_value) / (self.peak_value + 1e-8)
        return {
            "portfolio_value": self.portfolio_value,
            "peak_value": self.peak_value,
            "drawdown": drawdown,
            "current_weights": self.current_weights.copy(),
            "step": self.current_step,
        }


# ============================================================
# Markowitz Baseline
# ============================================================

class MarkowitzBaseline:
    """
    Classical Markowitz mean-variance optimization baseline.
    Solves: max w'μ - (λ/2) w'Σw  s.t. Σw_i = 1, w_i >= 0
    """

    def __init__(self, risk_aversion: float = 1.0, lookback: int = 60):
        self.risk_aversion = risk_aversion
        self.lookback = lookback

    def compute_weights(self, returns: np.ndarray) -> np.ndarray:
        """
        Compute optimal portfolio weights from historical returns.

        Args:
            returns: (T, N) array of asset returns

        Returns:
            weights: (N,) optimal portfolio weights
        """
        recent = returns[-self.lookback:]
        mu = np.mean(recent, axis=0)
        sigma = np.cov(recent.T)

        # Regularize covariance
        sigma += np.eye(len(mu)) * 1e-6

        # Analytical solution for unconstrained case
        sigma_inv = np.linalg.inv(sigma)
        raw_weights = sigma_inv @ mu / self.risk_aversion

        # Project to simplex (long-only, fully invested)
        weights = self._project_to_simplex(raw_weights)
        return weights

    @staticmethod
    def _project_to_simplex(v: np.ndarray) -> np.ndarray:
        """Project a vector onto the probability simplex."""
        n = len(v)
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u) - 1
        rho = np.nonzero(u * np.arange(1, n + 1) > cssv)[0][-1]
        theta = cssv[rho] / (rho + 1.0)
        return np.maximum(v - theta, 0)


# ============================================================
# Policy Network Architecture
# ============================================================

class PortfolioPolicyNetwork(nn.Module):
    """
    Policy network for portfolio management.
    Maps state to portfolio weights with built-in constraint satisfaction.
    """

    def __init__(
        self,
        state_dim: int,
        num_assets: int,
        hidden_dims: List[int] = None,
        allow_short: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        self.allow_short = allow_short
        self.num_assets = num_assets

        # Build encoder
        layers = []
        in_dim = state_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim

        self.encoder = nn.Sequential(*layers)

        # Policy head (mean)
        self.policy_mean = nn.Linear(in_dim, num_assets)

        # Log std (learnable)
        self.log_std = nn.Parameter(torch.zeros(num_assets) - 1.0)

        # Value head (for actor-critic)
        self.value_head = nn.Linear(in_dim, 1)

    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            state: (batch, state_dim) state tensor

        Returns:
            dict with 'weights', 'log_prob', 'value', 'entropy'
        """
        features = self.encoder(state)

        # Policy
        raw_action = self.policy_mean(features)

        if self.allow_short:
            weights = torch.tanh(raw_action)
            # Normalize to leverage constraint
            weights = weights / (torch.sum(torch.abs(weights), dim=-1, keepdim=True) + 1e-8)
        else:
            weights = F.softmax(raw_action, dim=-1)

        # Value
        value = self.value_head(features)

        # Log probability (Gaussian policy for training)
        std = torch.exp(self.log_std).expand_as(raw_action)
        dist = torch.distributions.Normal(raw_action, std)
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return {
            "weights": weights,
            "raw_action": raw_action,
            "log_prob": log_prob,
            "value": value.squeeze(-1),
            "entropy": entropy,
        }

    def get_weights(self, state: torch.Tensor) -> np.ndarray:
        """Get deterministic portfolio weights for evaluation."""
        with torch.no_grad():
            output = self.forward(state)
            return output["weights"].cpu().numpy()


# ============================================================
# Kelly Criterion Connection
# ============================================================

class KellyCriterion:
    """
    Kelly criterion for optimal bet sizing.
    Maximizes expected log-growth of the portfolio.

    For N assets with Gaussian returns:
        w* = Σ^{-1} μ  (unconstrained)
    """

    def __init__(self, fraction: float = 0.5):
        """
        Args:
            fraction: Kelly fraction (0.5 = half-Kelly for robustness)
        """
        self.fraction = fraction

    def compute_weights(self, returns: np.ndarray) -> np.ndarray:
        """Compute Kelly-optimal weights."""
        mu = np.mean(returns, axis=0)
        sigma = np.cov(returns.T)
        sigma += np.eye(len(mu)) * 1e-6

        sigma_inv = np.linalg.inv(sigma)
        kelly_weights = sigma_inv @ mu * self.fraction

        # Clip to reasonable range
        kelly_weights = np.clip(kelly_weights, -1.0, 1.0)

        return kelly_weights


# ============================================================
# Demonstration
# ============================================================

def generate_synthetic_prices(
    num_assets: int = 5,
    num_steps: int = 1000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic asset prices with realistic properties."""
    np.random.seed(seed)

    # Annual parameters
    annual_returns = np.random.uniform(0.02, 0.15, num_assets)
    annual_vols = np.random.uniform(0.10, 0.35, num_assets)

    # Correlation matrix
    L = np.random.randn(num_assets, num_assets) * 0.3
    corr = L @ L.T
    d = np.sqrt(np.diag(corr))
    corr = corr / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)

    # Daily parameters
    daily_returns = annual_returns / 252
    daily_vols = annual_vols / np.sqrt(252)

    # Covariance
    cov = np.outer(daily_vols, daily_vols) * corr

    # Generate returns
    returns = np.random.multivariate_normal(daily_returns, cov, size=num_steps)

    # Convert to prices
    prices = 100.0 * np.exp(np.cumsum(returns, axis=0))

    # Synthetic volume
    volumes = np.abs(np.random.lognormal(mean=15, sigma=1.0, size=(num_steps, num_assets)))

    return prices, volumes


def demo_portfolio_mdp():
    """Demonstrate the portfolio management MDP."""
    print("=" * 70)
    print("Portfolio Management MDP Demonstration")
    print("=" * 70)

    # Generate data
    num_assets = 5
    prices, volumes = generate_synthetic_prices(num_assets=num_assets, num_steps=1000)
    print(f"\nSynthetic data: {prices.shape[0]} steps, {num_assets} assets")
    print(f"Price range: [{prices.min():.2f}, {prices.max():.2f}]")

    # Create environment
    config = PortfolioMDPConfig(
        num_assets=num_assets,
        lookback=60,
        episode_length=200,
        transaction_cost=0.001,
        reward_type="log_return",
    )
    env = PortfolioManagementEnv(prices, config, volumes=volumes)

    # --- Run episode with random policy ---
    print("\n--- Random Policy ---")
    state, info = env.reset(seed=42)
    print(f"State dim: {len(state)}")
    total_reward = 0.0

    for step in range(config.episode_length):
        action = np.random.randn(num_assets)
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    print(f"Steps: {step + 1}")
    print(f"Total reward: {total_reward:.4f}")
    print(f"Final value: ${info['portfolio_value']:,.2f}")
    print(f"Return: {(info['portfolio_value'] / config.initial_capital - 1) * 100:.2f}%")
    print(f"Max drawdown: {info['drawdown'] * 100:.2f}%")

    # --- Run episode with Markowitz baseline ---
    print("\n--- Markowitz Baseline ---")
    state, info = env.reset(seed=42)
    markowitz = MarkowitzBaseline(risk_aversion=2.0, lookback=60)
    total_reward = 0.0

    for step in range(config.episode_length):
        data_idx = env.start_idx + config.lookback + env.current_step
        if data_idx < config.lookback:
            action = np.zeros(num_assets)
        else:
            hist_returns = np.diff(np.log(prices[data_idx - config.lookback:data_idx] + 1e-8), axis=0)
            weights = markowitz.compute_weights(hist_returns)
            # Convert weights to raw action (inverse softmax approximation)
            action = np.log(weights + 1e-8)

        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    print(f"Steps: {step + 1}")
    print(f"Total reward: {total_reward:.4f}")
    print(f"Final value: ${info['portfolio_value']:,.2f}")
    print(f"Return: {(info['portfolio_value'] / config.initial_capital - 1) * 100:.2f}%")
    print(f"Max drawdown: {info['drawdown'] * 100:.2f}%")
    print(f"Final weights: {info['current_weights']}")

    # --- Kelly Criterion ---
    print("\n--- Kelly Criterion (Half-Kelly) ---")
    kelly = KellyCriterion(fraction=0.5)
    returns = np.diff(np.log(prices + 1e-8), axis=0)
    kelly_weights = kelly.compute_weights(returns[-252:])
    print(f"Kelly weights: {kelly_weights}")
    print(f"Sum of weights: {np.sum(kelly_weights):.4f}")
    print(f"Max position: {np.max(np.abs(kelly_weights)):.4f}")

    # --- Policy Network ---
    print("\n--- Policy Network Architecture ---")
    state_dim = env.state_dim
    policy = PortfolioPolicyNetwork(
        state_dim=state_dim,
        num_assets=num_assets,
        hidden_dims=[256, 128, 64],
        allow_short=False,
    )
    print(f"State dim: {state_dim}")
    print(f"Total parameters: {sum(p.numel() for p in policy.parameters()):,}")

    # Forward pass
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    output = policy(state_tensor)
    print(f"Output weights: {output['weights'].detach().numpy().flatten()}")
    print(f"Value estimate: {output['value'].item():.4f}")
    print(f"Entropy: {output['entropy'].item():.4f}")

    # --- Reward type comparison ---
    print("\n--- Reward Function Comparison ---")
    reward_types = ["log_return", "sharpe", "sortino", "risk_adjusted"]

    for rtype in reward_types:
        cfg = PortfolioMDPConfig(
            num_assets=num_assets,
            lookback=60,
            episode_length=200,
            reward_type=rtype,
        )
        test_env = PortfolioManagementEnv(prices, cfg, volumes=volumes)
        state, _ = test_env.reset(seed=42)
        total_r = 0.0

        for _ in range(200):
            action = np.random.randn(num_assets) * 0.1
            state, r, term, trunc, _ = test_env.step(action)
            total_r += r
            if term or trunc:
                break

        print(f"  {rtype:15s}: total_reward = {total_r:8.4f}")


if __name__ == "__main__":
    demo_portfolio_mdp()
