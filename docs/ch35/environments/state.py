"""
Chapter 35.1.2: State Representations for Financial RL
======================================================
Feature engineering and state construction for trading agents.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


# ============================================================
# Feature Computation Functions
# ============================================================

def compute_log_returns(prices: np.ndarray) -> np.ndarray:
    """Compute log returns from prices. Returns (T-1, N) array."""
    return np.diff(np.log(prices), axis=0)


def compute_rsi(prices: np.ndarray, window: int = 14) -> np.ndarray:
    """
    Compute Relative Strength Index.
    RSI = 100 - 100 / (1 + RS), RS = avg_gain / avg_loss.
    Returns (T, N) array with values 0-100.
    """
    returns = np.diff(prices, axis=0)
    gains = np.maximum(returns, 0)
    losses = np.maximum(-returns, 0)

    T, N = prices.shape
    rsi = np.full((T, N), 50.0)

    for t in range(window, T):
        avg_gain = gains[t - window:t].mean(axis=0)
        avg_loss = losses[t - window:t].mean(axis=0)
        rs = avg_gain / (avg_loss + 1e-10)
        rsi[t] = 100 - 100 / (1 + rs)

    return rsi


def compute_macd(prices: np.ndarray, fast: int = 12, slow: int = 26,
                 signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute MACD line, signal line, and histogram. Each (T, N)."""
    def ema(data, span):
        alpha = 2 / (span + 1)
        result = np.zeros_like(data)
        result[0] = data[0]
        for t in range(1, len(data)):
            result[t] = alpha * data[t] + (1 - alpha) * result[t - 1]
        return result

    T, N = prices.shape
    macd_line = np.zeros((T, N))
    signal_line = np.zeros((T, N))

    for i in range(N):
        fast_ema = ema(prices[:, i], fast)
        slow_ema = ema(prices[:, i], slow)
        macd_line[:, i] = fast_ema - slow_ema
        signal_line[:, i] = ema(macd_line[:, i], signal)

    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger_position(prices: np.ndarray, window: int = 20,
                                num_std: float = 2.0) -> np.ndarray:
    """Compute position within Bollinger Bands (0=lower, 1=upper). (T, N)."""
    T, N = prices.shape
    position = np.full((T, N), 0.5)

    for t in range(window - 1, T):
        w = prices[t - window + 1:t + 1]
        middle = w.mean(axis=0)
        std = w.std(axis=0) + 1e-10
        upper = middle + num_std * std
        lower = middle - num_std * std
        position[t] = (prices[t] - lower) / (upper - lower + 1e-10)

    return np.clip(position, 0, 1)


def compute_realized_volatility(returns: np.ndarray, window: int = 20) -> np.ndarray:
    """Compute rolling annualized volatility. (T, N)."""
    T, N = returns.shape
    vol = np.zeros((T, N))
    for t in range(window, T):
        vol[t] = returns[t - window:t].std(axis=0) * np.sqrt(252)
    return vol


def compute_momentum(returns: np.ndarray,
                     windows: List[int] = None) -> np.ndarray:
    """Compute cumulative momentum over multiple windows. (T, N*W)."""
    if windows is None:
        windows = [5, 10, 21, 63]
    T, N = returns.shape
    features = []
    for w in windows:
        mom = np.zeros((T, N))
        for t in range(w, T):
            mom[t] = returns[t - w:t].sum(axis=0)
        features.append(mom)
    return np.concatenate(features, axis=-1)


def compute_rolling_correlation(returns: np.ndarray,
                                 window: int = 60) -> np.ndarray:
    """Compute rolling pairwise correlation matrix. (T, N, N)."""
    T, N = returns.shape
    corr = np.zeros((T, N, N))
    for t in range(window, T):
        corr[t] = np.corrcoef(returns[t - window:t].T)
    return corr


# ============================================================
# Normalization Strategies
# ============================================================

class RollingZScoreNormalizer:
    """Z-score normalization using rolling statistics."""

    def __init__(self, window: int = 252, epsilon: float = 1e-8):
        self.window = window
        self.epsilon = epsilon
        self._history = []

    def normalize(self, features: np.ndarray) -> np.ndarray:
        self._history.append(features.copy())
        if len(self._history) > self.window:
            self._history = self._history[-self.window:]
        history = np.array(self._history)
        mean = history.mean(axis=0)
        std = history.std(axis=0) + self.epsilon
        return (features - mean) / std

    def reset(self):
        self._history = []


class RankNormalizer:
    """Cross-sectional rank normalization to [0, 1]."""

    def normalize(self, features: np.ndarray) -> np.ndarray:
        if features.ndim == 1:
            ranks = features.argsort().argsort().astype(float)
            return ranks / (len(ranks) - 1 + 1e-10)
        result = np.zeros_like(features)
        for j in range(features.shape[1]):
            ranks = features[:, j].argsort().argsort().astype(float)
            result[:, j] = ranks / (len(ranks) - 1 + 1e-10)
        return result


class AdaptiveNormalizer:
    """Exponential moving average normalization."""

    def __init__(self, alpha: float = 0.01, epsilon: float = 1e-8):
        self.alpha = alpha
        self.epsilon = epsilon
        self._mean = None
        self._var = None

    def normalize(self, features: np.ndarray) -> np.ndarray:
        if self._mean is None:
            self._mean = features.copy()
            self._var = np.ones_like(features)
            return np.zeros_like(features)
        self._mean = self.alpha * features + (1 - self.alpha) * self._mean
        self._var = self.alpha * (features - self._mean) ** 2 + (1 - self.alpha) * self._var
        return (features - self._mean) / (np.sqrt(self._var) + self.epsilon)

    def reset(self):
        self._mean = None
        self._var = None


# ============================================================
# State Builder
# ============================================================

@dataclass
class StateConfig:
    """Configuration for state representation."""
    lookback: int = 60
    include_returns: bool = True
    include_volatility: bool = True
    include_momentum: bool = True
    include_rsi: bool = True
    include_macd: bool = True
    include_bollinger: bool = True
    include_correlation: bool = False
    momentum_windows: List[int] = field(default_factory=lambda: [5, 10, 21])
    normalization: str = "zscore"


class StateBuilder:
    """Constructs state observations from raw market data."""

    def __init__(self, config: StateConfig):
        self.config = config
        if config.normalization == "zscore":
            self.normalizer = RollingZScoreNormalizer(window=252)
        elif config.normalization == "rank":
            self.normalizer = RankNormalizer()
        elif config.normalization == "adaptive":
            self.normalizer = AdaptiveNormalizer()
        else:
            self.normalizer = None

    def compute_features(self, prices: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute all features from price data."""
        T, N = prices.shape
        features = {}
        log_rets = compute_log_returns(prices)
        padded = np.vstack([np.zeros((1, N)), log_rets])

        if self.config.include_returns:
            features['returns'] = padded
        if self.config.include_volatility:
            features['volatility'] = compute_realized_volatility(padded, window=20)
        if self.config.include_momentum:
            features['momentum'] = compute_momentum(padded, self.config.momentum_windows)
        if self.config.include_rsi:
            features['rsi'] = compute_rsi(prices, window=14) / 100.0
        if self.config.include_macd:
            _, _, histogram = compute_macd(prices)
            features['macd'] = histogram
        if self.config.include_bollinger:
            features['bollinger'] = compute_bollinger_position(prices)
        if self.config.include_correlation:
            corr = compute_rolling_correlation(padded, window=60)
            idx = np.triu_indices(N, k=1)
            features['correlation'] = corr[:, idx[0], idx[1]]

        return features

    def build_state(self, features: Dict[str, np.ndarray], time_idx: int,
                    portfolio_weights: np.ndarray, portfolio_value: float,
                    initial_capital: float) -> Dict[str, np.ndarray]:
        """Build the full state observation at a given time index."""
        lookback = self.config.lookback
        start = max(0, time_idx - lookback + 1)
        end = time_idx + 1

        market_parts = [feat[start:end] for feat in features.values()]
        market = np.concatenate(market_parts, axis=-1)

        if market.shape[0] < lookback:
            pad = np.zeros((lookback - market.shape[0], market.shape[1]))
            market = np.vstack([pad, market])

        account = np.array([
            portfolio_value / initial_capital,
            1.0 - np.abs(portfolio_weights).sum(),
            0.0,
        ])

        return {
            'market': market.astype(np.float32),
            'portfolio': portfolio_weights.astype(np.float32),
            'account': account.astype(np.float32),
        }

    def get_feature_dim(self, num_assets: int) -> int:
        dim = 0
        if self.config.include_returns:
            dim += num_assets
        if self.config.include_volatility:
            dim += num_assets
        if self.config.include_momentum:
            dim += num_assets * len(self.config.momentum_windows)
        if self.config.include_rsi:
            dim += num_assets
        if self.config.include_macd:
            dim += num_assets
        if self.config.include_bollinger:
            dim += num_assets
        if self.config.include_correlation:
            dim += num_assets * (num_assets - 1) // 2
        return dim

    def reset(self):
        if hasattr(self.normalizer, 'reset'):
            self.normalizer.reset()


# ============================================================
# Demo
# ============================================================

def demo_state_representations():
    """Demonstrate state representation building."""
    print("=" * 60)
    print("State Representations Demo")
    print("=" * 60)

    np.random.seed(42)
    T, N = 500, 4
    mu = np.array([0.0003, 0.0002, 0.0004, 0.0001])
    sigma = np.array([0.02, 0.015, 0.025, 0.01])
    returns = np.random.randn(T, N) * sigma + mu
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    asset_names = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
    print(f"\nData: {T} steps, {N} assets: {asset_names}")

    # Individual features
    print("\n--- Individual Features ---")
    log_rets = compute_log_returns(prices)
    print(f"Log returns shape: {log_rets.shape}, mean: {log_rets.mean(axis=0).round(6)}")

    rsi = compute_rsi(prices)
    print(f"RSI shape: {rsi.shape}, current: {rsi[-1].round(1)}")

    padded = np.vstack([np.zeros((1, N)), log_rets])
    vol = compute_realized_volatility(padded)
    print(f"Volatility shape: {vol.shape}, current: {vol[-1].round(4)}")

    _, _, hist = compute_macd(prices)
    print(f"MACD histogram shape: {hist.shape}")

    bb = compute_bollinger_position(prices)
    print(f"Bollinger position: {bb[-1].round(3)}")

    mom = compute_momentum(padded)
    print(f"Momentum shape: {mom.shape}")

    # State Builder
    print("\n--- State Builder ---")
    config = StateConfig(lookback=60, include_correlation=True, normalization="zscore")
    builder = StateBuilder(config)
    print(f"Feature dim: {builder.get_feature_dim(N)}")

    features = builder.compute_features(prices)
    for name, feat in features.items():
        print(f"  {name}: {feat.shape}")

    state = builder.build_state(features, time_idx=200,
                                 portfolio_weights=np.array([0.25]*4),
                                 portfolio_value=1_050_000, initial_capital=1_000_000)
    print(f"\nState shapes:")
    for k, v in state.items():
        print(f"  {k}: {v.shape}, range=[{v.min():.3f}, {v.max():.3f}]")

    # Normalization comparison
    print("\n--- Normalization Comparison ---")
    test = log_rets[-1]
    print(f"Raw: {test.round(6)}")

    zn = RollingZScoreNormalizer(252)
    for t in range(200, 499):
        zn.normalize(log_rets[t])
    print(f"Z-score: {zn.normalize(test).round(4)}")

    rn = RankNormalizer()
    print(f"Rank:    {rn.normalize(test).round(4)}")

    an = AdaptiveNormalizer(0.01)
    for t in range(200, 499):
        an.normalize(log_rets[t])
    print(f"Adaptive: {an.normalize(test).round(4)}")

    print("\nState representations demo complete!")


if __name__ == "__main__":
    demo_state_representations()
