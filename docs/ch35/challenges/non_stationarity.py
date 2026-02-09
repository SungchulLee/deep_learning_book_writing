"""
Chapter 35.5.1: Non-Stationarity
===================================
Detection and adaptation techniques for non-stationary financial markets.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque


@dataclass
class NonStationarityConfig:
    window_size: int = 60
    detection_threshold: float = 2.0
    ema_alpha: float = 0.05
    cusum_threshold: float = 5.0


class CUSUMDetector:
    """Cumulative Sum change-point detector."""

    def __init__(self, threshold: float = 5.0, drift: float = 0.5):
        self.threshold = threshold
        self.drift = drift
        self.reset()

    def reset(self):
        self.s_pos = 0.0
        self.s_neg = 0.0
        self.mean = 0.0
        self.count = 0

    def update(self, x: float) -> Dict[str, float]:
        self.count += 1
        if self.count < 10:
            self.mean += (x - self.mean) / self.count
            return {"change_detected": False, "s_pos": 0, "s_neg": 0}

        self.s_pos = max(0, self.s_pos + x - self.mean - self.drift)
        self.s_neg = min(0, self.s_neg + x - self.mean + self.drift)

        detected = self.s_pos > self.threshold or abs(self.s_neg) > self.threshold
        if detected:
            self.s_pos = 0
            self.s_neg = 0
            self.mean = x  # Reset reference

        return {
            "change_detected": detected,
            "s_pos": self.s_pos,
            "s_neg": self.s_neg,
        }


class DistributionShiftDetector:
    """Detect distribution shift using KS-like test on rolling windows."""

    def __init__(self, reference_window: int = 120, test_window: int = 30,
                 threshold: float = 0.2):
        self.ref_window = reference_window
        self.test_window = test_window
        self.threshold = threshold
        self.buffer = deque(maxlen=reference_window + test_window)

    def update(self, x: float) -> Dict:
        self.buffer.append(x)
        if len(self.buffer) < self.ref_window + self.test_window:
            return {"shift_detected": False, "distance": 0.0}

        data = np.array(self.buffer)
        ref = data[:self.ref_window]
        test = data[self.ref_window:]

        # Simple Wasserstein-like distance
        ref_sorted = np.sort(ref)
        test_sorted = np.sort(test)
        # Interpolate to same size
        ref_quantiles = np.quantile(ref_sorted, np.linspace(0, 1, 50))
        test_quantiles = np.quantile(test_sorted, np.linspace(0, 1, 50))
        distance = np.mean(np.abs(ref_quantiles - test_quantiles))

        return {
            "shift_detected": distance > self.threshold,
            "distance": float(distance),
            "ref_mean": float(np.mean(ref)),
            "test_mean": float(np.mean(test)),
        }


class AdaptivePolicy:
    """Policy that adapts to non-stationarity via exponential weighting."""

    def __init__(self, num_assets: int, ema_alpha: float = 0.05):
        self.num_assets = num_assets
        self.ema_alpha = ema_alpha
        self.ema_returns = np.zeros(num_assets)
        self.ema_var = np.ones(num_assets) * 0.01

    def update(self, returns: np.ndarray):
        self.ema_returns = (1 - self.ema_alpha) * self.ema_returns + self.ema_alpha * returns
        self.ema_var = (1 - self.ema_alpha) * self.ema_var + self.ema_alpha * (returns - self.ema_returns) ** 2

    def get_weights(self) -> np.ndarray:
        # Inverse volatility weighting with exponential adaptation
        inv_vol = 1.0 / (np.sqrt(self.ema_var) + 1e-8)
        weights = inv_vol / np.sum(inv_vol)
        return weights


def demo_non_stationarity():
    """Demonstrate non-stationarity detection and adaptation."""
    print("=" * 70)
    print("Non-Stationarity Detection & Adaptation")
    print("=" * 70)

    np.random.seed(42)
    # Generate data with regime change at t=200
    T = 400
    returns = np.concatenate([
        np.random.randn(200) * 0.01 + 0.001,   # Regime 1: low vol, positive
        np.random.randn(200) * 0.025 - 0.002,   # Regime 2: high vol, negative
    ])

    # CUSUM
    print("\n--- CUSUM Change Detection ---")
    cusum = CUSUMDetector(threshold=3.0)
    changes = []
    for t in range(T):
        result = cusum.update(returns[t])
        if result["change_detected"]:
            changes.append(t)
    print(f"Changes detected at: {changes}")
    print(f"True change at: 200")

    # Distribution shift
    print("\n--- Distribution Shift Detection ---")
    shift_det = DistributionShiftDetector(reference_window=100, test_window=30, threshold=0.005)
    shift_times = []
    for t in range(T):
        result = shift_det.update(returns[t])
        if result["shift_detected"] and (not shift_times or t - shift_times[-1] > 20):
            shift_times.append(t)
            print(f"  Shift at t={t}: distance={result['distance']:.6f}")

    # Adaptive policy
    print("\n--- Adaptive Policy ---")
    N = 5
    multi_returns = np.random.randn(T, N) * 0.01
    # Regime change in asset correlations
    multi_returns[200:] = multi_returns[200:] * 2.5  # Vol doubles

    adaptive = AdaptivePolicy(N, ema_alpha=0.05)
    print(f"{'Step':>5} {'Weights':>50}")
    for t in range(T):
        adaptive.update(multi_returns[t])
        if t in [0, 100, 199, 200, 250, 350]:
            w = adaptive.get_weights()
            print(f"{t:>5}  {np.array2string(w, precision=3)}")


if __name__ == "__main__":
    demo_non_stationarity()
