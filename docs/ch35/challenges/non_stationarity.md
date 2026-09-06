# Non-Stationarity

Financial markets are inherently non-stationary: the statistical properties of returns change over time as market regimes shift. Detection and adaptation techniques are essential for RL trading systems, which must recognize when their learned policies no longer match current market conditions and adjust accordingly.

## Code

```python
"""
Chapter 35.5.1: Non-Stationarity
===================================
Detection and adaptation techniques for non-stationary financial markets.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque

# ========================================================================
# Main
# ========================================================================


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
```

## Discussion

The CUSUM (Cumulative Sum) detector monitors for mean shifts by accumulating deviations from a reference mean. When the cumulative sum exceeds a threshold, a change point is declared. The algorithm maintains two statistics tracking positive and negative deviations, enabling detection of both upward and downward shifts. The drift parameter controls sensitivity: smaller drift detects smaller changes but produces more false alarms.

Distribution shift detection uses a sliding window approach, comparing the distribution of recent observations against a reference window. The Wasserstein distance (or similar metrics) quantifies the difference between the two empirical distributions. When this distance exceeds a threshold, a regime change is signaled. This approach captures changes in volatility, skewness, and tail behavior, not just mean shifts.

Adaptive policies respond to detected non-stationarity by continuously updating their parameters using exponential moving averages. Inverse volatility weighting, where allocation to each asset is proportional to the inverse of its recent volatility, automatically reduces exposure to assets experiencing volatility spikes. The EMA decay parameter controls how quickly the policy forgets old data, with faster decay adapting more quickly but also being more reactive to noise.

## Exercises

**Exercise 1.**
Implement a regime detection algorithm that classifies market conditions into "low volatility trending," "high volatility trending," and "mean-reverting" based on rolling statistics. Use a 60-day window for volatility and a 20-day window for trend detection.

??? success "Solution to Exercise 1"
    ```python
    def detect_regime(returns, vol_window=60, trend_window=20):
        vol = np.std(returns[-vol_window:])
        trend = np.mean(returns[-trend_window:])
        vol_threshold = np.median(np.std(returns[i:i+vol_window]) 
                                  for i in range(len(returns)-vol_window))
        
        if abs(trend) > 2 * np.std(returns[-trend_window:]) / np.sqrt(trend_window):
            if vol > vol_threshold:
                return "high_vol_trending"
            return "low_vol_trending"
        return "mean_reverting"
    ```
    The trend significance is tested against the standard error of the mean. High volatility is defined relative to the historical median volatility. This simple classifier enables regime-dependent strategy selection.

---


**Exercise 2.**
Explain why exponential weighting of historical data is preferable to a fixed rolling window for non-stationary financial data. What is the effective sample size of an EMA with decay parameter $\alpha$?

??? success "Solution to Exercise 2"
    A fixed rolling window assigns equal weight to all observations within the window and zero weight outside, creating discontinuities when old data drops out. Exponential weighting assigns smoothly decaying weights $w_t = \alpha(1-\alpha)^t$, providing a gradual transition that avoids jumps.

    The effective sample size of an EMA is approximately $N_{\text{eff}} = 2/\alpha - 1$. For $\alpha = 0.05$, $N_{\text{eff}} \approx 39$ observations. This means the EMA behaves roughly like a 39-day rolling window but with smoother transitions. Smaller $\alpha$ gives more smoothing (larger effective window) while larger $\alpha$ reacts faster to changes.

---


**Exercise 3.**
Design an adaptive trading policy that switches between momentum and mean-reversion strategies based on detected market regime. Implement the switching logic and discuss potential pitfalls.

??? success "Solution to Exercise 3"
    ```python
    class AdaptiveStrategyPolicy:
        def __init__(self, num_assets, switch_threshold=0.005):
            self.cusum = CUSUMDetector(threshold=3.0)
            self.regime = "trending"
            self.switch_threshold = switch_threshold
        
        def get_weights(self, returns_history):
            recent = returns_history[-60:]
            autocorr = np.corrcoef(recent[:-1], recent[1:])[0,1]
            
            if autocorr > self.switch_threshold:
                # Positive autocorrelation: momentum
                signal = np.mean(returns_history[-20:], axis=0)
                w = np.maximum(signal, 0)
            else:
                # Negative autocorrelation: mean-reversion
                signal = -np.mean(returns_history[-5:], axis=0)
                w = np.maximum(signal, 0)
            
            return w / (np.sum(w) + 1e-8)
    ```
    Key pitfalls: (1) Regime detection lag causes the policy to switch after the regime has already changed. (2) Frequent switching between strategies incurs high transaction costs. (3) Autocorrelation estimation is noisy with short windows. Mitigations include hysteresis in the switching rule, transaction cost penalties, and ensemble approaches that blend rather than switch strategies.

