# Low Signal-to-Noise Ratio

Financial markets exhibit extremely low signal-to-noise ratios, with typical daily Sharpe ratios around 0.02-0.04. This means that genuine predictive signals are buried under massive noise, requiring specialized techniques such as ensemble methods, data augmentation, and careful statistical analysis to extract and verify any edge.

## Code

```python
"""
Chapter 35.5.3: Low Signal-to-Noise
=====================================
Techniques for extracting signal from noisy financial data.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional
from dataclasses import dataclass

# ========================================================================
# Main
# ========================================================================


@dataclass
class SNRConfig:
    num_ensemble: int = 5
    noise_std: float = 0.001
    bootstrap_samples: int = 100


class SNRAnalyzer:
    """Analyze and quantify signal-to-noise ratio."""

    @staticmethod
    def compute_snr(returns: np.ndarray) -> Dict[str, float]:
        mean = np.mean(returns)
        std = np.std(returns) + 1e-8
        daily_snr = abs(mean) / std
        annual_sharpe = mean / std * np.sqrt(252)
        return {
            "daily_snr": float(daily_snr),
            "annual_sharpe": float(annual_sharpe),
            "mean": float(mean),
            "std": float(std),
            "required_days_for_significance": int((2.0 / daily_snr) ** 2) if daily_snr > 0 else 999999,
        }


class EnsembleAgent:
    """Ensemble of diverse agents for noise reduction."""

    def __init__(self, state_dim: int, action_dim: int, num_agents: int = 5):
        self.agents = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim, 64), nn.ReLU(),
                nn.Linear(64, 32), nn.ReLU(),
                nn.Linear(32, action_dim), nn.Tanh(),
            ) for _ in range(num_agents)
        ])

    def predict(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            predictions = torch.stack([agent(state) for agent in self.agents])
            mean = predictions.mean(dim=0)
            std = predictions.std(dim=0)
            return {"mean": mean, "std": std, "individual": predictions}


class DataAugmenter:
    """Data augmentation for financial time series."""

    @staticmethod
    def add_noise(data: np.ndarray, noise_std: float = 0.001) -> np.ndarray:
        return data + np.random.randn(*data.shape) * noise_std

    @staticmethod
    def bootstrap_sample(data: np.ndarray, block_size: int = 20) -> np.ndarray:
        T = len(data)
        n_blocks = T // block_size + 1
        indices = np.random.randint(0, T - block_size, n_blocks)
        blocks = [data[i:i + block_size] for i in indices]
        return np.concatenate(blocks)[:T]

    @staticmethod
    def time_reversal(returns: np.ndarray) -> np.ndarray:
        return returns[::-1].copy()


def demo_snr():
    """Demonstrate SNR analysis and mitigation."""
    print("=" * 70)
    print("Low Signal-to-Noise Ratio Analysis")
    print("=" * 70)

    np.random.seed(42)
    # Realistic financial returns (low SNR)
    T = 1000
    signal = 0.0003  # ~7.5% annual return
    noise = 0.015    # ~24% annual vol
    returns = np.random.randn(T) * noise + signal

    analyzer = SNRAnalyzer()
    snr = analyzer.compute_snr(returns)
    print(f"\n--- SNR Analysis ---")
    print(f"Daily SNR: {snr['daily_snr']:.4f}")
    print(f"Annual Sharpe: {snr['annual_sharpe']:.4f}")
    print(f"Days needed for significance: {snr['required_days_for_significance']}")

    # Ensemble
    print("\n--- Ensemble Agent ---")
    ensemble = EnsembleAgent(state_dim=10, action_dim=5, num_agents=5)
    state = torch.randn(1, 10)
    result = ensemble.predict(state)
    print(f"Ensemble mean: {result['mean'][0].numpy()}")
    print(f"Ensemble std:  {result['std'][0].numpy()}")
    print(f"Disagreement:  {result['std'].mean().item():.4f}")

    # Data augmentation
    print("\n--- Data Augmentation ---")
    aug = DataAugmenter()
    noisy = aug.add_noise(returns, 0.001)
    boot = aug.bootstrap_sample(returns, block_size=20)
    rev = aug.time_reversal(returns)
    print(f"Original mean: {np.mean(returns):.6f}")
    print(f"Noisy mean:    {np.mean(noisy):.6f}")
    print(f"Bootstrap mean:{np.mean(boot):.6f}")
    print(f"Reversed mean: {np.mean(rev):.6f}")


if __name__ == "__main__":
    demo_snr()
```

## Discussion

SNR analysis quantifies the fundamental difficulty of financial prediction. With a daily SNR of 0.02 (corresponding to an annual Sharpe of approximately 0.3), a strategy needs thousands of independent observations to establish statistical significance. The required days formula $T = (z / \text{SNR})^2$ reveals that detecting a Sharpe of 0.5 at 95% confidence requires over 40 years of daily data -- far more than most practitioners have available.

Ensemble agents reduce prediction variance by combining multiple diverse models. Each agent in the ensemble uses a different architecture or initialization, producing independent estimates. The ensemble mean has lower variance by a factor of $1/\sqrt{N}$ where $N$ is the ensemble size, effectively boosting the SNR of the combined prediction. The ensemble standard deviation also provides a natural uncertainty estimate that can be used for position sizing.

Data augmentation for financial time series requires domain-specific techniques. Adding small Gaussian noise preserves the overall distribution while creating new training examples. Block bootstrap resampling generates synthetic paths that maintain short-term correlation structure. Time reversal creates a valid augmented return series (under the assumption of time-reversibility of log returns) that doubles the effective dataset size.

## Exercises

**Exercise 1.**
Given a strategy with annual expected return of 8% and annual volatility of 25%, compute the daily SNR and the minimum number of trading days needed for statistical significance at 99% confidence.

??? success "Solution to Exercise 1"
    Daily expected return: $\mu = 0.08/252 \approx 0.000317$. Daily volatility: $\sigma = 0.25/\sqrt{252} \approx 0.01575$.

    Daily SNR: $\mu/\sigma \approx 0.0201$.
    
    For 99% confidence: $z = 2.576$.
    
    Minimum days: $T = (z/\text{SNR})^2 = (2.576/0.0201)^2 \approx 16,425$ days $\approx 65$ years.
    
    This illustrates the fundamental challenge: even a strategy with a reasonable 0.32 annual Sharpe ratio requires over 65 years of data to confirm significance at 99% confidence.

---


**Exercise 2.**
Explain why ensemble disagreement (the standard deviation across ensemble predictions) is a useful signal for position sizing in low-SNR environments.

??? success "Solution to Exercise 2"
    When ensemble agents disagree strongly (high standard deviation), the prediction is uncertain -- different models see different patterns in the noise. When they agree (low standard deviation), the signal is more likely to be genuine.

    Position sizing proportional to inverse disagreement -- larger positions when the ensemble agrees, smaller when it disagrees -- effectively implements adaptive confidence weighting. In low-SNR environments, this filtering mechanism avoids taking large bets on noisy predictions while concentrating capital on high-confidence signals, improving the realized Sharpe ratio.

---


**Exercise 3.**
Implement a block bootstrap data augmentation function that generates synthetic return series preserving both the marginal distribution and autocorrelation structure of the original data.

??? success "Solution to Exercise 3"
    ```python
    def block_bootstrap_augment(returns, n_synthetic=5, block_size=20):
        T = len(returns)
        synthetic_series = []
        
        for _ in range(n_synthetic):
            n_blocks = T // block_size + 1
            block_starts = np.random.randint(0, T - block_size, n_blocks)
            blocks = [returns[s:s+block_size] for s in block_starts]
            synthetic = np.concatenate(blocks)[:T]
            synthetic_series.append(synthetic)
        
        return synthetic_series
    ```
    Block bootstrapping preserves within-block temporal dependencies (autocorrelation, volatility clustering) while randomizing the sequence of blocks. Block size should match the correlation timescale: too small destroys serial dependence, too large reduces the number of unique blocks and limits diversity of synthetic paths.

