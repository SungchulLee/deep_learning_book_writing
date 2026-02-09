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
