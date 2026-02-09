"""
Chapter 35.5.2: Regime Changes
================================
Hidden Markov Model regime detection and regime-conditioned policies.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class RegimeConfig:
    num_regimes: int = 3
    lookback: int = 60
    transition_prior: float = 0.95  # Self-transition probability


class SimpleHMM:
    """
    Simple Hidden Markov Model with Gaussian emissions.
    Uses Baum-Welch (EM) for parameter estimation.
    """

    def __init__(self, n_states: int = 3, n_iter: int = 50):
        self.n_states = n_states
        self.n_iter = n_iter
        self.means = None
        self.stds = None
        self.transition = None
        self.initial = None

    def fit(self, returns: np.ndarray):
        """Fit HMM to return series using simplified EM."""
        K = self.n_states
        T = len(returns)

        # Initialize with quantile-based clustering
        quantiles = np.quantile(returns, np.linspace(0, 1, K + 1))
        self.means = np.array([(quantiles[i] + quantiles[i+1]) / 2 for i in range(K)])
        self.stds = np.full(K, np.std(returns))
        self.transition = np.eye(K) * 0.9 + np.ones((K, K)) * 0.1 / K
        self.transition /= self.transition.sum(axis=1, keepdims=True)
        self.initial = np.ones(K) / K

        for _ in range(self.n_iter):
            # E-step: forward-backward
            gamma = self._forward_backward(returns)

            # M-step
            for k in range(K):
                w = gamma[:, k]
                w_sum = w.sum() + 1e-8
                self.means[k] = np.sum(w * returns) / w_sum
                self.stds[k] = np.sqrt(np.sum(w * (returns - self.means[k])**2) / w_sum + 1e-8)

    def _emission_prob(self, x: float) -> np.ndarray:
        probs = np.zeros(self.n_states)
        for k in range(self.n_states):
            probs[k] = np.exp(-0.5 * ((x - self.means[k]) / self.stds[k])**2) / (self.stds[k] * np.sqrt(2 * np.pi) + 1e-8)
        return probs + 1e-300

    def _forward_backward(self, returns: np.ndarray) -> np.ndarray:
        T = len(returns)
        K = self.n_states

        # Forward
        alpha = np.zeros((T, K))
        alpha[0] = self.initial * self._emission_prob(returns[0])
        alpha[0] /= alpha[0].sum() + 1e-8

        for t in range(1, T):
            for k in range(K):
                alpha[t, k] = self._emission_prob(returns[t])[k] * np.sum(alpha[t-1] * self.transition[:, k])
            alpha[t] /= alpha[t].sum() + 1e-8

        # Backward
        beta = np.ones((T, K))
        for t in range(T - 2, -1, -1):
            for k in range(K):
                beta[t, k] = np.sum(self.transition[k] * self._emission_prob(returns[t+1]) * beta[t+1])
            beta[t] /= beta[t].sum() + 1e-8

        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True) + 1e-8
        return gamma

    def predict(self, returns: np.ndarray) -> np.ndarray:
        """Get regime probabilities for each time step."""
        return self._forward_backward(returns)

    def decode(self, returns: np.ndarray) -> np.ndarray:
        """Most likely regime sequence (Viterbi approximation)."""
        gamma = self.predict(returns)
        return np.argmax(gamma, axis=1)


class RegimeConditionedPolicy(nn.Module):
    """Mixture-of-experts policy conditioned on regime probabilities."""

    def __init__(self, state_dim: int, num_assets: int, num_regimes: int = 3,
                 hidden_dim: int = 128):
        super().__init__()
        self.num_regimes = num_regimes

        # Per-regime expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, num_assets),
            ) for _ in range(num_regimes)
        ])

        # Gating network (uses regime probs + state)
        self.gate = nn.Sequential(
            nn.Linear(state_dim + num_regimes, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_regimes),
        )

    def forward(self, state: torch.Tensor,
                regime_probs: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Expert outputs
        expert_outputs = torch.stack([exp(state) for exp in self.experts], dim=1)
        expert_weights = F.softmax(expert_outputs, dim=-1)  # (batch, K, N)

        # Gating
        gate_input = torch.cat([state, regime_probs], dim=-1)
        gate_weights = F.softmax(self.gate(gate_input), dim=-1)  # (batch, K)

        # Weighted combination
        weights = torch.sum(gate_weights.unsqueeze(-1) * expert_weights, dim=1)

        return {"weights": weights, "gate_weights": gate_weights}


def demo_regimes():
    """Demonstrate regime detection and conditioned policies."""
    print("=" * 70)
    print("Regime Detection & Conditioned Policies")
    print("=" * 70)

    np.random.seed(42)
    # 3-regime data
    T = 600
    regimes = np.concatenate([np.zeros(200), np.ones(200), 2 * np.ones(200)]).astype(int)
    means = [-0.001, 0.002, 0.0005]
    stds = [0.025, 0.008, 0.015]
    returns = np.array([np.random.normal(means[r], stds[r]) for r in regimes])

    # HMM
    print("\n--- HMM Regime Detection ---")
    hmm = SimpleHMM(n_states=3, n_iter=30)
    hmm.fit(returns)
    print(f"Estimated means: {np.round(hmm.means, 5)}")
    print(f"Estimated stds:  {np.round(hmm.stds, 5)}")

    decoded = hmm.decode(returns)
    print(f"\nDecoded regime distribution:")
    for k in range(3):
        pct = np.mean(decoded == k) * 100
        print(f"  Regime {k}: {pct:.1f}%")

    # Check accuracy (regimes may be permuted)
    print(f"\nSample decoded regimes: {decoded[:10]}... (true: {regimes[:10]})")

    # Regime-conditioned policy
    print("\n--- Regime-Conditioned Policy ---")
    policy = RegimeConditionedPolicy(state_dim=10, num_assets=5, num_regimes=3)
    params = sum(p.numel() for p in policy.parameters())
    print(f"Parameters: {params:,}")

    state = torch.randn(2, 10)
    probs = torch.FloatTensor([[0.8, 0.1, 0.1], [0.1, 0.2, 0.7]])
    with torch.no_grad():
        out = policy(state, probs)
    print(f"Weights (regime 0 dominant): {out['weights'][0].numpy()}")
    print(f"Weights (regime 2 dominant): {out['weights'][1].numpy()}")
    print(f"Gate weights: {out['gate_weights'].numpy()}")


if __name__ == "__main__":
    demo_regimes()
