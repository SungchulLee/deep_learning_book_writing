"""
Chapter 35.4.4: CVaR Optimization
====================================
Conditional Value-at-Risk estimation, Rockafellar-Uryasev optimization,
distributional RL for CVaR, and CVaR-constrained policy gradient.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class CVaRConfig:
    alpha: float = 0.05           # Tail probability
    cvar_limit: float = 0.03      # Max CVaR
    lambda_cvar: float = 5.0      # Constraint penalty weight
    num_quantiles: int = 51       # For distributional RL


# ============================================================
# CVaR Estimators
# ============================================================

class SampleCVaR:
    """Sample-based CVaR estimation."""

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def compute(self, returns: np.ndarray) -> Dict[str, float]:
        sorted_r = np.sort(returns)
        n_tail = max(1, int(len(sorted_r) * self.alpha))
        cvar = -np.mean(sorted_r[:n_tail])
        var = -sorted_r[n_tail - 1]
        return {"cvar": float(cvar), "var": float(var), "n_tail": n_tail}

    def compute_portfolio(self, weights: np.ndarray, returns: np.ndarray) -> Dict:
        port_returns = returns @ weights
        return self.compute(port_returns)


class RockafellarUryasev:
    """
    CVaR via Rockafellar-Uryasev optimization.
    CVaR_alpha = min_nu { nu + (1/alpha) * E[max(0, -R - nu)] }
    """

    def __init__(self, alpha: float = 0.05, num_steps: int = 100, lr: float = 0.01):
        self.alpha = alpha
        self.num_steps = num_steps
        self.lr = lr

    def compute(self, returns: np.ndarray) -> Dict[str, float]:
        nu = np.median(returns)
        for _ in range(self.num_steps):
            losses = np.maximum(0, -returns - nu)
            objective = nu + np.mean(losses) / self.alpha
            gradient = 1.0 - np.mean((-returns - nu) > 0) / self.alpha
            nu -= self.lr * gradient

        losses = np.maximum(0, -returns - nu)
        cvar = nu + np.mean(losses) / self.alpha
        return {"cvar": float(-cvar), "var": float(-nu)}  # Sign convention

    def optimize_portfolio(self, returns: np.ndarray, num_assets: int,
                           max_iter: int = 200, lr: float = 0.01) -> np.ndarray:
        """Find minimum CVaR portfolio weights."""
        w = np.ones(num_assets) / num_assets
        nu = 0.0

        for _ in range(max_iter):
            port_ret = returns @ w
            losses = np.maximum(0, -port_ret - nu)
            tail_mask = (-port_ret - nu) > 0

            # Gradient w.r.t. weights
            grad_w = -returns[tail_mask].mean(axis=0) / self.alpha if tail_mask.any() else np.zeros(num_assets)
            grad_nu = 1.0 - tail_mask.mean() / self.alpha

            w -= lr * grad_w
            nu -= lr * grad_nu

            # Project to simplex
            w = np.maximum(w, 0)
            w /= w.sum() + 1e-8

        return w


# ============================================================
# Distributional Q-Network for CVaR
# ============================================================

class QuantileNetwork(nn.Module):
    """
    Quantile Regression DQN for distributional RL.
    Learns the return distribution for direct CVaR computation.
    """

    def __init__(self, state_dim: int, action_dim: int,
                 num_quantiles: int = 51, hidden_dim: int = 128):
        super().__init__()
        self.num_quantiles = num_quantiles
        self.action_dim = action_dim

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.quantile_head = nn.Linear(hidden_dim, action_dim * num_quantiles)

        # Fixed quantile midpoints: tau_i = (2i-1) / (2N)
        taus = (2 * torch.arange(num_quantiles).float() + 1) / (2 * num_quantiles)
        self.register_buffer("taus", taus)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Returns (batch, action_dim, num_quantiles)."""
        features = self.encoder(state)
        quantiles = self.quantile_head(features)
        return quantiles.view(-1, self.action_dim, self.num_quantiles)

    def compute_cvar(self, state: torch.Tensor, alpha: float = 0.05) -> torch.Tensor:
        """Compute CVaR for each action."""
        quantiles = self.forward(state)  # (batch, actions, N)
        n_tail = max(1, int(self.num_quantiles * alpha))
        sorted_q, _ = torch.sort(quantiles, dim=-1)
        cvar = sorted_q[:, :, :n_tail].mean(dim=-1)
        return cvar

    def select_action_cvar(self, state: torch.Tensor, alpha: float = 0.05,
                           risk_aversion: float = 0.5) -> int:
        """Select action balancing mean return and CVaR."""
        with torch.no_grad():
            quantiles = self.forward(state)
            mean_q = quantiles.mean(dim=-1)
            cvar = self.compute_cvar(state, alpha)
            score = (1 - risk_aversion) * mean_q + risk_aversion * cvar
            return score.argmax(dim=-1).item()

    @staticmethod
    def quantile_huber_loss(predictions: torch.Tensor, targets: torch.Tensor,
                            taus: torch.Tensor, kappa: float = 1.0) -> torch.Tensor:
        """Quantile Huber loss for training."""
        diff = targets.unsqueeze(-1) - predictions.unsqueeze(-2)
        huber = torch.where(diff.abs() <= kappa,
                            0.5 * diff ** 2,
                            kappa * (diff.abs() - 0.5 * kappa))
        tau_weight = (taus.unsqueeze(0) - (diff < 0).float()).abs()
        loss = (tau_weight * huber).mean()
        return loss


# ============================================================
# CVaR-Constrained Policy
# ============================================================

class CVaRConstrainedPolicy(nn.Module):
    """Policy with CVaR constraint via Lagrangian relaxation."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim) - 1.0)
        self.value_head = nn.Linear(hidden_dim, 1)

        # Learnable Lagrangian multiplier
        self.log_lambda = nn.Parameter(torch.tensor(1.0))

    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.encoder(state)
        mean = torch.tanh(self.mean_head(features))
        value = self.value_head(features).squeeze(-1)
        return {"mean": mean, "log_std": self.log_std, "value": value,
                "lambda": torch.exp(self.log_lambda)}


# ============================================================
# Demonstration
# ============================================================

def demo_cvar_optimization():
    """Demonstrate CVaR optimization."""
    print("=" * 70)
    print("CVaR Optimization Demonstration")
    print("=" * 70)

    np.random.seed(42)
    N = 5
    T = 500

    # Generate returns with fat tails
    returns = np.random.standard_t(df=5, size=(T, N)) * 0.015 + 0.0003

    config = CVaRConfig(alpha=0.05)
    weights = np.array([0.3, 0.25, 0.2, 0.15, 0.1])

    # Sample CVaR
    print("\n--- Sample CVaR ---")
    sample_cvar = SampleCVaR(alpha=0.05)
    result = sample_cvar.compute_portfolio(weights, returns)
    print(f"VaR (5%): {result['var']*100:.2f}%")
    print(f"CVaR (5%): {result['cvar']*100:.2f}%")
    print(f"Tail samples: {result['n_tail']}")

    # Rockafellar-Uryasev
    print("\n--- Rockafellar-Uryasev ---")
    ru = RockafellarUryasev(alpha=0.05)
    port_returns = returns @ weights
    ru_result = ru.compute(port_returns)
    print(f"VaR: {-ru_result['var']*100:.2f}%")
    print(f"CVaR: {-ru_result['cvar']*100:.2f}%")

    # Minimum CVaR portfolio
    print("\n--- Minimum CVaR Portfolio ---")
    opt_weights = ru.optimize_portfolio(returns, N)
    opt_result = sample_cvar.compute_portfolio(opt_weights, returns)
    print(f"Optimal weights: {np.round(opt_weights, 3)}")
    print(f"Optimal CVaR: {opt_result['cvar']*100:.2f}%")
    print(f"Original CVaR: {result['cvar']*100:.2f}%")

    # Quantile Network
    print("\n--- Quantile Network (Distributional RL) ---")
    qnet = QuantileNetwork(state_dim=10, action_dim=5, num_quantiles=51)
    params = sum(p.numel() for p in qnet.parameters())
    print(f"Parameters: {params:,}")

    state = torch.randn(1, 10)
    quantiles = qnet(state)
    print(f"Quantiles shape: {quantiles.shape}")

    cvar = qnet.compute_cvar(state, alpha=0.05)
    print(f"CVaR per action: {cvar[0].detach().numpy()}")

    action = qnet.select_action_cvar(state, alpha=0.05, risk_aversion=0.5)
    print(f"Selected action (risk_aversion=0.5): {action}")

    # CVaR across confidence levels
    print("\n--- CVaR at Different Confidence Levels ---")
    print(f"{'Alpha':>8} {'VaR%':>10} {'CVaR%':>10}")
    print("-" * 30)
    for alpha in [0.01, 0.025, 0.05, 0.10, 0.20]:
        est = SampleCVaR(alpha=alpha)
        r = est.compute_portfolio(weights, returns)
        print(f"{alpha:>8.3f} {r['var']*100:>9.2f} {r['cvar']*100:>9.2f}")


if __name__ == "__main__":
    demo_cvar_optimization()
