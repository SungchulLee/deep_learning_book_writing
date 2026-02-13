"""
Chapter 35.2.2: Multi-Asset Allocation
=======================================
Deep RL architectures for multi-asset portfolio allocation including
EIIE, attention-based, and hierarchical approaches.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


# ============================================================
# Configuration
# ============================================================

@dataclass
class AllocationConfig:
    """Configuration for multi-asset allocation."""
    num_assets: int = 10
    lookback: int = 60
    feature_dim: int = 5        # Features per asset per time step
    hidden_dim: int = 128
    num_heads: int = 4           # For attention
    dropout: float = 0.1
    include_cash: bool = True
    max_position: float = 0.30
    allow_short: bool = False


# ============================================================
# EIIE Architecture (Ensemble of Identical Independent Evaluators)
# ============================================================

class EIIENetwork(nn.Module):
    """
    EIIE architecture (Jiang et al., 2017).
    Uses weight-shared sub-networks for each asset, making it
    scalable to arbitrary portfolio sizes.

    Architecture:
        Per-asset features → Shared CNN → Score → Softmax → Weights
    """

    def __init__(self, config: AllocationConfig):
        super().__init__()
        self.config = config
        self.num_assets = config.num_assets
        total_assets = config.num_assets + (1 if config.include_cash else 0)

        # Shared temporal convolution (processes each asset independently)
        self.conv1 = nn.Conv1d(config.feature_dim, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 1, kernel_size=1)  # Reduce to single score

        # Previous weights embedding
        self.weight_embed = nn.Linear(1, 16)

        # Score combiner
        self.score_net = nn.Sequential(
            nn.Linear(config.lookback + 16, 64),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, 1),
        )

        # Cash bias (learnable)
        if config.include_cash:
            self.cash_bias = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        asset_features: torch.Tensor,
        prev_weights: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            asset_features: (batch, num_assets, lookback, feature_dim)
            prev_weights: (batch, num_assets) previous portfolio weights

        Returns:
            dict with 'weights', 'scores'
        """
        batch_size = asset_features.shape[0]
        N = self.num_assets
        L = self.config.lookback

        # Process each asset through shared CNN
        # Reshape: (batch * N, feature_dim, lookback)
        x = asset_features.reshape(batch_size * N, L, -1).permute(0, 2, 1)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x).squeeze(1)  # (batch * N, lookback)

        x = x.reshape(batch_size, N, L)

        # Embed previous weights
        w_embed = self.weight_embed(prev_weights.unsqueeze(-1))  # (batch, N, 16)

        # Combine temporal features with weight embedding
        combined = torch.cat([x, w_embed], dim=-1)  # (batch, N, L+16)

        # Compute scores
        scores = self.score_net(combined).squeeze(-1)  # (batch, N)

        # Add cash option
        if self.config.include_cash:
            cash_score = self.cash_bias.expand(batch_size, 1)
            scores = torch.cat([scores, cash_score], dim=-1)

        # Softmax to get weights
        weights = F.softmax(scores, dim=-1)

        # Apply position limits
        if not self.config.include_cash:
            weights = self._apply_position_limits(weights)

        return {"weights": weights, "scores": scores}

    def _apply_position_limits(self, weights: torch.Tensor) -> torch.Tensor:
        """Apply maximum position limits with re-normalization."""
        clamped = torch.clamp(weights, max=self.config.max_position)
        return clamped / (clamped.sum(dim=-1, keepdim=True) + 1e-8)


# ============================================================
# Attention-Based Allocation
# ============================================================

class CrossAssetAttention(nn.Module):
    """Multi-head attention for cross-asset dependency modeling."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_assets, embed_dim)
        Returns:
            (batch, num_assets, embed_dim)
        """
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(x + attn_out)

        # FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class AttentionAllocationNetwork(nn.Module):
    """
    Transformer-based allocation network.
    Each asset attends to all others to capture cross-asset dependencies.
    """

    def __init__(self, config: AllocationConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_dim

        # Per-asset temporal encoder (LSTM)
        self.temporal_encoder = nn.LSTM(
            input_size=config.feature_dim,
            hidden_size=embed_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout,
        )

        # Cross-asset attention layers
        self.attention_layers = nn.ModuleList([
            CrossAssetAttention(embed_dim, config.num_heads, config.dropout)
            for _ in range(3)
        ])

        # Portfolio context embedding
        self.portfolio_embed = nn.Linear(1, embed_dim)

        # Score head
        self.score_head = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(embed_dim, 1),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim * config.num_assets, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        if config.include_cash:
            self.cash_bias = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        asset_features: torch.Tensor,
        prev_weights: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            asset_features: (batch, num_assets, lookback, feature_dim)
            prev_weights: (batch, num_assets)
        """
        batch_size = asset_features.shape[0]
        N = self.config.num_assets
        L = self.config.lookback

        # Temporal encoding per asset
        x = asset_features.reshape(batch_size * N, L, -1)
        temporal_out, _ = self.temporal_encoder(x)
        # Take last hidden state
        asset_embeds = temporal_out[:, -1, :]  # (batch*N, embed_dim)
        asset_embeds = asset_embeds.reshape(batch_size, N, -1)

        # Cross-asset attention
        for attn_layer in self.attention_layers:
            asset_embeds = attn_layer(asset_embeds)

        # Combine with portfolio context
        w_embed = self.portfolio_embed(prev_weights.unsqueeze(-1))
        combined = torch.cat([asset_embeds, w_embed], dim=-1)

        # Score per asset
        scores = self.score_head(combined).squeeze(-1)

        if self.config.include_cash:
            cash_score = self.cash_bias.expand(batch_size, 1)
            scores = torch.cat([scores, cash_score], dim=-1)

        weights = F.softmax(scores, dim=-1)

        # Value estimate
        flat = asset_embeds.reshape(batch_size, -1)
        value = self.value_head(flat).squeeze(-1)

        return {"weights": weights, "scores": scores, "value": value}


# ============================================================
# Hierarchical Allocation
# ============================================================

class HierarchicalAllocation(nn.Module):
    """
    Two-level hierarchical allocation:
    1. Sector/cluster level allocation
    2. Within-sector asset allocation

    Final weight: w_i = w_sector(i) * w_within(i)
    """

    def __init__(
        self,
        config: AllocationConfig,
        sector_assignments: List[int],
    ):
        """
        Args:
            config: allocation config
            sector_assignments: list mapping each asset to a sector index
        """
        super().__init__()
        self.config = config
        self.sector_assignments = sector_assignments
        self.num_sectors = max(sector_assignments) + 1
        self.sector_assets = self._group_by_sector()

        # Sector-level allocation
        self.sector_encoder = nn.Sequential(
            nn.Linear(config.hidden_dim * self.num_sectors, 256),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_sectors),
        )

        # Per-sector within-sector allocation
        self.within_encoders = nn.ModuleList()
        for sector_id in range(self.num_sectors):
            n_assets = len(self.sector_assets[sector_id])
            self.within_encoders.append(nn.Sequential(
                nn.Linear(config.hidden_dim * n_assets, 128),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(128, n_assets),
            ))

        # Per-asset feature extractor (shared)
        self.asset_encoder = nn.Sequential(
            nn.Linear(config.lookback * config.feature_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )

    def _group_by_sector(self) -> Dict[int, List[int]]:
        """Group asset indices by sector."""
        groups: Dict[int, List[int]] = {}
        for i, s in enumerate(self.sector_assignments):
            groups.setdefault(s, []).append(i)
        return groups

    def forward(
        self,
        asset_features: torch.Tensor,
        prev_weights: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            asset_features: (batch, num_assets, lookback, feature_dim)
            prev_weights: (batch, num_assets)
        """
        batch_size = asset_features.shape[0]
        N = self.config.num_assets

        # Encode each asset
        flat = asset_features.reshape(batch_size, N, -1)
        asset_embeds = self.asset_encoder(flat)  # (batch, N, hidden)

        # Aggregate sector representations
        sector_embeds = []
        for s in range(self.num_sectors):
            indices = self.sector_assets[s]
            sector_embed = asset_embeds[:, indices, :].mean(dim=1)
            sector_embeds.append(sector_embed)
        sector_input = torch.cat(sector_embeds, dim=-1)

        # Sector-level weights
        sector_scores = self.sector_encoder(sector_input)
        sector_weights = F.softmax(sector_scores, dim=-1)  # (batch, num_sectors)

        # Within-sector weights
        final_weights = torch.zeros(batch_size, N, device=asset_features.device)

        for s in range(self.num_sectors):
            indices = self.sector_assets[s]
            within_input = asset_embeds[:, indices, :].reshape(batch_size, -1)
            within_scores = self.within_encoders[s](within_input)
            within_weights = F.softmax(within_scores, dim=-1)  # (batch, n_in_sector)

            for j, asset_idx in enumerate(indices):
                final_weights[:, asset_idx] = (
                    sector_weights[:, s] * within_weights[:, j]
                )

        return {
            "weights": final_weights,
            "sector_weights": sector_weights,
        }


# ============================================================
# Equal Risk Contribution Baseline
# ============================================================

class EqualRiskContribution:
    """
    Equal Risk Contribution (ERC) portfolio.
    Each asset contributes equally to total portfolio risk.

    w_i * (Σw)_i = (1/N) * w'Σw  for all i
    """

    def __init__(self, max_iter: int = 1000, tol: float = 1e-8):
        self.max_iter = max_iter
        self.tol = tol

    def compute_weights(self, returns: np.ndarray) -> np.ndarray:
        """
        Compute ERC weights using iterative algorithm.

        Args:
            returns: (T, N) array of returns
        Returns:
            weights: (N,) ERC weights
        """
        cov = np.cov(returns.T)
        N = cov.shape[0]

        # Regularize
        cov += np.eye(N) * 1e-6

        # Initialize equal weights
        w = np.ones(N) / N

        for _ in range(self.max_iter):
            risk_contrib = w * (cov @ w)
            total_risk = w @ cov @ w

            target = total_risk / N
            gradient = risk_contrib - target

            # Update
            w_new = w * np.exp(-0.5 * gradient / (np.abs(risk_contrib) + 1e-8))
            w_new = w_new / np.sum(w_new)

            if np.max(np.abs(w_new - w)) < self.tol:
                break
            w = w_new

        return w

    def verify_erc(self, weights: np.ndarray, returns: np.ndarray) -> Dict:
        """Verify that the ERC condition holds."""
        cov = np.cov(returns.T)
        risk_contrib = weights * (cov @ weights)
        total_risk = weights @ cov @ weights
        pct_contrib = risk_contrib / total_risk

        return {
            "risk_contributions": risk_contrib,
            "pct_contributions": pct_contrib,
            "total_risk": total_risk,
            "max_deviation": float(np.max(np.abs(pct_contrib - 1.0 / len(weights)))),
        }


# ============================================================
# Curriculum Learning Trainer
# ============================================================

class CurriculumTrainer:
    """
    Curriculum learning for multi-asset allocation.
    Gradually increases the number of assets during training.
    """

    def __init__(
        self,
        all_asset_features: np.ndarray,
        schedule: List[Tuple[int, int]],
    ):
        """
        Args:
            all_asset_features: (T, N_total, lookback, feature_dim)
            schedule: list of (num_assets, num_episodes) for each phase
        """
        self.all_features = all_asset_features
        self.schedule = schedule
        self.current_phase = 0

    def get_current_config(self) -> Tuple[int, np.ndarray]:
        """Get the number of assets and data for the current phase."""
        num_assets, _ = self.schedule[self.current_phase]
        # Select the first num_assets
        features = self.all_features[:, :num_assets]
        return num_assets, features

    def should_advance(self, episode: int) -> bool:
        """Check if we should advance to the next curriculum phase."""
        total_episodes = 0
        for i, (_, n_ep) in enumerate(self.schedule):
            total_episodes += n_ep
            if episode < total_episodes:
                if i != self.current_phase:
                    self.current_phase = i
                    return True
                return False
        return False


# ============================================================
# Demonstration
# ============================================================

def generate_multi_asset_data(
    num_assets: int = 10,
    num_steps: int = 500,
    lookback: int = 60,
    feature_dim: int = 5,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Generate synthetic multi-asset data for testing."""
    np.random.seed(seed)

    # Generate synthetic features (already in feature form)
    features = np.random.randn(num_steps, num_assets, lookback, feature_dim) * 0.1

    # Generate returns for baselines
    returns = np.random.randn(num_steps, num_assets) * 0.01 + 0.0001

    # Convert to tensors
    features_tensor = torch.FloatTensor(features)
    returns_array = returns

    # Initial equal weights
    prev_weights = torch.ones(num_assets) / num_assets

    return features_tensor, prev_weights, returns_array


def demo_multi_asset_allocation():
    """Demonstrate multi-asset allocation architectures."""
    print("=" * 70)
    print("Multi-Asset Allocation Demonstration")
    print("=" * 70)

    num_assets = 10
    config = AllocationConfig(
        num_assets=num_assets,
        lookback=60,
        feature_dim=5,
        hidden_dim=128,
        include_cash=True,
    )

    features, prev_weights, returns = generate_multi_asset_data(num_assets=num_assets)
    batch_features = features[:4]  # batch of 4
    batch_weights = prev_weights.unsqueeze(0).expand(4, -1)

    # --- EIIE Network ---
    print("\n--- EIIE Architecture ---")
    eiie = EIIENetwork(config)
    params = sum(p.numel() for p in eiie.parameters())
    print(f"Parameters: {params:,}")

    with torch.no_grad():
        output = eiie(batch_features, batch_weights)
    w = output["weights"][0].numpy()
    print(f"Output weights (first sample): {w[:5]}... (showing 5 of {len(w)})")
    print(f"Weight sum: {w.sum():.6f}")
    print(f"Max weight: {w.max():.4f}, Min weight: {w.min():.4f}")

    # --- Attention Network ---
    print("\n--- Attention-Based Allocation ---")
    attn_net = AttentionAllocationNetwork(config)
    params = sum(p.numel() for p in attn_net.parameters())
    print(f"Parameters: {params:,}")

    with torch.no_grad():
        output = attn_net(batch_features, batch_weights)
    w = output["weights"][0].numpy()
    print(f"Output weights (first sample): {w[:5]}... (showing 5 of {len(w)})")
    print(f"Weight sum: {w.sum():.6f}")
    print(f"Value estimate: {output['value'][0].item():.4f}")

    # --- Hierarchical Allocation ---
    print("\n--- Hierarchical Allocation ---")
    # Assign assets to 3 sectors
    sector_assignments = [0, 0, 0, 1, 1, 1, 1, 2, 2, 2]
    config_no_cash = AllocationConfig(
        num_assets=num_assets, lookback=60, feature_dim=5,
        hidden_dim=128, include_cash=False,
    )
    hier_net = HierarchicalAllocation(config_no_cash, sector_assignments)
    params = sum(p.numel() for p in hier_net.parameters())
    print(f"Parameters: {params:,}")
    print(f"Sectors: {hier_net.num_sectors}")
    print(f"Assets per sector: {[len(v) for v in hier_net.sector_assets.values()]}")

    with torch.no_grad():
        output = hier_net(batch_features, batch_weights)
    w = output["weights"][0].numpy()
    sw = output["sector_weights"][0].numpy()
    print(f"Sector weights: {sw}")
    print(f"Asset weights: {w}")
    print(f"Weight sum: {w.sum():.6f}")

    # --- Equal Risk Contribution Baseline ---
    print("\n--- Equal Risk Contribution Baseline ---")
    erc = EqualRiskContribution()
    erc_weights = erc.compute_weights(returns[-252:])
    print(f"ERC weights: {erc_weights}")
    print(f"Weight sum: {erc_weights.sum():.6f}")

    verification = erc.verify_erc(erc_weights, returns[-252:])
    print(f"Risk contributions (%): {verification['pct_contributions'] * 100}")
    print(f"Max deviation from equal: {verification['max_deviation']:.6f}")

    # --- Architecture Comparison ---
    print("\n--- Architecture Comparison ---")
    print(f"{'Architecture':<25} {'Parameters':>12} {'Weight Range':>15}")
    print("-" * 55)

    architectures = {
        "EIIE": eiie,
        "Attention": attn_net,
        "Hierarchical": hier_net,
    }
    for name, net in architectures.items():
        p = sum(p.numel() for p in net.parameters())
        with torch.no_grad():
            if name == "Hierarchical":
                out = net(batch_features, batch_weights)
            else:
                out = net(batch_features, batch_weights)
        w = out["weights"][0].numpy()
        print(f"{name:<25} {p:>12,} [{w.min():.4f}, {w.max():.4f}]")

    # --- Curriculum Learning ---
    print("\n--- Curriculum Learning Schedule ---")
    schedule = [
        (3, 100),   # Phase 1: 3 assets, 100 episodes
        (5, 200),   # Phase 2: 5 assets, 200 episodes
        (10, 500),  # Phase 3: 10 assets, 500 episodes
    ]
    print("Phase | Assets | Episodes")
    print("-" * 30)
    for i, (n, ep) in enumerate(schedule):
        print(f"  {i+1}   |   {n:2d}   |   {ep}")

    print("\n--- Constraint Satisfaction Check ---")
    test_configs = [
        ("Long-only + cash", AllocationConfig(num_assets=5, include_cash=True, allow_short=False)),
        ("Long-only no cash", AllocationConfig(num_assets=5, include_cash=False, allow_short=False)),
    ]

    small_features = features[:4, :5]
    small_weights = torch.ones(4, 5) / 5

    for name, cfg in test_configs:
        net = EIIENetwork(cfg)
        with torch.no_grad():
            out = net(small_features, small_weights)
        w = out["weights"][0].numpy()
        print(f"\n{name}:")
        print(f"  Weights: {w}")
        print(f"  Sum: {w.sum():.6f}")
        print(f"  All non-negative: {(w >= -1e-6).all()}")


if __name__ == "__main__":
    demo_multi_asset_allocation()
