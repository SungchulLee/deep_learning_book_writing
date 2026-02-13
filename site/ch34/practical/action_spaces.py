"""
Chapter 34.6.2: Action Spaces
===============================
Action space wrappers, portfolio constraints, and
action transformation utilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import numpy as np


class PortfolioActionHead(nn.Module):
    """
    Portfolio allocation action head with various constraint modes.
    
    Modes:
    - 'softmax': Long-only, weights sum to 1
    - 'long_short': Weights sum to 0 (market neutral)
    - 'leverage': L1 norm bounded by max_leverage
    """
    
    def __init__(self, hidden_dim, n_assets, mode="softmax", max_leverage=1.0):
        super().__init__()
        self.mode = mode
        self.max_leverage = max_leverage
        self.head = nn.Linear(hidden_dim, n_assets)
    
    def forward(self, features):
        raw = self.head(features)
        
        if self.mode == "softmax":
            return F.softmax(raw, dim=-1)
        
        elif self.mode == "long_short":
            weights = torch.tanh(raw)
            weights = weights - weights.mean(dim=-1, keepdim=True)
            return weights
        
        elif self.mode == "leverage":
            weights = torch.tanh(raw)
            l1_norm = weights.abs().sum(dim=-1, keepdim=True)
            scale = torch.clamp(l1_norm / self.max_leverage, min=1.0)
            return weights / scale
        
        return raw


class DiscreteActionWrapper:
    """Discretize a continuous action space into bins."""
    
    def __init__(self, n_bins_per_dim, low, high):
        self.n_bins = n_bins_per_dim
        self.low = np.array(low)
        self.high = np.array(high)
        self.dim = len(low)
        self.total_actions = n_bins_per_dim ** self.dim
    
    def discrete_to_continuous(self, action_idx):
        """Convert discrete action index to continuous action."""
        indices = []
        idx = action_idx
        for _ in range(self.dim):
            indices.append(idx % self.n_bins)
            idx //= self.n_bins
        
        continuous = np.array([
            self.low[d] + (self.high[d] - self.low[d]) * i / (self.n_bins - 1)
            for d, i in enumerate(indices)
        ])
        return continuous


class ContinuousActionRescaler:
    """Rescale actions from [-1, 1] to actual bounds."""
    
    def __init__(self, low, high):
        self.low = torch.FloatTensor(low)
        self.high = torch.FloatTensor(high)
        self.center = (self.high + self.low) / 2
        self.scale = (self.high - self.low) / 2
    
    def scale_action(self, action):
        """[-1, 1] → [low, high]"""
        return action * self.scale + self.center
    
    def unscale_action(self, action):
        """[low, high] → [-1, 1]"""
        return (action - self.center) / self.scale


def demo_action_spaces():
    print("=" * 60)
    print("Action Space Demonstrations")
    print("=" * 60)
    
    # Portfolio action heads
    features = torch.randn(4, 64)  # Batch of 4
    n_assets = 5
    
    for mode in ["softmax", "long_short", "leverage"]:
        head = PortfolioActionHead(64, n_assets, mode=mode, max_leverage=1.5)
        weights = head(features)
        print(f"\n{mode} mode:")
        print(f"  Weights[0]: {weights[0].detach().numpy().round(3)}")
        print(f"  Sum: {weights[0].sum().item():.4f}")
        print(f"  L1 norm: {weights[0].abs().sum().item():.4f}")
    
    # Discretization
    print("\n" + "-" * 40)
    wrapper = DiscreteActionWrapper(n_bins_per_dim=5, low=[-1, -1], high=[1, 1])
    print(f"Discrete actions: {wrapper.total_actions} (5 bins × 2 dims)")
    for idx in [0, 6, 12, 24]:
        cont = wrapper.discrete_to_continuous(idx)
        print(f"  Action {idx:>2d} → {cont.round(3)}")
    
    # Rescaling
    print("\n" + "-" * 40)
    rescaler = ContinuousActionRescaler(low=[-2.0], high=[2.0])
    for a in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        scaled = rescaler.scale_action(torch.tensor([a]))
        print(f"  {a:>5.1f} → {scaled.item():>5.1f}")


if __name__ == "__main__":
    demo_action_spaces()
