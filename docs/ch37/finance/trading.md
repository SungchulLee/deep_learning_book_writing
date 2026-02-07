# Fairness in Algorithmic Trading

## Overview

While fairness in trading differs from classification-based fairness (there are no "protected groups" among securities), fairness concerns arise in **market access**, **execution quality**, and **information asymmetry**. ML-driven trading systems must ensure equitable treatment across client types, market participants, and order characteristics.

## Fairness Dimensions in Trading

### Execution Quality Fairness

All clients should receive comparable execution quality for similar orders:

$$\text{Slippage}(A=\text{retail}) \approx \text{Slippage}(A=\text{institutional})$$

### Market Access Fairness

ML-driven order routing should not systematically disadvantage certain participant types:

```python
import torch
import torch.nn as nn
import numpy as np
from typing import Dict

class FairOrderRouter(nn.Module):
    """
    Order routing model that ensures fair execution quality
    across client categories.
    """
    
    def __init__(self, n_features: int, n_venues: int = 5, fairness_weight: float = 1.0):
        super().__init__()
        self.fairness_weight = fairness_weight
        self.network = nn.Sequential(
            nn.Linear(n_features, 32), nn.ReLU(),
            nn.Linear(32, n_venues), nn.Softmax(dim=-1),
        )
    
    def forward(self, x):
        return self.network(x)
    
    def compute_loss(self, X, execution_quality, client_type):
        """
        Optimize routing for execution quality while ensuring
        fairness across client types.
        """
        venue_probs = self.forward(X)
        
        # Expected execution quality (higher = better)
        expected_quality = (venue_probs * execution_quality).sum(dim=1)
        quality_loss = -expected_quality.mean()
        
        # Fairness: equalize expected quality across client types
        types = torch.unique(client_type)
        avg_qualities = []
        for t in types:
            mask = client_type == t
            avg_qualities.append(expected_quality[mask].mean())
        
        if len(avg_qualities) >= 2:
            fair_loss = sum(
                (avg_qualities[i] - avg_qualities[j]) ** 2
                for i in range(len(avg_qualities))
                for j in range(i + 1, len(avg_qualities))
            )
        else:
            fair_loss = torch.tensor(0.0)
        
        return quality_loss + self.fairness_weight * fair_loss


def demo():
    torch.manual_seed(42)
    n = 2000
    n_venues = 5
    
    # Features: order size, volatility, spread, time of day
    X = torch.randn(n, 4)
    client_type = torch.randint(0, 3, (n,))  # retail, institutional, HFT
    
    # Venue execution quality (varies by order characteristics)
    execution_quality = torch.rand(n, n_venues) * 0.5 + 0.5
    
    model = FairOrderRouter(4, n_venues, fairness_weight=2.0)
    opt = torch.optim.Adam(model.parameters(), lr=0.005)
    
    for epoch in range(200):
        opt.zero_grad()
        loss = model.compute_loss(X, execution_quality, client_type)
        loss.backward()
        opt.step()
    
    model.eval()
    with torch.no_grad():
        probs = model(X)
        eq = (probs * execution_quality).sum(dim=1)
    
    print("Fair Order Routing")
    print("=" * 50)
    for ct, name in [(0, "Retail"), (1, "Institutional"), (2, "HFT")]:
        mask = client_type == ct
        print(f"  {name}: avg execution quality = {eq[mask].mean():.4f}")

if __name__ == "__main__":
    demo()
```

## Summary

- Trading fairness focuses on **execution quality equity** across participant types
- ML routing models should not systematically disadvantage retail or smaller participants
- **Best execution** regulations (MiFID II, Reg NMS) implicitly require fair treatment
- Fairness constraints can be integrated into venue selection optimization

## Next Steps

- [Regulatory Framework](regulatory.md): Comprehensive view of financial regulation and AI fairness
