# Longitudinal Fairness Analysis

## Overview

Fairness is not a one-time property—it must be monitored continuously in production. **Longitudinal analysis** tracks fairness metrics over time to detect drift, identify emerging biases, and trigger re-evaluation when thresholds are violated.

## Monitoring Framework

```python
import torch
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import deque

@dataclass
class FairnessSnapshot:
    """Fairness metrics at a point in time."""
    timestamp: int
    spd: float
    dir_ratio: float
    tpr_diff: float
    accuracy: float
    n_samples: int

class FairnessMonitor:
    """
    Production fairness monitoring with drift detection.
    
    Tracks fairness metrics over time windows and triggers alerts
    when metrics exceed configured thresholds.
    """
    
    def __init__(
        self,
        window_size: int = 100,
        alert_thresholds: Optional[Dict[str, float]] = None,
    ):
        self.window_size = window_size
        self.thresholds = alert_thresholds or {
            'spd': 0.1, 'dir': 0.8, 'tpr_diff': 0.1,
        }
        self.history: List[FairnessSnapshot] = []
        self.alerts: List[Dict] = []
    
    def update(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        A: torch.Tensor,
        timestamp: int,
    ) -> FairnessSnapshot:
        """Record a new fairness snapshot."""
        p0 = y_pred[A == 0].float().mean().item()
        p1 = y_pred[A == 1].float().mean().item()
        spd = abs(p0 - p1)
        dir_r = min(p0, p1) / max(p0, p1) if max(p0, p1) > 0 else 0
        
        pos0 = (A == 0) & (y_true == 1)
        pos1 = (A == 1) & (y_true == 1)
        tpr0 = y_pred[pos0].float().mean().item() if pos0.any() else 0
        tpr1 = y_pred[pos1].float().mean().item() if pos1.any() else 0
        
        snapshot = FairnessSnapshot(
            timestamp=timestamp,
            spd=spd,
            dir_ratio=dir_r,
            tpr_diff=abs(tpr0 - tpr1),
            accuracy=(y_pred == y_true).float().mean().item(),
            n_samples=len(y_true),
        )
        self.history.append(snapshot)
        
        # Check alerts
        if spd > self.thresholds['spd']:
            self.alerts.append({'time': timestamp, 'metric': 'spd', 'value': spd})
        if dir_r < self.thresholds['dir']:
            self.alerts.append({'time': timestamp, 'metric': 'dir', 'value': dir_r})
        
        return snapshot
    
    def get_trend(self, metric: str, last_n: int = 10) -> List[float]:
        """Get recent trend for a metric."""
        recent = self.history[-last_n:]
        return [getattr(s, metric) for s in recent]
    
    def detect_drift(self, metric: str = 'spd', window: int = 5) -> bool:
        """Detect if a metric is trending upward (worsening)."""
        if len(self.history) < 2 * window:
            return False
        recent = self.get_trend(metric, window)
        earlier = [getattr(s, metric) for s in self.history[-(2*window):-window]]
        return np.mean(recent) > np.mean(earlier) * 1.2  # 20% increase

# Demonstration
def demo():
    torch.manual_seed(42)
    monitor = FairnessMonitor()
    
    print("Longitudinal Fairness Monitoring")
    print("=" * 60)
    
    for t in range(20):
        n = 500
        A = torch.randint(0, 2, (n,))
        y = torch.randint(0, 2, (n,))
        
        # Gradually increasing bias over time
        drift = t * 0.01
        bias = torch.where(A == 0, torch.tensor(drift), torch.tensor(-drift))
        y_pred = (torch.rand(n) + bias > 0.5).long()
        
        snap = monitor.update(y, y_pred, A, timestamp=t)
        if (t + 1) % 5 == 0:
            print(f"  t={t:2d}: SPD={snap.spd:.4f}, DIR={snap.dir_ratio:.4f}, "
                  f"acc={snap.accuracy:.4f}")
    
    print(f"\nAlerts triggered: {len(monitor.alerts)}")
    for alert in monitor.alerts[-3:]:
        print(f"  t={alert['time']}: {alert['metric']}={alert['value']:.4f}")
    
    drift = monitor.detect_drift('spd')
    print(f"\nFairness drift detected: {drift}")

if __name__ == "__main__":
    demo()
```

## Summary

- **Continuous monitoring** detects fairness degradation before it causes harm
- **Drift detection** compares recent metric windows to historical baselines
- **Alerting** triggers when thresholds are violated, enabling rapid intervention
- Essential for regulatory compliance in finance (SR 11-7 model risk management)

## Next Steps

- [Credit Scoring](../finance/credit_scoring.md): Applying fairness to financial models
