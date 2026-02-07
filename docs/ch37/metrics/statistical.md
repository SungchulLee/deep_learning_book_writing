# Statistical Fairness Metrics

## Overview

Statistical fairness metrics quantify bias using aggregate statistics across protected groups. This section provides a unified PyTorch framework for computing all major metrics in a single pass.

## Comprehensive Metrics Framework

```python
import numpy as np
import torch
from typing import Dict, Optional
from dataclasses import dataclass, field

@dataclass
class FairnessReport:
    """Comprehensive fairness evaluation report."""
    demographic_parity: Dict[str, float] = field(default_factory=dict)
    equal_opportunity: Dict[str, float] = field(default_factory=dict)
    equalized_odds: Dict[str, float] = field(default_factory=dict)
    calibration: Dict[str, float] = field(default_factory=dict)
    overall_accuracy: float = 0.0
    group_accuracies: Dict[int, float] = field(default_factory=dict)

class ComprehensiveFairnessMetrics:
    """
    Unified calculator for all major statistical fairness metrics.
    
    Computes demographic parity, equal opportunity, equalized odds,
    predictive parity, and accuracy disparity in a single pass from
    per-group confusion matrices.
    """
    
    def compute(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        sensitive_attr: torch.Tensor,
    ) -> FairnessReport:
        report = FairnessReport()
        groups = torch.unique(sensitive_attr)
        
        group_stats = {}
        for g in groups:
            gv = g.item()
            mask = sensitive_attr == g
            tp = ((y_pred[mask] == 1) & (y_true[mask] == 1)).sum().float()
            fp = ((y_pred[mask] == 1) & (y_true[mask] == 0)).sum().float()
            fn = ((y_pred[mask] == 0) & (y_true[mask] == 1)).sum().float()
            n_pos = (y_true[mask] == 1).sum().float()
            n_neg = (y_true[mask] == 0).sum().float()
            n_pred_pos = (y_pred[mask] == 1).sum().float()
            
            group_stats[gv] = {
                'tpr': (tp / n_pos).item() if n_pos > 0 else 0.0,
                'fpr': (fp / n_neg).item() if n_neg > 0 else 0.0,
                'ppv': (tp / n_pred_pos).item() if n_pred_pos > 0 else 0.0,
                'positive_rate': y_pred[mask].float().mean().item(),
                'accuracy': (y_pred[mask] == y_true[mask]).float().mean().item(),
            }
        
        gvals = list(group_stats.values())
        rates = [g['positive_rate'] for g in gvals]
        tprs = [g['tpr'] for g in gvals]
        fprs = [g['fpr'] for g in gvals]
        ppvs = [g['ppv'] for g in gvals]
        
        report.demographic_parity = {
            'spd': abs(rates[0] - rates[1]),
            'dir': min(rates) / max(rates) if max(rates) > 0 else 0.0,
        }
        report.equal_opportunity = {'tpr_diff': abs(tprs[0] - tprs[1])}
        report.equalized_odds = {
            'tpr_diff': abs(tprs[0] - tprs[1]),
            'fpr_diff': abs(fprs[0] - fprs[1]),
            'max_violation': max(abs(tprs[0] - tprs[1]), abs(fprs[0] - fprs[1])),
        }
        report.calibration = {'ppv_diff': abs(ppvs[0] - ppvs[1])}
        report.overall_accuracy = (y_pred == y_true).float().mean().item()
        report.group_accuracies = {k: v['accuracy'] for k, v in group_stats.items()}
        
        return report
```

## Key Metrics Summary

| Metric | Formula | Threshold |
|--------|---------|-----------|
| Statistical Parity Difference | $\|P(\hat{Y}=1 \mid A=0) - P(\hat{Y}=1 \mid A=1)\|$ | < 0.1 |
| Disparate Impact Ratio | $\min(r_0, r_1) / \max(r_0, r_1)$ | ≥ 0.8 |
| TPR Difference | $\|\text{TPR}_0 - \text{TPR}_1\|$ | < 0.1 |
| FPR Difference | $\|\text{FPR}_0 - \text{FPR}_1\|$ | < 0.1 |
| PPV Difference | $\|\text{PPV}_0 - \text{PPV}_1\|$ | < 0.1 |
| Accuracy Gap | $\|\text{Acc}_0 - \text{Acc}_1\|$ | < 0.05 |

## Summary

- Always compute **multiple metrics** even when optimizing for one
- Per-group **confusion matrices** are the foundation for all rate-based metrics
- Use the unified `ComprehensiveFairnessMetrics` class for consistent evaluation

## Next Steps

- [Causal Metrics](causal.md): Metrics grounded in causal reasoning
- [Multi-Group Metrics](multi_group.md): Beyond binary group comparisons
