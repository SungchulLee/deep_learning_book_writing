# Fairness Audits

## Overview

A **fairness audit** is a systematic evaluation of an ML model for bias across protected groups. It combines quantitative metrics with qualitative analysis to assess whether a model meets fairness requirements before or after deployment.

## Audit Framework

A comprehensive fairness audit proceeds through five stages:

1. **Scope definition**: Identify protected attributes, fairness criteria, and acceptable thresholds
2. **Data analysis**: Examine training and test data for representation and label bias
3. **Model evaluation**: Compute fairness metrics on held-out data
4. **Stress testing**: Evaluate on adversarial or edge-case scenarios
5. **Documentation**: Record findings, decisions, and recommendations

## PyTorch Implementation

```python
import torch
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class AuditResult:
    """Complete fairness audit result."""
    timestamp: str
    model_name: str
    dataset_name: str
    protected_attributes: List[str]
    data_analysis: Dict[str, float]
    fairness_metrics: Dict[str, Dict[str, float]]
    pass_fail: Dict[str, bool]
    recommendations: List[str]
    overall_pass: bool

class FairnessAuditor:
    """
    Conduct a comprehensive fairness audit on a classification model.
    """
    
    def __init__(
        self,
        thresholds: Optional[Dict[str, float]] = None,
    ):
        self.thresholds = thresholds or {
            'spd': 0.1,
            'dir': 0.8,
            'tpr_diff': 0.1,
            'fpr_diff': 0.1,
            'accuracy_gap': 0.05,
        }
    
    def audit_data(
        self, y: torch.Tensor, A: torch.Tensor,
    ) -> Dict[str, float]:
        """Analyze training data for bias indicators."""
        groups = torch.unique(A).tolist()
        analysis = {}
        
        for g in groups:
            mask = A == g
            analysis[f'group_{g}_count'] = mask.sum().item()
            analysis[f'group_{g}_fraction'] = mask.float().mean().item()
            analysis[f'group_{g}_positive_rate'] = y[mask].float().mean().item()
        
        counts = [analysis[f'group_{g}_count'] for g in groups]
        analysis['representation_ratio'] = min(counts) / max(counts)
        
        rates = [analysis[f'group_{g}_positive_rate'] for g in groups]
        analysis['base_rate_gap'] = max(rates) - min(rates)
        
        return analysis
    
    def audit_model(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        A: torch.Tensor,
    ) -> Dict[str, Dict[str, float]]:
        """Compute all fairness metrics."""
        groups = torch.unique(A).tolist()
        
        stats = {}
        for g in groups:
            mask = A == g
            pos = mask & (y_true == 1)
            neg = mask & (y_true == 0)
            pred_pos = mask & (y_pred == 1)
            
            stats[g] = {
                'positive_rate': y_pred[mask].float().mean().item(),
                'tpr': y_pred[pos].float().mean().item() if pos.any() else 0,
                'fpr': y_pred[neg].float().mean().item() if neg.any() else 0,
                'accuracy': (y_pred[mask] == y_true[mask]).float().mean().item(),
            }
        
        vals = list(stats.values())
        metrics = {
            'demographic_parity': {
                'spd': abs(vals[0]['positive_rate'] - vals[1]['positive_rate']),
                'dir': min(v['positive_rate'] for v in vals) / max(v['positive_rate'] for v in vals) if max(v['positive_rate'] for v in vals) > 0 else 0,
            },
            'equal_opportunity': {
                'tpr_diff': abs(vals[0]['tpr'] - vals[1]['tpr']),
            },
            'equalized_odds': {
                'fpr_diff': abs(vals[0]['fpr'] - vals[1]['fpr']),
            },
            'accuracy': {
                'accuracy_gap': abs(vals[0]['accuracy'] - vals[1]['accuracy']),
                'overall': (y_pred == y_true).float().mean().item(),
            },
        }
        
        return metrics
    
    def run_audit(
        self,
        model_name: str,
        dataset_name: str,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        A: torch.Tensor,
        attr_name: str = 'protected',
    ) -> AuditResult:
        """Run a complete fairness audit."""
        data_analysis = self.audit_data(y_true, A)
        metrics = self.audit_model(y_true, y_pred, A)
        
        pass_fail = {
            'demographic_parity': metrics['demographic_parity']['spd'] < self.thresholds['spd'],
            'disparate_impact': metrics['demographic_parity']['dir'] >= self.thresholds['dir'],
            'equal_opportunity': metrics['equal_opportunity']['tpr_diff'] < self.thresholds['tpr_diff'],
            'equalized_odds': metrics['equalized_odds']['fpr_diff'] < self.thresholds['fpr_diff'],
            'accuracy_parity': metrics['accuracy']['accuracy_gap'] < self.thresholds['accuracy_gap'],
        }
        
        recommendations = []
        if not pass_fail['demographic_parity']:
            recommendations.append("Consider reweighing or threshold optimization for DP")
        if not pass_fail['equal_opportunity']:
            recommendations.append("Consider EO regularization or adversarial debiasing")
        if data_analysis['representation_ratio'] < 0.5:
            recommendations.append("Significant group imbalance — consider oversampling")
        if data_analysis['base_rate_gap'] > 0.15:
            recommendations.append("Large base rate gap — note impossibility theorem implications")
        
        return AuditResult(
            timestamp=datetime.now().isoformat(),
            model_name=model_name,
            dataset_name=dataset_name,
            protected_attributes=[attr_name],
            data_analysis=data_analysis,
            fairness_metrics=metrics,
            pass_fail=pass_fail,
            recommendations=recommendations,
            overall_pass=all(pass_fail.values()),
        )
    
    def print_audit(self, result: AuditResult):
        """Print formatted audit report."""
        print("=" * 65)
        print(f"FAIRNESS AUDIT REPORT — {result.model_name}")
        print(f"Dataset: {result.dataset_name}")
        print(f"Date: {result.timestamp}")
        print("=" * 65)
        
        print("\n--- Data Analysis ---")
        for k, v in result.data_analysis.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        
        print("\n--- Fairness Metrics ---")
        for category, metrics in result.fairness_metrics.items():
            print(f"  {category}:")
            for k, v in metrics.items():
                print(f"    {k}: {v:.4f}")
        
        print("\n--- Pass/Fail ---")
        for criterion, passed in result.pass_fail.items():
            print(f"  {criterion}: {'✓ PASS' if passed else '✗ FAIL'}")
        
        print(f"\n--- Overall: {'✓ PASS' if result.overall_pass else '✗ FAIL'} ---")
        
        if result.recommendations:
            print("\n--- Recommendations ---")
            for rec in result.recommendations:
                print(f"  • {rec}")


# Demonstration
def demo():
    torch.manual_seed(42)
    n = 2000
    A = torch.randint(0, 2, (n,))
    y = torch.where(A == 0,
        torch.tensor(np.random.choice([0,1], n, p=[0.35, 0.65])),
        torch.tensor(np.random.choice([0,1], n, p=[0.50, 0.50])))
    y_pred = torch.where(A == 0,
        torch.tensor(np.random.choice([0,1], n, p=[0.30, 0.70])),
        torch.tensor(np.random.choice([0,1], n, p=[0.55, 0.45])))
    
    auditor = FairnessAuditor()
    result = auditor.run_audit("CreditModel_v2", "lending_2024", y, y_pred, A, "race")
    auditor.print_audit(result)

if __name__ == "__main__":
    demo()
```

## Summary

- A fairness audit is a **systematic, documented evaluation** of model bias
- Combines **data analysis**, **model metrics**, **pass/fail assessment**, and **recommendations**
- Should be conducted before deployment and periodically thereafter
- The `FairnessAuditor` class provides a reusable, standardized audit framework

## Next Steps

- [Disparate Impact Testing](testing.md): Statistical testing for adverse impact
- [Longitudinal Analysis](longitudinal.md): Monitoring fairness over time
