# OOD Detection Evaluation Metrics

## Overview

OOD detection is a binary classification problem (in-distribution vs OOD), evaluated with standard detection metrics.

## Key Metrics

### AUROC (Area Under ROC Curve)

Threshold-independent measure of detection quality. AUROC = 1.0 is perfect, 0.5 is random.

### FPR@95TPR

False positive rate when true positive rate is 95%. Measures: "When we detect 95% of OOD samples, what fraction of in-distribution samples are falsely flagged?"

### AUPR (Area Under Precision-Recall Curve)

More informative than AUROC when classes are imbalanced.

## Implementation

```python
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc


def evaluate_ood_detection(
    in_scores: np.ndarray, ood_scores: np.ndarray
) -> dict:
    """
    Evaluate OOD detection performance.
    
    Args:
        in_scores: OOD scores for in-distribution data (should be low)
        ood_scores: OOD scores for OOD data (should be high)
    """
    labels = np.concatenate([np.zeros(len(in_scores)), np.ones(len(ood_scores))])
    scores = np.concatenate([in_scores, ood_scores])
    
    # AUROC
    auroc = roc_auc_score(labels, scores)
    
    # FPR@95TPR
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.argmin(np.abs(tpr - 0.95))
    fpr95 = fpr[idx]
    
    # AUPR
    precision, recall, _ = precision_recall_curve(labels, scores)
    aupr = auc(recall, precision)
    
    return {
        'auroc': auroc,
        'fpr@95tpr': fpr95,
        'aupr': aupr,
        'in_mean': in_scores.mean(),
        'ood_mean': ood_scores.mean(),
    }
```

## Benchmark Practices

- Always report multiple metrics (AUROC alone can be misleading)
- Test against multiple OOD datasets (near-OOD and far-OOD)
- Report confidence intervals when possible
- Compare against MSP baseline

## References

- Hendrycks, D., & Gimpel, K. (2017). "A Baseline for Detecting Misclassified and Out-of-Distribution Examples." ICLR.
