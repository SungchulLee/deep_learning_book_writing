# ROC Curve and AUC: Comprehensive Guide

## Overview

The Receiver Operating Characteristic (ROC) curve and Area Under the Curve (AUC) are fundamental tools for evaluating binary classifiers. Unlike threshold-dependent metrics, ROC-AUC evaluates a classifier's ability to rank positive examples higher than negative examples across all possible thresholds.

---

## The ROC Curve

### Definition

The ROC curve plots **True Positive Rate (TPR)** against **False Positive Rate (FPR)** at various thresholds:

- **X-axis**: FPR = $\frac{FP}{FP + TN}$
- **Y-axis**: TPR = $\frac{TP}{TP + FN}$ = Recall

### Key Points

| Point | Coordinates | Meaning |
|-------|-------------|---------|
| (0, 0) | FPR=0, TPR=0 | Predict all negative |
| (1, 1) | FPR=1, TPR=1 | Predict all positive |
| (0, 1) | FPR=0, TPR=1 | **Perfect classifier** |
| Diagonal | TPR=FPR | Random classifier |

### Probabilistic Interpretation

$$\text{AUC} = P(\hat{p}_{+} > \hat{p}_{-})$$

AUC equals the probability that a randomly chosen positive ranks higher than a randomly chosen negative.

---

## PyTorch Implementation

### Computing ROC Curve

```python
import torch
from typing import Tuple, List

def compute_roc_curve(y_true: torch.Tensor, 
                     y_scores: torch.Tensor,
                     num_thresholds: int = 200) -> Tuple[List, List, List]:
    """
    Compute ROC curve points.
    
    Args:
        y_true: Ground truth labels (0 or 1)
        y_scores: Predicted probability scores
        num_thresholds: Number of threshold points
        
    Returns:
        (fpr_list, tpr_list, thresholds_list)
    """
    y_true = y_true.long()
    thresholds = torch.linspace(0, 1, num_thresholds)
    
    fpr_list, tpr_list = [], []
    total_pos = (y_true == 1).sum().item()
    total_neg = (y_true == 0).sum().item()
    
    for thresh in thresholds:
        y_pred = (y_scores >= thresh).long()
        tp = ((y_true == 1) & (y_pred == 1)).sum().item()
        fp = ((y_true == 0) & (y_pred == 1)).sum().item()
        
        tpr = tp / total_pos if total_pos > 0 else 0
        fpr = fp / total_neg if total_neg > 0 else 0
        
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    
    return fpr_list, tpr_list, thresholds.tolist()
```

### Computing AUC

```python
def compute_auc(fpr: List[float], tpr: List[float]) -> float:
    """Compute AUC using trapezoidal rule."""
    sorted_pairs = sorted(zip(fpr, tpr))
    fpr_sorted = [p[0] for p in sorted_pairs]
    tpr_sorted = [p[1] for p in sorted_pairs]
    
    auc = 0.0
    for i in range(1, len(fpr_sorted)):
        dx = fpr_sorted[i] - fpr_sorted[i-1]
        avg_height = (tpr_sorted[i] + tpr_sorted[i-1]) / 2
        auc += dx * avg_height
    
    return auc


def roc_auc_score(y_true: torch.Tensor, y_scores: torch.Tensor) -> float:
    """Compute ROC-AUC score."""
    fpr, tpr, _ = compute_roc_curve(y_true, y_scores)
    return compute_auc(fpr, tpr)
```

### Fast AUC (Ranking-Based)

```python
def roc_auc_fast(y_true: torch.Tensor, y_scores: torch.Tensor) -> float:
    """Fast AUC using Mann-Whitney U statistic."""
    pos_scores = y_scores[y_true == 1]
    neg_scores = y_scores[y_true == 0]
    
    n_pos, n_neg = len(pos_scores), len(neg_scores)
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    
    wins = sum((pos > neg_scores).sum().item() for pos in pos_scores)
    ties = sum((pos == neg_scores).sum().item() for pos in pos_scores)
    
    return (wins + 0.5 * ties) / (n_pos * n_neg)
```

---

## AUC Interpretation

| AUC Range | Interpretation |
|-----------|----------------|
| 0.90 - 1.00 | Excellent |
| 0.80 - 0.90 | Good |
| 0.70 - 0.80 | Fair |
| 0.60 - 0.70 | Poor |
| 0.50 - 0.60 | Near random |
| < 0.50 | Worse than random |

### What AUC Measures

- **Ranking quality**: How well does the model separate classes?
- **AUC = 1.0**: Perfect separation
- **AUC = 0.5**: Random ranking

### What AUC Does NOT Tell You

1. Probability calibration
2. Which threshold to use
3. FP/FN cost weighting

---

## ROC vs Precision-Recall Curves

### When to Use Each

| Scenario | Recommended | Reason |
|----------|-------------|--------|
| Balanced classes | ROC-AUC | Both work well |
| Imbalanced classes | PR-AUC | ROC can be optimistic |
| FP matters | PR curve | Shows precision |

### PR Curve Implementation

```python
def compute_pr_curve(y_true: torch.Tensor, 
                    y_scores: torch.Tensor,
                    num_thresholds: int = 200) -> Tuple[List, List, List]:
    """Compute Precision-Recall curve."""
    thresholds = torch.linspace(1, 0, num_thresholds)
    precision_list, recall_list = [], []
    
    total_pos = (y_true == 1).sum().item()
    
    for thresh in thresholds:
        y_pred = (y_scores >= thresh).long()
        tp = ((y_true == 1) & (y_pred == 1)).sum().item()
        fp = ((y_true == 0) & (y_pred == 1)).sum().item()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / total_pos if total_pos > 0 else 0.0
        
        precision_list.append(precision)
        recall_list.append(recall)
    
    return precision_list, recall_list, thresholds.tolist()


def pr_auc_score(y_true: torch.Tensor, y_scores: torch.Tensor) -> float:
    """Compute PR-AUC score."""
    precision, recall, _ = compute_pr_curve(y_true, y_scores)
    sorted_pairs = sorted(zip(recall, precision))
    
    auc = 0.0
    for i in range(1, len(sorted_pairs)):
        dx = sorted_pairs[i][0] - sorted_pairs[i-1][0]
        avg_h = (sorted_pairs[i][1] + sorted_pairs[i-1][1]) / 2
        auc += dx * avg_h
    
    return auc
```

---

## Multi-Class ROC-AUC

### One-vs-Rest Approach

```python
def multiclass_roc_auc(y_true: torch.Tensor, 
                      y_scores: torch.Tensor,
                      average: str = 'macro') -> float:
    """
    Multi-class ROC-AUC using One-vs-Rest.
    
    Args:
        y_true: Labels (0 to K-1)
        y_scores: Shape (n_samples, n_classes)
        average: 'macro' or 'weighted'
    """
    num_classes = y_scores.shape[1]
    aucs, supports = [], []
    
    for k in range(num_classes):
        y_true_k = (y_true == k).long()
        y_scores_k = y_scores[:, k]
        
        if y_true_k.sum() > 0 and (1 - y_true_k).sum() > 0:
            aucs.append(roc_auc_fast(y_true_k, y_scores_k))
            supports.append(y_true_k.sum().item())
    
    if average == 'macro':
        return sum(aucs) / len(aucs)
    else:  # weighted
        total = sum(supports)
        return sum(a * s for a, s in zip(aucs, supports)) / total
```

---

## Optimal Threshold Selection

### Youden's J Statistic

Maximize J = TPR - FPR:

```python
def optimal_threshold_youden(y_true: torch.Tensor, 
                            y_scores: torch.Tensor) -> Tuple[float, float, float]:
    """Find threshold maximizing Youden's J."""
    fpr, tpr, thresholds = compute_roc_curve(y_true, y_scores)
    j_scores = [t - f for t, f in zip(tpr, fpr)]
    best_idx = j_scores.index(max(j_scores))
    
    return thresholds[best_idx], fpr[best_idx], tpr[best_idx]
```

### Cost-Based Selection

For costs $c_{FP}$ (false positive) and $c_{FN}$ (false negative):

```python
def optimal_threshold_cost(y_true: torch.Tensor, 
                          y_scores: torch.Tensor,
                          cost_fp: float = 1.0,
                          cost_fn: float = 1.0) -> float:
    """Find threshold minimizing total cost."""
    fpr, tpr, thresholds = compute_roc_curve(y_true, y_scores)
    
    n_pos = (y_true == 1).sum().item()
    n_neg = (y_true == 0).sum().item()
    
    costs = []
    for f, t, thresh in zip(fpr, tpr, thresholds):
        fnr = 1 - t
        cost = cost_fp * f * n_neg + cost_fn * fnr * n_pos
        costs.append((cost, thresh))
    
    return min(costs)[1]
```

---

## Comprehensive ROC Analyzer

```python
import torch
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class ROCReport:
    """ROC analysis results."""
    roc_auc: float
    pr_auc: float
    optimal_threshold: float
    optimal_fpr: float
    optimal_tpr: float


class ROCAnalyzer:
    """
    Comprehensive ROC analysis.
    
    Example:
        >>> analyzer = ROCAnalyzer(y_true, y_scores)
        >>> print(f"AUC: {analyzer.roc_auc():.4f}")
    """
    
    def __init__(self, y_true: torch.Tensor, y_scores: torch.Tensor):
        self.y_true = y_true.long()
        self.y_scores = y_scores.float()
        self.fpr, self.tpr, self.thresholds = compute_roc_curve(y_true, y_scores)
    
    def roc_auc(self) -> float:
        return compute_auc(self.fpr, self.tpr)
    
    def pr_auc(self) -> float:
        return pr_auc_score(self.y_true, self.y_scores)
    
    def optimal_threshold(self) -> Tuple[float, float, float]:
        j = [t - f for t, f in zip(self.tpr, self.fpr)]
        idx = j.index(max(j))
        return self.thresholds[idx], self.fpr[idx], self.tpr[idx]
    
    def report(self) -> ROCReport:
        thresh, fpr, tpr = self.optimal_threshold()
        return ROCReport(
            roc_auc=self.roc_auc(),
            pr_auc=self.pr_auc(),
            optimal_threshold=thresh,
            optimal_fpr=fpr,
            optimal_tpr=tpr
        )


# Example
if __name__ == "__main__":
    torch.manual_seed(42)
    y_true = torch.cat([torch.zeros(100), torch.ones(100)]).long()
    y_scores = torch.cat([torch.rand(100) * 0.6, 0.4 + torch.rand(100) * 0.6])
    
    analyzer = ROCAnalyzer(y_true, y_scores)
    report = analyzer.report()
    
    print(f"ROC-AUC: {report.roc_auc:.4f}")
    print(f"PR-AUC: {report.pr_auc:.4f}")
    print(f"Optimal threshold: {report.optimal_threshold:.3f}")
```

---

## Visualization

```python
import matplotlib.pyplot as plt

def plot_roc(y_true: torch.Tensor, y_scores: torch.Tensor, 
            title: str = 'ROC Curve') -> plt.Figure:
    """Plot ROC curve."""
    fpr, tpr, _ = compute_roc_curve(y_true, y_scores)
    auc = compute_auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, 'b-', lw=2, label=f'ROC (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.fill_between(fpr, tpr, alpha=0.3)
    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    return fig
```

---

## Best Practices

### Do's
- ✅ Use ROC-AUC for overall ranking evaluation
- ✅ Consider PR-AUC for imbalanced data
- ✅ Report confidence intervals
- ✅ Visualize the full curve

### Don'ts
- ❌ Assume high AUC = good calibration
- ❌ Ignore class imbalance
- ❌ Compare AUC across different datasets

---

## Summary

| Metric | Range | Perfect | Baseline | Best For |
|--------|-------|---------|----------|----------|
| ROC-AUC | [0, 1] | 1.0 | 0.5 | Ranking |
| PR-AUC | [0, 1] | 1.0 | Prevalence | Imbalanced |

**Key insight**: ROC-AUC = P(positive ranked higher than negative)

---

## References

1. Fawcett, T. (2006). "An introduction to ROC analysis."
2. Davis, J. & Goadrich, M. (2006). "The relationship between Precision-Recall and ROC curves."
3. Saito, T. & Rehmsmeier, M. (2015). "The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets."
