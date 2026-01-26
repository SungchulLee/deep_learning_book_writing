# Precision, Recall, and F1-Score: In-Depth Analysis

## Overview

Precision, recall, and F1-score are the core trinity of classification metrics. While accuracy tells us "how often are we correct?", these metrics answer more nuanced questions about the types of errors a model makes and their relative costs.

This section provides mathematical foundations, practical implementations, and guidance for choosing among these metrics based on business context.

---

## The Fundamental Trade-off

### Why Accuracy Isn't Enough

Consider fraud detection with 1% fraud rate:
- Predicting "not fraud" always → 99% accuracy
- But catches 0% of actual fraud → useless

**Key insight**: We need metrics that separately evaluate performance on each class.

### Two Types of Errors

| Error | Name | Meaning | Alternative Names |
|-------|------|---------|-------------------|
| FP | Type I Error | Predicting positive when actually negative | False Alarm |
| FN | Type II Error | Predicting negative when actually positive | Miss |

Different applications prioritize different errors:
- **Spam filter**: FP (blocking legitimate email) is worse → prioritize precision
- **Medical screening**: FN (missing disease) is worse → prioritize recall

---

## Precision

### Definition

**Question**: Of all positive predictions, how many were correct?

$$\text{Precision} = \frac{TP}{TP + FP} = \frac{\text{True Positives}}{\text{All Positive Predictions}}$$

### Intuition

Precision measures the **reliability** of positive predictions:
- High precision → When model says "positive," it's usually right
- Low precision → Many false alarms

### Mathematical Properties

- **Range**: $[0, 1]$
- **Perfect score**: 1.0 (no false positives)
- **Undefined**: When $TP + FP = 0$ (no positive predictions)

### PyTorch Implementation

```python
import torch
from typing import Optional

def precision_score(y_true: torch.Tensor, 
                   y_pred: torch.Tensor,
                   zero_division: float = 0.0) -> float:
    """
    Calculate precision: TP / (TP + FP)
    
    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted labels (0 or 1)
        zero_division: Value to return when no positive predictions
        
    Returns:
        Precision score in [0, 1]
    """
    y_true = y_true.long()
    y_pred = y_pred.long()
    
    tp = ((y_true == 1) & (y_pred == 1)).sum().item()
    fp = ((y_true == 0) & (y_pred == 1)).sum().item()
    
    if tp + fp == 0:
        return zero_division
    
    return tp / (tp + fp)


# Example
y_true = torch.tensor([0, 1, 1, 0, 1, 1, 0, 0, 1, 0])
y_pred = torch.tensor([0, 1, 1, 0, 0, 1, 0, 1, 1, 0])

prec = precision_score(y_true, y_pred)
print(f"Precision: {prec:.4f}")  # 0.8000
```

### When High Precision Matters

| Application | Why Precision Matters |
|-------------|----------------------|
| Spam filtering | Don't block legitimate emails |
| Search engines | Show relevant results |
| Drug approval | Don't approve ineffective treatments |
| Trading signals | Minimize false buy signals |

---

## Recall (Sensitivity)

### Definition

**Question**: Of all actual positives, how many did we find?

$$\text{Recall} = \frac{TP}{TP + FN} = \frac{\text{True Positives}}{\text{All Actual Positives}}$$

### Alternative Names

- **Sensitivity** (medical)
- **True Positive Rate (TPR)**
- **Hit Rate**
- **Detection Rate**

### Intuition

Recall measures **coverage** of actual positives:
- High recall → We find most positive cases
- Low recall → We miss many positive cases

### Mathematical Properties

- **Range**: $[0, 1]$
- **Perfect score**: 1.0 (no false negatives)
- **Undefined**: When $TP + FN = 0$ (no actual positives)

### PyTorch Implementation

```python
import torch

def recall_score(y_true: torch.Tensor, 
                y_pred: torch.Tensor,
                zero_division: float = 0.0) -> float:
    """
    Calculate recall: TP / (TP + FN)
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        zero_division: Value when no actual positives
        
    Returns:
        Recall score in [0, 1]
    """
    y_true = y_true.long()
    y_pred = y_pred.long()
    
    tp = ((y_true == 1) & (y_pred == 1)).sum().item()
    fn = ((y_true == 1) & (y_pred == 0)).sum().item()
    
    if tp + fn == 0:
        return zero_division
    
    return tp / (tp + fn)


# Example
recall = recall_score(y_true, y_pred)
print(f"Recall: {recall:.4f}")  # 0.8000
```

### When High Recall Matters

| Application | Why Recall Matters |
|-------------|-------------------|
| Disease screening | Don't miss sick patients |
| Fraud detection | Catch all fraud cases |
| Security threats | Detect all intrusions |
| Credit risk | Identify all potential defaults |

---

## The Precision-Recall Trade-off

### Inverse Relationship

Precision and recall typically have an **inverse relationship**:

```
           Precision ↑
                │
        *       │
            *   │   More selective predictions
                │
                │       *
                │           *
       ─────────┼──────────────→ Recall ↑
                │
        More inclusive predictions
```

### Threshold Effect

Most classifiers output probabilities. The threshold determines the trade-off:

```python
import torch
import numpy as np

def precision_recall_at_thresholds(y_true: torch.Tensor,
                                   y_proba: torch.Tensor,
                                   thresholds: list) -> dict:
    """
    Calculate precision and recall at different thresholds.
    """
    results = []
    
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).long()
        
        tp = ((y_true == 1) & (y_pred == 1)).sum().item()
        fp = ((y_true == 0) & (y_pred == 1)).sum().item()
        fn = ((y_true == 1) & (y_pred == 0)).sum().item()
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        results.append({
            'threshold': thresh,
            'precision': prec,
            'recall': rec,
            'tp': tp, 'fp': fp, 'fn': fn
        })
    
    return results


# Example
y_true = torch.tensor([0, 1, 1, 0, 1, 1, 0, 0, 1, 0])
y_proba = torch.tensor([0.1, 0.9, 0.8, 0.2, 0.4, 0.7, 0.3, 0.6, 0.85, 0.15])

thresholds = [0.3, 0.5, 0.7, 0.9]
results = precision_recall_at_thresholds(y_true, y_proba, thresholds)

print("Threshold  Precision  Recall")
print("-" * 30)
for r in results:
    print(f"   {r['threshold']:.1f}       {r['precision']:.3f}     {r['recall']:.3f}")
```

### Choosing a Threshold

| Priority | Strategy | Threshold |
|----------|----------|-----------|
| High Precision | Conservative | Higher (e.g., 0.7-0.9) |
| High Recall | Inclusive | Lower (e.g., 0.3-0.5) |
| Balance | Optimize F1 | Find max F1 |
| Custom | Cost-sensitive | Based on FP/FN costs |

---

## F1-Score

### Definition

The **harmonic mean** of precision and recall:

$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$

### Why Harmonic Mean?

The harmonic mean penalizes extreme values more than arithmetic mean:

| Precision | Recall | Arithmetic Mean | Harmonic Mean (F1) |
|-----------|--------|-----------------|-------------------|
| 0.9 | 0.9 | 0.90 | 0.90 |
| 0.9 | 0.1 | 0.50 | 0.18 |
| 0.5 | 0.5 | 0.50 | 0.50 |

**Key insight**: F1 requires BOTH precision AND recall to be high.

### Derivation

Starting from the harmonic mean definition for two numbers:

$$H(a, b) = \frac{2}{\frac{1}{a} + \frac{1}{b}} = \frac{2ab}{a + b}$$

Substituting $a = \text{Precision}$, $b = \text{Recall}$:

$$F_1 = \frac{2 \cdot P \cdot R}{P + R}$$

### PyTorch Implementation

```python
import torch

def f1_score(y_true: torch.Tensor, 
            y_pred: torch.Tensor,
            zero_division: float = 0.0) -> float:
    """
    Calculate F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        zero_division: Value when undefined
        
    Returns:
        F1 score in [0, 1]
    """
    prec = precision_score(y_true, y_pred, zero_division)
    rec = recall_score(y_true, y_pred, zero_division)
    
    if prec + rec == 0:
        return zero_division
    
    return 2 * (prec * rec) / (prec + rec)


# Alternative: Direct computation
def f1_score_direct(y_true: torch.Tensor, 
                   y_pred: torch.Tensor) -> float:
    """F1 computed directly from TP, FP, FN."""
    y_true = y_true.long()
    y_pred = y_pred.long()
    
    tp = ((y_true == 1) & (y_pred == 1)).sum().item()
    fp = ((y_true == 0) & (y_pred == 1)).sum().item()
    fn = ((y_true == 1) & (y_pred == 0)).sum().item()
    
    if 2*tp + fp + fn == 0:
        return 0.0
    
    return (2 * tp) / (2 * tp + fp + fn)


# Example
f1 = f1_score(y_true, y_pred)
print(f"F1 Score: {f1:.4f}")  # 0.8000
```

---

## F-beta Score

### Generalization

F-beta allows weighted importance of precision vs recall:

$$F_\beta = (1 + \beta^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{\beta^2 \cdot \text{Precision} + \text{Recall}}$$

### Beta Interpretation

| β Value | Weights | Use Case |
|---------|---------|----------|
| β = 0 | Pure precision | Not practical |
| β = 0.5 | Precision 2× recall | Spam filtering |
| β = 1 | Equal (F1) | Balanced tasks |
| β = 2 | Recall 2× precision | Medical screening |
| β → ∞ | Pure recall | Not practical |

### Mathematical Insight

The factor $(1 + \beta^2)$ normalizes so that $F_\beta$ ranges in $[0, 1]$:

When $P = R$: $F_\beta = P = R$ regardless of $\beta$.

### PyTorch Implementation

```python
import torch

def f_beta_score(y_true: torch.Tensor, 
                y_pred: torch.Tensor,
                beta: float = 1.0,
                zero_division: float = 0.0) -> float:
    """
    Calculate F-beta Score.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        beta: Weight factor (β > 1 favors recall, β < 1 favors precision)
        zero_division: Value when undefined
        
    Returns:
        F-beta score in [0, 1]
    """
    prec = precision_score(y_true, y_pred, zero_division)
    rec = recall_score(y_true, y_pred, zero_division)
    
    if prec + rec == 0:
        return zero_division
    
    beta_sq = beta ** 2
    return (1 + beta_sq) * (prec * rec) / (beta_sq * prec + rec)


# Examples
y_true = torch.tensor([0, 1, 1, 0, 1, 1, 0, 0, 1, 0])
y_pred = torch.tensor([0, 1, 1, 0, 0, 1, 0, 1, 1, 0])

print(f"F0.5 (precision-weighted): {f_beta_score(y_true, y_pred, 0.5):.4f}")
print(f"F1 (balanced):             {f_beta_score(y_true, y_pred, 1.0):.4f}")
print(f"F2 (recall-weighted):      {f_beta_score(y_true, y_pred, 2.0):.4f}")
```

---

## Multi-Class Extensions

### Averaging Strategies

For $K$ classes, we need to aggregate per-class metrics:

#### Macro Average

Simple mean across classes (treats all classes equally):

$$\text{Macro-Precision} = \frac{1}{K} \sum_{k=1}^{K} \text{Precision}_k$$

#### Micro Average

Global computation using total TP, FP, FN:

$$\text{Micro-Precision} = \frac{\sum_k TP_k}{\sum_k (TP_k + FP_k)}$$

#### Weighted Average

Weighted by class support (number of samples):

$$\text{Weighted-Precision} = \sum_{k=1}^{K} \frac{n_k}{n} \cdot \text{Precision}_k$$

### Comparison

| Method | When to Use |
|--------|-------------|
| **Macro** | All classes equally important |
| **Micro** | Overall performance, large datasets |
| **Weighted** | Account for class imbalance |

### PyTorch Implementation

```python
import torch
from typing import Literal, Dict

def multiclass_precision_recall_f1(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    num_classes: int,
    average: Literal['macro', 'micro', 'weighted'] = 'macro'
) -> Dict[str, float]:
    """
    Calculate precision, recall, F1 for multi-class classification.
    
    Args:
        y_true: Ground truth labels (0 to num_classes-1)
        y_pred: Predicted labels
        num_classes: Number of classes
        average: Averaging strategy
        
    Returns:
        Dictionary with precision, recall, f1
    """
    y_true = y_true.long()
    y_pred = y_pred.long()
    
    # Per-class counts
    tp_per_class = torch.zeros(num_classes)
    fp_per_class = torch.zeros(num_classes)
    fn_per_class = torch.zeros(num_classes)
    support = torch.zeros(num_classes)
    
    for k in range(num_classes):
        tp_per_class[k] = ((y_true == k) & (y_pred == k)).sum()
        fp_per_class[k] = ((y_true != k) & (y_pred == k)).sum()
        fn_per_class[k] = ((y_true == k) & (y_pred != k)).sum()
        support[k] = (y_true == k).sum()
    
    if average == 'macro':
        prec_per_class = tp_per_class / (tp_per_class + fp_per_class + 1e-10)
        rec_per_class = tp_per_class / (tp_per_class + fn_per_class + 1e-10)
        f1_per_class = 2 * prec_per_class * rec_per_class / (prec_per_class + rec_per_class + 1e-10)
        
        return {
            'precision': prec_per_class.mean().item(),
            'recall': rec_per_class.mean().item(),
            'f1': f1_per_class.mean().item()
        }
    
    elif average == 'micro':
        total_tp = tp_per_class.sum()
        total_fp = fp_per_class.sum()
        total_fn = fn_per_class.sum()
        
        prec = total_tp / (total_tp + total_fp + 1e-10)
        rec = total_tp / (total_tp + total_fn + 1e-10)
        f1 = 2 * prec * rec / (prec + rec + 1e-10)
        
        return {
            'precision': prec.item(),
            'recall': rec.item(),
            'f1': f1.item()
        }
    
    elif average == 'weighted':
        prec_per_class = tp_per_class / (tp_per_class + fp_per_class + 1e-10)
        rec_per_class = tp_per_class / (tp_per_class + fn_per_class + 1e-10)
        f1_per_class = 2 * prec_per_class * rec_per_class / (prec_per_class + rec_per_class + 1e-10)
        
        total = support.sum()
        weights = support / total
        
        return {
            'precision': (weights * prec_per_class).sum().item(),
            'recall': (weights * rec_per_class).sum().item(),
            'f1': (weights * f1_per_class).sum().item()
        }


# Example: 3-class classification
y_true = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
y_pred = torch.tensor([0, 1, 2, 0, 2, 2, 0, 1, 1, 0, 1, 2])

for avg in ['macro', 'micro', 'weighted']:
    result = multiclass_precision_recall_f1(y_true, y_pred, 3, avg)
    print(f"{avg.capitalize():8} - P: {result['precision']:.3f}, R: {result['recall']:.3f}, F1: {result['f1']:.3f}")
```

---

## Comprehensive Metrics Class

```python
import torch
from dataclasses import dataclass
from typing import Optional, Dict

@dataclass
class PRF1Report:
    """Container for precision-recall-F1 metrics."""
    precision: float
    recall: float
    f1: float
    f05: float  # F0.5 (precision-weighted)
    f2: float   # F2 (recall-weighted)
    support_positive: int
    support_negative: int
    
    def __str__(self) -> str:
        return (
            f"Precision: {self.precision:.4f}\n"
            f"Recall:    {self.recall:.4f}\n"
            f"F1 Score:  {self.f1:.4f}\n"
            f"F0.5:      {self.f05:.4f} (precision-weighted)\n"
            f"F2:        {self.f2:.4f} (recall-weighted)\n"
            f"Positive samples: {self.support_positive}\n"
            f"Negative samples: {self.support_negative}"
        )


class PrecisionRecallF1:
    """
    Complete precision-recall-F1 calculator.
    
    Example:
        >>> y_true = torch.tensor([0, 1, 1, 0, 1])
        >>> y_pred = torch.tensor([0, 1, 0, 0, 1])
        >>> metrics = PrecisionRecallF1(y_true, y_pred)
        >>> print(metrics.report())
    """
    
    def __init__(self, y_true: torch.Tensor, y_pred: torch.Tensor,
                 y_proba: Optional[torch.Tensor] = None):
        self.y_true = y_true.long()
        self.y_pred = y_pred.long()
        self.y_proba = y_proba
        
        # Compute counts
        self.tp = ((self.y_true == 1) & (self.y_pred == 1)).sum().item()
        self.tn = ((self.y_true == 0) & (self.y_pred == 0)).sum().item()
        self.fp = ((self.y_true == 0) & (self.y_pred == 1)).sum().item()
        self.fn = ((self.y_true == 1) & (self.y_pred == 0)).sum().item()
    
    def precision(self) -> float:
        """TP / (TP + FP)"""
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0
    
    def recall(self) -> float:
        """TP / (TP + FN)"""
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0
    
    def f1(self) -> float:
        """Harmonic mean of precision and recall"""
        p, r = self.precision(), self.recall()
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    
    def f_beta(self, beta: float) -> float:
        """F-beta score"""
        p, r = self.precision(), self.recall()
        if p + r == 0:
            return 0.0
        beta_sq = beta ** 2
        return (1 + beta_sq) * p * r / (beta_sq * p + r)
    
    def report(self) -> PRF1Report:
        """Generate complete report"""
        return PRF1Report(
            precision=self.precision(),
            recall=self.recall(),
            f1=self.f1(),
            f05=self.f_beta(0.5),
            f2=self.f_beta(2.0),
            support_positive=self.tp + self.fn,
            support_negative=self.tn + self.fp
        )
    
    def precision_recall_curve(self) -> Optional[Dict]:
        """Compute PR curve if probabilities available"""
        if self.y_proba is None:
            return None
        
        thresholds = torch.linspace(0, 1, 101)
        precisions, recalls = [], []
        
        for thresh in thresholds:
            y_pred_t = (self.y_proba >= thresh).long()
            tp = ((self.y_true == 1) & (y_pred_t == 1)).sum().item()
            fp = ((self.y_true == 0) & (y_pred_t == 1)).sum().item()
            fn = ((self.y_true == 1) & (y_pred_t == 0)).sum().item()
            
            prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            precisions.append(prec)
            recalls.append(rec)
        
        return {
            'thresholds': thresholds.tolist(),
            'precision': precisions,
            'recall': recalls
        }


# Example usage
if __name__ == "__main__":
    y_true = torch.tensor([0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0])
    y_pred = torch.tensor([0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0])
    
    metrics = PrecisionRecallF1(y_true, y_pred)
    report = metrics.report()
    
    print("=" * 40)
    print("PRECISION-RECALL-F1 REPORT")
    print("=" * 40)
    print(report)
```

---

## Metric Selection Guide

### Decision Framework

```
┌──────────────────────────────────────────────────────────────┐
│              WHICH METRIC TO PRIORITIZE?                      │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  Question                          →  Metric                  │
│  ─────────────────────────────────────────────────────────    │
│  Are false positives costly?       →  Precision, F0.5         │
│  Are false negatives costly?       →  Recall, F2              │
│  Need balance?                     →  F1                      │
│  Classes imbalanced?               →  F1, Weighted F1         │
│  All classes equally important?    →  Macro F1                │
│  Performance on majority matters?  →  Micro F1                │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### Industry-Specific Guidance

| Domain | Priority | Reason |
|--------|----------|--------|
| **Medical Screening** | Recall, F2 | Don't miss diseases |
| **Fraud Detection** | Recall | Catch all fraud |
| **Spam Filtering** | Precision, F0.5 | Don't block legitimate |
| **Search Ranking** | Precision | Show relevant results |
| **Credit Risk** | F1, Recall | Balance + catch defaults |
| **Trading Signals** | Precision | Quality over quantity |

---

## Summary

| Metric | Formula | Range | Best For |
|--------|---------|-------|----------|
| Precision | $\frac{TP}{TP+FP}$ | [0,1] | FP costly |
| Recall | $\frac{TP}{TP+FN}$ | [0,1] | FN costly |
| F1 | $\frac{2PR}{P+R}$ | [0,1] | Balance |
| F0.5 | Precision-weighted F | [0,1] | More precision |
| F2 | Recall-weighted F | [0,1] | More recall |

**Key Relationships:**
- Precision ↔ Recall: Inverse trade-off
- F1: Harmonic mean (requires both high)
- F-beta: Tunable balance

---

## References

1. Van Rijsbergen, C.J. (1979). *Information Retrieval*. Butterworths.
2. Powers, D.M.W. (2011). "Evaluation: From Precision, Recall and F-Measure to ROC, Informedness, Markedness & Correlation."
3. Sokolova, M. & Lapalme, G. (2009). "A systematic analysis of performance measures for classification tasks."
