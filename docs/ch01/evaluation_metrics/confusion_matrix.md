# Confusion Matrix: Deep Dive

## Overview

The confusion matrix is the foundational structure for understanding classification performance. It provides a complete picture of how predictions relate to actual classes, revealing not just overall accuracy but the specific types of errors a model makes.

---

## Binary Classification Confusion Matrix

### Structure

For binary classification with classes **Negative (0)** and **Positive (1)**:

$$
\begin{array}{c|cc}
& \text{Predicted Negative} & \text{Predicted Positive} \\
\hline
\text{Actual Negative} & \text{TN} & \text{FP} \\
\text{Actual Positive} & \text{FN} & \text{TP} \\
\end{array}
$$

### Terminology

| Term | Abbreviation | Meaning | Error Type |
|------|--------------|---------|------------|
| **True Positive** | TP | Correctly predicted positive | — |
| **True Negative** | TN | Correctly predicted negative | — |
| **False Positive** | FP | Incorrectly predicted positive | Type I Error |
| **False Negative** | FN | Incorrectly predicted negative | Type II Error |

### Visual Memory Aid

```
                        PREDICTION
                    Negative  Positive
                   ┌─────────┬─────────┐
         Negative  │   TN    │   FP    │  ← False Alarm (Type I)
ACTUAL             ├─────────┼─────────┤
         Positive  │   FN    │   TP    │  ← Miss (Type II)
                   └─────────┴─────────┘
```

---

## PyTorch Implementation

### Basic Confusion Matrix

```python
import torch
from typing import Tuple, Dict

def compute_confusion_matrix(y_true: torch.Tensor, 
                            y_pred: torch.Tensor) -> torch.Tensor:
    """
    Compute confusion matrix for binary classification.
    
    Args:
        y_true: Ground truth labels (0 or 1), shape (n_samples,)
        y_pred: Predicted labels (0 or 1), shape (n_samples,)
        
    Returns:
        2x2 confusion matrix tensor [[TN, FP], [FN, TP]]
    """
    y_true = y_true.long()
    y_pred = y_pred.long()
    
    tn = ((y_true == 0) & (y_pred == 0)).sum().item()
    fp = ((y_true == 0) & (y_pred == 1)).sum().item()
    fn = ((y_true == 1) & (y_pred == 0)).sum().item()
    tp = ((y_true == 1) & (y_pred == 1)).sum().item()
    
    return torch.tensor([[tn, fp], [fn, tp]])


def confusion_counts(y_true: torch.Tensor, 
                    y_pred: torch.Tensor) -> Dict[str, int]:
    """Return confusion matrix as dictionary."""
    cm = compute_confusion_matrix(y_true, y_pred)
    return {
        'TN': cm[0, 0].item(),
        'FP': cm[0, 1].item(),
        'FN': cm[1, 0].item(),
        'TP': cm[1, 1].item()
    }


# Example
y_true = torch.tensor([1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0])
y_pred = torch.tensor([1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0])

cm = compute_confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

counts = confusion_counts(y_true, y_pred)
print(f"\nTP: {counts['TP']}, TN: {counts['TN']}")
print(f"FP: {counts['FP']}, FN: {counts['FN']}")
```

### Multi-Class Confusion Matrix

```python
import torch

def compute_multiclass_cm(y_true: torch.Tensor, 
                         y_pred: torch.Tensor,
                         num_classes: int) -> torch.Tensor:
    """
    Compute confusion matrix for multi-class classification.
    
    Args:
        y_true: Ground truth labels (0 to num_classes-1)
        y_pred: Predicted labels
        num_classes: Number of classes
        
    Returns:
        num_classes x num_classes confusion matrix
        Entry [i, j] = samples with true class i predicted as class j
    """
    y_true = y_true.long()
    y_pred = y_pred.long()
    
    # Vectorized using bincount
    indices = y_true * num_classes + y_pred
    cm = torch.bincount(indices, minlength=num_classes**2)
    return cm.reshape(num_classes, num_classes)


# Example: 3-class classification
y_true = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
y_pred = torch.tensor([0, 1, 2, 0, 2, 2, 0, 1, 1, 0, 1, 2])

cm = compute_multiclass_cm(y_true, y_pred, num_classes=3)
print("Multi-class Confusion Matrix:")
print(cm)
```

---

## Normalization

### Why Normalize?

Raw counts can be misleading when:
- Classes have very different sizes
- Comparing models on different datasets
- Want to understand error patterns as proportions

### Normalization Types

| Type | Formula | Shows |
|------|---------|-------|
| **By True (rows)** | $\frac{CM[i,j]}{\sum_k CM[i,k]}$ | Recall per class |
| **By Pred (columns)** | $\frac{CM[i,j]}{\sum_k CM[k,j]}$ | Precision per class |
| **Global** | $\frac{CM[i,j]}{\sum_{k,l} CM[k,l]}$ | Overall distribution |

### PyTorch Implementation

```python
import torch
from typing import Literal

def normalize_cm(cm: torch.Tensor,
                normalize: Literal['true', 'pred', 'all'] = 'true') -> torch.Tensor:
    """
    Normalize confusion matrix.
    
    Args:
        cm: Confusion matrix tensor
        normalize: 'true' (rows), 'pred' (columns), or 'all'
            
    Returns:
        Normalized confusion matrix
    """
    cm_float = cm.float()
    
    if normalize == 'true':
        row_sums = cm_float.sum(dim=1, keepdim=True)
        row_sums[row_sums == 0] = 1
        return cm_float / row_sums
    
    elif normalize == 'pred':
        col_sums = cm_float.sum(dim=0, keepdim=True)
        col_sums[col_sums == 0] = 1
        return cm_float / col_sums
    
    elif normalize == 'all':
        total = cm_float.sum()
        return cm_float / total if total > 0 else cm_float
    
    else:
        raise ValueError(f"Unknown normalize: {normalize}")


# Example
cm = torch.tensor([[45, 5], [8, 42]])

print("Raw counts:")
print(cm)

print("\nNormalized by true label (shows recall):")
print(normalize_cm(cm, 'true'))

print("\nNormalized by prediction (shows precision):")
print(normalize_cm(cm, 'pred'))
```

---

## Deriving Metrics from Confusion Matrix

### Binary Classification Metrics

```python
import torch
import math
from typing import Dict

def derive_metrics(cm: torch.Tensor) -> Dict[str, float]:
    """
    Derive all classification metrics from 2x2 confusion matrix.
    
    Args:
        cm: 2x2 confusion matrix [[TN, FP], [FN, TP]]
        
    Returns:
        Dictionary of all derived metrics
    """
    tn, fp = cm[0, 0].item(), cm[0, 1].item()
    fn, tp = cm[1, 0].item(), cm[1, 1].item()
    n = tn + fp + fn + tp
    
    # Derived counts
    actual_pos = tp + fn
    actual_neg = tn + fp
    pred_pos = tp + fp
    pred_neg = tn + fn
    
    # Core metrics
    accuracy = (tp + tn) / n if n > 0 else 0
    precision = tp / pred_pos if pred_pos > 0 else 0
    recall = tp / actual_pos if actual_pos > 0 else 0
    specificity = tn / actual_neg if actual_neg > 0 else 0
    npv = tn / pred_neg if pred_neg > 0 else 0
    
    # Error rates
    fpr = fp / actual_neg if actual_neg > 0 else 0
    fnr = fn / actual_pos if actual_pos > 0 else 0
    
    # Composite metrics
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # MCC
    mcc_num = (tp * tn) - (fp * fn)
    mcc_denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = mcc_num / mcc_denom if mcc_denom > 0 else 0
    
    # Balanced accuracy
    balanced_acc = (recall + specificity) / 2
    
    return {
        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall (Sensitivity)': recall,
        'Specificity': specificity,
        'NPV': npv,
        'FPR': fpr,
        'FNR': fnr,
        'F1 Score': f1,
        'MCC': mcc,
        'Balanced Accuracy': balanced_acc
    }


# Example
cm = torch.tensor([[45, 5], [8, 42]])
metrics = derive_metrics(cm)

print("METRICS FROM CONFUSION MATRIX")
print("=" * 40)
for name, value in metrics.items():
    if isinstance(value, float):
        print(f"{name}: {value:.4f}")
    else:
        print(f"{name}: {value}")
```

### Multi-Class Per-Class Metrics

```python
import torch
from typing import Dict

def multiclass_metrics(cm: torch.Tensor) -> Dict[int, Dict[str, float]]:
    """
    Derive per-class metrics from multi-class confusion matrix.
    """
    num_classes = cm.shape[0]
    total = cm.sum().item()
    
    per_class = {}
    
    for k in range(num_classes):
        tp = cm[k, k].item()
        fp = cm[:, k].sum().item() - tp
        fn = cm[k, :].sum().item() - tp
        tn = total - tp - fp - fn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        per_class[k] = {
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'Support': cm[k, :].sum().item()
        }
    
    return per_class


# Aggregate metrics
def aggregate_metrics(cm: torch.Tensor) -> Dict[str, float]:
    """Compute macro, micro, weighted averages."""
    per_class = multiclass_metrics(cm)
    num_classes = len(per_class)
    total = cm.sum().item()
    
    # Macro average
    macro_p = sum(c['Precision'] for c in per_class.values()) / num_classes
    macro_r = sum(c['Recall'] for c in per_class.values()) / num_classes
    macro_f1 = sum(c['F1'] for c in per_class.values()) / num_classes
    
    # Weighted average
    total_support = sum(c['Support'] for c in per_class.values())
    weighted_f1 = sum(c['F1'] * c['Support'] for c in per_class.values()) / total_support
    
    # Accuracy
    accuracy = sum(cm[k, k].item() for k in range(num_classes)) / total
    
    return {
        'Accuracy': accuracy,
        'Macro F1': macro_f1,
        'Weighted F1': weighted_f1
    }
```

---

## Visualization

### Text-Based Visualization

```python
import torch
from typing import List, Optional

def print_cm(cm: torch.Tensor, 
            class_names: Optional[List[str]] = None,
            normalize: Optional[str] = None) -> None:
    """Print formatted confusion matrix."""
    num_classes = cm.shape[0]
    
    if class_names is None:
        class_names = [f"C{i}" for i in range(num_classes)]
    
    if normalize:
        cm_display = normalize_cm(cm, normalize)
        fmt = ".2f"
    else:
        cm_display = cm.float()
        fmt = ".0f"
    
    # Header
    header = "       " + " ".join(f"{name:>8}" for name in class_names)
    print(header)
    print("-" * len(header))
    
    # Rows
    for i, name in enumerate(class_names):
        row_vals = " ".join(f"{cm_display[i,j].item():>8{fmt}}" for j in range(num_classes))
        print(f"{name:>6} {row_vals}")
    
    if normalize:
        print(f"\n(Normalized by {normalize})")


# Example
cm = torch.tensor([[45, 3, 2], [5, 42, 3], [4, 6, 40]])
print_cm(cm, ['Setosa', 'Versi.', 'Virgin.'])
print()
print_cm(cm, ['Setosa', 'Versi.', 'Virgin.'], normalize='true')
```

### Matplotlib Visualization

```python
import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional

def plot_cm(cm: torch.Tensor,
           class_names: Optional[List[str]] = None,
           normalize: Optional[str] = None,
           cmap: str = 'Blues',
           figsize: tuple = (8, 6)) -> plt.Figure:
    """
    Plot confusion matrix as heatmap.
    """
    num_classes = cm.shape[0]
    
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]
    
    if normalize:
        cm_plot = normalize_cm(cm, normalize).numpy()
        fmt = '.2f'
    else:
        cm_plot = cm.numpy()
        fmt = 'd'
    
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_plot, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(num_classes),
           yticks=np.arange(num_classes),
           xticklabels=class_names,
           yticklabels=class_names,
           ylabel='True Label',
           xlabel='Predicted Label')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Add annotations
    thresh = cm_plot.max() / 2.
    for i in range(num_classes):
        for j in range(num_classes):
            val = cm_plot[i, j]
            text = f'{val:{fmt}}' if normalize else f'{int(val)}'
            ax.text(j, i, text, ha='center', va='center',
                   color='white' if val > thresh else 'black')
    
    fig.tight_layout()
    return fig
```

---

## Interpretation Guide

### Reading the Matrix

**Diagonal Elements (cm[i,i])**: Correct predictions
- High values = good performance
- Should be the largest values in each row/column

**Off-Diagonal Elements (cm[i,j], i≠j)**: Errors
- Class i misclassified as class j
- Patterns reveal systematic confusions

### Key Questions to Ask

1. **Is the diagonal dominant?** → Good overall accuracy
2. **Which classes have low recall?** → Check row sums vs diagonal
3. **Which classes have low precision?** → Check column sums vs diagonal
4. **Are there systematic confusions?** → Look for consistent off-diagonal patterns
5. **Is there class imbalance?** → Compare row sums

### Error Analysis

```python
def find_confusions(cm: torch.Tensor, 
                   class_names: List[str],
                   top_k: int = 5) -> List:
    """Find top confusion pairs."""
    num_classes = cm.shape[0]
    confusions = []
    
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i, j] > 0:
                confusions.append({
                    'true': class_names[i],
                    'pred': class_names[j],
                    'count': cm[i, j].item()
                })
    
    confusions.sort(key=lambda x: x['count'], reverse=True)
    return confusions[:top_k]


# Example
cm = torch.tensor([[45, 3, 2], [5, 42, 3], [4, 6, 40]])
class_names = ['Setosa', 'Versicolor', 'Virginica']

print("Top Confusion Pairs:")
for c in find_confusions(cm, class_names):
    print(f"  {c['true']} → {c['pred']}: {c['count']}")
```

---

## Comprehensive Analyzer Class

```python
import torch
from typing import Dict, List, Optional

class ConfusionMatrixAnalyzer:
    """
    Comprehensive confusion matrix analysis.
    
    Example:
        >>> analyzer = ConfusionMatrixAnalyzer(y_true, y_pred, ['A', 'B', 'C'])
        >>> print(analyzer.summary())
        >>> analyzer.plot()
    """
    
    def __init__(self, y_true: torch.Tensor, y_pred: torch.Tensor,
                 class_names: Optional[List[str]] = None):
        self.y_true = y_true.long()
        self.y_pred = y_pred.long()
        self.num_classes = max(y_true.max(), y_pred.max()).item() + 1
        
        self.class_names = class_names or [f"Class {i}" for i in range(self.num_classes)]
        self.cm = self._compute_cm()
    
    def _compute_cm(self) -> torch.Tensor:
        indices = self.y_true * self.num_classes + self.y_pred
        cm = torch.bincount(indices, minlength=self.num_classes**2)
        return cm.reshape(self.num_classes, self.num_classes)
    
    def normalize(self, by: str = 'true') -> torch.Tensor:
        return normalize_cm(self.cm, by)
    
    def per_class(self) -> Dict[str, Dict[str, float]]:
        metrics = multiclass_metrics(self.cm)
        return {self.class_names[k]: v for k, v in metrics.items()}
    
    def summary(self) -> str:
        agg = aggregate_metrics(self.cm)
        per = self.per_class()
        
        lines = [
            "=" * 50,
            "CONFUSION MATRIX ANALYSIS",
            "=" * 50,
            f"\nOverall Accuracy: {agg['Accuracy']:.2%}",
            f"Macro F1: {agg['Macro F1']:.4f}",
            f"Weighted F1: {agg['Weighted F1']:.4f}",
            "\nPer-Class Performance:"
        ]
        
        for name, m in per.items():
            lines.append(f"  {name}: P={m['Precision']:.3f}, R={m['Recall']:.3f}, F1={m['F1']:.3f}")
        
        lines.append("=" * 50)
        return "\n".join(lines)
    
    def plot(self, normalize: Optional[str] = None, **kwargs):
        return plot_cm(self.cm, self.class_names, normalize, **kwargs)


# Example usage
if __name__ == "__main__":
    y_true = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
    y_pred = torch.tensor([0, 1, 2, 0, 2, 2, 0, 1, 1, 0, 1, 2])
    
    analyzer = ConfusionMatrixAnalyzer(y_true, y_pred, ['A', 'B', 'C'])
    print(analyzer.summary())
```

---

## Best Practices

### Do's
- ✅ Show both raw counts and normalized versions
- ✅ Include class names on axes
- ✅ Annotate cells with values
- ✅ Compare to random baseline
- ✅ Consider business impact of different errors

### Don'ts
- ❌ Rely only on raw counts with imbalanced classes
- ❌ Ignore off-diagonal patterns
- ❌ Use wrong normalization for your question
- ❌ Skip visualization for complex matrices

---

## Summary

| Component | Meaning |
|-----------|---------|
| Diagonal | Correct predictions (larger = better) |
| Row sums | True class support |
| Column sums | Prediction distribution |
| Off-diagonal | Specific error patterns |
| Row-normalized | Recall per class |
| Column-normalized | Precision per class |

The confusion matrix is the **foundation** for understanding classification performance—always start here before computing aggregate metrics.
