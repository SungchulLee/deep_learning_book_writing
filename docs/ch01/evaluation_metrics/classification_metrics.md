# Classification Metrics: A Comprehensive Guide

## Overview

Classification metrics evaluate how well a model assigns discrete class labels to inputs. Unlike regression where we measure error magnitude, classification requires us to count correct and incorrect assignments across categories.

This section provides a rigorous treatment of classification metrics, from basic accuracy to sophisticated measures designed for imbalanced datasets and multi-class problems.

---

## The Classification Problem

### Binary Classification Setup

For binary classification with classes $\{0, 1\}$ (or $\{-, +\}$):

- **True Labels**: $y \in \{0, 1\}^n$
- **Predictions**: $\hat{y} \in \{0, 1\}^n$
- **Probability Scores**: $\hat{p} \in [0, 1]^n$ (optional)

### Fundamental Counts

All binary classification metrics derive from four fundamental counts:

| Prediction \ Actual | Negative (0) | Positive (1) |
|---------------------|--------------|--------------|
| **Negative (0)** | True Negative (TN) | False Negative (FN) |
| **Positive (1)** | False Positive (FP) | True Positive (TP) |

**Definitions:**
- **TP**: Correctly predicted positives
- **TN**: Correctly predicted negatives  
- **FP**: Incorrectly predicted positives (Type I Error)
- **FN**: Incorrectly predicted negatives (Type II Error)

**Constraint**: $\text{TP} + \text{TN} + \text{FP} + \text{FN} = n$

---

## Accuracy

### Definition

Accuracy measures the proportion of correct predictions:

$$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}} = \frac{\text{Correct Predictions}}{n}$$

### Properties

| Property | Value |
|----------|-------|
| **Range** | $[0, 1]$ |
| **Perfect Score** | 1.0 |
| **Random Baseline** | Class prior probability |

### PyTorch Implementation

```python
import torch

def accuracy(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Calculate classification accuracy.
    
    Args:
        y_true: Ground truth labels, shape (n_samples,)
        y_pred: Predicted labels, shape (n_samples,)
        
    Returns:
        Accuracy as a float in [0, 1]
    """
    return (y_true == y_pred).float().mean().item()

# Example
y_true = torch.tensor([0, 1, 1, 0, 1, 1, 0, 0, 1, 0])
y_pred = torch.tensor([0, 1, 1, 0, 0, 1, 0, 1, 1, 0])

acc = accuracy(y_true, y_pred)
print(f"Accuracy: {acc:.2%}")  # Accuracy: 80.00%
```

### The Accuracy Paradox

**Critical Warning**: Accuracy is misleading for imbalanced datasets.

**Example**: Fraud detection with 99% legitimate transactions:
- A model predicting "legitimate" for everything achieves 99% accuracy
- But catches 0% of fraud (useless for the actual task)

```python
# Imbalanced dataset example
y_true = torch.tensor([0] * 990 + [1] * 10)  # 1% positive class
y_pred = torch.tensor([0] * 1000)  # Predict all negative

acc = accuracy(y_true, y_pred)
print(f"Accuracy: {acc:.2%}")  # Accuracy: 99.00% (but misses ALL fraud!)
```

---

## Error Rates

### Misclassification Rate

$$\text{Error Rate} = 1 - \text{Accuracy} = \frac{\text{FP} + \text{FN}}{n}$$

### Specific Error Types

**False Positive Rate (FPR)** - Type I Error Rate:
$$\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}} = \frac{\text{FP}}{\text{Actual Negatives}}$$

**False Negative Rate (FNR)** - Type II Error Rate:
$$\text{FNR} = \frac{\text{FN}}{\text{FN} + \text{TP}} = \frac{\text{FN}}{\text{Actual Positives}}$$

---

## Precision, Recall, and Specificity

### Precision (Positive Predictive Value)

**Question answered**: Of all positive predictions, how many were correct?

$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$

**Use when**: False positives are costly
- Spam detection (blocking legitimate email is bad)
- Drug approval (approving ineffective drugs is costly)

### Recall (Sensitivity, True Positive Rate)

**Question answered**: Of all actual positives, how many did we catch?

$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

**Use when**: False negatives are costly
- Disease screening (missing a disease is dangerous)
- Fraud detection (missing fraud is expensive)

### Specificity (True Negative Rate)

**Question answered**: Of all actual negatives, how many did we correctly identify?

$$\text{Specificity} = \frac{\text{TN}}{\text{TN} + \text{FP}} = 1 - \text{FPR}$$

### PyTorch Implementation

```python
import torch
from typing import Tuple

def confusion_counts(y_true: torch.Tensor, 
                     y_pred: torch.Tensor) -> Tuple[int, int, int, int]:
    """
    Calculate TP, TN, FP, FN counts.
    
    Returns:
        Tuple of (TP, TN, FP, FN)
    """
    tp = ((y_true == 1) & (y_pred == 1)).sum().item()
    tn = ((y_true == 0) & (y_pred == 0)).sum().item()
    fp = ((y_true == 0) & (y_pred == 1)).sum().item()
    fn = ((y_true == 1) & (y_pred == 0)).sum().item()
    return tp, tn, fp, fn


def precision(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Calculate precision: TP / (TP + FP)."""
    tp, tn, fp, fn = confusion_counts(y_true, y_pred)
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Calculate recall: TP / (TP + FN)."""
    tp, tn, fp, fn = confusion_counts(y_true, y_pred)
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def specificity(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Calculate specificity: TN / (TN + FP)."""
    tp, tn, fp, fn = confusion_counts(y_true, y_pred)
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


# Example
y_true = torch.tensor([0, 1, 1, 0, 1, 1, 0, 0, 1, 0])
y_pred = torch.tensor([0, 1, 1, 0, 0, 1, 0, 1, 1, 0])

print(f"Precision: {precision(y_true, y_pred):.4f}")  # 0.8000
print(f"Recall: {recall(y_true, y_pred):.4f}")        # 0.8000
print(f"Specificity: {specificity(y_true, y_pred):.4f}")  # 0.8000
```

### The Precision-Recall Trade-off

Precision and recall typically have an inverse relationship:

- **Increasing threshold** (more selective): ↑ Precision, ↓ Recall
- **Decreasing threshold** (more inclusive): ↓ Precision, ↑ Recall

```
High Threshold (conservative):
  - Fewer positive predictions
  - Higher precision (predictions are reliable)
  - Lower recall (misses many positives)

Low Threshold (aggressive):
  - More positive predictions
  - Lower precision (many false positives)
  - Higher recall (catches most positives)
```

---

## F1 Score and F-beta

### F1 Score

The harmonic mean of precision and recall:

$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \cdot \text{TP}}{2 \cdot \text{TP} + \text{FP} + \text{FN}}$$

### Why Harmonic Mean?

The harmonic mean penalizes extreme values more than arithmetic mean:

$$\text{Arithmetic}: \frac{0.9 + 0.1}{2} = 0.5$$
$$\text{Harmonic}: \frac{2 \cdot 0.9 \cdot 0.1}{0.9 + 0.1} = 0.18$$

This ensures both precision and recall must be reasonably high for a good F1.

### F-beta Score

Generalization allowing weighted importance:

$$F_\beta = (1 + \beta^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{\beta^2 \cdot \text{Precision} + \text{Recall}}$$

| β Value | Interpretation |
|---------|----------------|
| β = 0.5 | Precision 2× more important than recall |
| β = 1 | Equal weight (standard F1) |
| β = 2 | Recall 2× more important than precision |

### PyTorch Implementation

```python
import torch

def f1_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Calculate F1 Score."""
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    
    if prec + rec == 0:
        return 0.0
    return 2 * (prec * rec) / (prec + rec)


def f_beta_score(y_true: torch.Tensor, y_pred: torch.Tensor, 
                 beta: float = 1.0) -> float:
    """
    Calculate F-beta Score.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        beta: Weight parameter (beta > 1 favors recall)
        
    Returns:
        F-beta score
    """
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    
    if prec + rec == 0:
        return 0.0
    
    beta_sq = beta ** 2
    return (1 + beta_sq) * (prec * rec) / (beta_sq * prec + rec)


# Example
y_true = torch.tensor([0, 1, 1, 0, 1, 1, 0, 0, 1, 0])
y_pred = torch.tensor([0, 1, 1, 0, 0, 1, 0, 1, 1, 0])

print(f"F1 Score: {f1_score(y_true, y_pred):.4f}")        # 0.8000
print(f"F0.5 Score: {f_beta_score(y_true, y_pred, 0.5):.4f}")  # Precision-weighted
print(f"F2 Score: {f_beta_score(y_true, y_pred, 2.0):.4f}")    # Recall-weighted
```

---

## Matthews Correlation Coefficient (MCC)

### Definition

MCC is a balanced measure that considers all four confusion matrix quadrants:

$$\text{MCC} = \frac{\text{TP} \cdot \text{TN} - \text{FP} \cdot \text{FN}}{\sqrt{(\text{TP}+\text{FP})(\text{TP}+\text{FN})(\text{TN}+\text{FP})(\text{TN}+\text{FN})}}$$

### Properties

| Property | Value |
|----------|-------|
| **Range** | $[-1, +1]$ |
| **Perfect** | +1 |
| **Random** | 0 |
| **Inverse** | -1 |

### Why Use MCC?

- **Balanced**: Works well with imbalanced classes
- **Symmetric**: Treats both classes equally
- **Informative**: A high MCC requires good performance on all four quadrants

### PyTorch Implementation

```python
import torch
import math

def matthews_correlation_coefficient(y_true: torch.Tensor, 
                                     y_pred: torch.Tensor) -> float:
    """
    Calculate Matthews Correlation Coefficient.
    
    MCC is particularly useful for imbalanced datasets where
    accuracy can be misleading.
    
    Returns:
        MCC in range [-1, +1]
    """
    tp, tn, fp, fn = confusion_counts(y_true, y_pred)
    
    numerator = (tp * tn) - (fp * fn)
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator


# Example
y_true = torch.tensor([0, 1, 1, 0, 1, 1, 0, 0, 1, 0])
y_pred = torch.tensor([0, 1, 1, 0, 0, 1, 0, 1, 1, 0])

mcc = matthews_correlation_coefficient(y_true, y_pred)
print(f"MCC: {mcc:.4f}")  # MCC: 0.6000
```

---

## Cohen's Kappa

### Definition

Cohen's Kappa measures agreement beyond chance:

$$\kappa = \frac{p_o - p_e}{1 - p_e}$$

where:
- $p_o$ = observed agreement (accuracy)
- $p_e$ = expected agreement by chance

### Computing Expected Agreement

$$p_e = \frac{(\text{TP}+\text{FP})(\text{TP}+\text{FN}) + (\text{TN}+\text{FN})(\text{TN}+\text{FP})}{n^2}$$

### Interpretation

| κ Value | Interpretation |
|---------|----------------|
| < 0 | Less than chance |
| 0.01 - 0.20 | Slight agreement |
| 0.21 - 0.40 | Fair agreement |
| 0.41 - 0.60 | Moderate agreement |
| 0.61 - 0.80 | Substantial agreement |
| 0.81 - 1.00 | Almost perfect |

### PyTorch Implementation

```python
import torch

def cohens_kappa(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Calculate Cohen's Kappa coefficient.
    
    Measures agreement accounting for chance agreement.
    """
    tp, tn, fp, fn = confusion_counts(y_true, y_pred)
    n = tp + tn + fp + fn
    
    # Observed agreement
    p_o = (tp + tn) / n
    
    # Expected agreement by chance
    p_yes = ((tp + fp) / n) * ((tp + fn) / n)
    p_no = ((tn + fn) / n) * ((tn + fp) / n)
    p_e = p_yes + p_no
    
    if p_e == 1:
        return 1.0  # Perfect agreement
    
    return (p_o - p_e) / (1 - p_e)


# Example
kappa = cohens_kappa(y_true, y_pred)
print(f"Cohen's Kappa: {kappa:.4f}")
```

---

## Log Loss (Cross-Entropy Loss)

### Definition

Log loss measures the quality of probability predictions:

$$\mathcal{L} = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\hat{p}_i) + (1-y_i) \log(1-\hat{p}_i) \right]$$

### Properties

| Property | Value |
|----------|-------|
| **Range** | $[0, \infty)$ |
| **Perfect** | 0 |
| **Interpretation** | Negative log-likelihood per sample |

### Why Log Loss?

- Evaluates **probability calibration**, not just correctness
- Heavily penalizes confident wrong predictions
- Directly related to training objective in logistic regression

### PyTorch Implementation

```python
import torch
import torch.nn.functional as F

def log_loss(y_true: torch.Tensor, y_pred_proba: torch.Tensor, 
             eps: float = 1e-15) -> float:
    """
    Calculate Log Loss (Binary Cross-Entropy).
    
    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred_proba: Predicted probabilities in [0, 1]
        eps: Small value to avoid log(0)
        
    Returns:
        Log loss value
    """
    # Clip predictions to avoid log(0)
    y_pred_proba = torch.clamp(y_pred_proba, eps, 1 - eps)
    
    loss = -(y_true * torch.log(y_pred_proba) + 
             (1 - y_true) * torch.log(1 - y_pred_proba))
    
    return loss.mean().item()


# Using PyTorch's built-in
y_true = torch.tensor([0., 1., 1., 0., 1.])
y_pred_proba = torch.tensor([0.1, 0.9, 0.8, 0.2, 0.7])

loss = log_loss(y_true, y_pred_proba)
print(f"Log Loss: {loss:.4f}")

# Using BCE
bce = F.binary_cross_entropy(y_pred_proba, y_true)
print(f"BCE (PyTorch): {bce.item():.4f}")
```

---

## Multi-Class Extensions

### Averaging Strategies

For $K$ classes, metrics can be aggregated in different ways:

**Macro Average**: Simple mean across classes
$$\text{Macro-Precision} = \frac{1}{K} \sum_{k=1}^{K} \text{Precision}_k$$

**Micro Average**: Global TP, FP, FN counts
$$\text{Micro-Precision} = \frac{\sum_k \text{TP}_k}{\sum_k (\text{TP}_k + \text{FP}_k)}$$

**Weighted Average**: Weighted by class support
$$\text{Weighted-Precision} = \sum_{k=1}^{K} w_k \cdot \text{Precision}_k$$

where $w_k = \frac{n_k}{n}$ (proportion of samples in class $k$)

### When to Use Each

| Strategy | Use Case |
|----------|----------|
| **Macro** | Equal importance for all classes |
| **Micro** | Large dataset, performance on majority matters |
| **Weighted** | Account for class imbalance |

### PyTorch Implementation

```python
import torch
from typing import Dict

def multiclass_metrics(y_true: torch.Tensor, y_pred: torch.Tensor, 
                       num_classes: int) -> Dict:
    """
    Calculate multi-class classification metrics.
    
    Args:
        y_true: Ground truth labels (integers 0 to num_classes-1)
        y_pred: Predicted labels
        num_classes: Number of classes
        
    Returns:
        Dictionary with per-class and aggregated metrics
    """
    per_class = {}
    
    for k in range(num_classes):
        # One-vs-rest binary classification
        y_true_k = (y_true == k).long()
        y_pred_k = (y_pred == k).long()
        
        tp = ((y_true_k == 1) & (y_pred_k == 1)).sum().item()
        fp = ((y_true_k == 0) & (y_pred_k == 1)).sum().item()
        fn = ((y_true_k == 1) & (y_pred_k == 0)).sum().item()
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        support = (y_true == k).sum().item()
        
        per_class[k] = {
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'support': support
        }
    
    # Macro average
    macro_precision = sum(c['precision'] for c in per_class.values()) / num_classes
    macro_recall = sum(c['recall'] for c in per_class.values()) / num_classes
    macro_f1 = sum(c['f1'] for c in per_class.values()) / num_classes
    
    # Weighted average
    total_support = sum(c['support'] for c in per_class.values())
    weighted_precision = sum(c['precision'] * c['support'] for c in per_class.values()) / total_support
    weighted_recall = sum(c['recall'] * c['support'] for c in per_class.values()) / total_support
    weighted_f1 = sum(c['f1'] * c['support'] for c in per_class.values()) / total_support
    
    return {
        'per_class': per_class,
        'macro': {'precision': macro_precision, 'recall': macro_recall, 'f1': macro_f1},
        'weighted': {'precision': weighted_precision, 'recall': weighted_recall, 'f1': weighted_f1},
        'accuracy': accuracy(y_true, y_pred)
    }


# Example: 3-class classification
y_true = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
y_pred = torch.tensor([0, 1, 2, 0, 2, 2, 0, 1, 1, 0, 1, 2])

results = multiclass_metrics(y_true, y_pred, num_classes=3)

print("Per-Class Metrics:")
for k, metrics in results['per_class'].items():
    print(f"  Class {k}: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")

print(f"\nMacro F1: {results['macro']['f1']:.4f}")
print(f"Weighted F1: {results['weighted']['f1']:.4f}")
print(f"Accuracy: {results['accuracy']:.4f}")
```

---

## Comprehensive Classification Metrics Class

```python
import torch
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

@dataclass
class BinaryClassificationReport:
    """Container for binary classification results."""
    accuracy: float
    precision: float
    recall: float
    specificity: float
    f1: float
    mcc: float
    kappa: float
    log_loss: Optional[float]
    confusion_matrix: Tuple[int, int, int, int]  # TP, TN, FP, FN
    
    def __str__(self) -> str:
        tp, tn, fp, fn = self.confusion_matrix
        lines = [
            "=" * 55,
            "BINARY CLASSIFICATION REPORT",
            "=" * 55,
            "",
            "Confusion Matrix:",
            f"                 Predicted",
            f"                 Neg    Pos",
            f"  Actual Neg     {tn:4d}   {fp:4d}",
            f"         Pos     {fn:4d}   {tp:4d}",
            "",
            "Metrics:",
            f"  Accuracy:      {self.accuracy:.4f}",
            f"  Precision:     {self.precision:.4f}",
            f"  Recall:        {self.recall:.4f}",
            f"  Specificity:   {self.specificity:.4f}",
            f"  F1 Score:      {self.f1:.4f}",
            f"  MCC:           {self.mcc:.4f}",
            f"  Cohen's Kappa: {self.kappa:.4f}",
        ]
        if self.log_loss is not None:
            lines.append(f"  Log Loss:      {self.log_loss:.4f}")
        lines.append("=" * 55)
        return "\n".join(lines)


class ClassificationMetrics:
    """
    Comprehensive binary classification metrics calculator.
    
    Example:
        >>> y_true = torch.tensor([0, 1, 1, 0, 1, 0])
        >>> y_pred = torch.tensor([0, 1, 0, 0, 1, 1])
        >>> y_proba = torch.tensor([0.2, 0.9, 0.4, 0.1, 0.8, 0.6])
        >>> metrics = ClassificationMetrics(y_true, y_pred, y_proba)
        >>> print(metrics.full_report())
    """
    
    def __init__(self, y_true: torch.Tensor, y_pred: torch.Tensor,
                 y_pred_proba: Optional[torch.Tensor] = None):
        """
        Initialize metrics calculator.
        
        Args:
            y_true: Ground truth labels (0 or 1)
            y_pred: Predicted labels (0 or 1)
            y_pred_proba: Predicted probabilities for positive class
        """
        self.y_true = y_true.long()
        self.y_pred = y_pred.long()
        self.y_pred_proba = y_pred_proba
        
        # Compute confusion matrix
        self._tp = ((self.y_true == 1) & (self.y_pred == 1)).sum().item()
        self._tn = ((self.y_true == 0) & (self.y_pred == 0)).sum().item()
        self._fp = ((self.y_true == 0) & (self.y_pred == 1)).sum().item()
        self._fn = ((self.y_true == 1) & (self.y_pred == 0)).sum().item()
    
    @property
    def confusion_counts(self) -> Tuple[int, int, int, int]:
        """Return (TP, TN, FP, FN)."""
        return (self._tp, self._tn, self._fp, self._fn)
    
    def accuracy(self) -> float:
        return (self._tp + self._tn) / (self._tp + self._tn + self._fp + self._fn)
    
    def precision(self) -> float:
        denom = self._tp + self._fp
        return self._tp / denom if denom > 0 else 0.0
    
    def recall(self) -> float:
        denom = self._tp + self._fn
        return self._tp / denom if denom > 0 else 0.0
    
    def specificity(self) -> float:
        denom = self._tn + self._fp
        return self._tn / denom if denom > 0 else 0.0
    
    def f1(self) -> float:
        p, r = self.precision(), self.recall()
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    
    def f_beta(self, beta: float) -> float:
        p, r = self.precision(), self.recall()
        if p + r == 0:
            return 0.0
        beta_sq = beta ** 2
        return (1 + beta_sq) * p * r / (beta_sq * p + r)
    
    def mcc(self) -> float:
        import math
        tp, tn, fp, fn = self._tp, self._tn, self._fp, self._fn
        numer = tp * tn - fp * fn
        denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        return numer / denom if denom > 0 else 0.0
    
    def cohens_kappa(self) -> float:
        tp, tn, fp, fn = self._tp, self._tn, self._fp, self._fn
        n = tp + tn + fp + fn
        p_o = (tp + tn) / n
        p_yes = ((tp + fp) / n) * ((tp + fn) / n)
        p_no = ((tn + fn) / n) * ((tn + fp) / n)
        p_e = p_yes + p_no
        return (p_o - p_e) / (1 - p_e) if p_e != 1 else 1.0
    
    def log_loss(self, eps: float = 1e-15) -> Optional[float]:
        if self.y_pred_proba is None:
            return None
        p = torch.clamp(self.y_pred_proba.float(), eps, 1 - eps)
        y = self.y_true.float()
        return -(y * torch.log(p) + (1 - y) * torch.log(1 - p)).mean().item()
    
    def full_report(self) -> BinaryClassificationReport:
        """Generate comprehensive classification report."""
        return BinaryClassificationReport(
            accuracy=self.accuracy(),
            precision=self.precision(),
            recall=self.recall(),
            specificity=self.specificity(),
            f1=self.f1(),
            mcc=self.mcc(),
            kappa=self.cohens_kappa(),
            log_loss=self.log_loss(),
            confusion_matrix=self.confusion_counts
        )


# Example usage
if __name__ == "__main__":
    # Simulated predictions
    y_true = torch.tensor([0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0])
    y_pred = torch.tensor([0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0])
    y_proba = torch.tensor([0.1, 0.9, 0.85, 0.2, 0.4, 0.75, 0.3, 0.6, 0.8, 
                            0.15, 0.7, 0.9, 0.25, 0.45, 0.1])
    
    metrics = ClassificationMetrics(y_true, y_pred, y_proba)
    report = metrics.full_report()
    print(report)
```

---

## Metric Selection Guide

### Decision Framework

```
┌────────────────────────────────────────────────────────────────┐
│              CLASSIFICATION METRIC SELECTION                    │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Problem Characteristic          →  Recommended Metric(s)       │
│  ────────────────────────────────────────────────────────────   │
│  Balanced dataset                →  Accuracy, F1                │
│  Imbalanced dataset              →  F1, MCC, PR-AUC             │
│  False positives costly          →  Precision, F0.5             │
│  False negatives costly          →  Recall, F2                  │
│  Probability calibration matters →  Log Loss, Brier Score       │
│  Model comparison                →  ROC-AUC, MCC                │
│  Clinical/medical                →  Sensitivity, Specificity    │
│  Multi-class balanced            →  Macro F1                    │
│  Multi-class imbalanced          →  Weighted F1                 │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### Finance-Specific Guidance

| Application | Priority Metric | Secondary |
|-------------|-----------------|-----------|
| Fraud Detection | Recall (catch fraud) | Precision (reduce false alarms) |
| Credit Scoring | ROC-AUC, Gini | Log Loss for calibration |
| Churn Prediction | F1, Recall | Precision for targeting |
| Trading Signals | Precision | Risk-adjusted returns |
| Default Prediction | MCC, ROC-AUC | Calibrated probabilities |

---

## Common Pitfalls

1. **Using accuracy on imbalanced data**: Misleading, use F1 or MCC
2. **Ignoring threshold selection**: Default 0.5 may not be optimal
3. **Not reporting multiple metrics**: Single metric can hide issues
4. **Confusion about averaging**: Know when to use macro vs weighted
5. **Ignoring probability calibration**: Especially for risk applications

---

## Summary

| Metric | Range | Perfect | Use Case |
|--------|-------|---------|----------|
| Accuracy | [0, 1] | 1 | Balanced data |
| Precision | [0, 1] | 1 | FP costly |
| Recall | [0, 1] | 1 | FN costly |
| F1 | [0, 1] | 1 | Balance P & R |
| MCC | [-1, 1] | 1 | Imbalanced data |
| Kappa | [-1, 1] | 1 | Inter-rater agreement |
| Log Loss | [0, ∞) | 0 | Probability calibration |

---

## References

1. Powers, D.M.W. (2011). "Evaluation: From Precision, Recall and F-Measure to ROC, Informedness, Markedness & Correlation."
2. Matthews, B.W. (1975). "Comparison of the predicted and observed secondary structure of T4 phage lysozyme."
3. Chicco, D. & Jurman, G. (2020). "The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation."
