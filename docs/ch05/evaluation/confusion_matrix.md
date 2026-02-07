# Confusion Matrix

## Overview

The confusion matrix is a tabular summary of classification predictions vs. actual labels. It provides the foundation from which all classification metrics (precision, recall, F1) are computed.

## Structure

For binary classification:

|  | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actual Positive** | True Positive (TP) | False Negative (FN) |
| **Actual Negative** | False Positive (FP) | True Negative (TN) |

## Computation

```python
from sklearn.metrics import confusion_matrix
import torch

def compute_confusion_matrix(preds, targets, num_classes):
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for t, p in zip(targets, preds):
        cm[t, p] += 1
    return cm

# Using sklearn
cm = confusion_matrix(y_true, y_pred)
```

## Derived Metrics

All standard classification metrics derive from the confusion matrix entries:

- Accuracy: $(TP + TN) / (TP + TN + FP + FN)$
- Precision: $TP / (TP + FP)$
- Recall: $TP / (TP + FN)$
- Specificity: $TN / (TN + FP)$
- False Positive Rate: $FP / (FP + TN) = 1 - \text{Specificity}$

## Multi-Class Confusion Matrix

For $C$ classes, the confusion matrix is $C \times C$. Entry $(i, j)$ counts samples with true class $i$ predicted as class $j$. The diagonal contains correct predictions; off-diagonal entries are errors.

## Interpretation

The confusion matrix reveals the *structure* of errors. A model may have high overall accuracy but systematically confuse certain class pairs. This structural information is invisible in aggregate metrics.

## Key Takeaways

- The confusion matrix is the fundamental object for classification evaluation.
- All classification metrics are functions of TP, TN, FP, FN.
- Multi-class confusion matrices reveal systematic misclassification patterns.
