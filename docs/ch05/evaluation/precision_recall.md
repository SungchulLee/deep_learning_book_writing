# Precision-Recall Curves

## Overview

Precision-recall (PR) curves plot precision against recall at varying classification thresholds. They are more informative than ROC curves for imbalanced datasets where the positive class is rare.

## PR Curve

```python
from sklearn.metrics import precision_recall_curve, average_precision_score

precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
ap = average_precision_score(y_true, y_scores)
```

## Average Precision (AP)

AP summarizes the PR curve as the weighted mean of precisions at each threshold, with the increase in recall as the weight:

$$\text{AP} = \sum_n (R_n - R_{n-1}) P_n$$

AP approximates the area under the PR curve.

## PR vs. ROC

For imbalanced datasets, ROC curves can be overly optimistic because TNs dominate the FPR calculation. PR curves focus exclusively on the positive class, making them more sensitive to the model's ability to correctly identify rare positives.

A model that appears excellent on ROC-AUC may have poor precision at the recall levels that matter operationally.

## Financial Application

In fraud detection or default prediction where the positive class (fraud/default) is rare (< 1%), PR curves reveal the practical tradeoff: how many true positives can we capture (recall) before the false positive rate makes the system unusable (low precision)?

## Key Takeaways

- PR curves are preferred over ROC for imbalanced classification problems.
- Average precision summarizes the PR curve in a single number.
- PR curves directly show the operational precision-recall tradeoff.
