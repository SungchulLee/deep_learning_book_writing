# ROC and AUC

## Overview

The Receiver Operating Characteristic (ROC) curve plots the True Positive Rate against the False Positive Rate at varying classification thresholds. The Area Under the Curve (AUC) summarizes discriminative ability in a single number.

## ROC Curve

At each threshold $\tau$, classify samples as positive if the predicted probability exceeds $\tau$:

- **TPR** (True Positive Rate) = $TP / (TP + FN)$ = Recall
- **FPR** (False Positive Rate) = $FP / (FP + TN)$ = 1 − Specificity

The ROC curve traces (FPR, TPR) as $\tau$ varies from 1 to 0.

```python
from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, thresholds = roc_curve(y_true, y_scores)
auc = roc_auc_score(y_true, y_scores)
```

## AUC Interpretation

- **AUC = 1.0**: Perfect classifier.
- **AUC = 0.5**: Random classifier (diagonal line).
- **AUC < 0.5**: Worse than random (predictions are inverted).

AUC equals the probability that a randomly chosen positive example is scored higher than a randomly chosen negative example.

## Advantages of AUC

AUC is **threshold-independent**: it evaluates the ranking quality of predictions without committing to a specific decision threshold. This makes it ideal for comparing models before choosing an operational threshold.

AUC is **scale-invariant**: it depends only on the ranking of scores, not their absolute values.

## Limitations

AUC can be misleading for heavily imbalanced datasets. When the negative class vastly outnumbers the positive class, a model can achieve high AUC by correctly ranking most negatives below most positives, even if its positive predictions are unreliable. In such cases, the precision-recall curve (next section) is more informative.

## Multi-Class AUC

For multi-class problems, compute AUC in a one-vs-rest or one-vs-one fashion:

```python
from sklearn.metrics import roc_auc_score

# One-vs-rest
auc_ovr = roc_auc_score(y_true, y_scores, multi_class='ovr')
```

## Key Takeaways

- ROC-AUC evaluates discriminative ability independent of the classification threshold.
- AUC = probability that a random positive is scored above a random negative.
- Prefer precision-recall curves over ROC for highly imbalanced datasets.
