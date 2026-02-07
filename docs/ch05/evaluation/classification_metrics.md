# Classification Metrics

## Overview

Classification metrics evaluate how well a model assigns discrete labels. The choice of metric depends on class balance, cost asymmetry, and whether the application requires hard predictions or probability estimates.

## Accuracy

$$\text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Predictions}}$$

Simple and intuitive, but misleading for imbalanced datasets. A model that always predicts the majority class achieves high accuracy while being useless.

```python
accuracy = (preds == targets).float().mean().item()
```

## Precision and Recall

**Precision** (positive predictive value): Of all predicted positives, how many are truly positive?

$$\text{Precision} = \frac{TP}{TP + FP}$$

**Recall** (sensitivity, true positive rate): Of all actual positives, how many are correctly identified?

$$\text{Recall} = \frac{TP}{TP + FN}$$

Precision and recall are inversely related: increasing the classification threshold raises precision but lowers recall.

## F1 Score

The harmonic mean of precision and recall:

$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

The $F_\beta$ score generalizes this with a parameter $\beta$ that controls the relative weight of recall:

$$F_\beta = (1 + \beta^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{\beta^2 \cdot \text{Precision} + \text{Recall}}$$

Use $F_2$ when recall is more important (e.g., fraud detection: missing a fraud is worse than a false alarm). Use $F_{0.5}$ when precision is more important.

## Multi-Class Averaging

For multi-class problems, per-class metrics are aggregated:

- **Macro**: Unweighted mean across classes. Treats all classes equally.
- **Weighted**: Weighted by class frequency. Accounts for class imbalance.
- **Micro**: Computed globally by counting total TP, FP, FN.

```python
from sklearn.metrics import classification_report

print(classification_report(y_true, y_pred))
```

## Financial Application

In credit default prediction, a false negative (missed default) is far more costly than a false positive (unnecessary credit denial). Optimize for recall or use a cost-sensitive $F_\beta$ with $\beta > 1$.

## Key Takeaways

- Accuracy is insufficient for imbalanced problems; use precision, recall, and F1.
- The precision-recall tradeoff is controlled by the classification threshold.
- Choose the averaging strategy (macro, weighted, micro) based on whether class balance matters.
