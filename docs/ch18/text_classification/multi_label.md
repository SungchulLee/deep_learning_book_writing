# Multi-Label Classification

## Overview

Each document can belong to multiple categories. Predict $\mathbf{y} \in \{0,1\}^k$:

$$P(y_j = 1 | x) = \sigma(\mathbf{w}_j^T \mathbf{h} + b_j)$$

Loss: Binary Cross-Entropy per label (BCEWithLogitsLoss).

```python
criterion = nn.BCEWithLogitsLoss()
logits = model(input_ids)  # (batch, num_labels)
loss = criterion(logits, labels.float())
```

## Threshold Tuning

Default threshold 0.5 may not be optimal. Tune per-label thresholds on validation set to maximize F1.

## References

1. Zhang, M. L., & Zhou, Z. H. (2014). A Review on Multi-Label Learning Algorithms. *TKDE*.
