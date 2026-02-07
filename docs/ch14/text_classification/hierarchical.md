# Hierarchical Classification

## Overview

Assign labels from a taxonomy tree while maintaining parent-child consistency.

## Approaches

- **Top-Down**: Cascaded classifiers at each hierarchy level
- **Flat + Constraints**: Multi-label classifier with hierarchy enforcement at inference
- **Hierarchy-Aware Loss**: Penalize violations of parent-child consistency during training

$$\mathcal{L} = \sum_j \mathcal{L}_{BCE}(\hat{y}_j, y_j) + \lambda \sum_{(p,c)} \max(0, \hat{y}_c - \hat{y}_p)$$

## References

1. Silla, C. N., & Freitas, A. A. (2011). A Survey of Hierarchical Classification. *Data Mining and Knowledge Discovery*.
