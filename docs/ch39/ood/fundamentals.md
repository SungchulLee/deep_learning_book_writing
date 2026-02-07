# OOD Detection Fundamentals

## Overview

Out-of-distribution (OOD) detection identifies test inputs that differ significantly from the training distribution. This is critical for safe deployment—a model trained on normal market conditions should flag novel regime changes rather than making confident predictions.

## Problem Formulation

Given a classifier trained on in-distribution data $\mathcal{D}_{\text{in}}$, OOD detection assigns each test input $\mathbf{x}$ a score $s(\mathbf{x})$ such that:

$$s(\mathbf{x}) \begin{cases} \text{low} & \text{if } \mathbf{x} \sim P_{\text{in}} \\ \text{high} & \text{if } \mathbf{x} \sim P_{\text{out}} \end{cases}$$

## Types of Distribution Shift

| Shift Type | Description | Finance Example |
|-----------|-------------|----------------|
| **Covariate shift** | Input distribution changes | New market regime |
| **Semantic shift** | New classes appear | Novel asset type |
| **Near-OOD** | Subtle distribution change | Gradual regime transition |
| **Far-OOD** | Clearly different domain | Text data to a vision model |

## Baseline Methods

The sections that follow cover specific detection methods:

1. **Softmax Baseline** (MSP) — maximum softmax probability
2. **Energy-Based** — log-sum-exp of logits
3. **Mahalanobis Distance** — feature-space distance from training clusters
4. **ODIN** — temperature scaling + input perturbation

## References

- Hendrycks, D., & Gimpel, K. (2017). "A Baseline for Detecting Misclassified and OOD Examples in Neural Networks." ICLR.
