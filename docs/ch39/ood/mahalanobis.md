# Mahalanobis Distance for OOD Detection

## Overview

The Mahalanobis distance method (Lee et al., 2018) detects OOD inputs by measuring the distance from test features to the nearest class-conditional Gaussian in the model's feature space.

## Method

1. Extract features from a penultimate layer for all training data
2. Fit class-conditional Gaussians: $p(\mathbf{z} | y=c) = \mathcal{N}(\boldsymbol{\mu}_c, \Sigma)$ with shared covariance
3. For a test input, compute the Mahalanobis distance to the nearest class:

$$M(\mathbf{x}) = \min_c (\mathbf{z} - \boldsymbol{\mu}_c)^T \Sigma^{-1} (\mathbf{z} - \boldsymbol{\mu}_c)$$

## Implementation

```python
import torch
import numpy as np
from typing import Tuple


class MahalanobisOOD:
    """Mahalanobis distance-based OOD detector."""
    
    def __init__(self):
        self.class_means = None
        self.precision = None
    
    def fit(self, features: torch.Tensor, labels: torch.Tensor):
        """Fit class-conditional Gaussians."""
        classes = labels.unique()
        self.class_means = []
        
        all_centered = []
        for c in classes:
            mask = labels == c
            class_features = features[mask]
            mean = class_features.mean(dim=0)
            self.class_means.append(mean)
            all_centered.append(class_features - mean)
        
        self.class_means = torch.stack(self.class_means)
        
        # Shared covariance
        centered = torch.cat(all_centered, dim=0)
        cov = (centered.T @ centered) / len(centered)
        self.precision = torch.linalg.inv(cov + 1e-5 * torch.eye(cov.size(0)))
    
    def score(self, features: torch.Tensor) -> torch.Tensor:
        """Compute Mahalanobis distance to nearest class."""
        distances = []
        for mean in self.class_means:
            diff = features - mean
            dist = torch.sum(diff @ self.precision * diff, dim=-1)
            distances.append(dist)
        
        distances = torch.stack(distances, dim=-1)
        return distances.min(dim=-1).values  # Min distance to any class
```

## References

- Lee, K., et al. (2018). "A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks." NeurIPS.
