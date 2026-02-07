# Feature Inversion

## Introduction

**Feature inversion** reconstructs inputs from intermediate representations, revealing what information the network preserves at each layer. This complementary approach to attribution shows not just *which* features matter, but *what the model actually sees* at different stages of processing.

## Mathematical Foundation

Given a feature representation $\Phi_l(\mathbf{x})$ at layer $l$, feature inversion finds:

$$
\mathbf{x}^* = \arg\min_{\mathbf{x}} \|\Phi_l(\mathbf{x}) - \Phi_l(\mathbf{x}_0)\|^2 + \lambda R(\mathbf{x})
$$

where $R(\mathbf{x})$ is a regularizer (total variation, $L^2$ norm) that encourages natural-looking images.

## Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureInversion:
    """Reconstruct input from intermediate features."""
    
    def __init__(self, model, target_layer, device):
        self.model = model
        self.target_layer = target_layer
        self.device = device
        self.target_features = None
        
        target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, 'current_features', o)
        )
    
    def total_variation(self, x):
        """Total variation regularizer for spatial smoothness."""
        diff_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        diff_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        return diff_h.mean() + diff_w.mean()
    
    def invert(
        self, target_input, n_steps=500, lr=0.05,
        tv_weight=1e-3, l2_weight=1e-5
    ):
        """Reconstruct input from layer features."""
        self.model.eval()
        
        with torch.no_grad():
            self.model(target_input.to(self.device))
            target_features = self.current_features.clone()
        
        # Start from noise
        x = torch.randn_like(target_input, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([x], lr=lr)
        
        for step in range(n_steps):
            optimizer.zero_grad()
            self.model(x)
            
            # Feature matching loss
            feat_loss = F.mse_loss(self.current_features, target_features)
            
            # Regularization
            tv_loss = tv_weight * self.total_variation(x)
            l2_loss = l2_weight * x.pow(2).mean()
            
            loss = feat_loss + tv_loss + l2_loss
            loss.backward()
            optimizer.step()
        
        return x.detach()
```

## Interpretation

Feature inversion reveals a fundamental insight: **early layers preserve spatial detail but lose semantic content, while later layers preserve semantic content but lose spatial detail.** This progressive abstraction is why Grad-CAM (targeting late layers) produces coarse but class-discriminative heatmaps.

## Summary

Feature inversion complements attribution methods by showing what information is preserved versus discarded at each network layer, providing a holistic understanding of the model's internal representations.

## References

1. Mahendran, A., & Vedaldi, A. (2015). "Understanding Deep Image Representations by Inverting Them." *CVPR*.

2. Dosovitskiy, A., & Brox, T. (2016). "Inverting Visual Representations with Convolutional Networks." *CVPR*.\n