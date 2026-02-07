# EfficientNet

## Learning Objectives

By the end of this section, you will be able to:

- Understand the key innovations introduced by EfficientNet (2019)
- Identify how EfficientNet influenced subsequent architecture design

## Overview

**Year**: 2019 | **Parameters**: 5.3M (B0) – 66M (B7) | **Key Innovation**: Compound scaling of depth, width, and resolution

EfficientNet (Tan & Le, 2019) introduced **compound scaling**—simultaneously scaling network depth, width, and input resolution using a principled formula. This produces a family of models (B0–B7) that dominate accuracy/efficiency trade-offs.

## Compound Scaling

$$\text{depth: } d = \alpha^\phi, \quad \text{width: } w = \beta^\phi, \quad \text{resolution: } r = \gamma^\phi$$

subject to $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$ (FLOPs roughly double per unit increase in $\phi$).

Optimal values found via NAS: $\alpha = 1.2, \beta = 1.1, \gamma = 1.15$.

| Model | Resolution | Params | Top-1 Acc |
|-------|-----------|--------|-----------|
| B0 | 224 | 5.3M | 77.1% |
| B3 | 300 | 12M | 81.6% |
| B5 | 456 | 30M | 83.6% |
| B7 | 600 | 66M | 84.3% |

```python
import torchvision.models as models
model = models.efficientnet_b0(weights='DEFAULT')
```

EfficientNet demonstrated that balanced scaling across all dimensions is strictly better than scaling any single dimension alone.

## References

1. Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ICML.
2. Tan, M., & Le, Q. V. (2021). EfficientNetV2: Smaller Models and Faster Training. ICML.
