# LeNet

## Learning Objectives

By the end of this section, you will be able to:

- Understand the key innovations introduced by LeNet (1998)
- Identify how LeNet influenced subsequent architecture design

## Overview

**Year**: 1998 | **Parameters**: 60K | **Key Innovation**: First successful CNN for digit recognition

LeNet-5 (LeCun et al., 1998) demonstrated that convolutional neural networks could learn spatial hierarchies for image recognition. The architecture introduced the fundamental CNN building blocks: convolutional layers for feature extraction, pooling for spatial invariance, and fully connected layers for classification.

## Architecture

```
Input (32×32×1)
  → Conv 5×5 (6 filters) → AvgPool 2×2
  → Conv 5×5 (16 filters) → AvgPool 2×2
  → FC(120) → FC(84) → FC(10)
```

### Key Design Principles

1. **Local connectivity**: Each neuron connects to a small spatial region
2. **Weight sharing**: Same filter applied across all positions
3. **Spatial subsampling**: Pooling reduces resolution and provides translation invariance
4. **Hierarchical features**: Early layers detect edges; later layers detect combinations

```python
import torch.nn as nn

class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
```

LeNet's principles remain the foundation of all modern CNNs. The shift from hand-crafted features to learned convolutional filters was the key paradigm change.

## References

1. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-Based Learning Applied to Document Recognition. Proceedings of the IEEE.
