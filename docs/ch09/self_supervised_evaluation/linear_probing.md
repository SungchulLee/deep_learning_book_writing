# Linear Probing

## Overview

Linear probing trains a linear classifier on frozen pre-trained features, directly measuring representation quality.

## Protocol

```python
class LinearProbe(nn.Module):
    def __init__(self, encoder, feature_dim, num_classes):
        super().__init__()
        self.encoder = encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():
            features = self.encoder(x)
        return self.classifier(features)
```

## Training Details

Optimizer: SGD with momentum. Learning rate: sweep over {0.01, 0.1, 1.0, 10.0}. Epochs: 100 on ImageNet. No data augmentation (only resize + center crop). L2-normalize features before the classifier.

## Interpretation

Near supervised baseline: excellent representations with linearly separable features. Significantly below supervised: features capture some structure but are not task-aligned.

## Limitations

Can underestimate representation quality when features are informative but not linearly separable — fine-tuning gives a more complete picture in such cases.
