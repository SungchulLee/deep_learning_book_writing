# Domain Adversarial Neural Networks (DANN)

Domain Adversarial Neural Networks learn domain-invariant features by training a feature extractor to simultaneously satisfy a label predictor and fool a domain classifier. The gradient reversal layer ensures that the shared features become indistinguishable between source and target domains.

## The Adversarial Approach

DANN jointly optimises three components:

```
                    ┌──────────────────┐
                    │  Label Predictor │ → Classification Loss
                    └────────┬─────────┘
                             │
Input ──► Feature Extractor ─┴──► Gradient Reversal ──► Domain Classifier → Domain Loss
                                        (flip gradient)
```

The feature extractor learns representations that are:

1. **Discriminative** for the main task (minimise classification loss)
2. **Domain-invariant** (maximise domain classifier confusion)

## Gradient Reversal Layer

The key innovation: during backpropagation, gradients from the domain classifier are *reversed*, so the feature extractor learns to make domains indistinguishable:

```python
import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReversal(Function):
    """Gradient Reversal Layer for domain adversarial training."""

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GRL(nn.Module):
    """Gradient Reversal Layer module."""

    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversal.apply(x, self.lambda_)

    def set_lambda(self, lambda_):
        self.lambda_ = lambda_
```

## DANN Architecture

```python
import numpy as np


class DANN(nn.Module):
    """Domain Adversarial Neural Network."""

    def __init__(self, backbone, num_classes, feature_dim=512):
        super().__init__()

        # Shared feature extractor
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        self.flatten = nn.Flatten()

        # Task-specific label predictor
        self.label_predictor = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

        # Domain classifier (with GRL)
        self.grl = GRL(lambda_=1.0)
        self.domain_classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)  # Binary: source vs target
        )

    def forward(self, x, alpha=1.0):
        features = self.flatten(self.feature_extractor(x))

        # Label prediction (normal gradient flow)
        class_output = self.label_predictor(features)

        # Domain prediction (reversed gradient flow)
        self.grl.set_lambda(alpha)
        reversed_features = self.grl(features)
        domain_output = self.domain_classifier(reversed_features)

        return class_output, domain_output
```

## Training DANN

### Progressive Lambda Schedule

The adversarial strength $\alpha$ increases during training using a sigmoid schedule:

$$\alpha_p = \frac{2}{1 + \exp(-10p)} - 1, \quad p = \frac{\text{epoch}}{\text{total epochs}}$$

This starts with weak adversarial signal (allowing the label predictor to learn) and gradually increases domain confusion.

```python
def train_dann(model, source_loader, target_loader, num_epochs=50, device='cuda'):
    """Train DANN model with progressive adversarial schedule."""
    model = model.to(device)

    class_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(num_epochs):
        model.train()

        # Progressive lambda schedule
        p = epoch / num_epochs
        alpha = 2. / (1. + np.exp(-10 * p)) - 1  # Grows from 0 to 1

        target_iter = iter(target_loader)

        for source_inputs, source_labels in source_loader:
            try:
                target_inputs, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_inputs, _ = next(target_iter)

            batch_size = source_inputs.size(0)
            inputs = torch.cat([source_inputs, target_inputs[:batch_size]], dim=0)
            inputs = inputs.to(device)
            source_labels = source_labels.to(device)

            # Domain labels: 0 = source, 1 = target
            domain_labels = torch.cat([
                torch.zeros(batch_size),
                torch.ones(min(batch_size, target_inputs.size(0)))
            ]).long().to(device)

            optimizer.zero_grad()

            class_output, domain_output = model(inputs, alpha=alpha)

            # Classification loss (source only)
            class_loss = class_criterion(class_output[:batch_size], source_labels)

            # Domain loss (both source and target)
            domain_loss = domain_criterion(domain_output, domain_labels)

            loss = class_loss + domain_loss
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Alpha: {alpha:.3f}")

    return model
```

## DANN Variants

### Conditional Domain Adversarial Network (CDAN)

CDAN conditions the domain classifier on class predictions for finer-grained alignment:

```python
class CDAN(nn.Module):
    """Conditional Domain Adversarial Network."""

    def __init__(self, backbone, num_classes, feature_dim=512):
        super().__init__()
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        self.flatten = nn.Flatten()
        self.label_predictor = nn.Linear(feature_dim, num_classes)

        self.grl = GRL()
        # Conditioned on feature ⊗ prediction outer product
        self.domain_classifier = nn.Sequential(
            nn.Linear(feature_dim * num_classes, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2)
        )
        self.num_classes = num_classes

    def forward(self, x, alpha=1.0):
        features = self.flatten(self.feature_extractor(x))
        class_output = self.label_predictor(features)

        # Multilinear conditioning: features ⊗ softmax(predictions)
        softmax_output = torch.softmax(class_output.detach(), dim=1)
        conditioned = torch.bmm(
            features.unsqueeze(2),
            softmax_output.unsqueeze(1)
        ).view(features.size(0), -1)

        self.grl.set_lambda(alpha)
        reversed_conditioned = self.grl(conditioned)
        domain_output = self.domain_classifier(reversed_conditioned)

        return class_output, domain_output
```

## When to Use DANN

| Scenario | Recommendation |
|----------|---------------|
| Large domain gap, no target labels | ✅ DANN is well-suited |
| Small domain gap | Overkill; use BN adaptation or MMD |
| Target labels available | Standard fine-tuning preferred |
| Multiple target domains | Consider multi-source DANN variants |

## Summary

DANN learns domain-invariant features through adversarial training:

1. The **gradient reversal layer** creates a minimax game between the feature extractor and domain classifier
2. The **progressive $\alpha$ schedule** stabilises training by gradually increasing adversarial strength
3. **CDAN** extends DANN with class-conditional domain alignment for finer-grained adaptation
4. DANN is most effective when domain gap is large and no target labels are available

## References

1. Ganin, Y., et al. (2016). "Domain-Adversarial Training of Neural Networks." *JMLR*.
2. Long, M., et al. (2018). "Conditional Adversarial Domain Adaptation." *NeurIPS*.
3. Tzeng, E., et al. (2017). "Adversarial Discriminative Domain Adaptation." *CVPR*.
