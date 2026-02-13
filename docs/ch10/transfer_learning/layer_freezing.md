# Layer Freezing

Layer freezing controls which parameters are updated during transfer learning. By selectively freezing pretrained layers and only training task-specific layers, we balance adaptation to new data against preservation of learned features.

## Freezing Fundamentals

### Why Freeze Layers?

Freezing serves three purposes:

1. **Prevent overfitting**: Fewer trainable parameters reduce the risk of memorising small datasets
2. **Preserve universal features**: Early layers capture edges, textures, and patterns that generalise across domains
3. **Reduce compute**: Frozen layers don't require gradient computation, speeding up training

### The Transferability Gradient

Feature transferability decreases with depth. Yosinski et al. (2014) showed that layer-wise transfer performance drops sharply in the final layers:

```
Layer 1  ████████████████████  Very transferable (edges, colours)
Layer 2  ███████████████████   Highly transferable (textures)
Layer 3  ████████████████      Transferable (patterns)
Layer 4  ██████████████        Moderately transferable (parts)
Layer 5  ██████████            Domain-dependent (objects)
FC       ████                  Task-specific (must replace)
```

## Selective Layer Unfreezing

Freeze early layers (generic features), unfreeze later layers (domain-specific features):

```python
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def create_selective_unfreeze_model(num_classes, unfreeze=('layer3', 'layer4')):
    """Freeze all layers except specified ones and the classifier."""
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    # Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze specified layers
    for name, param in model.named_parameters():
        if any(layer in name for layer in unfreeze):
            param.requires_grad = True

    # Replace classifier (always trainable)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# Verify what's trainable
model = create_selective_unfreeze_model(10, unfreeze=('layer4',))
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"  Trainable: {name}")
```

### Impact of Freezing Depth

| Configuration | Trainable % | Accuracy* | Overfitting risk |
|--------------|-------------|-----------|------------------|
| Only FC | <1% | 80–85% | Very low |
| layer4 + FC | ~25% | 83–87% | Low |
| layer3 + layer4 + FC | ~50% | 85–90% | Medium |
| All layers | 100% | 88–92% | High |

*Typical ranges on CIFAR-10 with ResNet-18 transfer from ImageNet.

## Gradual Unfreezing

Start frozen, progressively unfreeze layers during training. This approach, popularised by ULMFiT (Howard & Ruder, 2018), lets the model stabilise task-specific layers before adapting earlier, more general features.

### Schedule-Based Unfreezing

```python
def unfreeze_layers(model, epoch, schedule):
    """
    Progressively unfreeze layers according to a schedule.
    
    Args:
        model: The model to unfreeze
        epoch: Current training epoch
        schedule: Dict mapping epoch -> list of layers to unfreeze
                  Example: {0: ['fc'], 5: ['layer4'], 10: ['layer3']}
    """
    if epoch in schedule:
        for name, param in model.named_parameters():
            if any(layer in name for layer in schedule[epoch]):
                param.requires_grad = True
                print(f"Epoch {epoch}: unfreezing {name}")
```

### Gradual Unfreezing for Transformer Models

```python
from typing import List


class GradualUnfreezeScheduler:
    """
    Gradually unfreeze layers from top to bottom.
    
    Top layers are more task-specific; bottom layers contain more
    general features. Unfreezing top-first allows the model to adapt
    gradually without disrupting lower-level representations.
    """
    
    def __init__(self, model, encoder_attr='encoder'):
        self.model = model
        self.encoder = getattr(model, encoder_attr)
        
        if hasattr(self.encoder, 'layer'):
            self.layers = list(self.encoder.layer)
        elif hasattr(self.encoder, 'layers'):
            self.layers = list(self.encoder.layers)
        else:
            raise ValueError("Could not find encoder layers")
        
        self.num_layers = len(self.layers)
        self.unfrozen_count = 0
        self._freeze_all()
    
    def _freeze_all(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_next(self) -> bool:
        """Unfreeze the next layer (top to bottom)."""
        if self.unfrozen_count >= self.num_layers:
            return False
        
        layer_idx = self.num_layers - 1 - self.unfrozen_count
        for param in self.layers[layer_idx].parameters():
            param.requires_grad = True
        self.unfrozen_count += 1
        
        print(f"Unfroze layer {layer_idx} ({self.unfrozen_count}/{self.num_layers})")
        return True
    
    def unfreeze_n_layers(self, n: int):
        """Unfreeze n layers at once."""
        for _ in range(n):
            if not self.unfreeze_next():
                break
    
    def unfreeze_embeddings(self):
        """Unfreeze embedding layer (usually done last)."""
        if hasattr(self.encoder, 'embeddings'):
            for param in self.encoder.embeddings.parameters():
                param.requires_grad = True
            print("Unfroze embeddings")
```

### Training with Gradual Unfreezing

```python
# Usage example
num_epochs = 12
unfreeze_every = 2  # Unfreeze a new layer every 2 epochs

scheduler = GradualUnfreezeScheduler(model)

for epoch in range(num_epochs):
    if epoch > 0 and epoch % unfreeze_every == 0:
        scheduler.unfreeze_next()
    
    # Optionally unfreeze embeddings near the end
    if epoch == num_epochs - 2:
        scheduler.unfreeze_embeddings()
    
    # train_epoch(model, train_loader, optimizer)
```

## Batch Normalization Considerations

Frozen batch normalization layers require special attention:

```python
def freeze_with_bn_eval(model, freeze_layers):
    """Freeze layers and ensure BN uses running statistics."""
    for name, module in model.named_modules():
        if any(layer in name for layer in freeze_layers):
            for param in module.parameters():
                param.requires_grad = False
            # Set BN to eval mode so it uses running stats
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                module.eval()


class FrozenBNModel(nn.Module):
    """Model that keeps frozen BN layers in eval mode during training."""
    
    def __init__(self, model, frozen_layers):
        super().__init__()
        self.model = model
        self.frozen_layers = frozen_layers
    
    def train(self, mode=True):
        super().train(mode)
        # Keep frozen BN in eval mode
        if mode:
            freeze_with_bn_eval(self.model, self.frozen_layers)
        return self
```

## Few-Shot Transfer with Heavy Freezing

For extremely limited data (<100 samples per class), freeze everything and add a regularised classifier:

```python
def few_shot_transfer(model, num_classes):
    """Heavy regularisation for very small datasets."""
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes),
    )
    return model
```

Combine with aggressive data augmentation and early stopping.

## Summary

| Approach | When to use | Risk |
|----------|------------|------|
| Freeze all, train FC only | Very small data, similar domain | Underfitting |
| Freeze early, unfreeze late | Moderate data, moderate shift | Balanced |
| Gradual unfreezing | Any size, careful adaptation | Slow training |
| Unfreeze all | Large data, different domain | Overfitting |

## References

1. Yosinski, J., et al. (2014). "How Transferable are Features in Deep Neural Networks?" *NeurIPS*.
2. Howard, J., & Ruder, S. (2018). "Universal Language Model Fine-tuning for Text Classification." *ACL*.
3. Raghu, M., et al. (2019). "Transfusion: Understanding Transfer Learning for Medical Imaging." *NeurIPS*.
