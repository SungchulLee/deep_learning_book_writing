# Discriminative Learning Rates

Discriminative learning rates assign different learning rates to different layers of a pretrained model. Lower layers (general features) receive smaller updates to preserve useful representations, while upper layers (task-specific features) receive larger updates for faster adaptation.

## Motivation

Standard fine-tuning applies a single learning rate across all parameters. This creates a tension:

- **Too high**: Destroys pretrained features in early layers
- **Too low**: Adapts task-specific layers too slowly

Discriminative learning rates resolve this by matching the update magnitude to each layer's transferability.

## Two-Group Strategy

The simplest approach: separate backbone and classifier into two parameter groups.

```python
import torch.optim as optim


def create_discriminative_optimizer(model, base_lr=1e-3, backbone_mult=0.1):
    """Different LR for backbone vs classifier."""
    backbone_params = []
    classifier_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'fc' in name:
            classifier_params.append(param)
        else:
            backbone_params.append(param)

    return optim.Adam([
        {'params': backbone_params, 'lr': base_lr * backbone_mult},
        {'params': classifier_params, 'lr': base_lr},
    ])
```

Typical ratios:

| Scenario | Backbone : Classifier | Example (base_lr=1e-3) |
|----------|----------------------|----------------------|
| Similar domain | 1:10 | 1e-4 : 1e-3 |
| Different domain | 1:100 | 1e-5 : 1e-3 |
| Very small data | 1:100+ or freeze | 0 : 1e-3 |

## Layer-Wise Learning Rates

For finer control, assign a different rate to each layer group:

```python
def create_layerwise_optimizer(model, base_lr=1e-3):
    """Per-layer learning rates with exponential decay."""
    layer_lrs = {
        'layer1': 0.01, 'layer2': 0.05,
        'layer3': 0.1, 'layer4': 0.5, 'fc': 1.0,
    }
    groups = []
    for layer, mult in layer_lrs.items():
        params = [p for n, p in model.named_parameters()
                  if layer in n and p.requires_grad]
        if params:
            groups.append({'params': params, 'lr': base_lr * mult})
    return optim.Adam(groups)
```

## Discriminative Rates for Transformers

For BERT-like models with many encoder layers, use exponential decay from the top:

```python
from typing import List, Dict, Any


def get_discriminative_lr_params(
    model,
    base_lr: float = 2e-5,
    lr_decay: float = 0.95,
    weight_decay: float = 0.01
) -> List[Dict[str, Any]]:
    """
    Create parameter groups with discriminative learning rates.
    
    Learning rate schedule:
    - Embeddings: base_lr * decay^num_layers (lowest)
    - Layer 0: base_lr * decay^(num_layers-1)
    - Layer N-1: base_lr * decay (second highest)
    - Classifier: base_lr (highest)
    """
    param_groups = []

    if hasattr(model.encoder, 'layer'):
        encoder_layers = model.encoder.layer
    elif hasattr(model.encoder, 'layers'):
        encoder_layers = model.encoder.layers
    else:
        raise ValueError("Could not find encoder layers")

    num_layers = len(encoder_layers)

    # Embeddings - lowest learning rate
    if hasattr(model.encoder, 'embeddings'):
        emb_lr = base_lr * (lr_decay ** (num_layers + 1))
        param_groups.append({
            'params': model.encoder.embeddings.parameters(),
            'lr': emb_lr,
            'weight_decay': weight_decay,
            'name': 'embeddings'
        })

    # Encoder layers - gradually increasing LR
    for i, layer in enumerate(encoder_layers):
        layer_lr = base_lr * (lr_decay ** (num_layers - i))
        param_groups.append({
            'params': layer.parameters(),
            'lr': layer_lr,
            'weight_decay': weight_decay,
            'name': f'layer_{i}'
        })

    # Classifier head - highest learning rate
    if hasattr(model, 'classifier'):
        param_groups.append({
            'params': model.classifier.parameters(),
            'lr': base_lr,
            'weight_decay': weight_decay,
            'name': 'classifier'
        })

    return param_groups


def print_lr_schedule(param_groups: List[Dict[str, Any]]):
    """Print the learning rate schedule."""
    print("Learning Rate Schedule:")
    print("-" * 40)
    for group in param_groups:
        print(f"  {group.get('name', 'unnamed'):15s}: {group['lr']:.2e}")
```

### Example Schedule (BERT-base, 12 layers)

| Component | Layer | LR (decay=0.9) | LR (decay=0.95) |
|-----------|-------|----------------|-----------------|
| Embeddings | - | 5.7e-7 | 1.1e-6 |
| Layer 0 | Bottom | 5.1e-6 | 6.9e-6 |
| Layer 5 | Middle | 1.2e-5 | 1.3e-5 |
| Layer 11 | Top | 1.8e-5 | 1.9e-5 |
| Classifier | - | 2.0e-5 | 2.0e-5 |

## Combining with Gradual Unfreezing

Discriminative learning rates pair naturally with [gradual unfreezing](layer_freezing.md):

```python
def create_staged_optimizer(model, epoch, base_lr=2e-5, lr_decay=0.9):
    """Adjust optimizer as layers are unfrozen."""
    param_groups = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Determine layer depth (higher = closer to output)
        if 'classifier' in name or 'fc' in name:
            lr = base_lr
        elif 'layer4' in name or 'layer.11' in name:
            lr = base_lr * lr_decay
        elif 'layer3' in name or 'layer.8' in name:
            lr = base_lr * lr_decay ** 2
        else:
            lr = base_lr * lr_decay ** 3
        
        param_groups.append({'params': [param], 'lr': lr})
    
    return optim.AdamW(param_groups, weight_decay=0.01)
```

## Choosing the Decay Factor

The decay factor $\gamma$ determines how aggressively learning rates decrease with depth:

| Decay ($\gamma$) | Bottom/Top LR ratio | Use case |
|-------------------|---------------------|----------|
| 0.99 | ~0.89× | Minimal discrimination, similar domain |
| 0.95 | ~0.54× | Standard fine-tuning |
| 0.90 | ~0.28× | Moderate domain shift |
| 0.80 | ~0.07× | Large domain shift, preserve early features |

## Summary

| Method | Complexity | Best for |
|--------|-----------|----------|
| Two-group (backbone/head) | Simple | Quick experiments |
| Layer-wise (CNN) | Moderate | Vision transfer |
| Exponential decay (Transformer) | Full | NLP/large models |
| Combined with unfreezing | Advanced | Maximum control |

## References

1. Howard, J., & Ruder, S. (2018). "Universal Language Model Fine-tuning for Text Classification." *ACL*.
2. Clark, K., et al. (2020). "ELECTRA: Pre-training Text Encoders as Discriminators." *ICLR*.
3. Sun, C., et al. (2019). "How to Fine-Tune BERT for Text Classification." *CCL*.
