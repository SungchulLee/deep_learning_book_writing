# Transfer Learning Save/Load Patterns

## Overview

**Transfer learning** involves leveraging pre-trained models as starting points for new tasks. This requires specialized save/load patterns to handle partial weight loading, layer freezing, architecture modifications, and selective parameter updates. Mastering these patterns is essential for efficient model development and deployment.

## Learning Objectives

By the end of this section, you will be able to:

- Load pre-trained models and modify architectures for new tasks
- Implement layer freezing and unfreezing strategies
- Handle state dictionary mismatches gracefully
- Save and restore fine-tuned models with full configuration
- Apply partial state dictionary loading for model surgery

## Mathematical Background

### Transfer Learning Framework

Given a source task $\mathcal{T}_S$ with data distribution $\mathcal{D}_S$ and a target task $\mathcal{T}_T$ with distribution $\mathcal{D}_T$, transfer learning assumes:

$$P(\mathcal{D}_S) \neq P(\mathcal{D}_T) \quad \text{or} \quad \mathcal{T}_S \neq \mathcal{T}_T$$

The pre-trained model $f_{\theta_S}$ parameterized by $\theta_S$ serves as initialization for the target model:

$$\theta_T^{(0)} = \phi(\theta_S)$$

where $\phi$ is a transformation function that may:
- Copy weights directly (when architectures match)
- Map weights partially (when architectures differ)
- Initialize new layers randomly

### Layer Freezing Formulation

For a model with $L$ layers, define trainable indicator $\tau_l \in \{0, 1\}$:

$$\theta_l^{(t+1)} = \theta_l^{(t)} - \tau_l \cdot \eta \nabla_{\theta_l} \mathcal{L}$$

Setting $\tau_l = 0$ freezes layer $l$, preventing gradient updates. The optimization then operates on the reduced parameter space:

$$\tilde{\theta} = \{\theta_l : \tau_l = 1\}$$

## Loading Pre-trained Models

### Using torchvision Models

```python
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

# Load pre-trained ResNet18 with ImageNet weights
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

# Alternative: Load without pre-trained weights
model_random = models.resnet18(weights=None)

# Examine model structure
print("ResNet18 Architecture:")
print(f"  Feature extractor output: {model.fc.in_features}")
print(f"  Original classifier: {model.fc.out_features} classes (ImageNet)")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"  Total parameters: {total_params:,}")
```

### Loading from Checkpoints

```python
def load_pretrained_checkpoint(
    model_class: type,
    checkpoint_path: str,
    strict: bool = True,
    map_location: str = 'cpu',
    **model_kwargs
) -> nn.Module:
    """
    Load a pre-trained model from checkpoint.
    
    Args:
        model_class: Model class to instantiate
        checkpoint_path: Path to checkpoint file
        strict: Whether to require exact key matching
        map_location: Device for loading weights
        **model_kwargs: Arguments for model instantiation
    
    Returns:
        Loaded model
    """
    # Create model instance
    model = model_class(**model_kwargs)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Load state dict
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
    
    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")
    
    return model
```

## Architecture Modification

### Replacing the Classifier Head

```python
def modify_classifier(
    model: nn.Module,
    num_classes: int,
    classifier_name: str = 'fc'
) -> nn.Module:
    """
    Replace the final classifier layer for a new number of classes.
    
    Args:
        model: Pre-trained model
        num_classes: Number of classes for new task
        classifier_name: Name of classifier attribute
    
    Returns:
        Modified model
    """
    # Get the classifier layer
    classifier = getattr(model, classifier_name)
    
    if isinstance(classifier, nn.Linear):
        in_features = classifier.in_features
        new_classifier = nn.Linear(in_features, num_classes)
        setattr(model, classifier_name, new_classifier)
        print(f"Replaced {classifier_name}: {classifier.out_features} → {num_classes} classes")
    
    elif isinstance(classifier, nn.Sequential):
        # Handle sequential classifier (e.g., VGG, AlexNet)
        last_layer = list(classifier.children())[-1]
        if isinstance(last_layer, nn.Linear):
            in_features = last_layer.in_features
            new_layer = nn.Linear(in_features, num_classes)
            # Replace last layer in sequential
            new_children = list(classifier.children())[:-1] + [new_layer]
            setattr(model, classifier_name, nn.Sequential(*new_children))
    
    return model


# Example: Modify ResNet for 10-class problem
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model = modify_classifier(model, num_classes=10, classifier_name='fc')
```

### Adding Custom Heads

```python
class CustomHead(nn.Module):
    """Custom classification head with dropout and hidden layers."""
    
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_dim: int = 512,
        dropout_rate: float = 0.5
    ):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


def add_custom_head(
    model: nn.Module,
    num_classes: int,
    head_config: dict = None
) -> nn.Module:
    """Add custom classification head to pre-trained model."""
    
    # Remove original classifier
    in_features = model.fc.in_features
    model.fc = nn.Identity()  # Passthrough
    
    # Create custom head
    config = head_config or {}
    custom_head = CustomHead(
        in_features=in_features,
        num_classes=num_classes,
        **config
    )
    
    # Wrap model with custom head
    class ModelWithCustomHead(nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone
            self.head = head
        
        def forward(self, x):
            features = self.backbone(x)
            return self.head(features)
    
    return ModelWithCustomHead(model, custom_head)


# Example usage
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model = add_custom_head(model, num_classes=10, head_config={'dropout_rate': 0.3})
```

## Layer Freezing Strategies

### Freeze All Except Classifier

```python
def freeze_backbone(model: nn.Module, classifier_name: str = 'fc') -> None:
    """
    Freeze all layers except the classifier.
    
    Args:
        model: Model to freeze
        classifier_name: Name of classifier to keep trainable
    """
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze classifier
    classifier = getattr(model, classifier_name)
    for param in classifier.parameters():
        param.requires_grad = True
    
    # Count trainable parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Frozen: {total - trainable:,} parameters")
    print(f"Trainable: {trainable:,} parameters ({100 * trainable / total:.1f}%)")


# Example
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(512, 10)
freeze_backbone(model, classifier_name='fc')
```

### Gradual Unfreezing

```python
class GradualUnfreezer:
    """
    Progressively unfreeze model layers during training.
    
    This implements the discriminative fine-tuning approach where
    later layers are unfrozen first, then earlier layers gradually.
    """
    
    def __init__(
        self,
        model: nn.Module,
        layer_groups: list,
        unfreeze_schedule: dict
    ):
        """
        Args:
            model: Model to manage
            layer_groups: List of parameter groups (ordered from earliest to latest)
            unfreeze_schedule: Dict mapping epoch to layer group index to unfreeze
        """
        self.model = model
        self.layer_groups = layer_groups
        self.unfreeze_schedule = unfreeze_schedule
        
        # Initially freeze all
        for group in layer_groups:
            for param in group:
                param.requires_grad = False
        
        # Unfreeze last group (classifier)
        for param in layer_groups[-1]:
            param.requires_grad = True
    
    def step(self, epoch: int) -> None:
        """Check and apply unfreezing for current epoch."""
        if epoch in self.unfreeze_schedule:
            group_idx = self.unfreeze_schedule[epoch]
            for param in self.layer_groups[group_idx]:
                param.requires_grad = True
            print(f"Epoch {epoch}: Unfroze layer group {group_idx}")
    
    def get_trainable_params(self):
        """Get currently trainable parameters."""
        return [p for p in self.model.parameters() if p.requires_grad]


def create_layer_groups_resnet(model: nn.Module) -> list:
    """Create layer groups for ResNet gradual unfreezing."""
    return [
        list(model.conv1.parameters()) + list(model.bn1.parameters()),
        list(model.layer1.parameters()),
        list(model.layer2.parameters()),
        list(model.layer3.parameters()),
        list(model.layer4.parameters()),
        list(model.fc.parameters()),
    ]


# Example: Gradual unfreezing schedule
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(512, 10)

layer_groups = create_layer_groups_resnet(model)
unfreezer = GradualUnfreezer(
    model,
    layer_groups,
    unfreeze_schedule={
        0: 5,   # Unfreeze fc (already done by default)
        5: 4,   # Unfreeze layer4
        10: 3,  # Unfreeze layer3
        15: 2,  # Unfreeze layer2
        20: 1,  # Unfreeze layer1
        25: 0,  # Unfreeze conv1
    }
)
```

### Discriminative Learning Rates

```python
def create_discriminative_optimizer(
    model: nn.Module,
    layer_groups: list,
    base_lr: float = 1e-3,
    lr_decay_factor: float = 2.6
) -> torch.optim.Optimizer:
    """
    Create optimizer with different learning rates per layer group.
    
    Earlier layers get smaller learning rates.
    
    Args:
        model: Model to optimize
        layer_groups: List of parameter groups (earliest to latest)
        base_lr: Learning rate for the last (classifier) layer
        lr_decay_factor: Factor to reduce LR for earlier layers
    
    Returns:
        Optimizer with discriminative learning rates
    """
    param_groups = []
    num_groups = len(layer_groups)
    
    for i, group in enumerate(layer_groups):
        # Calculate LR: highest for last group, decreasing for earlier
        lr = base_lr / (lr_decay_factor ** (num_groups - 1 - i))
        
        param_groups.append({
            'params': group,
            'lr': lr,
            'name': f'group_{i}'
        })
        print(f"Group {i}: lr={lr:.6f}")
    
    return torch.optim.AdamW(param_groups, weight_decay=0.01)


# Example
layer_groups = create_layer_groups_resnet(model)
optimizer = create_discriminative_optimizer(
    model,
    layer_groups,
    base_lr=1e-3,
    lr_decay_factor=2.6
)
```

## Partial State Dict Loading

### Filtering and Mapping Keys

```python
def load_partial_state_dict(
    model: nn.Module,
    state_dict: dict,
    exclude_patterns: list = None,
    include_patterns: list = None,
    key_mapping: dict = None
) -> tuple:
    """
    Load state dict with filtering and key mapping.
    
    Args:
        model: Target model
        state_dict: Source state dict
        exclude_patterns: Patterns to exclude (regex)
        include_patterns: Only include matching patterns
        key_mapping: Dict mapping old keys to new keys
    
    Returns:
        Tuple of (loaded_keys, skipped_keys)
    """
    import re
    
    exclude_patterns = exclude_patterns or []
    include_patterns = include_patterns or ['.*']
    key_mapping = key_mapping or {}
    
    model_state = model.state_dict()
    loaded_keys = []
    skipped_keys = []
    
    for key, value in state_dict.items():
        # Apply key mapping
        target_key = key_mapping.get(key, key)
        
        # Check exclusion patterns
        excluded = any(re.match(p, target_key) for p in exclude_patterns)
        if excluded:
            skipped_keys.append((key, "excluded by pattern"))
            continue
        
        # Check inclusion patterns
        included = any(re.match(p, target_key) for p in include_patterns)
        if not included:
            skipped_keys.append((key, "not included by pattern"))
            continue
        
        # Check if key exists in model
        if target_key not in model_state:
            skipped_keys.append((key, "not in model"))
            continue
        
        # Check shape compatibility
        if model_state[target_key].shape != value.shape:
            skipped_keys.append((key, f"shape mismatch: {value.shape} vs {model_state[target_key].shape}"))
            continue
        
        # Copy value
        model_state[target_key] = value
        loaded_keys.append(target_key)
    
    # Load filtered state dict
    model.load_state_dict(model_state, strict=False)
    
    print(f"Loaded {len(loaded_keys)} parameters")
    if skipped_keys:
        print(f"Skipped {len(skipped_keys)} parameters:")
        for key, reason in skipped_keys[:5]:
            print(f"  {key}: {reason}")
        if len(skipped_keys) > 5:
            print(f"  ... and {len(skipped_keys) - 5} more")
    
    return loaded_keys, skipped_keys


# Example: Load backbone weights only (exclude classifier)
pretrained_state = torch.load('pretrained.pth')
load_partial_state_dict(
    model,
    pretrained_state,
    exclude_patterns=[r'fc\..*', r'classifier\..*']
)
```

### Handling Architecture Changes

```python
def adapt_state_dict_for_architecture(
    source_state: dict,
    target_model: nn.Module,
    adaptation_rules: dict = None
) -> dict:
    """
    Adapt state dict from one architecture to another.
    
    Args:
        source_state: Source model state dict
        target_model: Target model instance
        adaptation_rules: Custom adaptation functions per layer
    
    Returns:
        Adapted state dict compatible with target model
    """
    target_state = target_model.state_dict()
    adapted_state = {}
    adaptation_rules = adaptation_rules or {}
    
    for target_key, target_tensor in target_state.items():
        if target_key in source_state:
            source_tensor = source_state[target_key]
            
            if source_tensor.shape == target_tensor.shape:
                # Direct copy
                adapted_state[target_key] = source_tensor
            elif target_key in adaptation_rules:
                # Apply custom adaptation
                adapted_state[target_key] = adaptation_rules[target_key](
                    source_tensor, target_tensor.shape
                )
            else:
                # Attempt automatic adaptation
                adapted_state[target_key] = _auto_adapt_tensor(
                    source_tensor, target_tensor.shape
                )
        else:
            # Keep target initialization
            adapted_state[target_key] = target_tensor
            print(f"No source for {target_key}, keeping initialization")
    
    return adapted_state


def _auto_adapt_tensor(source: torch.Tensor, target_shape: tuple) -> torch.Tensor:
    """Automatically adapt tensor shape through truncation or padding."""
    result = torch.zeros(target_shape, dtype=source.dtype)
    
    # Calculate overlap region
    slices = tuple(
        slice(0, min(s, t))
        for s, t in zip(source.shape, target_shape)
    )
    
    # Copy overlapping region
    result[slices] = source[slices]
    
    return result
```

## Saving Fine-tuned Models

### Complete Fine-tuning Checkpoint

```python
from datetime import datetime

def save_finetuned_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    filepath: str,
    config: dict,
    base_model_info: dict = None,
    additional_state: dict = None
) -> None:
    """
    Save fine-tuned model with complete provenance information.
    
    Args:
        model: Fine-tuned model
        optimizer: Optimizer state
        filepath: Save path
        config: Training configuration
        base_model_info: Information about the base pre-trained model
        additional_state: Any additional state to save
    """
    checkpoint = {
        # Model state
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        
        # Fine-tuning configuration
        'config': config,
        'frozen_layers': _get_frozen_layers(model),
        
        # Provenance
        'base_model_info': base_model_info or {
            'name': 'unknown',
            'source': 'unknown'
        },
        
        # Metadata
        'pytorch_version': torch.__version__,
        'timestamp': datetime.now().isoformat(),
    }
    
    if additional_state:
        checkpoint['additional_state'] = additional_state
    
    torch.save(checkpoint, filepath)
    print(f"Fine-tuned model saved to {filepath}")


def _get_frozen_layers(model: nn.Module) -> list:
    """Get list of frozen layer names."""
    return [
        name for name, param in model.named_parameters()
        if not param.requires_grad
    ]


def load_finetuned_model(
    model_class: type,
    filepath: str,
    device: torch.device = None,
    restore_frozen: bool = True,
    **model_kwargs
) -> tuple:
    """
    Load fine-tuned model with full state restoration.
    
    Args:
        model_class: Model class to instantiate
        filepath: Checkpoint path
        device: Target device
        restore_frozen: Whether to restore frozen layer state
        **model_kwargs: Arguments for model instantiation
    
    Returns:
        Tuple of (model, checkpoint_data)
    """
    device = device or torch.device('cpu')
    checkpoint = torch.load(filepath, map_location=device)
    
    # Instantiate model
    model = model_class(**model_kwargs)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Restore frozen state if requested
    if restore_frozen and 'frozen_layers' in checkpoint:
        for name, param in model.named_parameters():
            if name in checkpoint['frozen_layers']:
                param.requires_grad = False
    
    print(f"Loaded fine-tuned model from {filepath}")
    print(f"  Base model: {checkpoint.get('base_model_info', {}).get('name', 'unknown')}")
    print(f"  Config: {checkpoint.get('config', {})}")
    
    return model, checkpoint
```

## Feature Extraction Mode

### Using Pre-trained Model as Feature Extractor

```python
class FeatureExtractor(nn.Module):
    """
    Wrapper to use pre-trained model as feature extractor.
    
    Removes the classification head and optionally extracts
    intermediate features from specified layers.
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        layer_names: list = None,
        pool_output: bool = True
    ):
        """
        Args:
            backbone: Pre-trained model
            layer_names: Names of layers to extract features from
            pool_output: Whether to apply global average pooling
        """
        super().__init__()
        self.backbone = backbone
        self.layer_names = layer_names or []
        self.pool_output = pool_output
        self._features = {}
        
        # Remove classifier
        if hasattr(self.backbone, 'fc'):
            self.backbone.fc = nn.Identity()
        
        # Register hooks for intermediate features
        if self.layer_names:
            self._register_hooks()
        
        # Freeze all parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def _register_hooks(self):
        """Register forward hooks for specified layers."""
        for name, module in self.backbone.named_modules():
            if name in self.layer_names:
                module.register_forward_hook(
                    lambda m, inp, out, name=name: self._features.update({name: out})
                )
    
    def forward(self, x: torch.Tensor) -> dict:
        """Extract features from input."""
        self._features = {}
        
        output = self.backbone(x)
        
        if self.pool_output and output.dim() == 4:
            output = nn.functional.adaptive_avg_pool2d(output, (1, 1))
            output = output.flatten(1)
        
        result = {'output': output}
        result.update(self._features)
        
        return result


# Example usage
backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
extractor = FeatureExtractor(
    backbone,
    layer_names=['layer3', 'layer4'],
    pool_output=True
)

# Extract features
x = torch.randn(4, 3, 224, 224)
features = extractor(x)
print(f"Output shape: {features['output'].shape}")
print(f"Layer3 shape: {features['layer3'].shape}")
print(f"Layer4 shape: {features['layer4'].shape}")
```

## Best Practices

### Transfer Learning Checklist

| Step | Action | Rationale |
|------|--------|-----------|
| 1 | Choose appropriate pre-trained model | Match domain and complexity |
| 2 | Modify architecture for target task | Adjust classifier head |
| 3 | Freeze backbone initially | Prevent destroying learned features |
| 4 | Train classifier with higher LR | Faster convergence for new layers |
| 5 | Gradually unfreeze backbone | Fine-tune from later to earlier layers |
| 6 | Use discriminative learning rates | Preserve early features, adapt later ones |
| 7 | Save complete checkpoint | Include frozen state and config |

### Common Pitfalls

1. **Forgetting to set eval mode** before extracting features
2. **Not normalizing inputs** according to pre-training statistics
3. **Using too high learning rate** for pre-trained layers
4. **Unfreezing too early** before classifier is trained
5. **Not saving frozen layer state** in checkpoints

## Summary

Transfer learning requires specialized save/load patterns:

1. **Partial loading**: Filter and map state dict keys for architecture changes
2. **Layer freezing**: Selectively disable gradient computation
3. **Discriminative fine-tuning**: Different learning rates per layer group
4. **Complete checkpointing**: Save frozen state and configuration for reproducibility

These patterns enable efficient adaptation of pre-trained models to new tasks while preserving valuable learned representations.

## References

- [PyTorch Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [ULMFiT Paper](https://arxiv.org/abs/1801.06146) - Gradual unfreezing methodology
- [torchvision Models](https://pytorch.org/vision/stable/models.html)
