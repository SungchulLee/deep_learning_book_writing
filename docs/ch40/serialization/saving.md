# Model Saving

## Overview

Model saving is the foundation of deployment—converting trained PyTorch models into persistent formats that can be loaded for inference, fine-tuning, or transfer across environments. PyTorch provides multiple serialization mechanisms, each suited to different deployment scenarios.

## PyTorch Native Saving

### Saving the Entire Model

The simplest approach saves the entire model object using Python's `pickle`:

```python
import torch
import torch.nn as nn

class AlphaModel(nn.Module):
    """Example model for quantitative finance alpha prediction."""
    
    def __init__(self, num_features: int = 50, hidden_size: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(num_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

model = AlphaModel(num_features=50, hidden_size=128)

# Save entire model (includes architecture + weights)
torch.save(model, 'alpha_model_full.pt')

# Load entire model
loaded_model = torch.load('alpha_model_full.pt')
loaded_model.eval()
```

**Advantages**: Simple, preserves everything including custom logic.

**Disadvantages**: Pickle is fragile—class definition must be importable at load time. If the source code changes (renamed module, moved file), loading fails. Not recommended for production.

### Safe Serialization with `weights_only`

Starting with PyTorch 2.0, `torch.load` supports `weights_only=True` to prevent arbitrary code execution from untrusted files:

```python
# Safe loading (recommended for production)
state_dict = torch.load('model.pt', weights_only=True)
```

This prevents pickle-based attacks where a malicious `.pt` file could execute arbitrary code during loading.

## Saving Metadata with Models

For production systems, save configuration alongside weights:

```python
import json
import os

def save_model_with_metadata(model, path, config, metrics=None):
    """Save model with reproducibility metadata."""
    os.makedirs(path, exist_ok=True)
    
    # Save weights
    torch.save(model.state_dict(), os.path.join(path, 'model.pt'))
    
    # Save config
    with open(os.path.join(path, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save metrics
    if metrics:
        with open(os.path.join(path, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)

def load_model_with_metadata(path, model_class):
    """Load model and metadata."""
    with open(os.path.join(path, 'config.json'), 'r') as f:
        config = json.load(f)
    
    model = model_class(**config)
    model.load_state_dict(
        torch.load(os.path.join(path, 'model.pt'), weights_only=True)
    )
    model.eval()
    
    return model, config

# Usage
config = {'num_features': 50, 'hidden_size': 128}
metrics = {'sharpe_ratio': 1.85, 'max_drawdown': -0.12}

save_model_with_metadata(model, 'model_v1/', config, metrics)
loaded_model, loaded_config = load_model_with_metadata('model_v1/', AlphaModel)
```

## Cross-Device Saving and Loading

Models trained on GPU must be mapped to the correct device when loaded:

```python
# Save GPU model
torch.save(model.state_dict(), 'model.pt')

# Load on CPU
model.load_state_dict(
    torch.load('model.pt', map_location='cpu', weights_only=True)
)

# Load on specific GPU
model.load_state_dict(
    torch.load('model.pt', map_location='cuda:0', weights_only=True)
)

# Load to available device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(
    torch.load('model.pt', map_location=device, weights_only=True)
)
```

## Format Comparison

| Format | Use Case | Python Required | Cross-Platform |
|--------|----------|----------------|----------------|
| `torch.save(model)` | Quick prototyping | Yes | No |
| `torch.save(state_dict)` | Standard training | Yes | No |
| TorchScript (`.pt`) | Production serving | No | Yes |
| ONNX (`.onnx`) | Cross-framework | No | Yes |
| SafeTensors | Secure sharing | Yes (loader) | Yes |

## Best Practices

- **Always save `state_dict`**, not the full model, for production workflows
- **Include metadata** (config, metrics, data version) alongside weights
- **Use `weights_only=True`** when loading untrusted model files
- **Specify `map_location`** explicitly to avoid device mismatch errors
- **Version your models** with timestamps or hash-based naming
- **Test loading** immediately after saving to catch serialization issues early

## References

1. PyTorch Serialization Documentation: https://pytorch.org/docs/stable/notes/serialization.html
2. PyTorch Save/Load Tutorial: https://pytorch.org/tutorials/beginner/saving_loading_models.html
