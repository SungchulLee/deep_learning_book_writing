# State Dict

## Overview

The `state_dict` is PyTorch's canonical mechanism for serializing model parameters. It is a Python dictionary that maps each layer's name to its parameter tensor. Understanding `state_dict` is essential for model saving, transfer learning, checkpoint management, and debugging parameter mismatches during deployment.

## What is a State Dict?

A `state_dict` is an `OrderedDict` mapping parameter names (strings) to tensors:

```python
import torch
import torch.nn as nn

class FactorModel(nn.Module):
    def __init__(self, num_factors: int = 10, hidden: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_factors, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden, 1)
    
    def forward(self, x):
        return self.head(self.encoder(x))

model = FactorModel(num_factors=10, hidden=64)

# Inspect state_dict
for name, tensor in model.state_dict().items():
    print(f"{name:40s} {str(tensor.shape):20s} {tensor.dtype}")
```

Output:
```
encoder.0.weight                         torch.Size([64, 10])  torch.float32
encoder.0.bias                           torch.Size([64])      torch.float32
encoder.2.weight                         torch.Size([64, 64])  torch.float32
encoder.2.bias                           torch.Size([64])      torch.float32
encoder.3.weight                         torch.Size([64])      torch.float32
encoder.3.bias                           torch.Size([64])      torch.float32
encoder.3.running_mean                   torch.Size([64])      torch.float32
encoder.3.running_var                    torch.Size([64])      torch.float32
encoder.3.num_batches_tracked            torch.Size([])        torch.int64
head.weight                              torch.Size([1, 64])   torch.float32
head.bias                                torch.Size([1])       torch.float32
```

Note that `state_dict` includes both **parameters** (learnable weights) and **buffers** (non-learnable state like BatchNorm running statistics).

## Save and Load State Dict

```python
# Save
torch.save(model.state_dict(), 'factor_model.pt')

# Load
model = FactorModel(num_factors=10, hidden=64)
model.load_state_dict(
    torch.load('factor_model.pt', weights_only=True)
)
model.eval()
```

## Strict vs Non-Strict Loading

By default, `load_state_dict` requires exact key matching. Use `strict=False` for partial loading:

```python
# Strict loading (default) - all keys must match
model.load_state_dict(state_dict, strict=True)

# Non-strict loading - allows missing/unexpected keys
missing, unexpected = model.load_state_dict(state_dict, strict=False)
print(f"Missing keys: {missing}")
print(f"Unexpected keys: {unexpected}")
```

**Common use cases for non-strict loading:**

- Transfer learning (loading a backbone without the classification head)
- Architecture evolution (adding new layers to an existing model)
- Loading partial checkpoints

## State Dict Manipulation

### Filtering and Renaming Keys

```python
def load_backbone_weights(model, pretrained_path):
    """Load only backbone weights, ignoring head."""
    pretrained = torch.load(pretrained_path, weights_only=True)
    
    # Filter keys
    backbone_state = {
        k: v for k, v in pretrained.items() 
        if k.startswith('encoder.')
    }
    
    model.load_state_dict(backbone_state, strict=False)
    return model

def rename_state_dict_keys(state_dict, old_prefix, new_prefix):
    """Rename keys when architecture changes."""
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith(old_prefix):
            new_key = new_prefix + k[len(old_prefix):]
            new_state[new_key] = v
        else:
            new_state[k] = v
    return new_state
```

### Comparing State Dicts

```python
def compare_state_dicts(sd1, sd2, rtol=1e-5, atol=1e-8):
    """Compare two state dicts for equality."""
    keys1, keys2 = set(sd1.keys()), set(sd2.keys())
    
    if keys1 != keys2:
        print(f"Missing in sd2: {keys1 - keys2}")
        print(f"Extra in sd2: {keys2 - keys1}")
        return False
    
    all_close = True
    for key in keys1:
        if not torch.allclose(sd1[key].float(), sd2[key].float(), 
                             rtol=rtol, atol=atol):
            diff = (sd1[key].float() - sd2[key].float()).abs().max()
            print(f"Mismatch in {key}: max_diff={diff:.2e}")
            all_close = False
    
    return all_close
```

## Optimizer State Dict

Optimizers also have state dicts, which store momentum buffers and learning rate schedules:

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Save both model and optimizer
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': 50,
}, 'checkpoint.pt')

# Load both
checkpoint = torch.load('checkpoint.pt', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
```

## Best Practices

- **Always save `state_dict()`**, not the model object, for long-term storage
- **Use `strict=True`** (default) unless you have a specific reason for partial loading
- **Log missing/unexpected keys** when using `strict=False` to catch silent failures
- **Compare state dicts** before and after save/load to validate round-trip fidelity
- **Save optimizer state** alongside model state for training resumption
- **Include metadata** (epoch, metrics, config) in checkpoint dictionaries

## References

1. PyTorch State Dict Documentation: https://pytorch.org/tutorials/beginner/saving_loading_models.html
2. PyTorch Serialization Semantics: https://pytorch.org/docs/stable/notes/serialization.html
