# State Dict

## Overview

The **state dictionary** (`state_dict`) is PyTorch's canonical approach to model persistence. It provides a lightweight, portable representation of a model's learnable parameters as a Python dictionary mapping layer names to tensor values. Understanding state dictionaries is fundamental to saving, loading, transferring, and debugging PyTorch models.

## Learning Objectives

By the end of this section, you will be able to:

- Explain what a state dictionary contains and its internal structure
- Save and load models using the state dictionary approach
- Understand the difference between complete model saving and state dict saving
- Handle device placement when loading state dictionaries
- Debug common state dictionary issues

## Mathematical Foundation

### Parameter Representation

A neural network $f_\theta(x)$ with parameters $\theta$ can be decomposed as:

$$\theta = \{\mathbf{W}^{(l)}, \mathbf{b}^{(l)}\}_{l=1}^{L}$$

where $\mathbf{W}^{(l)} \in \mathbb{R}^{n_{l} \times n_{l-1}}$ are weight matrices and $\mathbf{b}^{(l)} \in \mathbb{R}^{n_l}$ are bias vectors for each layer $l$.

The state dictionary provides a bijective mapping:

$$\text{state\_dict}: \theta \to \{(\text{name}_i, \mathbf{T}_i)\}_{i=1}^{|\theta|}$$

where each parameter tensor $\mathbf{T}_i$ is associated with a hierarchical name reflecting the model architecture.

### Serialization Protocol

PyTorch uses Python's pickle protocol for serialization, combined with custom handlers for tensor storage:

$$\text{serialize}(\text{state\_dict}) = \text{pickle}(\{k: \text{storage}(v) \mid (k, v) \in \text{state\_dict}\})$$

The storage format preserves:
- Tensor dtype ($\texttt{float32}$, $\texttt{float16}$, etc.)
- Tensor shape and stride information
- Device information (CPU/CUDA)
- Requires grad flag (though typically false in state dicts)

## Core API

### Essential Functions

PyTorch provides three essential functions for model persistence:

```python
# Save any Python object to disk
torch.save(obj, path)

# Load saved object from disk
obj = torch.load(path, map_location=device)

# Load parameters into a model
model.load_state_dict(state_dict, strict=True)
```

### State Dictionary Structure

```python
import torch
import torch.nn as nn

class SimpleNetwork(nn.Module):
    """Example network for state dict demonstration."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.bn1(self.fc1(x)))
        return self.fc2(x)

# Create model instance
model = SimpleNetwork(input_dim=784, hidden_dim=256, output_dim=10)

# Examine state dictionary
state_dict = model.state_dict()

print("State Dictionary Keys:")
for key in state_dict.keys():
    print(f"  {key}: {state_dict[key].shape}")
```

**Output:**
```
State Dictionary Keys:
  fc1.weight: torch.Size([256, 784])
  fc1.bias: torch.Size([256])
  bn1.weight: torch.Size([256])
  bn1.bias: torch.Size([256])
  bn1.running_mean: torch.Size([256])
  bn1.running_var: torch.Size([256])
  bn1.num_batches_tracked: torch.Size([])
  fc2.weight: torch.Size([10, 256])
  fc2.bias: torch.Size([10])
```

!!! note "Buffers vs Parameters"
    The state dictionary contains both **parameters** (learnable weights updated by gradient descent) and **buffers** (non-learnable tensors like BatchNorm running statistics). Use `model.named_parameters()` for only learnable parameters and `model.named_buffers()` for only buffers.

## Saving and Loading Methods

### Method 1: Save Complete Model (Not Recommended)

```python
# Save entire model (architecture + parameters)
torch.save(model, 'model_complete.pth')

# Load entire model
model = torch.load('model_complete.pth')
model.eval()
```

!!! warning "Limitations of Complete Model Saving"
    - **Larger file size**: Includes architecture definition
    - **Code dependency**: Model class must be available and unchanged when loading
    - **Version brittleness**: May break with PyTorch updates or code refactoring
    - **Not portable**: Tied to specific module paths

### Method 2: Save State Dict (Recommended)

```python
# Save only the state dictionary
torch.save(model.state_dict(), 'model_weights.pth')

# Load state dictionary into a new model instance
model = SimpleNetwork(input_dim=784, hidden_dim=256, output_dim=10)
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()
```

**Advantages:**
- Smaller file size (only parameters, no architecture)
- Portable across code versions
- Allows loading into modified architectures
- Industry standard for production deployment

## Device Management

### Cross-Device Loading

When models are trained on one device and deployed on another, explicit device mapping is required.

```python
# Scenario 1: GPU → CPU
# Model was trained on GPU, loading for CPU inference
device = torch.device('cpu')
state_dict = torch.load('model_weights.pth', map_location=device)
model = SimpleNetwork(784, 256, 10)
model.load_state_dict(state_dict)

# Scenario 2: CPU → GPU
# Model was trained on CPU, loading for GPU inference
device = torch.device('cuda:0')
state_dict = torch.load('model_weights.pth', map_location=device)
model = SimpleNetwork(784, 256, 10)
model.load_state_dict(state_dict)
model.to(device)

# Scenario 3: GPU:0 → GPU:1
# Model trained on one GPU, loading to different GPU
state_dict = torch.load('model_weights.pth', map_location='cuda:1')
model = SimpleNetwork(784, 256, 10)
model.load_state_dict(state_dict)
model.to('cuda:1')
```

### Memory-Efficient Loading

For large models, load to CPU first to avoid GPU memory issues:

```python
def load_model_safely(model_class, weights_path, device, *args, **kwargs):
    """Load model with memory-efficient device transfer."""
    # Always load to CPU first
    state_dict = torch.load(weights_path, map_location='cpu')
    
    # Create model on CPU
    model = model_class(*args, **kwargs)
    model.load_state_dict(state_dict)
    
    # Move to target device
    model.to(device)
    
    return model

# Usage
model = load_model_safely(
    SimpleNetwork,
    'model_weights.pth',
    device='cuda:0',
    input_dim=784,
    hidden_dim=256,
    output_dim=10
)
```

## State Dict Inspection Utilities

### Examining State Dictionary Contents

```python
def inspect_state_dict(state_dict: dict) -> None:
    """Display comprehensive state dictionary information."""
    print("=" * 70)
    print("STATE DICTIONARY INSPECTION")
    print("=" * 70)
    
    total_params = 0
    total_bytes = 0
    
    for name, tensor in state_dict.items():
        num_params = tensor.numel()
        num_bytes = tensor.element_size() * num_params
        total_params += num_params
        total_bytes += num_bytes
        
        print(f"\n{name}")
        print(f"  Shape: {list(tensor.shape)}")
        print(f"  Dtype: {tensor.dtype}")
        print(f"  Parameters: {num_params:,}")
        print(f"  Size: {num_bytes / 1024:.2f} KB")
        print(f"  Device: {tensor.device}")
        print(f"  Stats: min={tensor.min():.4f}, max={tensor.max():.4f}, mean={tensor.mean():.4f}")
    
    print("\n" + "=" * 70)
    print(f"Total Parameters: {total_params:,}")
    print(f"Total Size: {total_bytes / (1024 * 1024):.2f} MB")
    print("=" * 70)

# Usage
inspect_state_dict(model.state_dict())
```

### Comparing State Dictionaries

```python
def compare_state_dicts(dict1: dict, dict2: dict, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """Compare two state dictionaries for equality."""
    
    # Check keys match
    if set(dict1.keys()) != set(dict2.keys()):
        print("Key mismatch!")
        print(f"  Only in dict1: {set(dict1.keys()) - set(dict2.keys())}")
        print(f"  Only in dict2: {set(dict2.keys()) - set(dict1.keys())}")
        return False
    
    # Check values match
    all_match = True
    for key in dict1.keys():
        if not torch.allclose(dict1[key], dict2[key], rtol=rtol, atol=atol):
            print(f"Value mismatch at '{key}'")
            diff = (dict1[key] - dict2[key]).abs()
            print(f"  Max difference: {diff.max():.6e}")
            all_match = False
    
    if all_match:
        print("State dictionaries are identical")
    
    return all_match
```

## Handling State Dict Mismatches

### Strict vs Non-Strict Loading

```python
# Strict loading (default) - raises error on mismatch
model.load_state_dict(state_dict, strict=True)

# Non-strict loading - allows partial matches
missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

print(f"Missing keys: {missing_keys}")      # Keys in model but not in state_dict
print(f"Unexpected keys: {unexpected_keys}") # Keys in state_dict but not in model
```

### Filtering State Dictionary

```python
def filter_state_dict(state_dict: dict, model: nn.Module) -> dict:
    """Filter state dict to match model architecture."""
    model_keys = set(model.state_dict().keys())
    
    filtered = {}
    skipped = []
    
    for key, value in state_dict.items():
        if key in model_keys:
            model_shape = model.state_dict()[key].shape
            if value.shape == model_shape:
                filtered[key] = value
            else:
                skipped.append(f"{key}: shape mismatch {value.shape} vs {model_shape}")
        else:
            skipped.append(f"{key}: not in model")
    
    if skipped:
        print("Skipped entries:")
        for msg in skipped:
            print(f"  {msg}")
    
    return filtered

# Usage for transfer learning
pretrained_state = torch.load('pretrained_weights.pth')
filtered_state = filter_state_dict(pretrained_state, model)
model.load_state_dict(filtered_state, strict=False)
```

## Optimizer State Dictionary

Optimizers also maintain state (momentum buffers, adaptive learning rate accumulators, etc.) that should be saved for training resumption.

```python
import torch.optim as optim

model = SimpleNetwork(784, 256, 10)
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

# Simulate training steps to populate optimizer state
for _ in range(10):
    x = torch.randn(32, 784)
    loss = model(x).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Examine optimizer state
opt_state = optimizer.state_dict()

print("Optimizer State Dictionary Structure:")
print(f"  param_groups: {len(opt_state['param_groups'])} groups")
print(f"  state entries: {len(opt_state['state'])} parameters")

# Inspect one parameter's optimizer state (Adam)
param_id = list(opt_state['state'].keys())[0]
param_state = opt_state['state'][param_id]
print(f"\nPer-parameter state (Adam):")
for key, value in param_state.items():
    if isinstance(value, torch.Tensor):
        print(f"  {key}: {value.shape}")
    else:
        print(f"  {key}: {value}")
```

**Output:**
```
Optimizer State Dictionary Structure:
  param_groups: 1 groups
  state entries: 6 parameters

Per-parameter state (Adam):
  step: tensor(10)
  exp_avg: torch.Size([256, 784])
  exp_avg_sq: torch.Size([256, 784])
```

## Best Practices

### Recommended Save Pattern

```python
def save_training_state(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    loss: float,
    path: str,
    **kwargs
) -> None:
    """Save complete training state with metadata."""
    
    checkpoint = {
        # Core state
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        
        # Training progress
        'epoch': epoch,
        'loss': loss,
        
        # Metadata
        'pytorch_version': torch.__version__,
        'timestamp': datetime.now().isoformat(),
        
        # Additional custom data
        **kwargs
    }
    
    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path}")
```

### Recommended Load Pattern

```python
def load_training_state(
    model: nn.Module,
    optimizer: optim.Optimizer,
    path: str,
    device: torch.device = None
) -> dict:
    """Load complete training state with validation."""
    
    # Load with device mapping
    map_location = device if device else 'cpu'
    checkpoint = torch.load(path, map_location=map_location)
    
    # Restore model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Restore optimizer state
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Move to device if specified
    if device:
        model.to(device)
    
    print(f"Checkpoint loaded: epoch {checkpoint['epoch']}, loss {checkpoint['loss']:.4f}")
    
    return checkpoint
```

## Common Issues and Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| `RuntimeError: size mismatch` | Model architecture differs from saved weights | Use `strict=False` or ensure architectures match |
| `CUDA out of memory` | Loading directly to GPU | Use `map_location='cpu'` then `.to(device)` |
| `Missing keys` | Model has new layers not in checkpoint | Initialize new layers separately, load with `strict=False` |
| `Unexpected keys` | Checkpoint has extra parameters | Filter state dict or use `strict=False` |
| `module. prefix` | DataParallel wrapping mismatch | Strip or add `module.` prefix as needed |

## Summary

The state dictionary is PyTorch's preferred mechanism for model serialization because it:

1. **Separates architecture from parameters**: Enables architectural modifications without breaking saved weights
2. **Provides portability**: Works across PyTorch versions and platforms
3. **Supports partial loading**: Facilitates transfer learning and model surgery
4. **Maintains efficiency**: Stores only necessary data without code artifacts

Always prefer `model.state_dict()` over `torch.save(model, ...)` for production deployments.

## References

- [PyTorch Serialization Semantics](https://pytorch.org/docs/stable/notes/serialization.html)
- [Saving and Loading Models](https://pytorch.org/tutorials/beginner/saving_loading_models.html)
- [torch.save Documentation](https://pytorch.org/docs/stable/generated/torch.save.html)
