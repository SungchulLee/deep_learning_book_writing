# State Dict

## Overview

A state dict is an ordered dictionary mapping parameter names to tensors. It is PyTorch's recommended serialization format because it is portable, flexible, and independent of the model's code structure.

## Model State Dict

```python
# Inspect state dict
for name, param in model.state_dict().items():
    print(f"{name}: {param.shape}")

# Save
torch.save(model.state_dict(), 'model_state.pt')

# Load
model = MyModel()  # Must create model instance first
model.load_state_dict(torch.load('model_state.pt', weights_only=True))
model.eval()
```

## Optimizer State Dict

Optimizers also have state dicts containing momentum buffers, adaptive learning rate accumulators, and step counts:

```python
# Save optimizer state
torch.save(optimizer.state_dict(), 'optimizer_state.pt')

# Load optimizer state
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer.load_state_dict(torch.load('optimizer_state.pt', weights_only=True))
```

## Partial Loading

Load only a subset of parameters (useful for transfer learning):

```python
pretrained_dict = torch.load('pretrained.pt', weights_only=True)
model_dict = model.state_dict()

# Filter out keys that don't match
pretrained_dict = {k: v for k, v in pretrained_dict.items()
                   if k in model_dict and v.shape == model_dict[k].shape}

model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
```

## Strict Loading

By default, `load_state_dict` requires exact key matching. Set `strict=False` to allow mismatches:

```python
model.load_state_dict(state_dict, strict=False)
# Missing keys: parameters in model but not in state_dict (initialized randomly)
# Unexpected keys: parameters in state_dict but not in model (ignored)
```

## Key Takeaways

- State dicts are the portable, recommended serialization format.
- Both models and optimizers have state dicts.
- Partial loading enables transfer learning with mismatched architectures.
- Use `weights_only=True` for security when loading state dicts.
