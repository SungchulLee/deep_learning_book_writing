# Save and Load Overview

## Overview

Model persistence—saving trained models and loading them later—is essential for deployment, checkpoint recovery, experiment reproducibility, and model sharing. PyTorch provides multiple serialization mechanisms, each with different tradeoffs.

## Approaches

| Approach | Saves | Portable | Use Case |
|---|---|---|---|
| State dict | Parameters only | Yes | Standard saving/loading |
| Full model (`torch.save(model)`) | Model + code | No | Quick prototyping |
| Checkpointing | State dict + optimizer + epoch | Yes | Training resumption |
| ONNX export | Computation graph | Cross-framework | Deployment |
| TorchScript | Graph + parameters | Cross-language | Production inference |

## Quick Reference

```python
# Save state dict (recommended)
torch.save(model.state_dict(), 'model.pt')

# Load state dict
model = MyModel()
model.load_state_dict(torch.load('model.pt', weights_only=True))
model.eval()

# Save full model (not recommended for production)
torch.save(model, 'full_model.pt')

# Load full model
model = torch.load('full_model.pt')
```

## Key Takeaways

- State dict saving is the recommended default for portability and reliability.
- Full model saving is convenient but fragile (depends on exact code structure).
- Checkpointing adds optimizer state for training resumption.
- ONNX and TorchScript target deployment scenarios.
