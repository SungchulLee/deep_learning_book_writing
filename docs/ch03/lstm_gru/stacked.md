# Stacked (Multi-layer) LSTM and GRU

## Introduction

Stacking multiple LSTM or GRU layers creates deep recurrent networks capable of learning hierarchical representations of sequential data.

## PyTorch Implementation

```python
import torch
import torch.nn as nn

# Multi-layer LSTM
lstm = nn.LSTM(input_size=32, hidden_size=64, num_layers=3, 
               batch_first=True, dropout=0.2)

x = torch.randn(16, 50, 32)
output, (h_n, c_n) = lstm(x)

print(f"Output: {output.shape}")   # (16, 50, 64) - from final layer only
print(f"Hidden: {h_n.shape}")      # (3, 16, 64) - one per layer
```

## Depth Guidelines

| Layers | Use Case |
|--------|----------|
| 1 | Simple tasks, limited data |
| 2 | Most applications (default) |
| 3-4 | Complex tasks, large data |

## Key Points

- Output is only from the final layer
- Hidden states available for all layers via `h_n[layer_idx]`
- Dropout applies between layers (requires num_layers > 1)
- Use `model.eval()` to disable dropout during inference

## References

1. Pascanu, R., et al. (2014). How to Construct Deep Recurrent Neural Networks.
