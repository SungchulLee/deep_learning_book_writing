# Modules and

Modern neural network libraries (e.g. PyTorch) organize models using **modules** that encapsulate parameters and computation.

---

## What is a module?

A module is:
- a callable object,
- containing parameters,
- possibly composed of submodules.

```python
import torch.nn as nn

class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)
```

---

## Parameters

Parameters are tensors registered inside a module:

```python
for p in model.parameters():
    print(p.shape)
```

They:
- have `requires_grad=True`,
- are optimized during training.

---

## Submodules and

Modules can contain other modules:

```python
self.net = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 1),
)
```

This enables hierarchical model design.

---

## Why modules matter

Modules provide:
- parameter management,
- clean model definitions,
- integration with optimizers.

---

## Key takeaways

- Modules encapsulate computation and parameters.
- Parameters are tensors tracked by autograd.
- Composition enables scalable architectures.
