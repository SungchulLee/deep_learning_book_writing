# Normalization Layers

Normalization Layers - Common variants used in deep learning Includes:

This implementation provides a concise, educational reference for Normalization Layers. The code focuses on the core architecture and forward pass, making it straightforward to study the key design patterns and adapt them for experimentation.

## Code

```python
#!/usr/bin/env python3
"""
Normalization Layers - Common variants used in deep learning
Includes:
  - BatchNorm1d/2d (PyTorch builtin wrapper)
  - LayerNorm (PyTorch builtin wrapper)
  - GroupNorm (PyTorch builtin wrapper)
  - RMSNorm (commonly used in modern LLMs)

File: appendix/utils/normalization.py
Note: Educational, with comments explaining when to use each.
"""

import torch
import torch.nn as nn

# ========================================================================
# Main
# ========================================================================


class RMSNorm(nn.Module):
    """
    RMSNorm (Root Mean Square Layer Normalization)

    Unlike LayerNorm, RMSNorm:
      - does NOT subtract mean
      - only divides by RMS (sqrt(mean(x^2)))

    Often used in LLaMA-like models.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: (..., dim)
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


# Note:
# - BatchNorm, LayerNorm, GroupNorm are in PyTorch.
# - You typically import and use:
#     nn.BatchNorm2d(C)
#     nn.LayerNorm(D)
#     nn.GroupNorm(num_groups, C)
#
# This file adds RMSNorm plus short usage notes.


if __name__ == "__main__":
    pass```

## Discussion

The `RMSNorm` class encapsulates the model architecture using PyTorch's `nn.Module` interface. The `forward` method defines the computational graph, allowing PyTorch's autograd system to handle gradient computation automatically during training. This modular design makes it straightforward to modify individual components or integrate the model into larger pipelines.

As a reference implementation, this code prioritizes clarity over optimization. Production systems would typically add mixed-precision training, distributed data parallelism, and more sophisticated data augmentation. Nevertheless, the core architectural ideas shown here remain the same regardless of scale.

## Exercises

**Exercise 1.**
Read through the code and identify the key design decisions. List three specific implementation choices and explain why each is appropriate for utility module.

??? success "Solution to Exercise 1"
    Design decisions vary by implementation but commonly include: (1) choice of activation functions -- ReLU variants provide non-saturating gradients for faster training; (2) normalization strategy -- batch normalization stabilizes training by reducing internal covariate shift; (3) residual connections -- when present, they enable gradient flow in deep networks by providing skip paths. Each choice reflects a trade-off between expressiveness, computational cost, and training stability.

---

**Exercise 2.**
Add input validation to the main function or class to check that inputs have the expected shape and dtype. Raise informative error messages for invalid inputs.

??? success "Solution to Exercise 2"
    At the start of the `forward` method (or relevant function), add checks like: `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'` and `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'`. For shape validation, check critical dimensions: `B, C, H, W = x.shape; assert C == self.expected_channels`. Informative error messages significantly speed up debugging and make the code more robust for reuse.

---

**Exercise 3.**
Describe two potential failure modes of this implementation and explain how you would diagnose and fix each one.

??? success "Solution to Exercise 3"
    Common failure modes include: (1) **Vanishing/exploding gradients** -- diagnosed by monitoring gradient norms (`torch.nn.utils.clip_grad_norm_` or logging `param.grad.norm()` per layer). Fix with gradient clipping, better initialization (Xavier/Kaiming), or architectural changes (residual connections, normalization). (2) **Overfitting** -- diagnosed when training loss decreases but validation loss increases. Fix with regularization (dropout, weight decay, data augmentation) or reducing model capacity. Always monitor both training and validation metrics to catch these issues early.

---

**Exercise 4.**
Extend `RMSNorm` to support a configurable number of layers or blocks. Add a `num_layers` parameter to `__init__` and use `nn.ModuleList` to create a variable-depth architecture. Test with 2, 4, and 8 layers.

??? success "Solution to Exercise 4"
    Replace the hardcoded layers with:
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    In the `forward` method, iterate: `for layer in self.layers: x = layer(x)`. Using `nn.ModuleList` (not a plain Python list) ensures PyTorch registers all parameters for optimization. Test with: `for n in [2, 4, 8]: model = RMSNorm(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.
