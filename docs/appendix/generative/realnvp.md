# RealNVP

RealNVP was introduced in the 2017 paper "Density Estimation using Real NVP." - Invertible transformations   - Exact likelihood via change-of-variables.

This implementation provides a concise, educational reference for RealNVP. The code focuses on the core architecture and forward pass, making it straightforward to study the key design patterns and adapt them for experimentation.

## Code

```python
#!/usr/bin/env python3
"""
RealNVP - Real-valued Non-Volume Preserving Flows
Paper: "Density Estimation using Real NVP" (2017)
Key idea:
  - Invertible transformations
  - Exact likelihood via change-of-variables

File: appendix/generative/realnvp.py
"""

import torch
import torch.nn as nn

# ========================================================================
# Main
# ========================================================================


class CouplingLayer(nn.Module):
    """Affine coupling layer."""
    def __init__(self, dim=784):
        super().__init__()
        self.scale = nn.Linear(dim // 2, dim // 2)
        self.shift = nn.Linear(dim // 2, dim // 2)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        s = self.scale(x1)
        t = self.shift(x1)
        y2 = x2 * torch.exp(s) + t
        return torch.cat([x1, y2], dim=1)


if __name__ == "__main__":
    pass```

## Discussion

The `CouplingLayer` class encapsulates the model architecture using PyTorch's `nn.Module` interface. The `forward` method defines the computational graph, allowing PyTorch's autograd system to handle gradient computation automatically during training. This modular design makes it straightforward to modify individual components or integrate the model into larger pipelines.

As a reference implementation, this code prioritizes clarity over optimization. Production systems would typically add mixed-precision training, distributed data parallelism, and more sophisticated data augmentation. Nevertheless, the core architectural ideas shown here remain the same regardless of scale.

## Exercises

**Exercise 1.**
Calculate the total number of learnable parameters in `CouplingLayer` with the default initialization. Break down the count by layer, including both weights and biases.

??? success "Solution to Exercise 1"
    For each `nn.Linear(in_features, out_features)`, there are `in_features * out_features` weight parameters plus `out_features` bias parameters (unless `bias=False`). For `nn.Conv2d(in_c, out_c, k)`, there are `in_c * out_c * k * k` weight parameters plus `out_c` bias parameters. For `nn.Embedding(num, dim)`, there are `num * dim` parameters. Sum across all layers. You can verify with `sum(p.numel() for p in model.parameters())`.

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
Extend `CouplingLayer` to support a configurable number of layers or blocks. Add a `num_layers` parameter to `__init__` and use `nn.ModuleList` to create a variable-depth architecture. Test with 2, 4, and 8 layers.

??? success "Solution to Exercise 4"
    Replace the hardcoded layers with:
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    In the `forward` method, iterate: `for layer in self.layers: x = layer(x)`. Using `nn.ModuleList` (not a plain Python list) ensures PyTorch registers all parameters for optimization. Test with: `for n in [2, 4, 8]: model = CouplingLayer(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.
