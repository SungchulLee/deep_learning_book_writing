# GIN

GIN was introduced in the 2019 paper "How Powerful are Graph Neural Networks?." - Use sum aggregation + MLP   - Update: h'_i = MLP( (1 + eps) * h_i + sum_{j in N(i)} h_j )   - Proven as powerful as the Weisfeiler-Lehman test (under assumptions).

This implementation provides a concise, educational reference for GIN. The code focuses on the core architecture and forward pass, making it straightforward to study the key design patterns and adapt them for experimentation.

## Code

```python
#!/usr/bin/env python3
"""
GIN - Graph Isomorphism Network
Paper: "How Powerful are Graph Neural Networks?" (2019)
Authors: Keyulu Xu et al.
Key idea:
  - Use sum aggregation + MLP
  - Update: h'_i = MLP( (1 + eps) * h_i + sum_{j in N(i)} h_j )
  - Proven as powerful as the Weisfeiler-Lehman test (under assumptions)

File: appendix/gnn/gin.py
Note: Educational implementation with dense adjacency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================================
# Main
# ========================================================================


class MLP(nn.Module):
    """Small MLP used inside GIN."""
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class GINLayer(nn.Module):
    """
    One GIN layer:
      h'_i = MLP( (1 + eps) * h_i + sum_{j in N(i)} h_j )

    eps can be fixed or learnable; here we make it learnable.
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = MLP(in_dim, out_dim, hidden_dim=2 * out_dim)

    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # For sum aggregation, we can use A @ X if A contains edges (optionally without self-loops)
        neigh_sum = A @ X  # (N, in_dim)
        out = (1.0 + self.eps) * X + neigh_sum
        return self.mlp(out)


class GIN(nn.Module):
    """2-layer GIN for node classification."""
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.gin1 = GINLayer(in_dim, hidden_dim)
        self.gin2 = GINLayer(hidden_dim, num_classes)

    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        H = F.relu(self.gin1(X, A))
        logits = self.gin2(H, A)
        return logits


if __name__ == "__main__":
    N, Fin, C = 4, 6, 3
    X = torch.randn(N, Fin)

    # Adjacency without self-loops is okay for GIN because (1+eps)*X provides self contribution
    A = torch.tensor([
        [0, 1, 1, 0],
        [1, 0, 1, 0],
        [1, 1, 0, 1],
        [0, 0, 1, 0],
    ], dtype=torch.float32)

    model = GIN(in_dim=Fin, hidden_dim=16, num_classes=C)
    logits = model(X, A)
    print("logits:", logits.shape)  # (4, 3)```

## Discussion

The implementation defines 3 classes (`MLP`, `GINLayer`, `GIN`) that work together to form the complete graph neural network architecture. Each class encapsulates a distinct component, making the code modular and easy to extend. The `forward` methods define the computational graph that PyTorch uses for automatic differentiation.

As a reference implementation, this code prioritizes clarity over optimization. Production systems would typically add mixed-precision training, distributed data parallelism, and more sophisticated data augmentation. Nevertheless, the core architectural ideas shown here remain the same regardless of scale.

## Exercises

**Exercise 1.**
Calculate the total number of learnable parameters in `MLP` with the default initialization. Break down the count by layer, including both weights and biases.

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
Extend `MLP` to support a configurable number of layers or blocks. Add a `num_layers` parameter to `__init__` and use `nn.ModuleList` to create a variable-depth architecture. Test with 2, 4, and 8 layers.

??? success "Solution to Exercise 4"
    Replace the hardcoded layers with:
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    In the `forward` method, iterate: `for layer in self.layers: x = layer(x)`. Using `nn.ModuleList` (not a plain Python list) ensures PyTorch registers all parameters for optimization. Test with: `for n in [2, 4, 8]: model = MLP(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.
