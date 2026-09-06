# GraphSAGE

GraphSAGE was introduced in the 2017 paper "Inductive Representation Learning on Large Graphs." - Sample and aggregate neighbors   - Inductive: can generalize to unseen nodes/graphs   - Typical update: h'_i = sigma( W [h_i || AGG({h_j, j in N(i)})] ).

This implementation provides a concise, educational reference for GraphSAGE. The code focuses on the core architecture and forward pass, making it straightforward to study the key design patterns and adapt them for experimentation.

## Code

```python
#!/usr/bin/env python3
"""
GraphSAGE - Inductive Representation Learning on Large Graphs
Paper: "Inductive Representation Learning on Large Graphs" (2017)
Authors: Will Hamilton, Zhitao Ying, Jure Leskovec
Key idea:
  - Sample and aggregate neighbors
  - Inductive: can generalize to unseen nodes/graphs
  - Typical update: h'_i = sigma( W [h_i || AGG({h_j, j in N(i)})] )

File: appendix/gnn/graphsage.py
Note: Educational implementation using mean aggregation with dense adjacency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================================
# Main
# ========================================================================


class MeanAggregator(nn.Module):
    """Compute mean of neighbor features."""
    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # A: (N, N) adjacency (0/1), assume includes self-loops if desired
        deg = A.sum(dim=1, keepdim=True).clamp(min=1.0)  # avoid divide-by-zero
        neigh_mean = (A @ X) / deg
        return neigh_mean


class GraphSAGELayer(nn.Module):
    """
    One GraphSAGE layer with mean aggregation.

    h'_i = sigma( W [h_i || mean_{j in N(i)} h_j] )
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.agg = MeanAggregator()
        self.lin = nn.Linear(2 * in_dim, out_dim)

    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        neigh = self.agg(X, A)                    # (N, in_dim)
        h_cat = torch.cat([X, neigh], dim=1)      # (N, 2*in_dim)
        return self.lin(h_cat)                    # (N, out_dim)


class GraphSAGE(nn.Module):
    """2-layer GraphSAGE for node classification."""
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.sage1 = GraphSAGELayer(in_dim, hidden_dim)
        self.sage2 = GraphSAGELayer(hidden_dim, num_classes)

    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        H = F.relu(self.sage1(X, A))
        logits = self.sage2(H, A)
        return logits


if __name__ == "__main__":
    N, Fin, C = 5, 8, 4
    X = torch.randn(N, Fin)

    # Include self-loops to make neighbor mean include node itself (common trick)
    A = torch.tensor([
        [1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1],
    ], dtype=torch.float32)

    model = GraphSAGE(in_dim=Fin, hidden_dim=16, num_classes=C)
    logits = model(X, A)
    print("logits:", logits.shape)  # (5, 4)```

## Discussion

The implementation defines 3 classes (`MeanAggregator`, `GraphSAGELayer`, `GraphSAGE`) that work together to form the complete graph neural network architecture. Each class encapsulates a distinct component, making the code modular and easy to extend. The `forward` methods define the computational graph that PyTorch uses for automatic differentiation.

As a reference implementation, this code prioritizes clarity over optimization. Production systems would typically add mixed-precision training, distributed data parallelism, and more sophisticated data augmentation. Nevertheless, the core architectural ideas shown here remain the same regardless of scale.

## Exercises

**Exercise 1.**
Calculate the total number of learnable parameters in `MeanAggregator` with the default initialization. Break down the count by layer, including both weights and biases.

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
Extend `MeanAggregator` to support a configurable number of layers or blocks. Add a `num_layers` parameter to `__init__` and use `nn.ModuleList` to create a variable-depth architecture. Test with 2, 4, and 8 layers.

??? success "Solution to Exercise 4"
    Replace the hardcoded layers with:
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    In the `forward` method, iterate: `for layer in self.layers: x = layer(x)`. Using `nn.ModuleList` (not a plain Python list) ensures PyTorch registers all parameters for optimization. Test with: `for n in [2, 4, 8]: model = MeanAggregator(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.
