# GCN

GCN was introduced in the 2017 paper "Semi-Supervised Classification with Graph Convolutional Networks." - Node features are updated by aggregating (normalized) neighbor features   - Uses normalized adjacency:  D^{-1/2} (A + I) D^{-1/2}.

This implementation provides a concise, educational reference for GCN. The code focuses on the core architecture and forward pass, making it straightforward to study the key design patterns and adapt them for experimentation.

## Code

```python
#!/usr/bin/env python3
"""
GCN - Graph Convolutional Network
Paper: "Semi-Supervised Classification with Graph Convolutional Networks" (2017)
Authors: Thomas N. Kipf, Max Welling
Key idea:
  - Node features are updated by aggregating (normalized) neighbor features
  - Uses normalized adjacency:  D^{-1/2} (A + I) D^{-1/2}

File: appendix/gnn/gcn.py
Note: Educational implementation using dense adjacency for clarity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================================
# Main
# ========================================================================


def normalize_adjacency(A: torch.Tensor) -> torch.Tensor:
    """
    Compute normalized adjacency:  D^{-1/2} (A + I) D^{-1/2}

    A: (N, N) adjacency matrix (0/1 or weighted)
    Returns:
      A_norm: (N, N)
    """
    N = A.size(0)

    # Add self-loops: A_hat = A + I
    A_hat = A + torch.eye(N, device=A.device)

    # Degree matrix: D_hat[i] = sum_j A_hat[i, j]
    deg = A_hat.sum(dim=1)  # (N,)

    # D^{-1/2}: careful about division by zero
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    # Normalize: D^{-1/2} A_hat D^{-1/2}
    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt
    return A_norm


class GCNLayer(nn.Module):
    """
    One GCN layer:
      H^{(l+1)} = sigma( A_norm H^{(l)} W )

    Where:
      - H^{(l)} is node feature matrix (N, Fin)
      - W is learnable weight (Fin, Fout)
      - A_norm is normalized adjacency (N, N)
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, X: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        # Multiply features by weight, then propagate via graph structure
        return A_norm @ self.lin(X)  # (N, out_dim)


class GCN(nn.Module):
    """
    A simple 2-layer GCN for node classification.

    Inputs:
      X: (N, Fin) node features
      A: (N, N) adjacency
    Output:
      logits: (N, num_classes)
    """
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.gcn1 = GCNLayer(in_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, num_classes)

    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        A_norm = normalize_adjacency(A)
        H = F.relu(self.gcn1(X, A_norm))
        logits = self.gcn2(H, A_norm)
        return logits


if __name__ == "__main__":
    # Toy example with 4 nodes
    N, Fin, C = 4, 8, 3
    X = torch.randn(N, Fin)

    # Simple undirected adjacency
    A = torch.tensor([
        [0, 1, 1, 0],
        [1, 0, 1, 0],
        [1, 1, 0, 1],
        [0, 0, 1, 0],
    ], dtype=torch.float32)

    model = GCN(in_dim=Fin, hidden_dim=16, num_classes=C)
    logits = model(X, A)
    print("logits:", logits.shape)  # (4, 3)```

## Discussion

The implementation defines 2 classes (`GCNLayer`, `GCN`) that work together to form the complete graph neural network architecture. Each class encapsulates a distinct component, making the code modular and easy to extend. The `forward` methods define the computational graph that PyTorch uses for automatic differentiation.

As a reference implementation, this code prioritizes clarity over optimization. Production systems would typically add mixed-precision training, distributed data parallelism, and more sophisticated data augmentation. Nevertheless, the core architectural ideas shown here remain the same regardless of scale.

## Exercises

**Exercise 1.**
Calculate the total number of learnable parameters in `GCNLayer` with the default initialization. Break down the count by layer, including both weights and biases.

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
Extend `GCNLayer` to support a configurable number of layers or blocks. Add a `num_layers` parameter to `__init__` and use `nn.ModuleList` to create a variable-depth architecture. Test with 2, 4, and 8 layers.

??? success "Solution to Exercise 4"
    Replace the hardcoded layers with:
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    In the `forward` method, iterate: `for layer in self.layers: x = layer(x)`. Using `nn.ModuleList` (not a plain Python list) ensures PyTorch registers all parameters for optimization. Test with: `for n in [2, 4, 8]: model = GCNLayer(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.
