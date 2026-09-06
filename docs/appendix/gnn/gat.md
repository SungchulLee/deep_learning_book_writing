# GAT

GAT was introduced in the 2018 paper "Graph Attention Networks." - Learn attention weights over neighbors instead of fixed normalization   - For node i: aggregate neighbors j with alpha_{ij} learned from features.

This implementation provides a concise, educational reference for GAT. The code focuses on the core architecture and forward pass, making it straightforward to study the key design patterns and adapt them for experimentation.

## Code

```python
#!/usr/bin/env python3
"""
GAT - Graph Attention Network
Paper: "Graph Attention Networks" (2018)
Authors: Petar Veličković et al.
Key idea:
  - Learn attention weights over neighbors instead of fixed normalization
  - For node i: aggregate neighbors j with alpha_{ij} learned from features

File: appendix/gnn/gat.py
Note: Educational implementation with dense adjacency, single-head attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================================
# Main
# ========================================================================


class GATLayer(nn.Module):
    """
    Single-head GAT layer (dense adjacency).

    Steps:
      1) Linear transform: Wh_i = W x_i
      2) Attention scores: e_{ij} = LeakyReLU(a^T [Wh_i || Wh_j])
      3) Mask non-edges
      4) alpha_{ij} = softmax_j(e_{ij})
      5) h'_i = sum_j alpha_{ij} Wh_j
    """
    def __init__(self, in_dim: int, out_dim: int, leaky_slope: float = 0.2):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)

        # a is split into two parts (equivalent to a^T [Wh_i || Wh_j])
        self.attn_l = nn.Linear(out_dim, 1, bias=False)
        self.attn_r = nn.Linear(out_dim, 1, bias=False)

        self.leaky_relu = nn.LeakyReLU(leaky_slope)

    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # X: (N, Fin), A: (N, N) adjacency (0/1)
        H = self.W(X)  # (N, Fout)

        # Compute attention logits efficiently:
        # e_{ij} = LeakyReLU( a_l(H_i) + a_r(H_j) )
        e_l = self.attn_l(H)  # (N, 1)
        e_r = self.attn_r(H)  # (N, 1)
        e = e_l + e_r.T        # (N, N)
        e = self.leaky_relu(e)

        # Mask out non-neighbors:
        # Use a very negative value so softmax ~ 0 for non-edges
        mask = (A == 0)
        e = e.masked_fill(mask, float("-inf"))

        # Normalize across neighbors j
        alpha = F.softmax(e, dim=1)  # (N, N)

        # Weighted sum of neighbor features
        H_out = alpha @ H  # (N, Fout)
        return H_out


class GAT(nn.Module):
    """2-layer GAT for node classification (single-head)."""
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.gat1 = GATLayer(in_dim, hidden_dim)
        self.gat2 = GATLayer(hidden_dim, num_classes)

    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        H = F.elu(self.gat1(X, A))
        logits = self.gat2(H, A)
        return logits


if __name__ == "__main__":
    N, Fin, C = 4, 8, 3
    X = torch.randn(N, Fin)
    A = torch.tensor([
        [1, 1, 1, 0],  # include self-loop in GAT for stability
        [1, 1, 1, 0],
        [1, 1, 1, 1],
        [0, 0, 1, 1],
    ], dtype=torch.float32)

    model = GAT(in_dim=Fin, hidden_dim=16, num_classes=C)
    logits = model(X, A)
    print("logits:", logits.shape)  # (4, 3)```

## Discussion

The implementation defines 2 classes (`GATLayer`, `GAT`) that work together to form the complete graph neural network architecture. Each class encapsulates a distinct component, making the code modular and easy to extend. The `forward` methods define the computational graph that PyTorch uses for automatic differentiation.

As a reference implementation, this code prioritizes clarity over optimization. Production systems would typically add mixed-precision training, distributed data parallelism, and more sophisticated data augmentation. Nevertheless, the core architectural ideas shown here remain the same regardless of scale.

## Exercises

**Exercise 1.**
Calculate the total number of learnable parameters in `GATLayer` with the default initialization. Break down the count by layer, including both weights and biases.

??? success "Solution to Exercise 1"
    For each `nn.Linear(in_features, out_features)`, there are `in_features * out_features` weight parameters plus `out_features` bias parameters (unless `bias=False`). For `nn.Conv2d(in_c, out_c, k)`, there are `in_c * out_c * k * k` weight parameters plus `out_c` bias parameters. For `nn.Embedding(num, dim)`, there are `num * dim` parameters. Sum across all layers. You can verify with `sum(p.numel() for p in model.parameters())`.

---

**Exercise 2.**
Add a dropout layer after the attention weights (before multiplying with values). Use a dropout rate of 0.1 during training. Explain why attention dropout helps with regularization.

??? success "Solution to Exercise 2"
    Add `self.attn_dropout = nn.Dropout(0.1)` in `__init__` and apply it after the softmax: `attn_weights = self.attn_dropout(F.softmax(scores, dim=-1))`. Attention dropout randomly zeroes some attention weights during training, preventing the model from relying too heavily on specific token-to-token relationships. This encourages the model to distribute attention more broadly and learn more robust representations, similar to how standard dropout prevents co-adaptation of neurons.

---

**Exercise 3.**
Explain the computational complexity of self-attention as a function of sequence length $n$ and model dimension $d$. Why does this motivate architectures like Longformer or Linformer for long sequences?

??? success "Solution to Exercise 3"
    Standard self-attention computes an $n \times n$ attention matrix, giving $O(n^2 d)$ time complexity and $O(n^2)$ memory for the attention weights. For long sequences (e.g., $n = 4096$), this becomes prohibitive. Longformer uses a combination of local sliding-window attention ($O(n \cdot w \cdot d)$ where $w$ is window size) and sparse global attention for selected tokens. Linformer projects keys and values to a lower dimension $k \ll n$, reducing complexity to $O(n \cdot k \cdot d)$. Both trade some expressiveness for practical efficiency on long inputs.

---

**Exercise 4.**
Extend `GATLayer` to support a configurable number of layers or blocks. Add a `num_layers` parameter to `__init__` and use `nn.ModuleList` to create a variable-depth architecture. Test with 2, 4, and 8 layers.

??? success "Solution to Exercise 4"
    Replace the hardcoded layers with:
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    In the `forward` method, iterate: `for layer in self.layers: x = layer(x)`. Using `nn.ModuleList` (not a plain Python list) ensures PyTorch registers all parameters for optimization. Test with: `for n in [2, 4, 8]: model = GATLayer(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.
