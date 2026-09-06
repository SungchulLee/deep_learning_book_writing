# MPNN

MPNN was introduced in the 2017 paper "Neural Message Passing for Quantum Chemistry." - Separate *message* function and *update* function   - Iterative propagation for T steps:       m_i^{t+1} = sum_{j in N(i)} M(h_i^t, h_j^t, e_{ij})       h_i^{t+1} = U(h_i^t, m_i^{t+1}).

This implementation provides a concise, educational reference for MPNN. The code focuses on the core architecture and forward pass, making it straightforward to study the key design patterns and adapt them for experimentation.

## Code

```python
#!/usr/bin/env python3
"""
MPNN - Message Passing Neural Network (General Framework)
Paper: "Neural Message Passing for Quantum Chemistry" (2017)
Authors: Justin Gilmer et al.
Key idea:
  - Separate *message* function and *update* function
  - Iterative propagation for T steps:
      m_i^{t+1} = sum_{j in N(i)} M(h_i^t, h_j^t, e_{ij})
      h_i^{t+1} = U(h_i^t, m_i^{t+1})

File: appendix/gnn/mpnn.py
Note: Educational implementation with:
  - Dense adjacency
  - Optional edge features matrix E (N, N, E_dim)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================================
# Main
# ========================================================================


class MessageFn(nn.Module):
    """
    Message function M(h_i, h_j, e_ij).
    Here: concatenate and pass through an MLP.
    """
    def __init__(self, node_dim: int, edge_dim: int, msg_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, msg_dim),
            nn.ReLU(inplace=True),
            nn.Linear(msg_dim, msg_dim),
        )

    def forward(self, h_i, h_j, e_ij):
        # h_i, h_j: (msg_dim?) but here node_dim
        # e_ij: (edge_dim)
        x = torch.cat([h_i, h_j, e_ij], dim=-1)
        return self.net(x)


class UpdateFn(nn.Module):
    """
    Update function U(h_i, m_i).
    Here: GRUCell-like update (simple and common).
    """
    def __init__(self, node_dim: int, msg_dim: int):
        super().__init__()
        self.gru = nn.GRUCell(input_size=msg_dim, hidden_size=node_dim)

    def forward(self, h_i, m_i):
        # m_i: (node_dim?) here msg_dim
        return self.gru(m_i, h_i)


class MPNN(nn.Module):
    """
    Generic message passing network.

    Inputs:
      X: (N, node_dim) node features
      A: (N, N) adjacency (0/1)
      E: (N, N, edge_dim) edge features (optional; if None, use zeros)

    Output:
      H: (N, node_dim) updated node representations after T steps
    """
    def __init__(self, node_dim: int, edge_dim: int = 0, msg_dim: int = 64, T: int = 3):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.msg_dim = msg_dim
        self.T = T

        self.message = MessageFn(node_dim, edge_dim, msg_dim)
        self.update = UpdateFn(node_dim, msg_dim)

    def forward(self, X: torch.Tensor, A: torch.Tensor, E: torch.Tensor | None = None):
        N = X.size(0)
        device = X.device

        # If no edge features are provided, treat edges as having empty/zero features
        if E is None:
            E = torch.zeros(N, N, self.edge_dim, device=device)

        H = X  # current node states (N, node_dim)

        # Perform T rounds of message passing
        for _ in range(self.T):
            messages = []

            # For each node i, aggregate messages from neighbors j
            for i in range(N):
                m_i_list = []

                for j in range(N):
                    # Only send message if edge exists (A[i, j] == 1)
                    if A[i, j] > 0:
                        m_ij = self.message(H[i], H[j], E[i, j])  # (msg_dim,)
                        m_i_list.append(m_ij)

                # Sum aggregation (common in MPNN)
                if len(m_i_list) == 0:
                    m_i = torch.zeros(self.msg_dim, device=device)
                else:
                    m_i = torch.stack(m_i_list, dim=0).sum(dim=0)

                messages.append(m_i)

            M = torch.stack(messages, dim=0)  # (N, msg_dim)

            # Update node states using update function
            H = self.update(H, M)  # (N, node_dim)

        return H


if __name__ == "__main__":
    N, node_dim, edge_dim = 4, 8, 3
    X = torch.randn(N, node_dim)

    A = torch.tensor([
        [0, 1, 1, 0],
        [1, 0, 1, 0],
        [1, 1, 0, 1],
        [0, 0, 1, 0],
    ], dtype=torch.float32)

    E = torch.randn(N, N, edge_dim)  # random edge features (only used where A=1)

    model = MPNN(node_dim=node_dim, edge_dim=edge_dim, msg_dim=16, T=2)
    H = model(X, A, E)
    print("H:", H.shape)  # (4, 8)```

## Discussion

The implementation defines 3 classes (`MessageFn`, `UpdateFn`, `MPNN`) that work together to form the complete graph neural network architecture. Each class encapsulates a distinct component, making the code modular and easy to extend. The `forward` methods define the computational graph that PyTorch uses for automatic differentiation.

As a reference implementation, this code prioritizes clarity over optimization. Production systems would typically add mixed-precision training, distributed data parallelism, and more sophisticated data augmentation. Nevertheless, the core architectural ideas shown here remain the same regardless of scale.

## Exercises

**Exercise 1.**
Calculate the total number of learnable parameters in `MessageFn` with the default initialization. Break down the count by layer, including both weights and biases.

??? success "Solution to Exercise 1"
    For each `nn.Linear(in_features, out_features)`, there are `in_features * out_features` weight parameters plus `out_features` bias parameters (unless `bias=False`). For `nn.Conv2d(in_c, out_c, k)`, there are `in_c * out_c * k * k` weight parameters plus `out_c` bias parameters. For `nn.Embedding(num, dim)`, there are `num * dim` parameters. Sum across all layers. You can verify with `sum(p.numel() for p in model.parameters())`.

---

**Exercise 2.**
Add input validation to the main function or class to check that inputs have the expected shape and dtype. Raise informative error messages for invalid inputs.

??? success "Solution to Exercise 2"
    At the start of the `forward` method (or relevant function), add checks like: `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'` and `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'`. For shape validation, check critical dimensions: `B, C, H, W = x.shape; assert C == self.expected_channels`. Informative error messages significantly speed up debugging and make the code more robust for reuse.

---

**Exercise 3.**
Compare the number of parameters in an LSTM cell versus a GRU cell with the same hidden size $h$ and input size $x$. Which has fewer parameters and why?

??? success "Solution to Exercise 3"
    An LSTM has 4 gates (input, forget, cell, output), each with weight matrices for both input and hidden state: $4 \times (x \cdot h + h \cdot h + h) = 4(xh + h^2 + h)$ parameters. A GRU has 3 gates (reset, update, new): $3 \times (x \cdot h + h \cdot h + h) = 3(xh + h^2 + h)$ parameters. The GRU has 75% of the LSTM's parameters because it uses 3 gates instead of 4 and merges the cell and hidden state. In practice, GRUs often perform comparably to LSTMs despite fewer parameters.

---

**Exercise 4.**
Extend `MessageFn` to support a configurable number of layers or blocks. Add a `num_layers` parameter to `__init__` and use `nn.ModuleList` to create a variable-depth architecture. Test with 2, 4, and 8 layers.

??? success "Solution to Exercise 4"
    Replace the hardcoded layers with:
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    In the `forward` method, iterate: `for layer in self.layers: x = layer(x)`. Using `nn.ModuleList` (not a plain Python list) ensures PyTorch registers all parameters for optimization. Test with: `for n in [2, 4, 8]: model = MessageFn(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.
