# Vanilla RNN

Vanilla RNN - Recurrent Neural Network (Elman RNN) Classic idea: maintain a hidden state that is updated sequentially over time.

This implementation provides a concise, educational reference for Vanilla RNN. The code focuses on the core architecture and forward pass, making it straightforward to study the key design patterns and adapt them for experimentation.

## Code

```python
#!/usr/bin/env python3
"""
Vanilla RNN - Recurrent Neural Network (Elman RNN)
Classic idea: maintain a hidden state that is updated sequentially over time.

Reference: "Finding Structure in Time" (1990), Jeffrey L. Elman (popularized simple RNN)
Key: h_t = tanh(W_x x_t + W_h h_{t-1} + b)

File: appendix/sequence/rnn.py
Note: Educational, fully commented implementation (single-layer, batch-first).
"""

import torch
import torch.nn as nn

# ========================================================================
# Main
# ========================================================================


class RNNCell(nn.Module):
    """
    A single vanilla RNN cell (one time step).

    Shapes:
      x_t     : (B, input_size)
      h_prev  : (B, hidden_size)
      h_t     : (B, hidden_size)

    Update:
      h_t = tanh( W_x * x_t + W_h * h_{t-1} + b )
    """
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Linear layers to transform input and previous hidden state
        self.Wx = nn.Linear(input_size, hidden_size, bias=True)
        self.Wh = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        # Combine transformed input and hidden state, then apply tanh nonlinearity
        h_t = torch.tanh(self.Wx(x_t) + self.Wh(h_prev))
        return h_t


class RNN(nn.Module):
    """
    Vanilla RNN (manual unroll across time).

    Input:
      x : (B, T, input_size)  batch-first sequence
    Output:
      y : (B, T, hidden_size) all hidden states across time
      h_T : (B, hidden_size) final hidden state
    """
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.cell = RNNCell(input_size, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor, h0: torch.Tensor | None = None):
        # Extract batch size and time length
        B, T, _ = x.shape
        device = x.device

        # Initialize hidden state (zeros) if not provided
        if h0 is None:
            h_t = torch.zeros(B, self.hidden_size, device=device)
        else:
            h_t = h0

        outputs = []
        for t in range(T):
            # Take input for time step t
            x_t = x[:, t, :]               # (B, input_size)

            # Update hidden state using the RNN cell
            h_t = self.cell(x_t, h_t)      # (B, hidden_size)

            # Store hidden state for this time step
            outputs.append(h_t)

        # Stack hidden states across time into a single tensor
        y = torch.stack(outputs, dim=1)     # (B, T, hidden_size)
        return y, h_t


if __name__ == "__main__":
    # Quick sanity check: run a forward pass and print shapes
    model = RNN(input_size=8, hidden_size=16)
    x = torch.randn(2, 5, 8)     # (B=2, T=5, input=8)
    y, hT = model(x)

    print("y :", y.shape)        # expected (2, 5, 16)
    print("hT:", hT.shape)       # expected (2, 16)```

## Discussion

The implementation defines 2 classes (`RNNCell`, `RNN`) that work together to form the complete sequence model architecture. Each class encapsulates a distinct component, making the code modular and easy to extend. The `forward` methods define the computational graph that PyTorch uses for automatic differentiation.

As a reference implementation, this code prioritizes clarity over optimization. Production systems would typically add mixed-precision training, distributed data parallelism, and more sophisticated data augmentation. Nevertheless, the core architectural ideas shown here remain the same regardless of scale.

## Exercises

**Exercise 1.**
Calculate the total number of learnable parameters in `RNNCell` with the default initialization. Break down the count by layer, including both weights and biases.

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
Extend `RNNCell` to support a configurable number of layers or blocks. Add a `num_layers` parameter to `__init__` and use `nn.ModuleList` to create a variable-depth architecture. Test with 2, 4, and 8 layers.

??? success "Solution to Exercise 4"
    Replace the hardcoded layers with:
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    In the `forward` method, iterate: `for layer in self.layers: x = layer(x)`. Using `nn.ModuleList` (not a plain Python list) ensures PyTorch registers all parameters for optimization. Test with: `for n in [2, 4, 8]: model = RNNCell(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.
