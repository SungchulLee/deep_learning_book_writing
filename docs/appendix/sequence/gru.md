# GRU

GRU was introduced in the 2014 paper "Learning Phrase Representations using RNN Encoder-Decoder." Simplified gating mechanism compared to LSTM, fewer parameters.

This implementation provides a concise, educational reference for GRU. The code focuses on the core architecture and forward pass, making it straightforward to study the key design patterns and adapt them for experimentation.

## Code

```python
#!/usr/bin/env python3
'''
GRU - Gated Recurrent Unit
Paper: "Learning Phrase Representations using RNN Encoder-Decoder" (2014)
Key: Simplified gating mechanism compared to LSTM, fewer parameters
'''
import torch
import torch.nn as nn

# ========================================================================
# Main
# ========================================================================

class GRUModel(nn.Module):
    def __init__(self, input_size=100, hidden_size=256, num_layers=2, num_classes=10):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # GRU forward pass
        out, _ = self.gru(x, h0)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

if __name__ == "__main__":
    model = GRUModel()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    x = torch.randn(32, 10, 100)
    print(f"Input: {x.shape}, Output: {model(x).shape}")```

## Discussion

The `GRUModel` class encapsulates the model architecture using PyTorch's `nn.Module` interface. The `forward` method defines the computational graph, allowing PyTorch's autograd system to handle gradient computation automatically during training. This modular design makes it straightforward to modify individual components or integrate the model into larger pipelines.

As a reference implementation, this code prioritizes clarity over optimization. Production systems would typically add mixed-precision training, distributed data parallelism, and more sophisticated data augmentation. Nevertheless, the core architectural ideas shown here remain the same regardless of scale.

## Exercises

**Exercise 1.**
Calculate the total number of learnable parameters in `GRUModel` with the default initialization. Break down the count by layer, including both weights and biases.

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
Extend `GRUModel` to support a configurable number of layers or blocks. Add a `num_layers` parameter to `__init__` and use `nn.ModuleList` to create a variable-depth architecture. Test with 2, 4, and 8 layers.

??? success "Solution to Exercise 4"
    Replace the hardcoded layers with:
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    In the `forward` method, iterate: `for layer in self.layers: x = layer(x)`. Using `nn.ModuleList` (not a plain Python list) ensures PyTorch registers all parameters for optimization. Test with: `for n in [2, 4, 8]: model = GRUModel(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.
