# Longformer

Longformer was introduced in the 2020 paper "Longformer: The Long-Document Transformer." - Replace full O(S^2) attention with:       (a) sliding-window local attention (O(S * window))       (b) optional global attention tokens (e.g., [CLS], question tokens).

This implementation provides a concise, educational reference for Longformer. The code focuses on the core architecture and forward pass, making it straightforward to study the key design patterns and adapt them for experimentation.

## Code

```python
#!/usr/bin/env python3
"""
Longformer - The Long-Document Transformer
Paper: "Longformer: The Long-Document Transformer" (2020)
Authors: Iz Beltagy, Matthew E. Peters, Arman Cohan
Key idea:
  - Replace full O(S^2) attention with:
      (a) sliding-window local attention (O(S * window))
      (b) optional global attention tokens (e.g., [CLS], question tokens)

File: appendix/transformers/longformer.py
Note: Educational implementation of *local attention* (windowed self-attention).
      This is NOT an optimized kernel; it's a clear reference implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================================
# Main
# ========================================================================


class WindowSelfAttention(nn.Module):
    """
    Naive windowed self-attention (single-head for clarity).

    For each position i, attend only to tokens in [i-w, i+w].
    Complexity becomes ~O(S * window) instead of O(S^2).
    """
    def __init__(self, d_model=256, window=4):
        super().__init__()
        self.window = window
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: (B, S, D)
        B, S, D = x.shape

        Q = self.q(x)  # (B, S, D)
        K = self.k(x)
        V = self.v(x)

        outputs = []
        for i in range(S):
            # Determine local window indices
            left = max(0, i - self.window)
            right = min(S, i + self.window + 1)

            # Compute attention for token i against local window tokens
            q_i = Q[:, i : i + 1, :]            # (B, 1, D)
            k_w = K[:, left:right, :]           # (B, W, D)
            v_w = V[:, left:right, :]           # (B, W, D)

            # Scaled dot-product attention
            scores = (q_i @ k_w.transpose(1, 2)) / (D ** 0.5)  # (B, 1, W)
            attn = F.softmax(scores, dim=-1)                   # (B, 1, W)
            out_i = attn @ v_w                                 # (B, 1, D)
            outputs.append(out_i)

        y = torch.cat(outputs, dim=1)  # (B, S, D)
        return self.out(y)


class LongformerBlock(nn.Module):
    """One transformer-like block using window attention + feedforward network."""
    def __init__(self, d_model=256, window=4, ff_dim=1024):
        super().__init__()
        self.attn = WindowSelfAttention(d_model, window)
        self.norm1 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ff_dim, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Attention + residual + norm
        x = self.norm1(x + self.attn(x))

        # FFN + residual + norm
        x = self.norm2(x + self.ff(x))
        return x


class Longformer(nn.Module):
    """
    Longformer-like encoder using windowed attention blocks.
    """
    def __init__(self, vocab_size=30522, d_model=256, window=4, num_layers=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([LongformerBlock(d_model, window) for _ in range(num_layers)])
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        x = self.embed(input_ids)  # (B, S, D)
        for blk in self.blocks:
            x = blk(x)
        logits = self.lm_head(x)   # (B, S, vocab)
        return logits


if __name__ == "__main__":
    model = Longformer(vocab_size=1000, d_model=128, window=2, num_layers=2)
    ids = torch.randint(0, 1000, (2, 20))
    logits = model(ids)
    print("logits:", logits.shape)  # (2, 20, 1000)```

## Discussion

The implementation defines 3 classes (`WindowSelfAttention`, `LongformerBlock`, `Longformer`) that work together to form the complete transformer architecture. Each class encapsulates a distinct component, making the code modular and easy to extend. The `forward` methods define the computational graph that PyTorch uses for automatic differentiation.

As a reference implementation, this code prioritizes clarity over optimization. Production systems would typically add mixed-precision training, distributed data parallelism, and more sophisticated data augmentation. Nevertheless, the core architectural ideas shown here remain the same regardless of scale.

## Exercises

**Exercise 1.**
Calculate the total number of learnable parameters in `WindowSelfAttention` with the default initialization. Break down the count by layer, including both weights and biases.

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
Extend `WindowSelfAttention` to support a configurable number of layers or blocks. Add a `num_layers` parameter to `__init__` and use `nn.ModuleList` to create a variable-depth architecture. Test with 2, 4, and 8 layers.

??? success "Solution to Exercise 4"
    Replace the hardcoded layers with:
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    In the `forward` method, iterate: `for layer in self.layers: x = layer(x)`. Using `nn.ModuleList` (not a plain Python list) ensures PyTorch registers all parameters for optimization. Test with: `for n in [2, 4, 8]: model = WindowSelfAttention(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.
