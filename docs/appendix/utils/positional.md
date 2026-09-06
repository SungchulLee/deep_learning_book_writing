# Positional Encodings

Positional Encodings - Common variants for sequence models Includes:

This implementation provides a concise, educational reference for Positional Encodings. The code focuses on the core architecture and forward pass, making it straightforward to study the key design patterns and adapt them for experimentation.

## Code

```python
#!/usr/bin/env python3
"""
Positional Encodings - Common variants for sequence models
Includes:
  - Sinusoidal positional encoding (Transformer)
  - Learnable positional embedding
  - Rotary positional embedding (RoPE) (conceptual helper)

File: appendix/utils/positional.py
Note: Educational reference; RoPE included as a conceptual minimal.
"""

import math
import torch
import torch.nn as nn

# ========================================================================
# Main
# ========================================================================


class SinusoidalPositionalEncoding(nn.Module):
    """
    Classic sinusoidal encoding from "Attention Is All You Need".

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)

        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        # Register as buffer: saved with model, not trainable
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        """
        x: (B, T, D)
        Returns:
          x + PE[:T]
        """
        T = x.size(1)
        return x + self.pe[:, :T, :]


class LearnablePositionalEmbedding(nn.Module):
    """Learnable position embeddings used in BERT/ViT-like models."""
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.pos = nn.Embedding(max_len, d_model)

    def forward(self, x):
        B, T, D = x.shape
        positions = torch.arange(T, device=x.device)
        return x + self.pos(positions)[None, :, :]


def rope_rotate_half(x):
    """
    Helper for RoPE: rotate last dimension pairs.
    If x = [..., 2i, 2i+1], rotate to [-x_{2i+1}, x_{2i}]
    """
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack([-x2, x1], dim=-1).flatten(-2)


def apply_rope(q, k, cos, sin):
    """
    Apply rotary positional embeddings to q and k.
    This is a conceptual helper used in LLaMA-like models.

    q, k: (..., D) where D is even
    cos, sin: (..., D) or broadcastable to q/k
    """
    q_rot = (q * cos) + (rope_rotate_half(q) * sin)
    k_rot = (k * cos) + (rope_rotate_half(k) * sin)
    return q_rot, k_rot


if __name__ == "__main__":
    pass```

## Discussion

The implementation defines 2 classes (`SinusoidalPositionalEncoding`, `LearnablePositionalEmbedding`) that work together to form the complete utility module architecture. Each class encapsulates a distinct component, making the code modular and easy to extend. The `forward` methods define the computational graph that PyTorch uses for automatic differentiation.

As a reference implementation, this code prioritizes clarity over optimization. Production systems would typically add mixed-precision training, distributed data parallelism, and more sophisticated data augmentation. Nevertheless, the core architectural ideas shown here remain the same regardless of scale.

## Exercises

**Exercise 1.**
Read through the code and identify the key design decisions. List three specific implementation choices and explain why each is appropriate for utility module.

??? success "Solution to Exercise 1"
    Design decisions vary by implementation but commonly include: (1) choice of activation functions -- ReLU variants provide non-saturating gradients for faster training; (2) normalization strategy -- batch normalization stabilizes training by reducing internal covariate shift; (3) residual connections -- when present, they enable gradient flow in deep networks by providing skip paths. Each choice reflects a trade-off between expressiveness, computational cost, and training stability.

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
Extend `SinusoidalPositionalEncoding` to support a configurable number of layers or blocks. Add a `num_layers` parameter to `__init__` and use `nn.ModuleList` to create a variable-depth architecture. Test with 2, 4, and 8 layers.

??? success "Solution to Exercise 4"
    Replace the hardcoded layers with:
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    In the `forward` method, iterate: `for layer in self.layers: x = layer(x)`. Using `nn.ModuleList` (not a plain Python list) ensures PyTorch registers all parameters for optimization. Test with: `for n in [2, 4, 8]: model = SinusoidalPositionalEncoding(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.
