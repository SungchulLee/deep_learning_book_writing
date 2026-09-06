# LLaMA

LLaMA was introduced in the 2023 paper "LLaMA: Open and Efficient Foundation Language Models."

This implementation provides a concise, educational reference for LLaMA. The code focuses on the core architecture and forward pass, making it straightforward to study the key design patterns and adapt them for experimentation.

## Code

```python
#!/usr/bin/env python3
"""
LLaMA - Large Language Model Meta AI
Paper: "LLaMA: Open and Efficient Foundation Language Models" (2023)
Authors: Meta AI
Key ideas (high-level):
  - Decoder-only Transformer (GPT-style)
  - RMSNorm instead of LayerNorm
  - SwiGLU feedforward
  - Rotary positional embeddings (RoPE) instead of learned absolute positions

File: appendix/transformers/llama.py
Note: Educational, commented implementation focusing on RMSNorm + SwiGLU + causal attention.
      This is NOT an optimized LLaMA; it's a readable reference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================================
# Main
# ========================================================================


class RMSNorm(nn.Module):
    """
    RMSNorm: normalize by root-mean-square (no mean subtraction).

    x_norm = x / sqrt(mean(x^2) + eps) * weight
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: (..., dim)
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class SwiGLU(nn.Module):
    """
    SwiGLU feedforward (used in many modern LLMs):

      FF(x) = (SiLU(xW1) * (xW3)) W2

    Compared to standard GELU FFN, this gating often improves quality/efficiency.
    """
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class CausalSelfAttention(nn.Module):
    """
    Causal (masked) self-attention (multi-head).
    For simplicity, we omit RoPE math and use a standard causal mask.
    """
    def __init__(self, dim: int, nhead: int):
        super().__init__()
        assert dim % nhead == 0
        self.dim = dim
        self.nhead = nhead
        self.head_dim = dim // nhead

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        # x: (B, S, D)
        B, S, D = x.shape

        qkv = self.qkv(x)                 # (B, S, 3D)
        q, k, v = qkv.chunk(3, dim=-1)    # each: (B, S, D)

        # Reshape into heads: (B, nhead, S, head_dim)
        q = q.view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.nhead, self.head_dim).transpose(1, 2)

        # Attention scores: (B, nhead, S, S)
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Causal mask: prevent attention to future tokens
        mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        out = attn @ v  # (B, nhead, S, head_dim)

        # Merge heads back: (B, S, D)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.out(out)


class LLaMABlock(nn.Module):
    """
    One LLaMA-style transformer block (simplified):
      - RMSNorm
      - Causal self-attention
      - RMSNorm
      - SwiGLU FFN
      - Residual connections
    """
    def __init__(self, dim: int, nhead: int, ff_hidden: int):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, nhead)
        self.norm2 = RMSNorm(dim)
        self.ff = SwiGLU(dim, ff_hidden)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class LLaMA(nn.Module):
    """
    Decoder-only language model (GPT-style).

    Inputs:
      input_ids: (B, S)
    Outputs:
      logits: (B, S, vocab_size)
    """
    def __init__(self, vocab_size=32000, dim=512, nhead=8, num_layers=8, ff_hidden=2048):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([LLaMABlock(dim, nhead, ff_hidden) for _ in range(num_layers)])
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embed(input_ids)  # (B, S, D)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        logits = self.lm_head(x)   # (B, S, vocab)
        return logits


if __name__ == "__main__":
    model = LLaMA(vocab_size=1000, dim=256, nhead=8, num_layers=2, ff_hidden=1024)
    ids = torch.randint(0, 1000, (2, 12))
    logits = model(ids)
    print("logits:", logits.shape)  # (2, 12, 1000)```

## Discussion

The implementation defines 5 classes (`RMSNorm`, `SwiGLU`, `CausalSelfAttention`, `LLaMABlock`, and 1 more) that work together to form the complete transformer architecture. Each class encapsulates a distinct component, making the code modular and easy to extend. The `forward` methods define the computational graph that PyTorch uses for automatic differentiation.

As a reference implementation, this code prioritizes clarity over optimization. Production systems would typically add mixed-precision training, distributed data parallelism, and more sophisticated data augmentation. Nevertheless, the core architectural ideas shown here remain the same regardless of scale.

## Exercises

**Exercise 1.**
Calculate the total number of learnable parameters in `RMSNorm` with the default initialization. Break down the count by layer, including both weights and biases.

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
Extend `RMSNorm` to support a configurable number of layers or blocks. Add a `num_layers` parameter to `__init__` and use `nn.ModuleList` to create a variable-depth architecture. Test with 2, 4, and 8 layers.

??? success "Solution to Exercise 4"
    Replace the hardcoded layers with:
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    In the `forward` method, iterate: `for layer in self.layers: x = layer(x)`. Using `nn.ModuleList` (not a plain Python list) ensures PyTorch registers all parameters for optimization. Test with: `for n in [2, 4, 8]: model = RMSNorm(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.
