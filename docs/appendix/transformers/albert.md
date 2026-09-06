# ALBERT

ALBERT was introduced in the 2019 paper "ALBERT: A Lite BERT for Self-supervised Learning of Language Representations." 1) Factorized embedding parameterization:        vocab embedding dim (E) smaller than hidden dim (H)        embed: vocab -> E, then project E -> H   2) Cross-layer parameter sharing:        reuse the same Transformer layer weights across all layers   3) Sentence-order prediction (SOP) (often discussed vs NSP).

This implementation provides a concise, educational reference for ALBERT. The code focuses on the core architecture and forward pass, making it straightforward to study the key design patterns and adapt them for experimentation.

## Code

```python
#!/usr/bin/env python3
"""
ALBERT - A Lite BERT
Paper: "ALBERT: A Lite BERT for Self-supervised Learning of Language Representations" (2019)
Authors: Zhenzhong Lan et al.
Key ideas:
  1) Factorized embedding parameterization:
       vocab embedding dim (E) smaller than hidden dim (H)
       embed: vocab -> E, then project E -> H
  2) Cross-layer parameter sharing:
       reuse the same Transformer layer weights across all layers
  3) Sentence-order prediction (SOP) (often discussed vs NSP)

File: appendix/transformers/albert.py
Note: Educational implementation showing factorized embeddings + shared layer.
"""

import torch
import torch.nn as nn

# ========================================================================
# Main
# ========================================================================


class SharedTransformerBlock(nn.Module):
    """One transformer encoder layer, intended to be reused multiple times."""
    def __init__(self, d_model=768, nhead=12):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)

    def forward(self, x, src_key_padding_mask=None):
        return self.layer(x, src_key_padding_mask=src_key_padding_mask)


class ALBERT(nn.Module):
    """
    ALBERT encoder-only model with:
      - factorized embeddings
      - shared transformer block repeated num_layers times
    """
    def __init__(self, vocab_size=30000, embed_dim=128, hidden_dim=768, nhead=12, num_layers=12):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)    # vocab -> E
        self.embed_proj = nn.Linear(embed_dim, hidden_dim)        # E -> H

        self.shared_block = SharedTransformerBlock(d_model=hidden_dim, nhead=nhead)
        self.num_layers = num_layers

        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        # Token embedding in low dimension E
        x = self.token_embed(input_ids)     # (B, S, E)

        # Project to hidden dimension H
        x = self.embed_proj(x)              # (B, S, H)

        # Padding mask (True = ignore)
        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = ~attention_mask.bool()

        # Reuse the same block multiple times (parameter sharing)
        for _ in range(self.num_layers):
            x = self.shared_block(x, src_key_padding_mask=src_key_padding_mask)

        # MLM logits
        logits = self.lm_head(x)            # (B, S, vocab)
        return logits


if __name__ == "__main__":
    model = ALBERT(vocab_size=1000, embed_dim=64, hidden_dim=256, nhead=8, num_layers=4)
    ids = torch.randint(0, 1000, (2, 9))
    mask = torch.ones(2, 9, dtype=torch.long)
    logits = model(ids, attention_mask=mask)
    print("logits:", logits.shape)  # (2, 9, 1000)```

## Discussion

The implementation defines 2 classes (`SharedTransformerBlock`, `ALBERT`) that work together to form the complete transformer architecture. Each class encapsulates a distinct component, making the code modular and easy to extend. The `forward` methods define the computational graph that PyTorch uses for automatic differentiation.

As a reference implementation, this code prioritizes clarity over optimization. Production systems would typically add mixed-precision training, distributed data parallelism, and more sophisticated data augmentation. Nevertheless, the core architectural ideas shown here remain the same regardless of scale.

## Exercises

**Exercise 1.**
Calculate the total number of learnable parameters in `SharedTransformerBlock` with the default initialization. Break down the count by layer, including both weights and biases.

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
Extend `SharedTransformerBlock` to support a configurable number of layers or blocks. Add a `num_layers` parameter to `__init__` and use `nn.ModuleList` to create a variable-depth architecture. Test with 2, 4, and 8 layers.

??? success "Solution to Exercise 4"
    Replace the hardcoded layers with:
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    In the `forward` method, iterate: `for layer in self.layers: x = layer(x)`. Using `nn.ModuleList` (not a plain Python list) ensures PyTorch registers all parameters for optimization. Test with: `for n in [2, 4, 8]: model = SharedTransformerBlock(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.
