# RoBERTa

RoBERTa was introduced in the 2019 paper "RoBERTa: A Robustly Optimized BERT Pretraining Approach."

This implementation provides a concise, educational reference for RoBERTa. The code focuses on the core architecture and forward pass, making it straightforward to study the key design patterns and adapt them for experimentation.

## Code

```python
#!/usr/bin/env python3
"""
RoBERTa - Robustly Optimized BERT Pretraining Approach
Paper: "RoBERTa: A Robustly Optimized BERT Pretraining Approach" (2019)
Authors: Yinhan Liu et al.
Key differences vs BERT (high-level):
  - Trains longer, with larger batches and more data
  - Removes next sentence prediction (NSP)
  - Uses dynamic masking

File: appendix/transformers/roberta.py
Note: Educational implementation of a BERT-like encoder-only Transformer.
"""

import torch
import torch.nn as nn

# ========================================================================
# Main
# ========================================================================


class RoBERTa(nn.Module):
    """
    Encoder-only Transformer for masked language modeling (MLM).

    Inputs:
      input_ids: (B, S)
      attention_mask: (B, S) 1 for tokens, 0 for padding

    Outputs:
      logits: (B, S, vocab_size) token-level vocab logits for MLM
    """
    def __init__(self, vocab_size=50265, d_model=768, nhead=12, num_layers=12):
        super().__init__()

        # Token embeddings (RoBERTa also uses learned positional embeddings; omitted for brevity)
        self.embed = nn.Embedding(vocab_size, d_model)

        # Encoder stack
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # MLM head: predict original token IDs
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        x = self.embed(input_ids)  # (B, S, D)

        # Convert attention_mask to src_key_padding_mask (True = ignore)
        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = ~attention_mask.bool()

        h = self.encoder(x, src_key_padding_mask=src_key_padding_mask)  # (B, S, D)
        logits = self.lm_head(h)                                        # (B, S, vocab)
        return logits


if __name__ == "__main__":
    model = RoBERTa(vocab_size=1000, d_model=256, nhead=8, num_layers=2)
    input_ids = torch.randint(0, 1000, (2, 8))
    mask = torch.ones(2, 8, dtype=torch.long)
    logits = model(input_ids, attention_mask=mask)
    print("logits:", logits.shape)  # (2, 8, 1000)```

## Discussion

The `RoBERTa` class encapsulates the model architecture using PyTorch's `nn.Module` interface. The `forward` method defines the computational graph, allowing PyTorch's autograd system to handle gradient computation automatically during training. This modular design makes it straightforward to modify individual components or integrate the model into larger pipelines.

As a reference implementation, this code prioritizes clarity over optimization. Production systems would typically add mixed-precision training, distributed data parallelism, and more sophisticated data augmentation. Nevertheless, the core architectural ideas shown here remain the same regardless of scale.

## Exercises

**Exercise 1.**
Calculate the total number of learnable parameters in `RoBERTa` with the default initialization. Break down the count by layer, including both weights and biases.

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
Extend `RoBERTa` to support a configurable number of layers or blocks. Add a `num_layers` parameter to `__init__` and use `nn.ModuleList` to create a variable-depth architecture. Test with 2, 4, and 8 layers.

??? success "Solution to Exercise 4"
    Replace the hardcoded layers with:
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    In the `forward` method, iterate: `for layer in self.layers: x = layer(x)`. Using `nn.ModuleList` (not a plain Python list) ensures PyTorch registers all parameters for optimization. Test with: `for n in [2, 4, 8]: model = RoBERTa(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.
