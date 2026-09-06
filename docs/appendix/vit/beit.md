# BEiT

BEiT was introduced in the 2021 paper "BEiT: BERT Pre-Training of Image Transformers." - Self-supervised pretraining   - Predict *discrete visual tokens* instead of pixels   - Inspired by BERT masked language modeling.

This implementation provides a concise, educational reference for BEiT. The code focuses on the core architecture and forward pass, making it straightforward to study the key design patterns and adapt them for experimentation.

## Code

```python
#!/usr/bin/env python3
"""
BEiT - BERT Pre-Training of Image Transformers
Paper: "BEiT: BERT Pre-Training of Image Transformers" (2021)
Authors: Hangbo Bao et al.
Key idea:
  - Self-supervised pretraining
  - Predict *discrete visual tokens* instead of pixels
  - Inspired by BERT masked language modeling

File: appendix/vit/beit.py
Note: Educational implementation (masked patch prediction).
"""

import torch
import torch.nn as nn

# ========================================================================
# Main
# ========================================================================


class BEiT(nn.Module):
    """
    BEiT pretraining model.

    Steps:
      1) Tokenize image into patches
      2) Mask some patches
      3) Predict their discrete visual tokens (codebook indices)
    """
    def __init__(self, vocab_size=8192, embed_dim=768, num_patches=196):
        super().__init__()

        self.patch_embed = nn.Linear(768, embed_dim)  # assume pre-extracted patch features
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=12, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=12)

        # Predict discrete token ids
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, patch_feats, mask):
        """
        patch_feats: (B, N, D) patch embeddings
        mask:        (B, N) boolean mask (True = masked)
        """
        x = self.patch_embed(patch_feats)

        # Replace masked patches with mask token
        mask_token = self.mask_token.expand(x.size(0), x.size(1), -1)
        x = torch.where(mask.unsqueeze(-1), mask_token, x)

        x = x + self.pos_embed
        x = self.encoder(x)

        # Predict visual tokens
        logits = self.head(x)  # (B, N, vocab_size)
        return logits


if __name__ == "__main__":
    pass```

## Discussion

The `BEiT` class encapsulates the model architecture using PyTorch's `nn.Module` interface. The `forward` method defines the computational graph, allowing PyTorch's autograd system to handle gradient computation automatically during training. This modular design makes it straightforward to modify individual components or integrate the model into larger pipelines.

As a reference implementation, this code prioritizes clarity over optimization. Production systems would typically add mixed-precision training, distributed data parallelism, and more sophisticated data augmentation. Nevertheless, the core architectural ideas shown here remain the same regardless of scale.

## Exercises

**Exercise 1.**
Calculate the total number of learnable parameters in `BEiT` with the default initialization. Break down the count by layer, including both weights and biases.

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
Extend `BEiT` to support a configurable number of layers or blocks. Add a `num_layers` parameter to `__init__` and use `nn.ModuleList` to create a variable-depth architecture. Test with 2, 4, and 8 layers.

??? success "Solution to Exercise 4"
    Replace the hardcoded layers with:
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    In the `forward` method, iterate: `for layer in self.layers: x = layer(x)`. Using `nn.ModuleList` (not a plain Python list) ensures PyTorch registers all parameters for optimization. Test with: `for n in [2, 4, 8]: model = BEiT(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.
