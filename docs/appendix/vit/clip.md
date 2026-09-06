# CLIP

CLIP was introduced in the 2021 paper "Learning Transferable Visual Models From Natural Language Supervision." - Jointly train image encoder + text encoder   - Align them using contrastive loss in a shared embedding space.

This implementation provides a concise, educational reference for CLIP. The code focuses on the core architecture and forward pass, making it straightforward to study the key design patterns and adapt them for experimentation.

## Code

```python
#!/usr/bin/env python3
"""
CLIP - Contrastive Language-Image Pretraining
Paper: "Learning Transferable Visual Models From Natural Language Supervision" (2021)
Authors: Alec Radford et al.
Key idea:
  - Jointly train image encoder + text encoder
  - Align them using contrastive loss in a shared embedding space

File: appendix/vit/clip.py
Note: Educational implementation (core idea only).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================================
# Main
# ========================================================================


class ImageEncoder(nn.Module):
    """Simple ViT-style image encoder."""
    def __init__(self, embed_dim=512):
        super().__init__()
        self.encoder = nn.Linear(768, embed_dim)  # assume patch pooled features

    def forward(self, img_feat):
        return self.encoder(img_feat)


class TextEncoder(nn.Module):
    """Simple Transformer-based text encoder."""
    def __init__(self, vocab_size=50000, embed_dim=512):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=8, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

    def forward(self, tokens):
        x = self.emb(tokens)
        x = self.encoder(x)
        return x[:, 0]  # use [CLS]-like token


class CLIP(nn.Module):
    """
    CLIP model: image encoder + text encoder with contrastive objective.
    """
    def __init__(self, embed_dim=512):
        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim)
        self.text_encoder = TextEncoder(embed_dim=embed_dim)

        # Temperature parameter (learned)
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.659)

    def forward(self, image_feat, text_tokens):
        img_emb = self.image_encoder(image_feat)
        txt_emb = self.text_encoder(text_tokens)

        # Normalize embeddings
        img_emb = F.normalize(img_emb, dim=-1)
        txt_emb = F.normalize(txt_emb, dim=-1)

        # Cosine similarity scaled by temperature
        scale = self.logit_scale.exp()
        logits = scale * img_emb @ txt_emb.t()

        return logits


if __name__ == "__main__":
    pass```

## Discussion

The implementation defines 3 classes (`ImageEncoder`, `TextEncoder`, `CLIP`) that work together to form the complete vision transformer architecture. Each class encapsulates a distinct component, making the code modular and easy to extend. The `forward` methods define the computational graph that PyTorch uses for automatic differentiation.

As a reference implementation, this code prioritizes clarity over optimization. Production systems would typically add mixed-precision training, distributed data parallelism, and more sophisticated data augmentation. Nevertheless, the core architectural ideas shown here remain the same regardless of scale.

## Exercises

**Exercise 1.**
Calculate the total number of learnable parameters in `ImageEncoder` with the default initialization. Break down the count by layer, including both weights and biases.

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
Extend `ImageEncoder` to support a configurable number of layers or blocks. Add a `num_layers` parameter to `__init__` and use `nn.ModuleList` to create a variable-depth architecture. Test with 2, 4, and 8 layers.

??? success "Solution to Exercise 4"
    Replace the hardcoded layers with:
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    In the `forward` method, iterate: `for layer in self.layers: x = layer(x)`. Using `nn.ModuleList` (not a plain Python list) ensures PyTorch registers all parameters for optimization. Test with: `for n in [2, 4, 8]: model = ImageEncoder(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.
