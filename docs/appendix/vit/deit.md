# DeiT

DeiT was introduced in the 2021 paper "Training data-efficient image transformers & distillation through attention." - Train ViT with less data using knowledge distillation   - Introduce a *distillation token* in addition to the class token.

This implementation provides a concise, educational reference for DeiT. The code focuses on the core architecture and forward pass, making it straightforward to study the key design patterns and adapt them for experimentation.

## Code

```python
#!/usr/bin/env python3
"""
DeiT - Data-efficient Image Transformers
Paper: "Training data-efficient image transformers & distillation through attention" (2021)
Authors: Hugo Touvron et al.
Key idea:
  - Train ViT with less data using knowledge distillation
  - Introduce a *distillation token* in addition to the class token

File: appendix/vit/deit.py
Note: Educational implementation (forward pass only).
"""

import torch
import torch.nn as nn

# ========================================================================
# Main
# ========================================================================


class PatchEmbedding(nn.Module):
    """Split image into patches and embed them."""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, 3, H, W)
        x = self.proj(x)                 # (B, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2) # (B, N, D)
        return x


class DeiT(nn.Module):
    """
    DeiT = ViT + distillation token.

    Tokens:
      - [CLS] token: standard classification token
      - [DIST] token: learns from teacher predictions
    """
    def __init__(self, num_classes=1000, embed_dim=768, num_patches=196):
        super().__init__()

        self.patch_embed = PatchEmbedding(embed_dim=embed_dim)

        # Learnable tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Position embedding includes both tokens
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, embed_dim))

        # Transformer encoder (simplified)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=12, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=12)

        # Two heads: one for cls, one for distillation
        self.head_cls = nn.Linear(embed_dim, num_classes)
        self.head_dist = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.size(0)

        x = self.patch_embed(x)  # (B, N, D)

        # Expand tokens for batch
        cls = self.cls_token.expand(B, -1, -1)
        dist = self.dist_token.expand(B, -1, -1)

        # Concatenate tokens with patch embeddings
        x = torch.cat([cls, dist, x], dim=1)
        x = x + self.pos_embed

        # Transformer encoding
        x = self.encoder(x)

        # Separate outputs
        cls_out = x[:, 0]
        dist_out = x[:, 1]

        # Two prediction heads
        logits_cls = self.head_cls(cls_out)
        logits_dist = self.head_dist(dist_out)

        return logits_cls, logits_dist


if __name__ == "__main__":
    pass```

## Discussion

The implementation defines 2 classes (`PatchEmbedding`, `DeiT`) that work together to form the complete vision transformer architecture. Each class encapsulates a distinct component, making the code modular and easy to extend. The `forward` methods define the computational graph that PyTorch uses for automatic differentiation.

As a reference implementation, this code prioritizes clarity over optimization. Production systems would typically add mixed-precision training, distributed data parallelism, and more sophisticated data augmentation. Nevertheless, the core architectural ideas shown here remain the same regardless of scale.

## Exercises

**Exercise 1.**
Trace the tensor shapes through the `PatchEmbedding` forward pass. For a batch of 4 input samples with the default parameters, write down the shape after each major operation (convolution, pooling, linear layer).

??? success "Solution to Exercise 1"
    Start from the input shape and apply each layer sequentially. For each `Conv2d(in_c, out_c, k)`, the spatial dimensions change as $H_{\text{out}} = H_{\text{in}} - k + 1$ (without padding) or remain the same (with `padding=k//2`). Pooling with kernel 2 halves spatial dimensions. Linear layers transform the last dimension. Track the batch dimension throughout: it remains unchanged. Write each intermediate shape as $(B, C, H, W)$ for convolutional layers and $(B, F)$ after flattening.

---

**Exercise 2.**
Modify the architecture to accept RGB images of size $64 \times 64$ (input shape: $3 \times 64 \times 64$). Update all layer dimensions accordingly and verify the model runs without errors.

??? success "Solution to Exercise 2"
    Change the first convolutional layer's `in_channels` from the current value to 3. Recalculate the spatial dimensions after each convolution and pooling layer using the formulas $H_{\text{out}} = \lfloor(H_{\text{in}} + 2p - k) / s\rfloor + 1$. Update the first linear layer's `in_features` to match the flattened output of the last convolutional/pooling layer. Verify with: `model = PatchEmbedding(...); x = torch.randn(1, 3, 64, 64); print(model(x).shape)`.

---

**Exercise 3.**
Explain the computational complexity of self-attention as a function of sequence length $n$ and model dimension $d$. Why does this motivate architectures like Longformer or Linformer for long sequences?

??? success "Solution to Exercise 3"
    Standard self-attention computes an $n \times n$ attention matrix, giving $O(n^2 d)$ time complexity and $O(n^2)$ memory for the attention weights. For long sequences (e.g., $n = 4096$), this becomes prohibitive. Longformer uses a combination of local sliding-window attention ($O(n \cdot w \cdot d)$ where $w$ is window size) and sparse global attention for selected tokens. Linformer projects keys and values to a lower dimension $k \ll n$, reducing complexity to $O(n \cdot k \cdot d)$. Both trade some expressiveness for practical efficiency on long inputs.

---

**Exercise 4.**
Extend `PatchEmbedding` to support a configurable number of layers or blocks. Add a `num_layers` parameter to `__init__` and use `nn.ModuleList` to create a variable-depth architecture. Test with 2, 4, and 8 layers.

??? success "Solution to Exercise 4"
    Replace the hardcoded layers with:
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    In the `forward` method, iterate: `for layer in self.layers: x = layer(x)`. Using `nn.ModuleList` (not a plain Python list) ensures PyTorch registers all parameters for optimization. Test with: `for n in [2, 4, 8]: model = PatchEmbedding(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.
