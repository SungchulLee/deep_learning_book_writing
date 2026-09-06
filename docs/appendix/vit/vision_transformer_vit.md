# Vision Transformer VIT

Vision Transformer (ViT) - An Image is Worth 16x16 Words Link: https://arxiv.org/abs/2010.11929

This implementation provides a concise, educational reference for Vision Transformer VIT. The code focuses on the core architecture and forward pass, making it straightforward to study the key design patterns and adapt them for experimentation.

## Code

```python
#!/usr/bin/env python3
"""
================================================================================
Vision Transformer (ViT) - An Image is Worth 16x16 Words
================================================================================

Paper: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
Authors: Alexey Dosovitskiy et al. (Google Research)
Link: https://arxiv.org/abs/2010.11929

================================================================================
HISTORICAL SIGNIFICANCE
================================================================================
ViT demonstrated that pure transformer architectures can achieve excellent 
performance on image classification when pre-trained on sufficient data.

- With enough data, ViT outperforms ResNet with 4× less compute
- Attention patterns show emergent object localization

================================================================================
KEY INSIGHT: IMAGES AS SEQUENCES OF PATCHES
================================================================================

For 16×16 patches on 224×224 image: (224/16)² = 196 patches
Each patch → Linear embedding → Sequence of "visual tokens"

================================================================================
CURRICULUM MAPPING
================================================================================

Related: swin_transformer.py, convnext.py
================================================================================
"""

import torch
import torch.nn as nn
from typing import Tuple

# ========================================================================
# Main
# ========================================================================


class PatchEmbedding(nn.Module):
    """Convert image into sequence of patch embeddings using Conv2d."""
    
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) for Image Classification
    
    Args:
        img_size: Input image size. Default: 224
        patch_size: Size of each patch. Default: 16
        num_classes: Number of output classes. Default: 1000
        embed_dim: Embedding dimension. Default: 768
        depth: Number of transformer layers. Default: 12
        num_heads: Number of attention heads. Default: 12
    """
    
    def __init__(self, img_size=224, patch_size=16, num_classes=1000, embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.num_patches, embed_dim))
        
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=embed_dim*4, batch_first=True)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        return self.head(x[:, 0])


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    print("=" * 70)
    print("Vision Transformer (ViT-B/16)")
    print("=" * 70)
    
    model = VisionTransformer()
    print(f"Parameters: {count_parameters(model):,}")
    
    x = torch.randn(2, 3, 224, 224)
    model.eval()
    with torch.no_grad():
        out = model(x)
    print(f"Input: {x.shape}, Output: {out.shape}")
    print("=" * 70)```

## Discussion

The implementation defines 2 classes (`PatchEmbedding`, `VisionTransformer`) that work together to form the complete vision transformer architecture. Each class encapsulates a distinct component, making the code modular and easy to extend. The `forward` methods define the computational graph that PyTorch uses for automatic differentiation.

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
