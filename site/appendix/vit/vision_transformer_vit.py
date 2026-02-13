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
    print("=" * 70)
