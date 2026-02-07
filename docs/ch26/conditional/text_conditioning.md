# Text-to-Image Generation

## Introduction

Text-to-image diffusion models generate images from natural language descriptions, combining diffusion models with text encoders like CLIP.

## Architecture Overview

```
Text Prompt → Text Encoder (CLIP) → Text Embeddings
                                          ↓
Noise → U-Net (with cross-attention) → Denoised Image
```

## Key Components

### Text Encoding

CLIP text encoder maps prompts to embeddings:
- Input: "A cat sitting on a couch"
- Output: Sequence of embeddings [77, 768]

### Cross-Attention

U-Net incorporates text via cross-attention:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$$

where Q comes from image features, K and V from text embeddings.

## Implementation

```python
"""
Text-to-Image Components
========================
"""

import torch
import torch.nn as nn


class CrossAttention(nn.Module):
    """Cross-attention for text conditioning."""
    
    def __init__(self, query_dim, context_dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = heads * dim_head
        
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)
    
    def forward(self, x, context):
        """
        Args:
            x: Image features [B, H*W, C]
            context: Text embeddings [B, L, D]
        """
        B, N, C = x.shape
        
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        
        # Reshape for multi-head attention
        q = q.view(B, N, self.heads, -1).transpose(1, 2)
        k = k.view(B, -1, self.heads, -1).transpose(1, 2)
        v = v.view(B, -1, self.heads, -1).transpose(1, 2)
        
        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, N, -1)
        
        return self.to_out(out)


class TextConditionedBlock(nn.Module):
    """U-Net block with text cross-attention."""
    
    def __init__(self, channels, context_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.LayerNorm(channels)
        self.cross_attn = CrossAttention(channels, context_dim)
    
    def forward(self, x, context):
        # Convolution
        h = self.conv(self.norm1(x))
        
        # Cross-attention with text
        B, C, H, W = h.shape
        h_flat = h.view(B, C, H*W).transpose(1, 2)
        h_flat = self.norm2(h_flat)
        h_attn = self.cross_attn(h_flat, context)
        h = h + h_attn.transpose(1, 2).view(B, C, H, W)
        
        return h
```

## Prompting Tips

| Technique | Example |
|-----------|---------|
| Style | "in the style of Van Gogh" |
| Quality | "highly detailed, 4k" |
| Lighting | "dramatic lighting, sunset" |
| Composition | "centered, symmetrical" |

## Summary

Text-to-image combines:
1. **Text encoder**: CLIP for semantic understanding
2. **Cross-attention**: Inject text into U-Net
3. **Classifier-free guidance**: Control fidelity/diversity

