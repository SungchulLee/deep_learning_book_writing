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
