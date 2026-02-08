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
