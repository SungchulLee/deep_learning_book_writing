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
