#!/usr/bin/env python3
"""
MAE - Masked Autoencoders
Paper: "Masked Autoencoders Are Scalable Vision Learners" (2021)
Authors: Kaiming He et al.
Key idea:
  - Mask most image patches (e.g., 75%)
  - Encode only visible patches
  - Lightweight decoder reconstructs masked patches

File: appendix/vit/mae.py
Note: Educational implementation (encoder-decoder structure).
"""

import torch
import torch.nn as nn


class MAE(nn.Module):
    """
    Masked Autoencoder with ViT-style encoder and decoder.
    """
    def __init__(self, embed_dim=768, decoder_dim=512, num_patches=196):
        super().__init__()

        # Encoder processes visible patches only
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=12, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=12)

        # Decoder reconstructs masked patches
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim, nhead=8, batch_first=True
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=8)

        self.enc_to_dec = nn.Linear(embed_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))

        self.head = nn.Linear(decoder_dim, embed_dim)

    def forward(self, x, mask):
        """
        x   : (B, N, D) patch embeddings
        mask: (B, N) True for masked patches
        """
        # Keep only visible patches
        visible = x[~mask].view(x.size(0), -1, x.size(-1))

        # Encode visible patches
        enc = self.encoder(visible)

        # Project to decoder dimension
        dec_input = self.enc_to_dec(enc)

        # Append mask tokens for reconstruction
        num_masked = mask.sum(dim=1).max()
        mask_tokens = self.mask_token.expand(x.size(0), num_masked, -1)

        dec_input = torch.cat([dec_input, mask_tokens], dim=1)

        # Decode
        dec = self.decoder(dec_input)

        # Predict reconstructed patch embeddings
        recon = self.head(dec)
        return recon
