#!/usr/bin/env python3
"""
Positional Encodings - Common variants for sequence models
Includes:
  - Sinusoidal positional encoding (Transformer)
  - Learnable positional embedding
  - Rotary positional embedding (RoPE) (conceptual helper)

File: appendix/utils/positional.py
Note: Educational reference; RoPE included as a conceptual minimal.
"""

import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """
    Classic sinusoidal encoding from "Attention Is All You Need".

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)

        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        # Register as buffer: saved with model, not trainable
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        """
        x: (B, T, D)
        Returns:
          x + PE[:T]
        """
        T = x.size(1)
        return x + self.pe[:, :T, :]


class LearnablePositionalEmbedding(nn.Module):
    """Learnable position embeddings used in BERT/ViT-like models."""
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.pos = nn.Embedding(max_len, d_model)

    def forward(self, x):
        B, T, D = x.shape
        positions = torch.arange(T, device=x.device)
        return x + self.pos(positions)[None, :, :]


def rope_rotate_half(x):
    """
    Helper for RoPE: rotate last dimension pairs.
    If x = [..., 2i, 2i+1], rotate to [-x_{2i+1}, x_{2i}]
    """
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack([-x2, x1], dim=-1).flatten(-2)


def apply_rope(q, k, cos, sin):
    """
    Apply rotary positional embeddings to q and k.
    This is a conceptual helper used in LLaMA-like models.

    q, k: (..., D) where D is even
    cos, sin: (..., D) or broadcastable to q/k
    """
    q_rot = (q * cos) + (rope_rotate_half(q) * sin)
    k_rot = (k * cos) + (rope_rotate_half(k) * sin)
    return q_rot, k_rot
