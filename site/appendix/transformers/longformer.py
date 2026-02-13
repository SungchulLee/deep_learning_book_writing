#!/usr/bin/env python3
"""
Longformer - The Long-Document Transformer
Paper: "Longformer: The Long-Document Transformer" (2020)
Authors: Iz Beltagy, Matthew E. Peters, Arman Cohan
Key idea:
  - Replace full O(S^2) attention with:
      (a) sliding-window local attention (O(S * window))
      (b) optional global attention tokens (e.g., [CLS], question tokens)

File: appendix/transformers/longformer.py
Note: Educational implementation of *local attention* (windowed self-attention).
      This is NOT an optimized kernel; it's a clear reference implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WindowSelfAttention(nn.Module):
    """
    Naive windowed self-attention (single-head for clarity).

    For each position i, attend only to tokens in [i-w, i+w].
    Complexity becomes ~O(S * window) instead of O(S^2).
    """
    def __init__(self, d_model=256, window=4):
        super().__init__()
        self.window = window
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: (B, S, D)
        B, S, D = x.shape

        Q = self.q(x)  # (B, S, D)
        K = self.k(x)
        V = self.v(x)

        outputs = []
        for i in range(S):
            # Determine local window indices
            left = max(0, i - self.window)
            right = min(S, i + self.window + 1)

            # Compute attention for token i against local window tokens
            q_i = Q[:, i : i + 1, :]            # (B, 1, D)
            k_w = K[:, left:right, :]           # (B, W, D)
            v_w = V[:, left:right, :]           # (B, W, D)

            # Scaled dot-product attention
            scores = (q_i @ k_w.transpose(1, 2)) / (D ** 0.5)  # (B, 1, W)
            attn = F.softmax(scores, dim=-1)                   # (B, 1, W)
            out_i = attn @ v_w                                 # (B, 1, D)
            outputs.append(out_i)

        y = torch.cat(outputs, dim=1)  # (B, S, D)
        return self.out(y)


class LongformerBlock(nn.Module):
    """One transformer-like block using window attention + feedforward network."""
    def __init__(self, d_model=256, window=4, ff_dim=1024):
        super().__init__()
        self.attn = WindowSelfAttention(d_model, window)
        self.norm1 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ff_dim, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Attention + residual + norm
        x = self.norm1(x + self.attn(x))

        # FFN + residual + norm
        x = self.norm2(x + self.ff(x))
        return x


class Longformer(nn.Module):
    """
    Longformer-like encoder using windowed attention blocks.
    """
    def __init__(self, vocab_size=30522, d_model=256, window=4, num_layers=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([LongformerBlock(d_model, window) for _ in range(num_layers)])
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        x = self.embed(input_ids)  # (B, S, D)
        for blk in self.blocks:
            x = blk(x)
        logits = self.lm_head(x)   # (B, S, vocab)
        return logits


if __name__ == "__main__":
    model = Longformer(vocab_size=1000, d_model=128, window=2, num_layers=2)
    ids = torch.randint(0, 1000, (2, 20))
    logits = model(ids)
    print("logits:", logits.shape)  # (2, 20, 1000)
