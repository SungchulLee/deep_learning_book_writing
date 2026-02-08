#!/usr/bin/env python3
"""
LLaMA - Large Language Model Meta AI
Paper: "LLaMA: Open and Efficient Foundation Language Models" (2023)
Authors: Meta AI
Key ideas (high-level):
  - Decoder-only Transformer (GPT-style)
  - RMSNorm instead of LayerNorm
  - SwiGLU feedforward
  - Rotary positional embeddings (RoPE) instead of learned absolute positions

File: appendix/transformers/llama.py
Note: Educational, commented implementation focusing on RMSNorm + SwiGLU + causal attention.
      This is NOT an optimized LLaMA; it's a readable reference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """
    RMSNorm: normalize by root-mean-square (no mean subtraction).

    x_norm = x / sqrt(mean(x^2) + eps) * weight
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: (..., dim)
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class SwiGLU(nn.Module):
    """
    SwiGLU feedforward (used in many modern LLMs):

      FF(x) = (SiLU(xW1) * (xW3)) W2

    Compared to standard GELU FFN, this gating often improves quality/efficiency.
    """
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class CausalSelfAttention(nn.Module):
    """
    Causal (masked) self-attention (multi-head).
    For simplicity, we omit RoPE math and use a standard causal mask.
    """
    def __init__(self, dim: int, nhead: int):
        super().__init__()
        assert dim % nhead == 0
        self.dim = dim
        self.nhead = nhead
        self.head_dim = dim // nhead

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        # x: (B, S, D)
        B, S, D = x.shape

        qkv = self.qkv(x)                 # (B, S, 3D)
        q, k, v = qkv.chunk(3, dim=-1)    # each: (B, S, D)

        # Reshape into heads: (B, nhead, S, head_dim)
        q = q.view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.nhead, self.head_dim).transpose(1, 2)

        # Attention scores: (B, nhead, S, S)
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Causal mask: prevent attention to future tokens
        mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        out = attn @ v  # (B, nhead, S, head_dim)

        # Merge heads back: (B, S, D)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.out(out)


class LLaMABlock(nn.Module):
    """
    One LLaMA-style transformer block (simplified):
      - RMSNorm
      - Causal self-attention
      - RMSNorm
      - SwiGLU FFN
      - Residual connections
    """
    def __init__(self, dim: int, nhead: int, ff_hidden: int):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, nhead)
        self.norm2 = RMSNorm(dim)
        self.ff = SwiGLU(dim, ff_hidden)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class LLaMA(nn.Module):
    """
    Decoder-only language model (GPT-style).

    Inputs:
      input_ids: (B, S)
    Outputs:
      logits: (B, S, vocab_size)
    """
    def __init__(self, vocab_size=32000, dim=512, nhead=8, num_layers=8, ff_hidden=2048):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([LLaMABlock(dim, nhead, ff_hidden) for _ in range(num_layers)])
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embed(input_ids)  # (B, S, D)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        logits = self.lm_head(x)   # (B, S, vocab)
        return logits


if __name__ == "__main__":
    model = LLaMA(vocab_size=1000, dim=256, nhead=8, num_layers=2, ff_hidden=1024)
    ids = torch.randint(0, 1000, (2, 12))
    logits = model(ids)
    print("logits:", logits.shape)  # (2, 12, 1000)
