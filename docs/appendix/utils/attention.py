#!/usr/bin/env python3
"""
Attention Mechanisms - Common building blocks
Includes:
  - Scaled Dot-Product Attention
  - Multi-Head Attention (minimal)
  - Additive (Bahdanau) Attention (for seq2seq)

File: appendix/utils/attention.py
Note: Educational, heavily commented reference implementations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Scaled dot-product attention (core Transformer attention).

    Inputs:
      Q: queries   (B, H, Tq, Dh)
      K: keys      (B, H, Tk, Dh)
      V: values    (B, H, Tk, Dh)
      mask: optional attention mask broadcastable to (B, H, Tq, Tk)
            - True/1 indicates "keep", False/0 indicates "mask out"

    Output:
      out: (B, H, Tq, Dh)
      attn: (B, H, Tq, Tk) attention weights
    """
    Dh = Q.size(-1)

    # Raw attention scores: (B, H, Tq, Tk)
    scores = (Q @ K.transpose(-2, -1)) / (Dh ** 0.5)

    # If mask is provided, set masked positions to -inf before softmax
    if mask is not None:
        # Convert mask to boolean where False means masked out
        scores = scores.masked_fill(~mask.bool(), float("-inf"))

    # Softmax across keys dimension
    attn = F.softmax(scores, dim=-1)

    # Weighted sum of values
    out = attn @ V
    return out, attn


class MultiHeadAttention(nn.Module):
    """
    Minimal Multi-Head Attention layer.

    Steps:
      1) Project input to Q,K,V
      2) Split into heads
      3) Apply scaled dot-product attention
      4) Merge heads + output projection
    """
    def __init__(self, d_model: int, nhead: int):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_model = d_model
        self.nhead = nhead
        self.dh = d_model // nhead

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, context=None, mask=None):
        """
        x: (B, Tq, D) queries come from x
        context: (B, Tk, D) keys/values come from context (if None, self-attention)
        mask: optional mask broadcastable to (B, H, Tq, Tk)
        """
        if context is None:
            context = x

        B, Tq, D = x.shape
        Tk = context.size(1)

        # Project to Q,K,V in model dimension
        Q = self.q_proj(x)        # (B, Tq, D)
        K = self.k_proj(context)  # (B, Tk, D)
        V = self.v_proj(context)  # (B, Tk, D)

        # Reshape into heads: (B, H, T, Dh)
        Q = Q.view(B, Tq, self.nhead, self.dh).transpose(1, 2)
        K = K.view(B, Tk, self.nhead, self.dh).transpose(1, 2)
        V = V.view(B, Tk, self.nhead, self.dh).transpose(1, 2)

        # Compute attention
        out, attn = scaled_dot_product_attention(Q, K, V, mask=mask)

        # Merge heads back: (B, Tq, D)
        out = out.transpose(1, 2).contiguous().view(B, Tq, D)
        out = self.out_proj(out)
        return out, attn


class AdditiveAttention(nn.Module):
    """
    Additive (Bahdanau) attention, common in seq2seq RNN models.

    Score:
      e_{t,s} = v^T tanh(W_h h_s + W_q q_t)

    Where:
      - h_s = encoder hidden at source position s
      - q_t = decoder hidden at target position t
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, encoder_outputs, query):
        """
        encoder_outputs: (B, S, H)
        query: (B, H)  (e.g., decoder hidden at time t)

        Returns:
          context: (B, H)
          alpha:   (B, S) attention weights over source positions
        """
        # Project encoder and query into same space
        h_proj = self.W_h(encoder_outputs)              # (B, S, H)
        q_proj = self.W_q(query).unsqueeze(1)           # (B, 1, H)

        # Scores: (B, S, 1) -> (B, S)
        scores = self.v(torch.tanh(h_proj + q_proj)).squeeze(-1)

        # Attention weights over source tokens
        alpha = F.softmax(scores, dim=1)                # (B, S)

        # Weighted sum of encoder outputs
        context = torch.bmm(alpha.unsqueeze(1), encoder_outputs).squeeze(1)  # (B, H)
        return context, alpha
