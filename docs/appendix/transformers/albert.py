#!/usr/bin/env python3
"""
ALBERT - A Lite BERT
Paper: "ALBERT: A Lite BERT for Self-supervised Learning of Language Representations" (2019)
Authors: Zhenzhong Lan et al.
Key ideas:
  1) Factorized embedding parameterization:
       vocab embedding dim (E) smaller than hidden dim (H)
       embed: vocab -> E, then project E -> H
  2) Cross-layer parameter sharing:
       reuse the same Transformer layer weights across all layers
  3) Sentence-order prediction (SOP) (often discussed vs NSP)

File: appendix/transformers/albert.py
Note: Educational implementation showing factorized embeddings + shared layer.
"""

import torch
import torch.nn as nn


class SharedTransformerBlock(nn.Module):
    """One transformer encoder layer, intended to be reused multiple times."""
    def __init__(self, d_model=768, nhead=12):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)

    def forward(self, x, src_key_padding_mask=None):
        return self.layer(x, src_key_padding_mask=src_key_padding_mask)


class ALBERT(nn.Module):
    """
    ALBERT encoder-only model with:
      - factorized embeddings
      - shared transformer block repeated num_layers times
    """
    def __init__(self, vocab_size=30000, embed_dim=128, hidden_dim=768, nhead=12, num_layers=12):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)    # vocab -> E
        self.embed_proj = nn.Linear(embed_dim, hidden_dim)        # E -> H

        self.shared_block = SharedTransformerBlock(d_model=hidden_dim, nhead=nhead)
        self.num_layers = num_layers

        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        # Token embedding in low dimension E
        x = self.token_embed(input_ids)     # (B, S, E)

        # Project to hidden dimension H
        x = self.embed_proj(x)              # (B, S, H)

        # Padding mask (True = ignore)
        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = ~attention_mask.bool()

        # Reuse the same block multiple times (parameter sharing)
        for _ in range(self.num_layers):
            x = self.shared_block(x, src_key_padding_mask=src_key_padding_mask)

        # MLM logits
        logits = self.lm_head(x)            # (B, S, vocab)
        return logits


if __name__ == "__main__":
    model = ALBERT(vocab_size=1000, embed_dim=64, hidden_dim=256, nhead=8, num_layers=4)
    ids = torch.randint(0, 1000, (2, 9))
    mask = torch.ones(2, 9, dtype=torch.long)
    logits = model(ids, attention_mask=mask)
    print("logits:", logits.shape)  # (2, 9, 1000)
