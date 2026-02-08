#!/usr/bin/env python3
"""
RoBERTa - Robustly Optimized BERT Pretraining Approach
Paper: "RoBERTa: A Robustly Optimized BERT Pretraining Approach" (2019)
Authors: Yinhan Liu et al.
Key differences vs BERT (high-level):
  - Trains longer, with larger batches and more data
  - Removes next sentence prediction (NSP)
  - Uses dynamic masking

File: appendix/transformers/roberta.py
Note: Educational implementation of a BERT-like encoder-only Transformer.
"""

import torch
import torch.nn as nn


class RoBERTa(nn.Module):
    """
    Encoder-only Transformer for masked language modeling (MLM).

    Inputs:
      input_ids: (B, S)
      attention_mask: (B, S) 1 for tokens, 0 for padding

    Outputs:
      logits: (B, S, vocab_size) token-level vocab logits for MLM
    """
    def __init__(self, vocab_size=50265, d_model=768, nhead=12, num_layers=12):
        super().__init__()

        # Token embeddings (RoBERTa also uses learned positional embeddings; omitted for brevity)
        self.embed = nn.Embedding(vocab_size, d_model)

        # Encoder stack
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # MLM head: predict original token IDs
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        x = self.embed(input_ids)  # (B, S, D)

        # Convert attention_mask to src_key_padding_mask (True = ignore)
        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = ~attention_mask.bool()

        h = self.encoder(x, src_key_padding_mask=src_key_padding_mask)  # (B, S, D)
        logits = self.lm_head(h)                                        # (B, S, vocab)
        return logits


if __name__ == "__main__":
    model = RoBERTa(vocab_size=1000, d_model=256, nhead=8, num_layers=2)
    input_ids = torch.randint(0, 1000, (2, 8))
    mask = torch.ones(2, 8, dtype=torch.long)
    logits = model(input_ids, attention_mask=mask)
    print("logits:", logits.shape)  # (2, 8, 1000)
