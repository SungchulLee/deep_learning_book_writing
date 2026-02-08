#!/usr/bin/env python3
"""
DistilBERT - Distilled version of BERT
Paper: "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter" (2019)
Authors: Victor Sanh et al.
Key idea:
  - Knowledge distillation: train a smaller student Transformer to match a larger teacher
  - Typically fewer layers, similar hidden size

File: appendix/transformers/distilbert.py
Note: Educational implementation of a smaller encoder-only Transformer.
"""

import torch
import torch.nn as nn


class DistilBERT(nn.Module):
    """
    DistilBERT-like encoder-only Transformer.

    This resembles BERT/RoBERTa but uses fewer encoder layers.
    Distillation happens during training (not shown here).
    """
    def __init__(self, vocab_size=30522, d_model=768, nhead=12, num_layers=6):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)

        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        x = self.embed(input_ids)

        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = ~attention_mask.bool()

        h = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        logits = self.lm_head(h)
        return logits


if __name__ == "__main__":
    model = DistilBERT(vocab_size=1000, d_model=256, nhead=8, num_layers=2)
    ids = torch.randint(0, 1000, (2, 10))
    mask = torch.ones(2, 10, dtype=torch.long)
    logits = model(ids, attention_mask=mask)
    print("logits:", logits.shape)  # (2, 10, 1000)
