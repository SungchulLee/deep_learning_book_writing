#!/usr/bin/env python3
"""
T5 - Text-to-Text Transfer Transformer
Paper: "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" (2020)
Authors: Colin Raffel et al.
Key idea:
  - Everything is cast as text-to-text (inputs and outputs are text)
  - Encoder-decoder Transformer
  - Uses *span corruption* (sentinel tokens) for pretraining

File: appendix/transformers/t5.py
Note: Educational/simplified implementation (architecture + forward shapes).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class T5Encoder(nn.Module):
    """Transformer encoder: token embeddings -> encoder hidden states."""
    def __init__(self, vocab_size=32128, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)

        # PyTorch encoder layer uses self-attention + FFN
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

    def forward(self, input_ids, attention_mask=None):
        # input_ids: (B, S)
        x = self.embed(input_ids)  # (B, S, D)

        # attention_mask (optional): True/1 for tokens to keep, False/0 for padding
        # PyTorch uses src_key_padding_mask where True means "ignore"
        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = ~attention_mask.bool()  # (B, S)

        h = self.encoder(x, src_key_padding_mask=src_key_padding_mask)  # (B, S, D)
        return h


class T5Decoder(nn.Module):
    """Transformer decoder: autoregressive self-attention + cross-attention to encoder."""
    def __init__(self, vocab_size=32128, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)

        dec_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)

        # Language modeling head: map decoder states -> vocab logits
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, decoder_input_ids, encoder_hidden_states, encoder_attention_mask=None):
        # decoder_input_ids: (B, T)
        y = self.embed(decoder_input_ids)  # (B, T, D)

        # Causal mask prevents looking ahead in the decoder
        T = decoder_input_ids.size(1)
        causal_mask = torch.triu(torch.ones(T, T, device=y.device), diagonal=1).bool()  # (T, T)

        # Encoder padding mask: True means ignore
        mem_key_padding_mask = None
        if encoder_attention_mask is not None:
            mem_key_padding_mask = ~encoder_attention_mask.bool()  # (B, S)

        # Decode with cross-attention to encoder_hidden_states
        dec_h = self.decoder(
            tgt=y,
            memory=encoder_hidden_states,
            tgt_mask=causal_mask,
            memory_key_padding_mask=mem_key_padding_mask,
        )  # (B, T, D)

        logits = self.lm_head(dec_h)  # (B, T, vocab)
        return logits


class T5(nn.Module):
    """T5 encoder-decoder wrapper (text-to-text)."""
    def __init__(self, vocab_size=32128, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        self.encoder = T5Encoder(vocab_size, d_model, nhead, num_layers)
        self.decoder = T5Decoder(vocab_size, d_model, nhead, num_layers)

    def forward(self, input_ids, decoder_input_ids, attention_mask=None):
        enc = self.encoder(input_ids, attention_mask=attention_mask)
        logits = self.decoder(decoder_input_ids, enc, encoder_attention_mask=attention_mask)
        return logits


if __name__ == "__main__":
    model = T5(vocab_size=1000, d_model=256, nhead=8, num_layers=2)
    input_ids = torch.randint(0, 1000, (2, 7))
    dec_in = torch.randint(0, 1000, (2, 5))
    attn = torch.ones(2, 7, dtype=torch.long)
    logits = model(input_ids, dec_in, attention_mask=attn)
    print("logits:", logits.shape)  # (2, 5, 1000)
