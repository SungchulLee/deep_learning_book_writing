# Encoder vs Decoder vs Encoder-Decoder

## Overview

| Architecture | Attention | Examples | Best For |
|--------------|-----------|----------|----------|
| Encoder-only | Bidirectional | BERT, RoBERTa | Understanding |
| Decoder-only | Causal | GPT, LLaMA | Generation |
| Encoder-Decoder | Both | T5, BART | Seq2Seq |

## Encoder-Only (BERT-style)

**Attention**: All positions see all positions (bidirectional)

```python
class EncoderOnly(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
    
    def forward(self, x):
        return self.encoder(self.embed(x))  # No causal mask
```

**Use cases**: Classification, NER, QA extraction, embeddings

## Decoder-Only (GPT-style)

**Attention**: Causal (each position sees only past)

```python
class DecoderOnly(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, num_heads, batch_first=True),
            num_layers
        )
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.register_buffer('mask', torch.triu(torch.ones(2048, 2048), 1).bool())
    
    def forward(self, x):
        seq_len = x.size(1)
        h = self.layers(self.embed(x), mask=self.mask[:seq_len, :seq_len])
        return self.lm_head(h)
```

**Use cases**: Text generation, chat, code, in-context learning

## Encoder-Decoder (T5-style)

**Attention**: Encoder bidirectional + Decoder causal + Cross-attention

```python
class EncoderDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, num_heads, batch_first=True),
            num_layers
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, num_heads, batch_first=True),
            num_layers
        )
        self.lm_head = nn.Linear(d_model, vocab_size)
    
    def forward(self, src, tgt):
        memory = self.encoder(self.embed(src))
        tgt_mask = torch.triu(torch.ones(tgt.size(1), tgt.size(1)), 1).bool()
        h = self.decoder(self.embed(tgt), memory, tgt_mask=tgt_mask)
        return self.lm_head(h)
```

**Use cases**: Translation, summarization, text-to-text tasks

## Comparison

| Aspect | Encoder | Decoder | Enc-Dec |
|--------|---------|---------|---------|
| Context | Bidirectional | Causal | Both |
| Generation | ✗ | ✓ | ✓ |
| Understanding | ✓✓ | ✓ | ✓ |
| Parameters | 1x | 1x | ~2x |
| Modern Popularity | Medium | **High** | Medium |

## When to Use

- **Encoder-only**: Classification, embeddings, extraction tasks
- **Decoder-only**: Generation, chat, few-shot learning (most popular today)
- **Encoder-Decoder**: Translation, summarization with distinct source/target

## References

1. Devlin, J., et al. (2019). "BERT."
2. Radford, A., et al. (2019). "GPT-2."
3. Raffel, C., et al. (2020). "T5."
