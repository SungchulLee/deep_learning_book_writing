# Transformer vs RNN vs CNN for Sequence Modeling

## Overview

| Architecture | Parallelization | Long-range | Complexity | Inductive Bias |
|--------------|-----------------|------------|------------|----------------|
| RNN/LSTM | Sequential | Difficult | O(n) | Temporal |
| CNN | Parallel | Limited | O(n) | Local patterns |
| Transformer | Parallel | Easy | O(n²) | None (learned) |

## RNN/LSTM

### Architecture

```
x₁ → [h₁] → x₂ → [h₂] → x₃ → [h₃] → ...
        ↓         ↓         ↓
       y₁        y₂        y₃
```

### Characteristics

```python
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        x = self.embed(x)
        out, (h, c) = self.lstm(x)
        return self.fc(out)
```

**Pros:**
- O(n) complexity
- Good inductive bias for sequences
- Memory efficient

**Cons:**
- Sequential processing (slow training)
- Vanishing gradients for long sequences
- Difficult to parallelize

## CNN for Sequences

### Architecture

```
[x₁ x₂ x₃ x₄ x₅]
   \_____/
    Conv1
      \_____/
       Conv2
         ...
```

### Characteristics

```python
class CNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_filters, kernel_sizes):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k, padding=k//2)
            for k in kernel_sizes
        ])
        self.fc = nn.Linear(num_filters * len(kernel_sizes), vocab_size)
    
    def forward(self, x):
        x = self.embed(x).transpose(1, 2)  # [B, C, L]
        conv_outs = [F.relu(conv(x)) for conv in self.convs]
        pooled = [F.max_pool1d(c, c.size(2)).squeeze(2) for c in conv_outs]
        return self.fc(torch.cat(pooled, dim=1))
```

**Pros:**
- Fully parallelizable
- Good for local patterns
- Efficient for fixed-size contexts

**Cons:**
- Limited receptive field (need many layers)
- Not ideal for very long dependencies

## Transformer

### Architecture

```
[x₁ x₂ x₃ x₄ x₅]
    ↓↓↓↓↓
Self-Attention (all-to-all)
    ↓↓↓↓↓
[y₁ y₂ y₃ y₄ y₅]
```

### Characteristics

```python
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(512, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        pos = torch.arange(x.size(1), device=x.device)
        x = self.embed(x) + self.pos(pos)
        return self.fc(self.transformer(x))
```

**Pros:**
- Fully parallelizable
- Direct long-range connections
- Highly expressive

**Cons:**
- O(n²) complexity
- No inherent sequential bias
- Memory intensive for long sequences

## Detailed Comparison

### Complexity Analysis

| Operation | RNN | CNN | Transformer |
|-----------|-----|-----|-------------|
| Sequential ops | O(n) | O(1) | O(1) |
| Per-layer complexity | O(n·d²) | O(k·n·d²) | O(n²·d) |
| Max path length | O(n) | O(log_k(n)) | O(1) |

### Memory Usage

For sequence length n, dimension d:

| Model | Training Memory | Inference Memory |
|-------|-----------------|------------------|
| RNN | O(n·d) | O(d) |
| CNN | O(n·d) | O(n·d) |
| Transformer | O(n²·h + n·d) | O(n²·h) with KV-cache |

### Performance by Sequence Length

```
Short (< 512):    Transformer ≈ CNN > RNN
Medium (512-2K):  Transformer > CNN > RNN
Long (2K-8K):     Efficient Transformers > CNN > RNN
Very Long (>8K):  State Space Models / Linear Attention
```

## Benchmarks (Approximate)

### Language Modeling (Perplexity, lower is better)

| Model | WikiText-103 | Parameters |
|-------|--------------|------------|
| LSTM | ~35 | 150M |
| Transformer | ~18 | 150M |
| GPT-2 | ~15 | 1.5B |

### Text Classification (Accuracy)

| Model | IMDB | SST-2 |
|-------|------|-------|
| LSTM | 89% | 87% |
| CNN | 90% | 88% |
| BERT | 95% | 94% |

### Machine Translation (BLEU)

| Model | WMT En-De |
|-------|-----------|
| LSTM Seq2Seq | 28.4 |
| Transformer | 34.4 |

## Hybrid Approaches

### Transformer + CNN

```python
class ConvTransformer(nn.Module):
    """Local convolution + global attention."""
    def __init__(self, d_model, num_heads, kernel_size=3):
        super().__init__()
        self.local_conv = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size//2)
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
    
    def forward(self, x):
        # Local features
        local = self.local_conv(x.transpose(1, 2)).transpose(1, 2)
        # Global attention
        global_out, _ = self.attention(x, x, x)
        return local + global_out
```

### Transformer + RNN

```python
class TransformerWithRecurrence(nn.Module):
    """Transformer with recurrent memory."""
    def __init__(self, d_model, num_heads, memory_size):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.memory_rnn = nn.GRU(d_model, d_model, batch_first=True)
        self.memory = None
    
    def forward(self, x):
        # Update memory
        if self.memory is not None:
            x = torch.cat([self.memory, x], dim=1)
        out, _ = self.attention(x, x, x)
        # Compress to memory
        self.memory, _ = self.memory_rnn(out)
        return out[:, -x.size(1):, :]
```

## When to Use Each

### Use RNN When:
- Streaming data (online processing)
- Very memory constrained
- Sequence order is crucial
- Short sequences (<100)

### Use CNN When:
- Local patterns matter most
- Fixed-size classification
- Fast inference needed
- Moderate sequence length

### Use Transformer When:
- Long-range dependencies important
- Parallel training available
- State-of-the-art needed
- Pre-trained models available

## Modern Alternatives

For very long sequences, consider:

| Model | Complexity | Notes |
|-------|------------|-------|
| Linear Attention | O(n) | Approximate attention |
| Mamba/S4 | O(n) | State space models |
| RWKV | O(n) | RNN-like Transformer |
| Longformer | O(n·w) | Sparse attention |

## Summary

- **RNNs**: Historical importance, still useful for streaming
- **CNNs**: Fast, local patterns, good for classification
- **Transformers**: Current SOTA, best for most NLP tasks

The field has largely converged on Transformers, but understanding all three helps choose the right tool.

## References

1. Vaswani, A., et al. (2017). "Attention Is All You Need."
2. Hochreiter, S., & Schmidhuber, J. (1997). "Long Short-Term Memory."
3. Kim, Y. (2014). "Convolutional Neural Networks for Sentence Classification."
4. Gu, A., et al. (2022). "Efficiently Modeling Long Sequences with Structured State Spaces."
