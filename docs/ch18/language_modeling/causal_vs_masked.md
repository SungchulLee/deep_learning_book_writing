# Causal vs Masked Language Modeling

## Learning Objectives

By the end of this section, you will be able to:

- Distinguish between causal and masked language modeling objectives
- Understand the architectural implications of each approach
- Implement both training objectives in PyTorch
- Select the appropriate objective for different tasks

---

## Overview

Language models can be trained with different objectives that fundamentally affect their capabilities:

| Aspect | Causal LM | Masked LM |
|--------|-----------|-----------|
| Direction | Left-to-right | Bidirectional |
| Masking | Future tokens | Random tokens |
| Primary use | Text generation | Understanding |
| Examples | GPT, LLaMA | BERT, RoBERTa |

---

## Causal Language Modeling (CLM)

### Definition

Causal LM predicts each token based only on **previous** tokens:

$$P(w_1, \ldots, w_n) = \prod_{i=1}^{n} P(w_i | w_1, \ldots, w_{i-1})$$

The model cannot "look ahead"—it's **autoregressive**.

### Training Objective

For a sequence $[w_1, w_2, \ldots, w_T]$:

$$\mathcal{L}_{CLM} = -\sum_{t=1}^{T} \log P(w_t | w_1, \ldots, w_{t-1})$$

### Causal Attention Mask

Implemented via an upper triangular mask that prevents attending to future positions:

```python
import torch
import torch.nn as nn

def create_causal_mask(seq_len: int) -> torch.Tensor:
    """Create causal attention mask."""
    # Upper triangular = True (masked), diagonal and below = False
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    return mask

# Example: seq_len = 4
# [[False, True,  True,  True ],   # Position 0 sees only itself
#  [False, False, True,  True ],   # Position 1 sees 0, 1
#  [False, False, False, True ],   # Position 2 sees 0, 1, 2
#  [False, False, False, False]]   # Position 3 sees all
```

### PyTorch Implementation

```python
class CausalLM(nn.Module):
    """GPT-style Causal Language Model."""
    
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Embedding(512, d_model)  # Learned positions
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        
        # Embeddings
        h = self.embedding(x) + self.pos_encoding(positions)
        
        # Causal mask
        mask = create_causal_mask(seq_len).to(x.device)
        
        # Transformer with causal attention
        h = self.transformer(h, h, tgt_mask=mask)
        
        return self.fc(h)


def train_causal_lm(model, data_loader, epochs=10):
    """Train causal language model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for inputs, targets in data_loader:
            # inputs: [w1, w2, ..., wT-1]
            # targets: [w2, w3, ..., wT]
            
            logits = model(inputs)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}: Loss = {total_loss/len(data_loader):.4f}")
```

### Advantages of Causal LM

1. **Natural for generation**: Matches how text is produced
2. **Efficient inference**: KV-cache for fast autoregressive decoding
3. **Simple training**: Next-token prediction is straightforward
4. **Scales well**: Foundation of modern LLMs

---

## Masked Language Modeling (MLM)

### Definition

MLM randomly masks tokens and predicts them from **bidirectional** context:

$$\mathcal{L}_{MLM} = -\sum_{i \in M} \log P(w_i | w_{\backslash M})$$

where $M$ is the set of masked positions and $w_{\backslash M}$ is the context (all unmasked tokens).

### BERT-style Masking

Standard BERT masking strategy (15% of tokens):
- 80%: Replace with `[MASK]`
- 10%: Replace with random token
- 10%: Keep original

```python
import random

def bert_masking(tokens, vocab_size, mask_prob=0.15, mask_token_id=103):
    """Apply BERT-style masking."""
    masked_tokens = tokens.clone()
    labels = torch.full_like(tokens, -100)  # -100 = ignore in loss
    
    # Randomly select positions to mask
    mask_indices = torch.bernoulli(
        torch.full(tokens.shape, mask_prob)
    ).bool()
    
    labels[mask_indices] = tokens[mask_indices]
    
    # 80%: [MASK] token
    mask_replace = mask_indices & (torch.rand(tokens.shape) < 0.8)
    masked_tokens[mask_replace] = mask_token_id
    
    # 10%: Random token
    random_replace = mask_indices & ~mask_replace & (torch.rand(tokens.shape) < 0.5)
    masked_tokens[random_replace] = torch.randint(0, vocab_size, (random_replace.sum(),))
    
    # 10%: Keep original (already set)
    
    return masked_tokens, labels
```

### PyTorch Implementation

```python
class MaskedLM(nn.Module):
    """BERT-style Masked Language Model."""
    
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Embedding(512, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)
    
    def forward(self, x, attention_mask=None):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        
        h = self.embedding(x) + self.pos_encoding(positions)
        
        # No causal mask - bidirectional attention
        h = self.transformer(h, src_key_padding_mask=attention_mask)
        
        return self.fc(h)


def train_masked_lm(model, data_loader, vocab_size, epochs=10):
    """Train masked language model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in data_loader:
            # Apply masking
            masked_inputs, labels = bert_masking(batch, vocab_size)
            
            logits = model(masked_inputs)
            loss = criterion(logits.view(-1, vocab_size), labels.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}: Loss = {total_loss/len(data_loader):.4f}")
```

### Advantages of Masked LM

1. **Bidirectional context**: Sees both left and right
2. **Better representations**: Captures full context
3. **Efficient training**: Predicts multiple tokens per sequence
4. **Strong for understanding**: Classification, QA, NER

---

## Comparison

### Information Flow

```
Causal LM (GPT):
Input:  [The] [cat] [sat] [on] [the] [mat]
Sees:   [The] → [The, cat] → [The, cat, sat] → ...

Masked LM (BERT):
Input:  [The] [MASK] [sat] [on] [the] [mat]
Sees:   All tokens except masked position
```

### Training Efficiency

| Aspect | Causal | Masked |
|--------|--------|--------|
| Tokens predicted | All | ~15% (masked only) |
| Context used | Partial | Full |
| Samples needed | More | Fewer |

### Task Suitability

| Task | Best Objective |
|------|----------------|
| Text generation | Causal |
| Text classification | Masked |
| Question answering | Masked |
| Summarization | Both |
| Translation | Encoder-Decoder |

---

## Hybrid Approaches

### Prefix LM (T5-style)

Combines bidirectional encoding with causal decoding:

```
Encoder (bidirectional): "Translate to French: Hello"
Decoder (causal): "Bonjour" (generated left-to-right)
```

### Permutation LM (XLNet)

Trains on all possible permutations, capturing bidirectional context while maintaining autoregressive factorization.

### Span Corruption (T5)

Masks contiguous spans rather than individual tokens:

```
Input:  "The [X] sat on [Y] mat"
Target: "[X] cat [Y] the"
```

---

## Summary

| Property | Causal LM | Masked LM |
|----------|-----------|-----------|
| Attention | Left-to-right | Bidirectional |
| Generation | Native | Difficult |
| Understanding | Limited | Strong |
| Training signal | Dense | Sparse |
| Modern examples | GPT-4, LLaMA | BERT, RoBERTa |

Choose **Causal LM** for generation tasks and **Masked LM** for understanding tasks.

---

## Exercises

1. Implement both objectives and compare training dynamics
2. Fine-tune a causal LM for classification (vs. masked LM)
3. Implement span corruption for T5-style training
4. Analyze attention patterns in both architectures

---

## References

1. Radford, A., et al. (2018). Improving language understanding with GPT.
2. Devlin, J., et al. (2019). BERT: Pre-training of deep bidirectional transformers.
3. Raffel, C., et al. (2020). Exploring the limits of transfer learning with T5.
