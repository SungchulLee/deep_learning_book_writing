# GPT: Generative Pre-trained Transformer

## Overview

GPT (Radford et al., 2018) is a **decoder-only** Transformer that pioneered large-scale language model pre-training. The GPT architecture has become the foundation for modern LLMs including GPT-3, GPT-4, and Claude.

## Architecture

GPT uses only the **Transformer decoder stack** with causal masking:

$$\text{Input} \xrightarrow{\text{Embed + PE}} \mathbf{X} \xrightarrow{\text{Decoder} \times N} \mathbf{H} \xrightarrow{\text{LM Head}} P(x_{n+1})$$

Key difference from encoder-decoder: No cross-attention (no encoder to attend to).

| Model | Layers | Hidden | Heads | Parameters |
|-------|--------|--------|-------|------------|
| GPT-1 | 12 | 768 | 12 | 117M |
| GPT-2 | 48 | 1600 | 25 | 1.5B |
| GPT-3 | 96 | 12288 | 96 | 175B |

## Pre-training Objective

**Autoregressive Language Modeling**: Predict the next token given previous tokens.

$$\mathcal{L} = -\sum_{i=1}^{n} \log P(x_i | x_1, \ldots, x_{i-1})$$

- Input: $(x_1, x_2, \ldots, x_n)$
- Target: $(x_2, x_3, \ldots, x_{n+1})$

Each position predicts the next token. The causal mask ensures position $i$ only sees tokens $1, \ldots, i$.

## Key Design Choices

### Causal Masking

The decoder uses masked self-attention:

$$\mathbf{A} = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}} + \mathbf{M}_{\text{causal}}\right)$$

This enables:
- **Parallel training**: All positions computed simultaneously
- **Autoregressive inference**: Generate one token at a time

### Unified Interface

All tasks become text completion:

| Task | Format |
|------|--------|
| Classification | "Review: Great movie! Sentiment:" → "positive" |
| Translation | "Translate to French: Hello" → "Bonjour" |
| QA | "Q: What is 2+2? A:" → "4" |
| Code | "def fibonacci(n):" → function body |

No task-specific heads—everything is generation.

## Generation Process

### Autoregressive Sampling

```
Step 1: Input "The" → Sample "cat"
Step 2: Input "The cat" → Sample "sat"
Step 3: Input "The cat sat" → Sample "on"
...
```

### Decoding Strategies

| Strategy | Description |
|----------|-------------|
| Greedy | Take argmax at each step |
| Temperature | Scale logits by $1/T$ before softmax |
| Top-k | Sample from top $k$ tokens |
| Top-p (nucleus) | Sample from smallest set with cumulative prob ≥ $p$ |

```python
# Temperature sampling
logits = model(input_ids)[:, -1, :]
probs = F.softmax(logits / temperature, dim=-1)
next_token = torch.multinomial(probs, num_samples=1)
```

## PyTorch Implementation (Simplified)

```python
import torch
import torch.nn as nn

class GPTBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))  # Pre-norm
        x = x + self.ffn(self.norm2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, max_len):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([
            GPTBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
    
    def forward(self, x):
        # x: (batch, seq_len)
        pos = torch.arange(x.size(1), device=x.device)
        x = self.token_emb(x) + self.pos_emb(pos)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        logits = self.lm_head(x)  # (batch, seq_len, vocab_size)
        return logits
```

## In-Context Learning

GPT's surprising capability: learning from examples in the prompt without weight updates.

### Zero-shot
```
Translate English to French:
sea otter =>
```

### Few-shot
```
Translate English to French:
sea otter => loutre de mer
cheese => fromage
hello =>
```

The model uses the examples to infer the task, despite never being explicitly trained for translation.

## Comparison with BERT

| Aspect | GPT | BERT |
|--------|-----|------|
| Architecture | Decoder-only | Encoder-only |
| Attention | Causal (unidirectional) | Bidirectional |
| Pre-training | Next token prediction | Masked token prediction |
| Strengths | Generation | Understanding |
| Context | Left context only | Full context |

## Evolution: GPT-1 → GPT-2 → GPT-3

### GPT-1 (2018)
- First to show pre-train + fine-tune works
- Task-specific fine-tuning still needed

### GPT-2 (2019)
- "Language models are unsupervised multitask learners"
- Zero-shot performance competitive with fine-tuned models
- Withheld release due to misuse concerns

### GPT-3 (2020)
- 175B parameters
- Few-shot learning without fine-tuning
- Emergent capabilities at scale

## Why Decoder-Only Dominates

1. **Unified interface**: All tasks become generation
2. **Simpler architecture**: Single stack, no encoder-decoder interaction
3. **Scales well**: Easier to train very large models
4. **Flexibility**: Can do both understanding and generation via prompting

## Summary

GPT's contributions:

1. **Decoder-only pre-training**: Simple, scalable architecture
2. **Autoregressive generation**: Flexible text production
3. **In-context learning**: Task adaptation without fine-tuning
4. **Unified paradigm**: Everything is text completion

The GPT architecture has become the default for modern LLMs, demonstrating that simple autoregressive modeling at scale produces remarkably capable systems.

## References

- Radford et al., "Improving Language Understanding by Generative Pre-Training" (GPT-1, 2018)
- Radford et al., "Language Models are Unsupervised Multitask Learners" (GPT-2, 2019)
- Brown et al., "Language Models are Few-Shot Learners" (GPT-3, 2020)
