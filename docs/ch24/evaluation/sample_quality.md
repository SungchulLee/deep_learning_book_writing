# Sample Quality

## Overview

While autoregressive models optimize likelihood, sample quality is the practical measure that matters for generation applications. High likelihood does not always translate to high-quality samples.

## Sampling Strategies

### Temperature Scaling
Divide logits by temperature $T$ before softmax:

$$p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

- $T < 1$: sharper distribution, more deterministic, higher quality but less diverse
- $T > 1$: flatter distribution, more random, more diverse but lower quality
- $T = 1$: original learned distribution

### Top-k Sampling
Sample only from the $k$ most probable tokens:

```python
def top_k_sample(logits, k=50):
    values, indices = logits.topk(k)
    probs = F.softmax(values, dim=-1)
    idx = torch.multinomial(probs, 1)
    return indices.gather(-1, idx)
```

### Nucleus (Top-p) Sampling
Sample from the smallest set of tokens whose cumulative probability exceeds $p$:

```python
def top_p_sample(logits, p=0.9):
    sorted_logits, sorted_indices = logits.sort(descending=True)
    cumulative_probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
    
    mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= p
    sorted_logits[mask] = float('-inf')
    
    probs = F.softmax(sorted_logits, dim=-1)
    idx = torch.multinomial(probs, 1)
    return sorted_indices.gather(-1, idx)
```

## Quality Metrics

| Metric | Domain | Measures |
|--------|--------|---------|
| FID | Images | Feature distribution match |
| IS | Images | Quality and diversity |
| BLEU | Text | N-gram overlap |
| MAUVE | Text | Distribution gap |
| MOS | Audio | Human perceptual quality |
