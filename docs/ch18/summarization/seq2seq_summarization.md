# Seq2Seq Summarization

## Overview

Sequence-to-sequence models with attention form the foundation for abstractive summarization.

## Architecture

```
Document tokens → Encoder → Hidden states → Attention → Decoder → Summary tokens
```

## Training

### Teacher Forcing

$$\mathcal{L} = -\sum_{t=1}^{T} \log P(y_t^* | y_{<t}^*, \mathbf{x})$$

### Label Smoothing

Prevent overconfidence by distributing probability mass:

$$y_i^{\text{smooth}} = (1 - \epsilon) \cdot y_i + \frac{\epsilon}{|V|}$$

## Decoding Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| Greedy | Select highest probability token | Fast, lower quality |
| Beam search | Maintain top-$k$ hypotheses | Standard for summarization |
| Top-k sampling | Sample from top-$k$ tokens | More diverse output |
| Length penalty | $\text{score} / T^\alpha$ | Control output length |

## Handling Long Documents

- **Truncation**: Process first 512/1024 tokens only
- **Hierarchical**: Encode sentence-level then document-level
- **Sliding window**: Process overlapping chunks
- **Longformer/LED**: Sparse attention for long sequences

## Summary

1. Seq2seq with attention is the base architecture for abstractive summarization
2. Beam search with length penalty is the standard decoding approach
3. Long document handling remains a key challenge
