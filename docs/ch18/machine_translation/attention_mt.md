# Attention Mechanisms in Translation

## Learning Objectives

- Understand the motivation and mechanics of attention
- Compare additive, multiplicative, and dot-product attention
- Implement attention in a seq2seq translation model

## Motivation

The encoder-decoder bottleneck forces the entire source sentence into a single fixed-length vector. Attention (Bahdanau et al., 2015) solves this by allowing the decoder to look at all encoder states at each generation step.

## Bahdanau (Additive) Attention

At each decoder step $t$, compute attention weights over all encoder states:

### Alignment Score

$$e_{t,i} = \mathbf{v}^T \tanh(\mathbf{W}_1 \mathbf{s}_{t-1} + \mathbf{W}_2 \mathbf{h}_i)$$

where $\mathbf{s}_{t-1}$ is the previous decoder state and $\mathbf{h}_i$ is the $i$-th encoder state.

### Attention Weights

$$\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_{k=1}^{T_x} \exp(e_{t,k})}$$

### Context Vector

$$\mathbf{c}_t = \sum_{i=1}^{T_x} \alpha_{t,i} \mathbf{h}_i$$

The context vector $\mathbf{c}_t$ is a weighted combination of encoder states, providing the decoder with relevant source information at each step.

## Attention Variants

### Luong (Multiplicative) Attention

$$e_{t,i} = \mathbf{s}_t^T \mathbf{W} \mathbf{h}_i$$

### Dot-Product Attention

$$e_{t,i} = \mathbf{s}_t^T \mathbf{h}_i$$

### Scaled Dot-Product Attention

$$e_{t,i} = \frac{\mathbf{s}_t^T \mathbf{h}_i}{\sqrt{d_k}}$$

Scaling prevents dot products from growing too large in high dimensions, which would push softmax into saturated regions with tiny gradients.

## Comparison

| Variant | Computation | Parameters | Speed |
|---------|------------|------------|-------|
| Additive | $\mathbf{v}^T \tanh(\mathbf{W}_1 \mathbf{s} + \mathbf{W}_2 \mathbf{h})$ | $O(d^2)$ | Slower |
| Multiplicative | $\mathbf{s}^T \mathbf{W} \mathbf{h}$ | $O(d^2)$ | Moderate |
| Dot-product | $\mathbf{s}^T \mathbf{h}$ | None | Fastest |
| Scaled dot-product | $\mathbf{s}^T \mathbf{h} / \sqrt{d}$ | None | Fastest |

## Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BahdanauAttention(nn.Module):
    def __init__(self, enc_dim, dec_dim, attn_dim=256):
        super().__init__()
        self.W1 = nn.Linear(dec_dim, attn_dim, bias=False)
        self.W2 = nn.Linear(enc_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, decoder_state, encoder_outputs, mask=None):
        # decoder_state: (B, dec_dim)
        # encoder_outputs: (B, T, enc_dim)
        scores = self.v(torch.tanh(
            self.W1(decoder_state).unsqueeze(1) + self.W2(encoder_outputs)
        )).squeeze(-1)  # (B, T)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)  # (B, T)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attn_weights
```

## Attention as Soft Alignment

Attention weights provide an implicit soft alignment between source and target tokens. Visualizing the attention matrix reveals which source words each target word "attends to" — often closely matching linguistic alignment.

This interpretability is valuable: in financial translation, verifying that numerical values and entity names are properly attended to helps ensure translation accuracy for critical content.

## Impact

Attention was arguably the single most important innovation in NMT. It eliminated the bottleneck, improved long-sentence translation, provided interpretability, and laid the groundwork for the Transformer architecture (which uses self-attention as its core mechanism).

## References

1. Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural Machine Translation by Jointly Learning to Align and Translate. *ICLR*.
2. Luong, M. T., Pham, H., & Manning, C. D. (2015). Effective Approaches to Attention-Based NMT. *EMNLP*.
