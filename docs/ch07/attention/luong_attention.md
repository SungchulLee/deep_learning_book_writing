# Luong Attention

## Introduction

Luong attention (Luong et al., 2015) introduced several simplifications and alternatives to Bahdanau attention for neural machine translation. The key contributions were: (1) simpler scoring functions that are faster to compute, (2) a comparison of **global** vs. **local** attention strategies, and (3) architectural simplifications that streamlined the encoder-decoder attention pipeline.

While Bahdanau attention uses an additive score with a feedforward network, Luong's primary variants use **multiplicative** scoring that can be computed via efficient matrix operations. This computational advantage, combined with comparable translation quality, made Luong-style attention widely adopted and directly influenced the scaled dot-product attention used in Transformers.

## Score Functions

Luong proposed three scoring alternatives:

### Dot-Product Score

$$\text{score}(\mathbf{s}_t, \mathbf{h}_i) = \mathbf{s}_t^\top \mathbf{h}_i$$

The simplest option—requires $d_{\text{dec}} = d_{\text{enc}}$. No learnable parameters in the score function itself. Efficient because it reduces to a single matrix multiplication across all source positions simultaneously.

### General (Bilinear) Score

$$\text{score}(\mathbf{s}_t, \mathbf{h}_i) = \mathbf{s}_t^\top \mathbf{W}_a \mathbf{h}_i$$

Introduces a learnable weight matrix $\mathbf{W}_a \in \mathbb{R}^{d_{\text{dec}} \times d_{\text{enc}}}$ that allows the model to learn an arbitrary bilinear interaction between decoder and encoder states. This accommodates different dimensions for $\mathbf{s}_t$ and $\mathbf{h}_i$ and is more expressive than the plain dot product.

### Concat Score

$$\text{score}(\mathbf{s}_t, \mathbf{h}_i) = \mathbf{v}^\top \tanh(\mathbf{W}_a [\mathbf{s}_t; \mathbf{h}_i])$$

This is essentially Bahdanau-style additive attention reformulated with concatenation instead of separate projections. The concatenation $[\mathbf{s}_t; \mathbf{h}_i]$ is fed through a single linear layer followed by $\tanh$.

### Comparison

| Score | Formula | Parameters | Computation | Expressiveness |
|-------|---------|------------|-------------|----------------|
| **Dot** | $\mathbf{s}^\top \mathbf{h}$ | 0 | $O(d)$ | Bilinear (identity) |
| **General** | $\mathbf{s}^\top \mathbf{W}_a \mathbf{h}$ | $d_s \times d_h$ | $O(d^2)$ | Bilinear (learned) |
| **Concat** | $\mathbf{v}^\top \tanh(\mathbf{W}_a [\mathbf{s}; \mathbf{h}])$ | $(d_s + d_h) \cdot d_a + d_a$ | $O(d \cdot d_a)$ | Nonlinear |

Luong found that the **general** and **dot** scores performed comparably to the more expensive concat score, demonstrating that simpler scoring functions suffice when combined with learned projections.

## Architectural Differences from Bahdanau

### Decoder State Used for Scoring

**Bahdanau:** Uses the decoder state from the **previous** step $\mathbf{s}_{t-1}$ to compute attention, then feeds the context vector into the current RNN step.

**Luong:** Uses the decoder state from the **current** step $\mathbf{s}_t$ (the "top" hidden state of a stacked LSTM), then combines the context with this state to form the attentional hidden state:

$$\tilde{\mathbf{h}}_t = \tanh(\mathbf{W}_c [\mathbf{c}_t; \mathbf{s}_t])$$

This attentional hidden state $\tilde{\mathbf{h}}_t$ is then used for prediction and fed to the next decoder step.

### Information Flow Comparison

```
Bahdanau:
  s_{t-1} ──→ attention ──→ c_t ──→ RNN(s_{t-1}, y_{t-1}, c_t) ──→ s_t ──→ output

Luong:
  s_{t-1} ──→ RNN(s_{t-1}, y_{t-1}) ──→ s_t ──→ attention ──→ c_t
                                          └──→ tanh(W_c [c_t; s_t]) ──→ output
```

The Luong formulation is simpler: the RNN step and attention step are decoupled, and the context vector is combined with the decoder state only at the output stage.

## Global vs. Local Attention

### Global Attention

Global attention attends to **all** source positions at every decoder step:

$$\alpha_{ti} = \frac{\exp(\text{score}(\mathbf{s}_t, \mathbf{h}_i))}{\sum_{j=1}^{T_s} \exp(\text{score}(\mathbf{s}_t, \mathbf{h}_j))}$$

$$\mathbf{c}_t = \sum_{i=1}^{T_s} \alpha_{ti} \mathbf{h}_i$$

This is equivalent to what Bahdanau attention does, just with different score functions.

### Local Attention

Local attention restricts the attention window to a **subset** of source positions around an aligned position $p_t$, reducing computational cost for long sequences:

**Step 1:** Predict alignment position:

- **Monotonic (local-m):** $p_t = t$ (assume roughly monotonic alignment)
- **Predictive (local-p):** $p_t = T_s \cdot \sigma(\mathbf{v}_p^\top \tanh(\mathbf{W}_p \mathbf{s}_t))$

**Step 2:** Attend within a window $[p_t - D, p_t + D]$ for some window size $D$:

$$\alpha_{ti} = \text{softmax}(\text{score}(\mathbf{s}_t, \mathbf{h}_i)) \cdot \exp\left(-\frac{(i - p_t)^2}{2\sigma^2}\right)$$

The Gaussian term favours positions near the predicted alignment, providing a soft locality bias.

### Global vs. Local Trade-offs

| Aspect | Global | Local |
|--------|--------|-------|
| Attention scope | All source positions | Window of size $2D + 1$ |
| Complexity per step | $O(T_s)$ | $O(D)$ |
| Long sequences | Expensive | Efficient |
| Non-monotonic alignment | Natural | Requires predictive $p_t$ |
| Gradient flow | To all encoder states | Only within window |

In practice, global attention with efficient implementations (batched matrix multiplication) proved sufficient for most sequence lengths encountered in NMT, and local attention saw less adoption. However, the idea of restricting the attention window resurfaced in efficient Transformer variants like sliding window attention (Beltagy et al., 2020).

## PyTorch Implementation

### Luong Attention Module

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class LuongAttention(nn.Module):
    """
    Luong Attention with multiple scoring options.
    
    Supports dot, general, and concat scoring functions.
    Uses the current decoder state (not previous) for scoring.
    """
    
    def __init__(
        self,
        decoder_dim: int,
        encoder_dim: int,
        score_type: str = 'general'
    ):
        """
        Args:
            decoder_dim: Dimension of decoder hidden state
            encoder_dim: Dimension of encoder hidden states
            score_type: One of 'dot', 'general', 'concat'
        """
        super().__init__()
        self.score_type = score_type
        self.decoder_dim = decoder_dim
        self.encoder_dim = encoder_dim
        
        if score_type == 'dot':
            assert decoder_dim == encoder_dim, \
                f"Dot score requires decoder_dim == encoder_dim, got {decoder_dim} != {encoder_dim}"
                
        elif score_type == 'general':
            self.W_a = nn.Linear(encoder_dim, decoder_dim, bias=False)
            
        elif score_type == 'concat':
            self.W_a = nn.Linear(decoder_dim + encoder_dim, decoder_dim, bias=False)
            self.v = nn.Linear(decoder_dim, 1, bias=False)
        else:
            raise ValueError(f"Unknown score type: {score_type}")
        
        # Attentional hidden state: combines context and decoder state
        self.W_c = nn.Linear(decoder_dim + encoder_dim, decoder_dim, bias=False)
    
    def score(self, decoder_state, encoder_outputs):
        """
        Compute alignment scores.
        
        Args:
            decoder_state: (batch, decoder_dim) — current decoder state
            encoder_outputs: (batch, src_len, encoder_dim)
            
        Returns:
            scores: (batch, src_len)
        """
        if self.score_type == 'dot':
            # s^T h for all encoder positions via batched matmul
            return torch.bmm(
                encoder_outputs, decoder_state.unsqueeze(-1)
            ).squeeze(-1)
            
        elif self.score_type == 'general':
            # s^T W_a h = (W_a h)^T s
            transformed = self.W_a(encoder_outputs)  # (batch, src_len, decoder_dim)
            return torch.bmm(
                transformed, decoder_state.unsqueeze(-1)
            ).squeeze(-1)
            
        elif self.score_type == 'concat':
            src_len = encoder_outputs.size(1)
            # Expand decoder state to match encoder sequence length
            dec_expanded = decoder_state.unsqueeze(1).expand(-1, src_len, -1)
            concat = torch.cat([dec_expanded, encoder_outputs], dim=-1)
            return self.v(torch.tanh(self.W_a(concat))).squeeze(-1)
    
    def forward(
        self,
        decoder_state: torch.Tensor,
        encoder_outputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            decoder_state: (batch, decoder_dim) — current decoder hidden state
            encoder_outputs: (batch, src_len, encoder_dim) — all encoder states
            mask: (batch, src_len) — 1 for valid, 0 for padding
            
        Returns:
            attentional_hidden: (batch, decoder_dim) — context-enriched state
            attn_weights: (batch, src_len)
        """
        # Compute scores and weights
        scores = self.score(decoder_state, encoder_outputs)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        
        # Context vector: weighted sum of encoder outputs
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        # Attentional hidden state: tanh(W_c [c_t; s_t])
        attentional_hidden = torch.tanh(
            self.W_c(torch.cat([context, decoder_state], dim=-1))
        )
        
        return attentional_hidden, attn_weights
```

### Local Attention

```python
class LuongLocalAttention(nn.Module):
    """
    Luong Local Attention (predictive alignment).
    
    Predicts an alignment position and attends within a
    Gaussian-weighted window around it.
    """
    
    def __init__(
        self,
        decoder_dim: int,
        encoder_dim: int,
        window_size: int = 5,
        sigma: float = None
    ):
        super().__init__()
        self.D = window_size
        self.sigma = sigma or (window_size / 2.0)
        
        # Alignment predictor: p_t = T_s * sigmoid(v_p^T tanh(W_p s_t))
        self.W_p = nn.Linear(decoder_dim, decoder_dim, bias=False)
        self.v_p = nn.Linear(decoder_dim, 1, bias=False)
        
        # Score function (general)
        self.W_a = nn.Linear(encoder_dim, decoder_dim, bias=False)
        
        # Output combination
        self.W_c = nn.Linear(decoder_dim + encoder_dim, decoder_dim, bias=False)
    
    def forward(
        self,
        decoder_state: torch.Tensor,
        encoder_outputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            decoder_state: (batch, decoder_dim)
            encoder_outputs: (batch, src_len, encoder_dim)
            
        Returns:
            attentional_hidden: (batch, decoder_dim)
            attn_weights: (batch, src_len) — full-length weights (zero outside window)
        """
        batch_size, src_len, _ = encoder_outputs.shape
        
        # Predict alignment position: p_t ∈ [0, src_len)
        p_t = src_len * torch.sigmoid(
            self.v_p(torch.tanh(self.W_p(decoder_state)))
        ).squeeze(-1)  # (batch,)
        
        # Compute general scores for all positions
        transformed = self.W_a(encoder_outputs)
        scores = torch.bmm(
            transformed, decoder_state.unsqueeze(-1)
        ).squeeze(-1)  # (batch, src_len)
        
        # Apply Gaussian weighting centred at p_t
        positions = torch.arange(src_len, device=scores.device).float()  # (src_len,)
        gaussian = torch.exp(
            -((positions.unsqueeze(0) - p_t.unsqueeze(1)) ** 2) / (2 * self.sigma ** 2)
        )  # (batch, src_len)
        
        # Zero out positions outside window [p_t - D, p_t + D]
        window_mask = (
            (positions.unsqueeze(0) >= (p_t.unsqueeze(1) - self.D)) &
            (positions.unsqueeze(0) <= (p_t.unsqueeze(1) + self.D))
        ).float()
        
        if mask is not None:
            window_mask = window_mask * mask
            
        scores = scores.masked_fill(window_mask == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1) * gaussian
        
        # Renormalise after Gaussian weighting
        attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-10)
        
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        attentional_hidden = torch.tanh(
            self.W_c(torch.cat([context, decoder_state], dim=-1))
        )
        
        return attentional_hidden, attn_weights
```

### Demonstration

```python
def demonstrate_luong_variants():
    """Compare Luong attention scoring variants."""
    torch.manual_seed(42)
    
    batch_size = 2
    src_len = 10
    decoder_dim = 256
    encoder_dim = 256  # Same dim for dot-product compatibility
    
    encoder_outputs = torch.randn(batch_size, src_len, encoder_dim)
    decoder_state = torch.randn(batch_size, decoder_dim)
    
    print("Luong Attention Variants")
    print("=" * 60)
    
    for score_type in ['dot', 'general', 'concat']:
        attn = LuongAttention(decoder_dim, encoder_dim, score_type)
        output, weights = attn(decoder_state, encoder_outputs)
        
        params = sum(p.numel() for p in attn.parameters())
        print(f"\n{score_type.upper()} score:")
        print(f"  Parameters: {params:,}")
        print(f"  Output shape: {output.shape}")
        print(f"  Weights (sample 1): {weights[0].detach().numpy().round(3)}")
        print(f"  Max attention: position {weights[0].argmax().item()}")


if __name__ == "__main__":
    demonstrate_luong_variants()
```

## Input Feeding

Luong introduced **input feeding**, where the attentional hidden state $\tilde{\mathbf{h}}_t$ from the previous step is concatenated with the current input embedding before being fed to the decoder RNN:

$$\text{RNN input}_t = [\text{Embed}(y_{t-1}); \tilde{\mathbf{h}}_{t-1}]$$

This allows the model to be aware of previous alignment decisions, improving translation of complex reordering patterns.

```python
class LuongDecoderWithInputFeeding(nn.Module):
    """Luong decoder with input feeding."""
    
    def __init__(self, vocab_size, embed_dim, hidden_dim, encoder_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = LuongAttention(hidden_dim, encoder_dim, score_type='general')
        
        # Input feeding: RNN receives [embedding; previous attentional state]
        self.rnn = nn.LSTM(embed_dim + hidden_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        
    def forward_step(self, input_token, hidden, cell, prev_attn, encoder_outputs, mask=None):
        embedded = self.embedding(input_token).unsqueeze(1)  # (batch, 1, embed_dim)
        
        # Input feeding: concatenate with previous attentional state
        rnn_input = torch.cat([embedded, prev_attn.unsqueeze(1)], dim=-1)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        
        # Compute attention using current hidden state (Luong-style)
        attn_hidden, weights = self.attention(
            hidden.squeeze(0), encoder_outputs, mask
        )
        
        logits = self.fc_out(attn_hidden)
        return logits, hidden, cell, attn_hidden, weights
```

## Connection to Scaled Dot-Product Attention

Luong's dot-product attention is a direct precursor to the Transformer's scaled dot-product attention:

| Feature | Luong Dot | Transformer Scaled Dot-Product |
|---------|-----------|-------------------------------|
| Score | $\mathbf{s}^\top \mathbf{h}$ | $\mathbf{q}^\top \mathbf{k} / \sqrt{d_k}$ |
| Scaling | None | $1/\sqrt{d_k}$ |
| Q, K, V projections | Implicit (RNN states) | Explicit learned projections |
| Multi-head | No | Yes |
| Context | RNN seq2seq | Self-attention / cross-attention |

The Transformer essentially takes Luong's dot-product score, adds scaling for gradient stability, wraps it in learned projections, and applies it in a multi-head, self-attention setting. The general score $\mathbf{s}^\top \mathbf{W}_a \mathbf{h}$ can be seen as a single-head version with the projection matrix factored differently.

## Summary

| Aspect | Detail |
|--------|--------|
| **Score variants** | Dot ($\mathbf{s}^\top \mathbf{h}$), General ($\mathbf{s}^\top \mathbf{W} \mathbf{h}$), Concat |
| **Key simplification** | Use current decoder state $\mathbf{s}_t$, not previous $\mathbf{s}_{t-1}$ |
| **Attentional state** | $\tilde{\mathbf{h}}_t = \tanh(\mathbf{W}_c[\mathbf{c}_t; \mathbf{s}_t])$ |
| **Global vs. Local** | Global attends to all; Local uses predicted window |
| **Input feeding** | Previous attentional state fed back to decoder input |
| **Legacy** | Dot-product scoring → scaled dot-product in Transformers |

## References

1. Luong, M.-T., Pham, H., & Manning, C. D. (2015). "Effective Approaches to Attention-based Neural Machine Translation." *EMNLP*.
2. Bahdanau, D., Cho, K., & Bengio, Y. (2014). "Neural Machine Translation by Jointly Learning to Align and Translate." *arXiv:1409.0473*.
3. Vaswani, A., et al. (2017). "Attention Is All You Need." *NeurIPS*.
4. Beltagy, I., Peters, M., & Cohan, A. (2020). "Longformer: The Long-Document Transformer." *arXiv:2004.05150*.
