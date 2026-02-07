# Bahdanau Attention

## Introduction

Bahdanau attention (Bahdanau et al., 2014), also called **additive attention**, was the first attention mechanism applied to neural machine translation. It solved the critical information bottleneck in encoder-decoder architectures by allowing the decoder to dynamically focus on different parts of the source sequence at each generation step.

The key insight was replacing the fixed context vector with a **dynamic, step-dependent** weighted sum of encoder hidden states. This seemingly simple modification dramatically improved translation quality, especially for long sentences, and laid the groundwork for all subsequent attention mechanisms including the Transformer.

## The Bottleneck Problem

In the original sequence-to-sequence architecture (Sutskever et al., 2014), the encoder compresses the entire source sequence into a single fixed-length vector:

$$\mathbf{c} = \mathbf{h}_T^{\text{enc}}$$

The decoder then generates every output token conditioned on this same $\mathbf{c}$. For long sequences, this creates a severe information bottleneck—all source information must be squeezed through a fixed-dimensional vector regardless of sequence length.

**Empirical evidence:** Cho et al. (2014) showed that seq2seq performance degrades sharply for sentences longer than about 20 tokens, consistent with the bottleneck hypothesis.

## Mathematical Formulation

### The Additive Score Function

Bahdanau attention computes compatibility between the decoder state $\mathbf{s}_{t-1}$ and each encoder hidden state $\mathbf{h}_i$ using a learned feedforward network:

$$e_{ti} = \mathbf{v}^\top \tanh(\mathbf{W}_s \mathbf{s}_{t-1} + \mathbf{W}_h \mathbf{h}_i)$$

where:

- $\mathbf{s}_{t-1} \in \mathbb{R}^{d_{\text{dec}}}$: Decoder hidden state at previous time step
- $\mathbf{h}_i \in \mathbb{R}^{d_{\text{enc}}}$: Encoder hidden state at position $i$
- $\mathbf{W}_s \in \mathbb{R}^{d_a \times d_{\text{dec}}}$: Decoder state projection
- $\mathbf{W}_h \in \mathbb{R}^{d_a \times d_{\text{enc}}}$: Encoder state projection
- $\mathbf{v} \in \mathbb{R}^{d_a}$: Learned weight vector
- $d_a$: Attention hidden dimension (a hyperparameter)

### Attention Weights

The scores are normalised via softmax to form a valid probability distribution:

$$\alpha_{ti} = \frac{\exp(e_{ti})}{\sum_{j=1}^{T_s} \exp(e_{tj})}$$

where $T_s$ is the source sequence length.

### Context Vector

The context vector is a weighted sum of encoder states:

$$\mathbf{c}_t = \sum_{i=1}^{T_s} \alpha_{ti} \mathbf{h}_i$$

Unlike the original seq2seq, this context vector is **different at each decoder step** $t$, allowing the model to focus on different parts of the source as needed.

### Decoder Update

The context vector is combined with the previous decoder state and previous output to produce the next state:

$$\mathbf{s}_t = f(\mathbf{s}_{t-1}, y_{t-1}, \mathbf{c}_t)$$

where $f$ is typically a GRU or LSTM cell. In Bahdanau's original formulation, the context vector feeds into the GRU alongside the previous output embedding.

## Why "Additive"?

The score function is called additive because it **sums** the transformed query and key before applying the nonlinearity:

$$e_{ti} = \mathbf{v}^\top \tanh(\underbrace{\mathbf{W}_s \mathbf{s}_{t-1}}_{\text{transformed query}} + \underbrace{\mathbf{W}_h \mathbf{h}_i}_{\text{transformed key}})$$

This contrasts with **multiplicative** (dot-product) attention where query and key interact through multiplication. The additive formulation is more expressive because the $\tanh$ nonlinearity allows the score function to learn complex, non-linear compatibility patterns between queries and keys.

### Expressiveness Analysis

The additive score function can be interpreted as a single-hidden-layer feedforward network:

$$e_{ti} = \text{MLP}([\mathbf{s}_{t-1}; \mathbf{h}_i])$$

This makes it a **universal function approximator** (given sufficient $d_a$) for the compatibility function, while dot-product attention is restricted to bilinear forms.

## Comparison with Other Score Functions

| Score Function | Formula | Parameters | Expressiveness |
|----------------|---------|------------|----------------|
| **Additive (Bahdanau)** | $\mathbf{v}^\top \tanh(\mathbf{W}_s \mathbf{s} + \mathbf{W}_h \mathbf{h})$ | $d_a(d_s + d_h) + d_a$ | High (nonlinear) |
| Dot-product | $\mathbf{s}^\top \mathbf{h}$ | 0 | Low (bilinear) |
| Scaled dot-product | $\mathbf{s}^\top \mathbf{h} / \sqrt{d}$ | 0 | Low (bilinear, stable) |
| General (Luong) | $\mathbf{s}^\top \mathbf{W} \mathbf{h}$ | $d_s \times d_h$ | Medium (learnable bilinear) |

Despite its greater expressiveness, Bahdanau attention is slower than dot-product attention because the score computation cannot be parallelised as a single matrix multiplication across all positions. This is the primary reason Transformers adopted scaled dot-product attention instead.

## Bidirectional Encoder

A distinctive feature of Bahdanau's architecture is the use of a **bidirectional RNN** encoder. Each encoder state $\mathbf{h}_i$ is the concatenation of forward and backward hidden states:

$$\mathbf{h}_i = [\overrightarrow{\mathbf{h}}_i; \overleftarrow{\mathbf{h}}_i]$$

This ensures each position captures both left and right context, making the encoder representations richer for the attention mechanism to query.

## PyTorch Implementation

### Core Bahdanau Attention

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    """
    Additive (Bahdanau) Attention for RNN Seq2Seq.
    
    Computes alignment scores using a learned feedforward network,
    allowing the decoder to focus on relevant encoder positions
    at each generation step.
    """

    def __init__(self, decoder_dim: int, encoder_dim: int, attention_dim: int = None):
        """
        Args:
            decoder_dim: Dimension of decoder hidden state
            encoder_dim: Dimension of encoder hidden states
            attention_dim: Hidden dimension of the score network (default: decoder_dim)
        """
        super().__init__()
        attention_dim = attention_dim or decoder_dim

        self.W_dec = nn.Linear(decoder_dim, attention_dim, bias=False)
        self.W_enc = nn.Linear(encoder_dim, attention_dim, bias=False)
        self.v = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, decoder_state, encoder_outputs, mask=None):
        """
        Args:
            decoder_state: (batch, decoder_dim) — current decoder hidden state
            encoder_outputs: (batch, src_len, encoder_dim) — all encoder states
            mask: (batch, src_len) — 1 for valid positions, 0 for padding

        Returns:
            context: (batch, encoder_dim) — weighted sum of encoder states
            weights: (batch, src_len) — attention distribution over source
        """
        # Project decoder state: (batch, 1, attn_dim)
        dec_proj = self.W_dec(decoder_state).unsqueeze(1)
        
        # Project encoder states: (batch, src_len, attn_dim)
        enc_proj = self.W_enc(encoder_outputs)
        
        # Additive score: v^T tanh(W_s * s + W_h * h_i)
        scores = self.v(torch.tanh(dec_proj + enc_proj)).squeeze(-1)  # (batch, src_len)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, weights
```

### Encoder-Decoder with Bahdanau Attention

```python
class BahdanauEncoder(nn.Module):
    """Bidirectional GRU encoder."""
    
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, n_layers: int = 1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(
            embed_dim, hidden_dim, n_layers,
            batch_first=True, bidirectional=True
        )
        # Project bidirectional state to decoder dimension
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        
    def forward(self, src, src_lengths=None):
        """
        Args:
            src: (batch, src_len) — source token indices
            src_lengths: (batch,) — actual lengths for packing
            
        Returns:
            outputs: (batch, src_len, hidden_dim * 2) — all hidden states
            hidden: (batch, hidden_dim) — initial decoder state
        """
        embedded = self.embedding(src)
        
        if src_lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            outputs, hidden = self.rnn(packed)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        else:
            outputs, hidden = self.rnn(embedded)
        
        # Combine forward and backward final states for decoder init
        # hidden: (2 * n_layers, batch, hidden_dim) → take last layer
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)  # (batch, hidden_dim * 2)
        hidden = torch.tanh(self.fc(hidden))  # (batch, hidden_dim)
        
        return outputs, hidden


class BahdanauDecoder(nn.Module):
    """GRU decoder with Bahdanau attention."""
    
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, encoder_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = BahdanauAttention(hidden_dim, encoder_dim)
        
        # GRU input: embedding + context vector
        self.rnn = nn.GRU(embed_dim + encoder_dim, hidden_dim, batch_first=True)
        
        # Output projection
        self.fc_out = nn.Linear(hidden_dim + encoder_dim + embed_dim, vocab_size)
        
    def forward(self, input_token, hidden, encoder_outputs, mask=None):
        """
        Single decoder step.
        
        Args:
            input_token: (batch,) — previous output token
            hidden: (batch, hidden_dim) — previous decoder state
            encoder_outputs: (batch, src_len, encoder_dim)
            mask: (batch, src_len)
            
        Returns:
            prediction: (batch, vocab_size) — output logits
            hidden: (batch, hidden_dim) — updated decoder state
            attn_weights: (batch, src_len) — attention distribution
        """
        embedded = self.embedding(input_token).unsqueeze(1)  # (batch, 1, embed_dim)
        
        # Compute attention
        context, attn_weights = self.attention(hidden, encoder_outputs, mask)
        context = context.unsqueeze(1)  # (batch, 1, encoder_dim)
        
        # GRU step with context
        rnn_input = torch.cat([embedded, context], dim=-1)
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        hidden = hidden.squeeze(0)
        
        # Combine for prediction
        prediction = self.fc_out(
            torch.cat([output.squeeze(1), context.squeeze(1), embedded.squeeze(1)], dim=-1)
        )
        
        return prediction, hidden, attn_weights
```

### Demonstration

```python
def demonstrate_bahdanau():
    """Demonstrate Bahdanau attention in a translation-like setting."""
    torch.manual_seed(42)
    
    batch_size = 2
    src_len = 8
    decoder_dim = 256
    encoder_dim = 512  # Bidirectional: 256 * 2
    
    # Simulated encoder outputs and decoder state
    encoder_outputs = torch.randn(batch_size, src_len, encoder_dim)
    decoder_state = torch.randn(batch_size, decoder_dim)
    
    # Padding mask: second sequence is shorter
    mask = torch.ones(batch_size, src_len)
    mask[1, 6:] = 0  # Second sequence has length 6
    
    # Compute attention
    attention = BahdanauAttention(decoder_dim, encoder_dim, attention_dim=128)
    context, weights = attention(decoder_state, encoder_outputs, mask)
    
    print("Bahdanau Attention Demonstration")
    print("-" * 50)
    print(f"Encoder outputs: {encoder_outputs.shape}")
    print(f"Decoder state:   {decoder_state.shape}")
    print(f"Context vector:  {context.shape}")
    print(f"Attention weights: {weights.shape}")
    print(f"\nWeights (sample 1): {weights[0].detach().numpy().round(3)}")
    print(f"Weights (sample 2): {weights[1].detach().numpy().round(3)}")
    print(f"  (Note: last 2 positions are 0 due to padding mask)")


if __name__ == "__main__":
    demonstrate_bahdanau()
```

## Precomputing Encoder Projections

An important optimisation: since encoder outputs do not change during decoding, $\mathbf{W}_h \mathbf{h}_i$ can be computed once and reused at every decoder step:

```python
class EfficientBahdanauAttention(nn.Module):
    """Bahdanau attention with precomputed encoder projections."""
    
    def __init__(self, decoder_dim: int, encoder_dim: int, attention_dim: int):
        super().__init__()
        self.W_dec = nn.Linear(decoder_dim, attention_dim, bias=False)
        self.W_enc = nn.Linear(encoder_dim, attention_dim, bias=False)
        self.v = nn.Linear(attention_dim, 1, bias=False)
        self._cached_enc_proj = None
        
    def precompute(self, encoder_outputs):
        """Call once before decoding loop."""
        self._cached_enc_proj = self.W_enc(encoder_outputs)
        
    def forward(self, decoder_state, encoder_outputs, mask=None):
        if self._cached_enc_proj is None:
            enc_proj = self.W_enc(encoder_outputs)
        else:
            enc_proj = self._cached_enc_proj
            
        dec_proj = self.W_dec(decoder_state).unsqueeze(1)
        scores = self.v(torch.tanh(dec_proj + enc_proj)).squeeze(-1)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, weights
```

This reduces the per-step cost from $O(T_s \cdot (d_{\text{dec}} + d_{\text{enc}}) \cdot d_a)$ to $O(T_s \cdot d_{\text{dec}} \cdot d_a)$—significant when decoding long sequences.

## Historical Significance

Bahdanau attention was transformative for several reasons:

**Solved the bottleneck.** Translation quality on long sentences improved dramatically. The BLEU score gap between short and long sentences narrowed substantially.

**Interpretable alignments.** Attention weights provide a soft alignment between source and target, revealing what the model "looks at" during translation. These alignments often match linguistically plausible word correspondences.

**Foundation for self-attention.** The idea that neural networks could dynamically route information based on content similarity led directly to self-attention and the Transformer. Bahdanau attention is cross-attention with an additive score function.

**Generalised beyond NMT.** The attention paradigm was rapidly adopted for image captioning (Xu et al., 2015), speech recognition, question answering, and many other tasks.

## Limitations

**Sequential computation.** The decoder generates tokens one at a time, and attention must be recomputed at each step. This prevents parallelisation during training (though teacher forcing helps).

**Quadratic scoring cost.** Computing additive scores requires a forward pass through a small network for each query-key pair, which is slower than batched matrix multiplication used in dot-product attention.

**No self-attention.** Bahdanau attention only connects encoder and decoder—it does not model relationships within the source or target sequence itself.

These limitations motivated the development of faster scoring functions (Luong attention) and eventually self-attention (Transformers).

## Summary

| Aspect | Detail |
|--------|--------|
| **Score function** | $\mathbf{v}^\top \tanh(\mathbf{W}_s \mathbf{s} + \mathbf{W}_h \mathbf{h})$ (additive) |
| **Context vector** | Step-dependent: $\mathbf{c}_t = \sum_i \alpha_{ti} \mathbf{h}_i$ |
| **Encoder** | Bidirectional RNN (concatenated forward/backward states) |
| **Key innovation** | Dynamic focus on source, solving information bottleneck |
| **Complexity** | $O(T_s \cdot d_a)$ per decoder step |
| **Legacy** | Foundation for all modern attention mechanisms |

## References

1. Bahdanau, D., Cho, K., & Bengio, Y. (2014). "Neural Machine Translation by Jointly Learning to Align and Translate." *arXiv:1409.0473*.
2. Sutskever, I., Vinyals, O., & Le, Q. V. (2014). "Sequence to Sequence Learning with Neural Networks." *NeurIPS*.
3. Cho, K., et al. (2014). "On the Properties of Neural Machine Translation: Encoder-Decoder Approaches." *SSST-8*.
4. Xu, K., et al. (2015). "Show, Attend and Tell: Neural Image Caption Generation with Visual Attention." *ICML*.
