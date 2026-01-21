# Cross-Attention

## Overview

Cross-attention is the mechanism that allows the decoder to reference the encoder's output in encoder-decoder Transformers. It serves as the bridge between understanding (encoder) and generation (decoder).

## Definition

In cross-attention, queries come from the decoder, but keys and values come from the encoder:

$$\mathbf{Q} = \mathbf{Y}'\mathbf{W}_Q \quad \text{(from decoder state)}$$
$$\mathbf{K} = \mathbf{M}\mathbf{W}_K \quad \text{(from encoder memory)}$$
$$\mathbf{V} = \mathbf{M}\mathbf{W}_V \quad \text{(from encoder memory)}$$

The attention computation:

$$\text{CrossAttention}(\mathbf{Y}', \mathbf{M}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

## The Memory $\mathbf{M}$

$\mathbf{M}$ is the **encoder's final output**—the "memory" that the decoder references:

$$\text{Source text} \xrightarrow{\text{Embed + PE}} \mathbf{X} \xrightarrow{\text{Encoder Layer 1}} \cdots \xrightarrow{\text{Encoder Layer N}} \mathbf{M}$$

After the source sequence passes through all $N$ encoder layers:

$$\mathbf{M} \in \mathbb{R}^{n_s \times d_{\text{model}}}$$

where $n_s$ is the source sequence length.

**Why "Memory"?** The name reflects its role: $\mathbf{M}$ stores the encoder's complete understanding of the source sequence. The decoder "queries" this memory to retrieve relevant information during generation.

## Cross-Attention vs Self-Attention

| Aspect | Self-Attention | Cross-Attention |
|--------|----------------|-----------------|
| Q source | Same sequence | Decoder |
| K, V source | Same sequence | Encoder memory |
| Mask | Optional (causal in decoder) | None |
| Purpose | Internal context | External reference |
| When used | Encoder, decoder self-attn | Decoder only |

## Dimensional Analysis

Let:
- Source sequence length: $n_s$
- Target sequence length: $n_t$
- Model dimension: $d_{\text{model}}$
- Key/value dimension per head: $d_k$

**Encoder output (memory)**: $\mathbf{M} \in \mathbb{R}^{n_s \times d_{\text{model}}}$

**Cross-attention shapes**:

$$\mathbf{Q} \in \mathbb{R}^{n_t \times d_k} \quad \text{(decoder positions)}$$
$$\mathbf{K}, \mathbf{V} \in \mathbb{R}^{n_s \times d_k} \quad \text{(encoder positions)}$$
$$\mathbf{Q}\mathbf{K}^T \in \mathbb{R}^{n_t \times n_s}$$

Entry $(i, j)$ of the attention matrix measures how much decoder position $i$ should attend to encoder position $j$.

## No Causal Mask in Cross-Attention

Unlike decoder self-attention, cross-attention has **no mask**.

Every decoder position can attend to **all** encoder positions. When generating the 5th target word, you can look at any source word—the source sentence isn't being generated, it's fully available.

This makes sense: the source is a complete, fixed input that the decoder can freely reference.

## A Translation Example

Source: "Le chat noir" → Encoder produces $\mathbf{M}$ with 3 rows (one per token)

Target so far: "\<start\> The" → Decoder state $\mathbf{Y}'$ with 2 rows

Cross-attention matrix:

$$\mathbf{A} = \begin{pmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \end{pmatrix}$$

- Row 1: "\<start\>" attends to "Le", "chat", "noir"
- Row 2: "The" attends to "Le", "chat", "noir"

When generating "black", the decoder might attend strongly to "noir" (high $a_{23}$), retrieving its semantic content to produce the correct translation.

## Information Flow Diagram

```
┌─────────────────────────────────────┐
│           Encoder                   │
│   "Le"  →  "chat"  →  "noir"        │
│     ↓        ↓         ↓            │
│   [M₁]     [M₂]      [M₃]           │
└─────────────────────────────────────┘
                  │
                  │ K, V (from encoder)
                  ▼
┌─────────────────────────────────────┐
│           Cross-Attention           │
│                                     │
│  Q (from decoder) → Attend → Output │
│                                     │
└─────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│           Decoder                   │
│   "<s>"  →  "The"  →  "black"  → ?  │
└─────────────────────────────────────┘
```

## Cross-Attention in the Decoder Layer

Each decoder layer in an encoder-decoder Transformer has three sublayers:

**Sublayer 1**: Masked Self-Attention
$$\mathbf{Y}' = \text{LayerNorm}(\mathbf{Y} + \text{MaskedSelfAttention}(\mathbf{Y}))$$

**Sublayer 2**: Cross-Attention
$$\mathbf{Y}'' = \text{LayerNorm}(\mathbf{Y}' + \text{CrossAttention}(\mathbf{Y}', \mathbf{M}))$$

**Sublayer 3**: Feed-Forward Network
$$\mathbf{Y}''' = \text{LayerNorm}(\mathbf{Y}'' + \text{FFN}(\mathbf{Y}''))$$

The cross-attention sublayer sits between self-attention and FFN, allowing the decoder to first process its own context, then consult the encoder, then refine.

## Independent Parameters

The $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V$ in cross-attention are **distinct** from those in self-attention:

| Attention Type | Parameters |
|----------------|------------|
| Encoder self-attention | $\mathbf{W}_Q^{\text{(enc-self)}}, \mathbf{W}_K^{\text{(enc-self)}}, \mathbf{W}_V^{\text{(enc-self)}}$ |
| Decoder self-attention | $\mathbf{W}_Q^{\text{(dec-self)}}, \mathbf{W}_K^{\text{(dec-self)}}, \mathbf{W}_V^{\text{(dec-self)}}$ |
| Cross-attention | $\mathbf{W}_Q^{\text{(cross)}}, \mathbf{W}_K^{\text{(cross)}}, \mathbf{W}_V^{\text{(cross)}}$ |

Each layer, each attention type has its own learned projections.

## Training Cross-Attention

The encoder and decoder are trained **simultaneously** with end-to-end backpropagation:

1. Loss is computed on decoder outputs
2. Gradients flow backward through decoder layers
3. At cross-attention, gradients flow into both:
   - The decoder's query projection (updating $\mathbf{W}_Q^{\text{(cross)}}$)
   - The encoder's key/value projections and through $\mathbf{M}$ to all encoder layers

This joint training ensures the encoder learns representations that the decoder finds useful.

## PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CrossAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Query from decoder, Key/Value from encoder
        self.W_q = nn.Linear(d_model, d_model)  # For decoder
        self.W_k = nn.Linear(d_model, d_model)  # For encoder
        self.W_v = nn.Linear(d_model, d_model)  # For encoder
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, decoder_state, encoder_memory, mask=None):
        """
        Args:
            decoder_state: Current decoder hidden state (batch, n_t, d_model)
            encoder_memory: Encoder output (batch, n_s, d_model)
            mask: Optional padding mask for encoder
        
        Returns:
            output: Cross-attended output (batch, n_t, d_model)
        """
        batch_size = decoder_state.size(0)
        n_t = decoder_state.size(1)
        n_s = encoder_memory.size(1)
        
        # Project: Q from decoder, K/V from encoder
        Q = self.W_q(decoder_state).view(batch_size, n_t, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(encoder_memory).view(batch_size, n_s, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(encoder_memory).view(batch_size, n_s, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention: (batch, heads, n_t, d_k) x (batch, heads, d_k, n_s) -> (batch, heads, n_t, n_s)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        
        # Weighted sum: (batch, heads, n_t, n_s) x (batch, heads, n_s, d_k) -> (batch, heads, n_t, d_k)
        context = torch.matmul(attn_weights, V)
        
        # Reshape and project
        context = context.transpose(1, 2).contiguous().view(batch_size, n_t, self.d_model)
        output = self.W_o(context)
        
        return output, attn_weights

# Example
d_model = 512
n_heads = 8
n_s, n_t = 20, 15  # Source length 20, target length 15

cross_attn = CrossAttention(d_model, n_heads)
encoder_memory = torch.randn(2, n_s, d_model)
decoder_state = torch.randn(2, n_t, d_model)

output, weights = cross_attn(decoder_state, encoder_memory)
print(f"Output shape: {output.shape}")    # (2, 15, 512)
print(f"Attention shape: {weights.shape}")  # (2, 8, 15, 20)
```

## Cross-Attention in Different Architectures

### Original Transformer (Encoder-Decoder)

- Used for sequence-to-sequence tasks (translation, summarization)
- Decoder cross-attends to encoder at every layer

### Decoder-Only (GPT)

- No encoder, no cross-attention
- Relies solely on causal self-attention

### Modern Multimodal Models

- Text decoder cross-attends to image/audio encoder
- Enables vision-language models, speech recognition

## Summary

Cross-attention is the mechanism that:

1. **Bridges encoder and decoder** in sequence-to-sequence models
2. **Enables information flow** from source to target
3. **Has no causal mask** because the source is fully available
4. **Uses separate Q projections** (from decoder) and K, V projections (from encoder)

Without cross-attention, the decoder would have no way to condition its generation on the source sequence—it's the essential ingredient for translation, summarization, and other transformation tasks.

## References

- Vaswani et al., "Attention Is All You Need" (2017)
- Bahdanau et al., "Neural Machine Translation by Jointly Learning to Align and Translate" (2015)
