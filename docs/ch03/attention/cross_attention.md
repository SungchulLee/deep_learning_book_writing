# Cross-Attention

## Overview

Cross-attention (also called encoder-decoder attention) is the mechanism that allows the decoder to reference the encoder's output in encoder-decoder Transformers. It serves as the bridge between understanding (encoder) and generation (decoder).

Unlike self-attention where a sequence attends to itself, cross-attention enables information flow between two different representations—queries come from one sequence while keys and values come from another. This makes it essential for tasks like machine translation, image captioning, and any encoder-decoder architecture.

## Mathematical Formulation

### Cross-Attention Definition

In cross-attention, queries come from the decoder, but keys and values come from the encoder:

$$\mathbf{Q} = \mathbf{Y}'\mathbf{W}_Q \quad \text{(from decoder state)}$$
$$\mathbf{K} = \mathbf{M}\mathbf{W}_K \quad \text{(from encoder memory)}$$
$$\mathbf{V} = \mathbf{M}\mathbf{W}_V \quad \text{(from encoder memory)}$$

The attention computation:

$$\text{CrossAttention}(\mathbf{Y}', \mathbf{M}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

### The Memory $\mathbf{M}$

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
| Attention matrix | $(n \times n)$ square | $(n_t \times n_s)$ rectangular |
| Mask | Optional (causal in decoder) | None (source fully available) |
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

Unlike decoder self-attention, cross-attention has **no causal mask**.

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

## PyTorch Implementation

### Basic Cross-Attention

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class CrossAttention(nn.Module):
    """
    Cross-Attention Layer
    
    Computes attention where queries come from one sequence (decoder)
    and keys/values come from another sequence (encoder).
    """
    
    def __init__(
        self, 
        query_dim: int, 
        key_dim: int, 
        embed_dim: int, 
        dropout: float = 0.0
    ):
        """
        Args:
            query_dim: Dimension of query vectors (decoder)
            key_dim: Dimension of key/value vectors (encoder)
            embed_dim: Output embedding dimension
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.scale = embed_dim ** -0.5
        
        # Queries from decoder, keys/values from encoder
        self.query_proj = nn.Linear(query_dim, embed_dim)
        self.key_proj = nn.Linear(key_dim, embed_dim)
        self.value_proj = nn.Linear(key_dim, embed_dim)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        decoder_state: torch.Tensor, 
        encoder_memory: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            decoder_state: (batch, n_t, query_dim) - Current decoder hidden state
            encoder_memory: (batch, n_s, key_dim) - Encoder output
            mask: (batch, n_t, n_s) - Optional padding mask for encoder
            
        Returns:
            output: (batch, n_t, embed_dim)
            attention_weights: (batch, n_t, n_s)
        """
        # Project queries (from decoder)
        Q = self.query_proj(decoder_state)
        
        # Project keys and values (from encoder)
        K = self.key_proj(encoder_memory)
        V = self.value_proj(encoder_memory)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        output = self.out_proj(attended)
        
        return output, attention_weights
```

### Multi-Head Cross-Attention

```python
class MultiHeadCrossAttention(nn.Module):
    """
    Multi-Head Cross-Attention
    
    Extends cross-attention with multiple heads for richer representations.
    Query from decoder, Key/Value from encoder.
    """
    
    def __init__(
        self, 
        d_model: int, 
        n_heads: int, 
        dropout: float = 0.0
    ):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = self.d_k ** -0.5
        
        # Query from decoder, Key/Value from encoder
        self.W_q = nn.Linear(d_model, d_model)  # For decoder
        self.W_k = nn.Linear(d_model, d_model)  # For encoder
        self.W_v = nn.Linear(d_model, d_model)  # For encoder
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        decoder_state: torch.Tensor, 
        encoder_memory: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            decoder_state: Current decoder hidden state (batch, n_t, d_model)
            encoder_memory: Encoder output (batch, n_s, d_model)
            mask: Optional padding mask for encoder
        
        Returns:
            output: Cross-attended output (batch, n_t, d_model)
            attn_weights: Attention weights (batch, n_heads, n_t, n_s)
        """
        batch_size = decoder_state.size(0)
        n_t = decoder_state.size(1)
        n_s = encoder_memory.size(1)
        
        # Project: Q from decoder, K/V from encoder
        Q = self.W_q(decoder_state).view(batch_size, n_t, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(encoder_memory).view(batch_size, n_s, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(encoder_memory).view(batch_size, n_s, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention: (batch, heads, n_t, d_k) x (batch, heads, d_k, n_s) -> (batch, heads, n_t, n_s)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Weighted sum: (batch, heads, n_t, n_s) x (batch, heads, n_s, d_k) -> (batch, heads, n_t, d_k)
        context = torch.matmul(attn_weights, V)
        
        # Reshape and project
        context = context.transpose(1, 2).contiguous().view(batch_size, n_t, self.d_model)
        output = self.W_o(context)
        
        return output, attn_weights
```

### Complete Decoder Block

```python
class TransformerDecoderBlock(nn.Module):
    """
    Complete Transformer Decoder Block
    
    Includes:
    1. Masked self-attention (causal)
    2. Cross-attention to encoder
    3. Feed-forward network
    """
    
    def __init__(
        self, 
        d_model: int, 
        n_heads: int, 
        ff_dim: int, 
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Masked self-attention
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        
        # Cross-attention to encoder
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(d_model)
        
    def forward(
        self, 
        x: torch.Tensor, 
        encoder_memory: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        cross_attn_mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Decoder input (batch, n_t, d_model)
            encoder_memory: Encoder output (batch, n_s, d_model)
            self_attn_mask: Causal mask for decoder self-attention
            cross_attn_mask: Padding mask for encoder
        """
        # Sublayer 1: Masked self-attention
        attn_out, self_attn_weights = self.self_attn(x, x, x, attn_mask=self_attn_mask)
        x = self.norm1(x + attn_out)
        
        # Sublayer 2: Cross-attention to encoder
        attn_out, cross_attn_weights = self.cross_attn(
            x, encoder_memory, encoder_memory, key_padding_mask=cross_attn_mask
        )
        x = self.norm2(x + attn_out)
        
        # Sublayer 3: Feed-forward
        x = self.norm3(x + self.ffn(x))
        
        return x, self_attn_weights, cross_attn_weights
```

## Training Cross-Attention

The encoder and decoder are trained **simultaneously** with end-to-end backpropagation:

1. Loss is computed on decoder outputs
2. Gradients flow backward through decoder layers
3. At cross-attention, gradients flow into both:
   - The decoder's query projection (updating $\mathbf{W}_Q^{\text{(cross)}}$)
   - The encoder's key/value projections and through $\mathbf{M}$ to all encoder layers

This joint training ensures the encoder learns representations that the decoder finds useful.

## Applications

### Machine Translation

Cross-attention enables the decoder to focus on relevant source words when generating each target word:

```python
def translation_example():
    """Demonstrate cross-attention in translation."""
    
    d_model = 64
    n_heads = 4
    
    # Simulated encoder output for "Le chat noir"
    encoder_memory = torch.randn(1, 3, d_model)  # (batch, n_s, d_model)
    
    # Decoder generating "The black cat"
    decoder_state = torch.randn(1, 3, d_model)   # (batch, n_t, d_model)
    
    cross_attn = MultiHeadCrossAttention(d_model, n_heads)
    output, weights = cross_attn(decoder_state, encoder_memory)
    
    print(f"Output shape: {output.shape}")      # (1, 3, 64)
    print(f"Attention shape: {weights.shape}")  # (1, 4, 3, 3) - (batch, heads, n_t, n_s)
    
    # weights[0, 0, :, :] shows which source words each target word attends to
    print("\nCross-attention weights (target -> source), head 0:")
    print(weights[0, 0].detach().numpy().round(3))
```

### Image Captioning

Cross-attention allows text decoder to attend to relevant image regions:

```python
class ImageCaptioningCrossAttention(nn.Module):
    """Cross-attention for image captioning: text attends to image patches."""
    
    def __init__(self, text_dim: int, image_dim: int, n_heads: int):
        super().__init__()
        # Project image features to text dimension if different
        self.image_proj = nn.Linear(image_dim, text_dim) if image_dim != text_dim else nn.Identity()
        self.cross_attn = MultiHeadCrossAttention(text_dim, n_heads)
        
    def forward(self, text_embeddings: torch.Tensor, image_features: torch.Tensor):
        """
        Args:
            text_embeddings: (batch, text_len, text_dim) - Caption tokens
            image_features: (batch, num_patches, image_dim) - Image patches from ViT
        """
        image_memory = self.image_proj(image_features)
        return self.cross_attn(text_embeddings, image_memory)
```

### Cross-Attention Patterns in Translation

Different language pairs exhibit characteristic alignment patterns:

| Pattern | Description | Example |
|---------|-------------|---------|
| **Monotonic** | Sequential alignment | English ↔ Spanish (similar word order) |
| **Diagonal** | One-to-one with shifts | Closely related languages |
| **Scattered** | Long-range reordering | English ↔ Japanese (SOV vs SVO) |
| **Many-to-one** | Multiple source → one target | Compound words, idioms |

### Visualization

```python
def visualize_cross_attention(
    weights: torch.Tensor,
    source_tokens: list,
    target_tokens: list
):
    """Visualize cross-attention alignment."""
    import matplotlib.pyplot as plt
    
    # Take first head's attention
    attn = weights[0, 0].detach().numpy()
    
    plt.figure(figsize=(8, 6))
    plt.imshow(attn, cmap='Blues', aspect='auto')
    plt.xticks(range(len(source_tokens)), source_tokens, rotation=45)
    plt.yticks(range(len(target_tokens)), target_tokens)
    plt.xlabel('Source (Encoder)')
    plt.ylabel('Target (Decoder)')
    plt.colorbar(label='Attention Weight')
    plt.title('Cross-Attention Alignment')
    plt.tight_layout()
```

## Cross-Attention in Different Architectures

| Architecture | Cross-Attention Usage |
|--------------|----------------------|
| **Original Transformer** | Decoder cross-attends to encoder at every layer |
| **Decoder-Only (GPT)** | No encoder, no cross-attention |
| **BART, T5** | Encoder-decoder with cross-attention for seq2seq |
| **Multimodal (Flamingo, LLaVA)** | Text decoder cross-attends to image/audio encoder |
| **Whisper** | Audio encoder + text decoder with cross-attention |

## Summary

Cross-attention is the mechanism that:

1. **Bridges encoder and decoder** in sequence-to-sequence models
2. **Enables information flow** from source to target representation
3. **Has no causal mask** because the source is fully available
4. **Uses separate projections**: Q from decoder, K/V from encoder
5. **Enables multimodal fusion**: Text can attend to images, audio, etc.

Without cross-attention, the decoder would have no way to condition its generation on the source sequence—it's the essential ingredient for translation, summarization, captioning, and other transformation tasks.

**Key insight**: Cross-attention decouples the "what to look for" (decoder queries) from "what's available" (encoder keys/values), allowing flexible information routing between different representations.

## References

1. Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS*.

2. Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural Machine Translation by Jointly Learning to Align and Translate. *ICLR*.

3. Lewis, M., et al. (2020). BART: Denoising Sequence-to-Sequence Pre-training. *ACL*.

4. Alayrac, J.-B., et al. (2022). Flamingo: a Visual Language Model for Few-Shot Learning. *NeurIPS*.
