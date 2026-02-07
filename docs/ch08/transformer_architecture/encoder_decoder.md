# Encoder-Decoder Structure

## Overview

The Transformer architecture (Vaswani et al., 2017) has been adapted into three main paradigms. Each makes different choices about attention directionality and information flow, leading to distinct strengths:

| Architecture | Attention | Examples | Best For |
|--------------|-----------|----------|----------|
| Encoder-only | Bidirectional | BERT, RoBERTa | Understanding |
| Decoder-only | Causal | GPT, LLaMA | Generation |
| Encoder-Decoder | Both | T5, BART | Seq2Seq |

## The Original Encoder-Decoder Architecture

The original Transformer was designed as an encoder-decoder model for machine translation. Understanding this full architecture is essential before examining the simplified variants.

### Encoder Stack

The encoder consists of $N$ identical layers (typically $N = 6$), each containing two sublayers:

**Sublayer 1 — Multi-Head Self-Attention:**

$$\mathbf{Z} = \text{LayerNorm}(\mathbf{X} + \text{MultiHead}(\mathbf{X}, \mathbf{X}, \mathbf{X}))$$

All positions attend to all other positions (bidirectional). No masking is applied, so every token can access the full source context.

**Sublayer 2 — Position-wise Feed-Forward Network:**

$$\text{FFN}(\mathbf{z}) = \text{ReLU}(\mathbf{z}\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2$$

$$\mathbf{H} = \text{LayerNorm}(\mathbf{Z} + \text{FFN}(\mathbf{Z}))$$

The FFN is applied identically and independently to each position—it adds per-token nonlinear capacity without any cross-position interaction.

### Decoder Stack

The decoder also has $N$ identical layers, but each contains **three** sublayers:

**Sublayer 1 — Masked Self-Attention:**

$$\mathbf{Y}' = \text{LayerNorm}(\mathbf{Y} + \text{MaskedMultiHead}(\mathbf{Y}, \mathbf{Y}, \mathbf{Y}))$$

Causal masking prevents attending to future positions, preserving the autoregressive property.

**Sublayer 2 — Cross-Attention:**

$$\mathbf{Y}'' = \text{LayerNorm}(\mathbf{Y}' + \text{MultiHead}(\mathbf{Y}', \mathbf{M}, \mathbf{M}))$$

Queries come from the decoder; keys and values come from the encoder output $\mathbf{M}$. This is the bridge between encoder and decoder.

**Sublayer 3 — Feed-Forward Network:**

$$\mathbf{Y}''' = \text{LayerNorm}(\mathbf{Y}'' + \text{FFN}(\mathbf{Y}''))$$

### Information Flow Diagram

```
Source tokens          Target tokens (shifted right)
     │                        │
  [Embedding + PE]        [Embedding + PE]
     │                        │
  ┌──▼──┐                ┌───▼───┐
  │ Self │                │Masked │
  │ Attn │                │ Self  │
  │(bidir)│               │ Attn  │
  └──┬──┘                └───┬───┘
     │                        │
  [Add & Norm]           [Add & Norm]
     │                        │
  ┌──▼──┐                ┌───▼───┐
  │ FFN │     Memory M    │Cross  │◄── K, V from encoder
  └──┬──┘  ──────────►   │ Attn  │
     │                    └───┬───┘
  [Add & Norm]           [Add & Norm]
     │                        │
     │    ×N layers       ┌───▼───┐
     │                    │  FFN  │
     ▼                    └───┬───┘
  Encoder                [Add & Norm]
  Output M                    │
                          ×N layers
                              │
                           [Linear]
                           [Softmax]
                              │
                          Output probs
```

### Residual Connections and Layer Normalization

Every sublayer in the Transformer uses a residual connection followed by layer normalization. This design choice is critical for training stability:

$$\text{output} = \text{LayerNorm}(\mathbf{x} + \text{Sublayer}(\mathbf{x}))$$

**Post-norm** (original paper): LayerNorm applied after the residual addition, as shown above.

**Pre-norm** (modern practice): LayerNorm applied before the sublayer, found to be more stable for deep models:

$$\text{output} = \mathbf{x} + \text{Sublayer}(\text{LayerNorm}(\mathbf{x}))$$

Pre-norm allows gradient to flow through the residual path without normalization, enabling training of much deeper networks. Most modern Transformers (GPT-2+, LLaMA, T5 v1.1) use pre-norm.

## Encoder-Only (BERT-style)

**Attention**: All positions see all positions (bidirectional)

The encoder-only architecture removes the decoder entirely. The model processes input through a stack of bidirectional self-attention layers, producing contextualized representations for each token.

```python
class EncoderOnly(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
    
    def forward(self, x):
        return self.encoder(self.embed(x))  # No causal mask
```

**Use cases**: Classification, NER, QA extraction, sentence embeddings

**Pre-training**: Masked Language Modeling (MLM) — randomly mask tokens and predict them from bidirectional context. This is not a generative objective; the model learns to understand, not to generate.

## Decoder-Only (GPT-style)

**Attention**: Causal (each position sees only past)

The decoder-only architecture removes the encoder and cross-attention layers entirely. What remains is a stack of causally-masked self-attention layers with FFN sublayers.

```python
class DecoderOnly(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, num_heads, batch_first=True),
            num_layers
        )
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.register_buffer('mask', torch.triu(torch.ones(2048, 2048), 1).bool())
    
    def forward(self, x):
        seq_len = x.size(1)
        h = self.layers(self.embed(x), mask=self.mask[:seq_len, :seq_len])
        return self.lm_head(h)
```

**Use cases**: Text generation, chat, code, in-context learning

**Pre-training**: Causal Language Modeling (CLM) — predict the next token given all previous tokens. Every token in the training sequence provides a training signal, making this highly data-efficient.

**Note on naming**: Despite using `nn.TransformerEncoder`, this is functionally a decoder because of the causal mask. PyTorch's `TransformerEncoderLayer` is simply a self-attention + FFN block; adding a causal mask makes it a decoder layer.

## Encoder-Decoder (T5-style)

**Attention**: Encoder bidirectional + Decoder causal + Cross-attention

The encoder-decoder architecture retains the full original Transformer design. The encoder processes the input bidirectionally, and the decoder generates output autoregressively while attending to the encoder's representation.

```python
class EncoderDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, num_heads, batch_first=True),
            num_layers
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, num_heads, batch_first=True),
            num_layers
        )
        self.lm_head = nn.Linear(d_model, vocab_size)
    
    def forward(self, src, tgt):
        memory = self.encoder(self.embed(src))
        tgt_mask = torch.triu(torch.ones(tgt.size(1), tgt.size(1)), 1).bool()
        h = self.decoder(self.embed(tgt), memory, tgt_mask=tgt_mask)
        return self.lm_head(h)
```

**Use cases**: Translation, summarization, text-to-text tasks

**Pre-training**: Span corruption (T5), denoising (BART) — corrupt the input, then reconstruct. The encoder sees the corrupted input bidirectionally; the decoder generates the original or the corrupted spans.

## Detailed Comparison

| Aspect | Encoder-Only | Decoder-Only | Enc-Dec |
|--------|-------------|-------------|---------|
| Context | Bidirectional | Causal | Both |
| Generation | ✗ | ✓ | ✓ |
| Understanding | ✓✓ | ✓ | ✓ |
| Cross-attention | ✗ | ✗ | ✓ |
| Parameters | 1x | 1x | ~2x |
| KV cache (inference) | N/A | Self-attn only | Self-attn + cross-attn |
| Modern Popularity | Medium | **High** | Medium |

### Key Trade-offs

**Bidirectional vs. Causal Context**: Encoder-only models see the full sequence when computing each token's representation, providing richer context for understanding tasks. Decoder-only models see only the left context, which is a natural fit for generation but means understanding tasks must rely on unidirectional representations. Encoder-decoder models get the best of both: bidirectional encoding of the source and causal decoding of the target.

**Parameter Efficiency**: Encoder-decoder models have roughly twice the parameters of encoder-only or decoder-only models at the same depth and width, since they contain two separate stacks. However, decoder-only models can be scaled more simply (just add layers) and have become the dominant paradigm for large-scale models.

**Training Objective Alignment**: Decoder-only models are trained with causal language modeling, which directly aligns with generation. Encoder-only models use masked language modeling, which aligns with understanding. Encoder-decoder models use span corruption or similar objectives that align with sequence-to-sequence tasks.

**Inference Efficiency**: Decoder-only models benefit from a single KV cache that grows with generation. Encoder-decoder models require two caches (decoder self-attention KV cache that grows, plus fixed cross-attention KV cache), but the cross-attention KV cache is computed once from the encoder and reused.

### Feed-Forward Network Role

Across all three architectures, the FFN sublayer serves the same purpose: per-token nonlinear transformation. Research suggests the FFN layers act as key-value memories, storing factual knowledge:

$$\text{FFN}(\mathbf{x}) = f(\mathbf{x}\mathbf{W}_1)\mathbf{W}_2$$

where $\mathbf{W}_1 \in \mathbb{R}^{d \times d_{ff}}$ computes "keys" (pattern detectors) and $\mathbf{W}_2 \in \mathbb{R}^{d_{ff} \times d}$ stores "values" (associated information). The inner dimension $d_{ff}$ is typically $4d$, giving the FFN four times as many parameters per layer as the attention mechanism.

## Historical Evolution

The field has evolved from encoder-only dominance (BERT era, 2018–2020) to decoder-only dominance (GPT-3 onward, 2020–present):

- **2017**: Original Transformer uses encoder-decoder for translation
- **2018**: BERT demonstrates encoder-only superiority for understanding benchmarks
- **2018**: GPT-1 shows decoder-only models can be fine-tuned for diverse tasks
- **2019**: GPT-2 shows decoder-only models can generate coherent text
- **2020**: T5 demonstrates encoder-decoder versatility across all NLP tasks
- **2020+**: GPT-3 and subsequent large decoder-only models demonstrate emergent few-shot capabilities, shifting the field toward decoder-only architectures
- **2023+**: Decoder-only models (LLaMA, Mistral, GPT-4) dominate both generation and understanding tasks at scale

### Why Decoder-Only Won

Several factors drove the convergence toward decoder-only:

1. **Simplicity**: One architecture, one objective, one scaling recipe
2. **In-context learning**: Emergent at scale, makes fine-tuning optional
3. **Unified interface**: Both understanding and generation through next-token prediction
4. **Scaling efficiency**: All parameters contribute to every prediction (no separate encoder sitting idle during generation)

## When to Use

- **Encoder-only**: Classification, embeddings, extraction tasks
- **Decoder-only**: Generation, chat, few-shot learning (most popular today)
- **Encoder-Decoder**: Translation, summarization with distinct source/target

## References

1. Vaswani, A., et al. (2017). "Attention Is All You Need." NeurIPS.
2. Devlin, J., et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers." NAACL.
3. Radford, A., et al. (2019). "Language Models are Unsupervised Multitask Learners." (GPT-2)
4. Raffel, C., et al. (2020). "Exploring the Limits of Transfer Learning with T5." JMLR.
5. Brown, T., et al. (2020). "Language Models are Few-Shot Learners." NeurIPS.

---

## Encoder Block: Deep Dive

##### Overview

The Transformer encoder block is the fundamental building unit of encoder-based architectures like BERT. Each block transforms its input through self-attention and feed-forward layers, maintaining the sequence length while refining representations.

##### Architecture

Each encoder block consists of two sub-layers with residual connections and layer normalization:

$$
\begin{aligned}
\mathbf{Z} &= \text{LayerNorm}(\mathbf{X} + \text{MultiHeadAttention}(\mathbf{X}, \mathbf{X}, \mathbf{X})) \\
\mathbf{X}' &= \text{LayerNorm}(\mathbf{Z} + \text{FFN}(\mathbf{Z}))
\end{aligned}
$$

##### Component Breakdown

1. **Multi-Head Self-Attention**: Captures dependencies between all positions
2. **Feed-Forward Network**: Position-wise transformation with expansion
3. **Residual Connections**: Enable gradient flow in deep networks
4. **Layer Normalization**: Stabilizes training

##### Multi-Head Self-Attention in Encoder

The encoder uses bidirectional self-attention where each position can attend to all positions:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

For self-attention, $Q$, $K$, $V$ are all derived from the same input:

$$
Q = XW^Q, \quad K = XW^K, \quad V = XW^V
$$

##### From Input to Attention: Step-by-Step

Understanding how a raw sentence flows through the encoder requires tracing each transformation:

**Step 1 — Tokenization**: The input sentence is split into tokens (words or subwords). For example, "The cat sat" becomes the token sequence $[\text{The}, \text{cat}, \text{sat}]$.

**Step 2 — Embedding lookup**: Each token is mapped to a dense vector via the embedding matrix $E \in \mathbb{R}^{V \times d_{\text{model}}}$, producing a matrix $X_{\text{embed}} \in \mathbb{R}^{n \times d_{\text{model}}}$ where $n$ is the sequence length. Embeddings are scaled by $\sqrt{d_{\text{model}}}$.

**Step 3 — Positional encoding**: Sinusoidal positional encodings are added element-wise:

$$
X = X_{\text{embed}} \cdot \sqrt{d_{\text{model}}} + \text{PE}
$$

The result $X \in \mathbb{R}^{n \times d_{\text{model}}}$ carries both semantic (from embeddings) and positional information.

**Step 4 — Linear projections to Q, K, V**: Three separate weight matrices project $X$ into query, key, and value spaces:

$$
Q = XW^Q, \quad K = XW^K, \quad V = XW^V
$$

where $W^Q, W^K, W^V \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$. These projections are linear (no bias terms in the standard formulation), allowing the model to learn different representations for what to look for (queries), what to advertise (keys), and what information to provide (values).

**Step 5 — Split into multiple heads**: Each of $Q$, $K$, $V$ is split along the last dimension into $h$ heads, each of dimension $d_k = d_{\text{model}} / h$:

$$
Q_i, K_i, V_i \in \mathbb{R}^{n \times d_k} \quad \text{for } i = 1, \ldots, h
$$

**Step 6 — Scaled dot-product attention per head**: Each head computes attention independently:

$$
\text{head}_i = \text{softmax}\left(\frac{Q_i K_i^T}{\sqrt{d_k}}\right) V_i
$$

The scaling by $\sqrt{d_k}$ prevents the dot products from growing too large in magnitude, which would push the softmax into regions with very small gradients.

**Step 7 — Concatenate and project**: Head outputs are concatenated and linearly projected:

$$
\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O
$$

where $W^O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$ (again, no bias term in the standard formulation). The output has the same shape as the input: $\mathbb{R}^{n \times d_{\text{model}}}$.

!!! note "No Bias Terms in Attention"
    The original Transformer formulation uses no bias terms in the $W^Q$, $W^K$, $W^V$, and $W^O$ projections. The attention mechanism includes scaling ($\sqrt{d_k}$) and normalization (softmax) that provide sufficient control over the signal, making additive biases unnecessary at this stage. Some implementations (e.g., PyTorch's `nn.MultiheadAttention`) include bias by default, but setting `bias=False` matches the original design.

##### Multi-Head Formulation

$$
\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O
$$

Where each head:

$$
\text{head}_i = \text{Attention}(XW_i^Q, XW_i^K, XW_i^V)
$$

##### Feed-Forward Network

The position-wise FFN applies the same transformation to each position independently:

$$
\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2
$$

Or with GELU activation (common in modern models):

$$
\text{FFN}(x) = \text{GELU}(xW_1 + b_1)W_2 + b_2
$$

##### Dimension Expansion

The FFN typically expands the hidden dimension by a factor of 4:

- Input: $d_{\text{model}}$
- Hidden: $d_{ff} = 4 \times d_{\text{model}}$
- Output: $d_{\text{model}}$

##### Pre-Norm vs Post-Norm

##### Post-Norm (Original Transformer)

$$
\mathbf{X}' = \text{LayerNorm}(\mathbf{X} + \text{SubLayer}(\mathbf{X}))
$$

##### Pre-Norm (Modern Standard)

$$
\mathbf{X}' = \mathbf{X} + \text{SubLayer}(\text{LayerNorm}(\mathbf{X}))
$$

Pre-norm is more stable for training deep models and doesn't require learning rate warmup.

##### PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention for Transformer Encoder.
    
    All positions can attend to all other positions (bidirectional).
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True
    ):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            bias: Whether to include bias in linear projections
        """
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Combined QKV projection for efficiency
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=bias)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            key_padding_mask: Mask for padding tokens [batch_size, seq_len]
                True for positions to mask (padding)
            return_attention: Whether to return attention weights
            
        Returns:
            output: Transformed tensor [batch_size, seq_len, d_model]
            attention_weights: Optional attention weights [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute Q, K, V in one projection
        qkv = self.qkv_proj(x)  # [batch, seq, 3 * d_model]
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention scores
        # [batch, heads, seq, seq]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply padding mask if provided
        if key_padding_mask is not None:
            # Expand mask: [batch, seq] -> [batch, 1, 1, seq]
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        # [batch, heads, seq, head_dim]
        context = torch.matmul(attn_weights, v)
        
        # Reshape and project
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, self.d_model)
        output = self.out_proj(context)
        output = self.proj_dropout(output)
        
        if return_attention:
            return output, attn_weights
        return output, None


class PositionWiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    
    FFN(x) = activation(xW1 + b1)W2 + b2
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        """
        Args:
            d_model: Model dimension
            d_ff: Feed-forward hidden dimension
            dropout: Dropout probability
            activation: Activation function ('relu' or 'gelu')
        """
        super().__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'gelu':
            self.activation = F.gelu
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class TransformerEncoderBlock(nn.Module):
    """
    Single Transformer Encoder Block.
    
    Consists of:
    1. Multi-Head Self-Attention + Residual + LayerNorm
    2. Position-wise FFN + Residual + LayerNorm
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'gelu',
        pre_norm: bool = True
    ):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
            dropout: Dropout probability
            activation: FFN activation function
            pre_norm: Use pre-norm (True) or post-norm (False)
        """
        super().__init__()
        
        self.pre_norm = pre_norm
        
        # Self-attention
        self.self_attention = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Feed-forward network
        self.feed_forward = PositionWiseFeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout for residual connections
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            key_padding_mask: Padding mask [batch_size, seq_len]
            return_attention: Whether to return attention weights
            
        Returns:
            output: Transformed tensor [batch_size, seq_len, d_model]
            attention_weights: Optional attention weights
        """
        if self.pre_norm:
            # Pre-norm: LayerNorm before sublayer
            # Attention block
            residual = x
            x = self.norm1(x)
            attn_output, attn_weights = self.self_attention(
                x, key_padding_mask, return_attention
            )
            x = residual + self.dropout(attn_output)
            
            # FFN block
            residual = x
            x = self.norm2(x)
            ff_output = self.feed_forward(x)
            x = residual + self.dropout(ff_output)
        else:
            # Post-norm: LayerNorm after sublayer
            # Attention block
            attn_output, attn_weights = self.self_attention(
                x, key_padding_mask, return_attention
            )
            x = self.norm1(x + self.dropout(attn_output))
            
            # FFN block
            ff_output = self.feed_forward(x)
            x = self.norm2(x + self.dropout(ff_output))
        
        return x, attn_weights


class TransformerEncoder(nn.Module):
    """
    Full Transformer Encoder stack.
    
    Consists of:
    1. Token embedding
    2. Positional encoding
    3. Stack of N encoder blocks
    4. Optional final layer norm (for pre-norm)
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_len: int = 512,
        dropout: float = 0.1,
        activation: str = 'gelu',
        pre_norm: bool = True
    ):
        """
        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of encoder blocks
            d_ff: Feed-forward hidden dimension
            max_len: Maximum sequence length
            dropout: Dropout probability
            activation: FFN activation function
            pre_norm: Use pre-norm architecture
        """
        super().__init__()
        
        self.d_model = d_model
        self.pre_norm = pre_norm
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding (learned)
        self.position_embedding = nn.Embedding(max_len, d_model)
        
        # Embedding dropout
        self.embedding_dropout = nn.Dropout(dropout)
        
        # Encoder blocks
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation,
                pre_norm=pre_norm
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm (for pre-norm architecture)
        if pre_norm:
            self.final_norm = nn.LayerNorm(d_model)
        else:
            self.final_norm = None
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_all_hidden_states: bool = False
    ) -> dict:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
                1 for real tokens, 0 for padding
            return_all_hidden_states: Whether to return all layer outputs
            
        Returns:
            Dictionary containing:
                - last_hidden_state: Final encoder output
                - all_hidden_states: Optional list of all layer outputs
        """
        batch_size, seq_len = input_ids.shape
        
        # Create position IDs
        position_ids = torch.arange(seq_len, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        
        # Scale token embeddings
        x = token_embeds * math.sqrt(self.d_model) + position_embeds
        x = self.embedding_dropout(x)
        
        # Convert attention mask to key padding mask
        # attention_mask: 1 for real, 0 for padding
        # key_padding_mask: True for padding (to be masked)
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)
        else:
            key_padding_mask = None
        
        # Pass through encoder layers
        all_hidden_states = [x] if return_all_hidden_states else None
        
        for layer in self.layers:
            x, _ = layer(x, key_padding_mask)
            if return_all_hidden_states:
                all_hidden_states.append(x)
        
        # Final layer norm
        if self.final_norm is not None:
            x = self.final_norm(x)
        
        return {
            'last_hidden_state': x,
            'all_hidden_states': all_hidden_states
        }


# Example usage and testing
if __name__ == "__main__":
    # Configuration
    vocab_size = 30522  # BERT-like vocabulary
    d_model = 768
    num_heads = 12
    num_layers = 12
    d_ff = 3072
    max_len = 512
    
    # Create encoder
    encoder = TransformerEncoder(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_len=max_len,
        dropout=0.1,
        pre_norm=True
    )
    
    # Sample input
    batch_size = 8
    seq_len = 128
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    attention_mask[:, -10:] = 0  # Simulate padding
    
    # Forward pass
    outputs = encoder(input_ids, attention_mask, return_all_hidden_states=True)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {outputs['last_hidden_state'].shape}")
    print(f"Number of hidden states: {len(outputs['all_hidden_states'])}")
    
    # Count parameters
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test single block
    print("\n--- Testing single encoder block ---")
    single_block = TransformerEncoderBlock(
        d_model=512,
        num_heads=8,
        d_ff=2048
    )
    
    test_input = torch.randn(4, 32, 512)
    output, attn = single_block(test_input, return_attention=True)
    print(f"Block input shape: {test_input.shape}")
    print(f"Block output shape: {output.shape}")
    print(f"Attention shape: {attn.shape}")
```

##### Gradient Flow Analysis

##### Residual Connection Benefits

Without residuals, gradients must flow through all sublayers:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{x}^{(0)}} = \prod_{l=0}^{L-1} \frac{\partial \text{SubLayer}^{(l)}}{\partial \mathbf{x}^{(l)}}
$$

With residuals, there's a direct gradient path:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{x}^{(0)}} = \frac{\partial \mathcal{L}}{\partial \mathbf{x}^{(L)}} + \text{other terms}
$$

##### Layer Normalization Effects

Layer norm normalizes across the feature dimension:

$$
\text{LayerNorm}(\mathbf{x}) = \gamma \odot \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

Where:
- $\mu = \frac{1}{d} \sum_{i=1}^{d} x_i$
- $\sigma^2 = \frac{1}{d} \sum_{i=1}^{d} (x_i - \mu)^2$

This stabilizes the gradient magnitude across layers.

##### Computational Complexity

##### Per-Layer Complexity

For sequence length $n$, model dimension $d$, and FFN dimension $d_{ff}$:

| Component | Time Complexity | Space Complexity |
|-----------|-----------------|------------------|
| Self-Attention | $O(n^2 d)$ | $O(n^2 + nd)$ |
| FFN | $O(n d \cdot d_{ff})$ | $O(n d_{ff})$ |
| Layer Norm | $O(nd)$ | $O(nd)$ |

##### Total Encoder Complexity

For $L$ layers:

$$
\text{Time: } O(L \cdot (n^2 d + n d \cdot d_{ff}))
$$

With typical $d_{ff} = 4d$:

$$
\text{Time: } O(L \cdot n \cdot d \cdot (n + 4d))
$$

##### Summary

The Transformer encoder block is a modular, powerful building block that:

1. **Captures Global Dependencies**: Self-attention connects all positions
2. **Applies Non-linear Transformations**: FFN adds representational capacity
3. **Maintains Gradient Flow**: Residuals enable deep architectures
4. **Stabilizes Training**: Layer normalization controls activations

Understanding the encoder block is essential for implementing models like BERT, RoBERTa, and encoder-based architectures.

##### References

1. Vaswani, A., et al. (2017). "Attention Is All You Need." NeurIPS.
2. Devlin, J., et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers." NAACL.
3. Xiong, R., et al. (2020). "On Layer Normalization in the Transformer Architecture." ICML.
4. He, K., et al. (2016). "Deep Residual Learning for Image Recognition." CVPR.

---

## Decoder Block: Deep Dive

##### Overview

The Transformer decoder block is the building unit for autoregressive models like GPT. Unlike the encoder, the decoder uses **causal (masked) self-attention** to prevent positions from attending to future positions, enabling left-to-right generation.

##### Architecture Variants

##### Full Encoder-Decoder (Original Transformer)

The original decoder has three sub-layers:

$$
\begin{aligned}
\mathbf{Z}_1 &= \text{LayerNorm}(\mathbf{Y} + \text{MaskedSelfAttn}(\mathbf{Y})) \\
\mathbf{Z}_2 &= \text{LayerNorm}(\mathbf{Z}_1 + \text{CrossAttn}(\mathbf{Z}_1, \mathbf{X}_{\text{enc}})) \\
\mathbf{Y}' &= \text{LayerNorm}(\mathbf{Z}_2 + \text{FFN}(\mathbf{Z}_2))
\end{aligned}
$$

##### Decoder-Only (GPT-style)

Modern language models use decoder-only architecture with two sub-layers:

$$
\begin{aligned}
\mathbf{Z} &= \text{LayerNorm}(\mathbf{X} + \text{MaskedSelfAttn}(\mathbf{X})) \\
\mathbf{X}' &= \text{LayerNorm}(\mathbf{Z} + \text{FFN}(\mathbf{Z}))
\end{aligned}
$$

##### Causal Self-Attention

The key difference from encoder self-attention is the **causal mask** that prevents attending to future positions.

##### Causal Mask

For sequence length $n$, the causal mask $M$ is:

$$
M_{ij} = \begin{cases} 0 & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}
$$

This creates a lower-triangular attention pattern:

$$
\text{MaskedAttn}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V
$$

##### Visual Representation

```
Position can attend to:
     1  2  3  4  5
1  [ ✓  ✗  ✗  ✗  ✗ ]
2  [ ✓  ✓  ✗  ✗  ✗ ]
3  [ ✓  ✓  ✓  ✗  ✗ ]
4  [ ✓  ✓  ✓  ✓  ✗ ]
5  [ ✓  ✓  ✓  ✓  ✓ ]
```

##### Cross-Attention (Encoder-Decoder)

In sequence-to-sequence models, the decoder attends to encoder outputs:

$$
\text{CrossAttn}(Z, X_{\text{enc}}) = \text{softmax}\left(\frac{Z W^Q (X_{\text{enc}} W^K)^T}{\sqrt{d_k}}\right) X_{\text{enc}} W^V
$$

Where:
- Query: from decoder ($Z W^Q$)
- Key, Value: from encoder ($X_{\text{enc}} W^K$, $X_{\text{enc}} W^V$)

##### PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class CausalSelfAttention(nn.Module):
    """
    Causal (Masked) Self-Attention for Transformer Decoder.
    
    Each position can only attend to itself and previous positions.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_len: int = 2048,
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()
        
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Combined QKV projection
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        
        # Pre-compute causal mask
        self.register_buffer(
            'causal_mask',
            torch.triu(torch.ones(max_len, max_len), diagonal=1).bool()
        )
    
    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with causal masking.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            key_padding_mask: Padding mask [batch_size, seq_len]
            return_attention: Whether to return attention weights
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply causal mask
        causal_mask = self.causal_mask[:seq_len, :seq_len]
        attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        
        # Apply padding mask if provided
        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, v)
        
        # Reshape and project
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, self.d_model)
        output = self.out_proj(context)
        output = self.proj_dropout(output)
        
        if return_attention:
            return output, attn_weights
        return output, None


class TransformerDecoderBlock(nn.Module):
    """
    Single Transformer Decoder Block (GPT-style).
    
    Consists of:
    1. Masked Self-Attention + Residual + LayerNorm
    2. Feed-Forward Network + Residual + LayerNorm
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'gelu',
        max_len: int = 2048,
        pre_norm: bool = True
    ):
        super().__init__()
        
        self.pre_norm = pre_norm
        
        # Masked self-attention
        self.self_attention = CausalSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_len=max_len,
            dropout=dropout
        )
        self.norm1 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass."""
        if self.pre_norm:
            # Pre-norm: LayerNorm before sublayer
            residual = x
            x = self.norm1(x)
            attn_out, _ = self.self_attention(x, key_padding_mask)
            x = residual + self.dropout(attn_out)
            
            residual = x
            x = self.norm2(x)
            ff_out = self.feed_forward(x)
            x = residual + ff_out
        else:
            # Post-norm: LayerNorm after sublayer
            attn_out, _ = self.self_attention(x, key_padding_mask)
            x = self.norm1(x + self.dropout(attn_out))
            
            ff_out = self.feed_forward(x)
            x = self.norm2(x + ff_out)
        
        return x


class TransformerDecoder(nn.Module):
    """
    Full Transformer Decoder (GPT-style, decoder-only).
    
    For language modeling and text generation.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        d_ff: int = 3072,
        max_len: int = 1024,
        dropout: float = 0.1,
        pre_norm: bool = True
    ):
        super().__init__()
        
        self.d_model = d_model
        self.pre_norm = pre_norm
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        
        self.embedding_dropout = nn.Dropout(dropout)
        
        # Decoder blocks
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
                max_len=max_len,
                pre_norm=pre_norm
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm (for pre-norm)
        self.final_norm = nn.LayerNorm(d_model) if pre_norm else None
        
        # Output projection (weight tying with embeddings)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Labels for LM loss [batch_size, seq_len]
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Position IDs
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        
        # Embeddings
        x = self.token_embedding(input_ids) + self.position_embedding(position_ids)
        x = self.embedding_dropout(x)
        
        # Padding mask
        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        
        # Pass through decoder layers
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        
        # Final norm
        if self.final_norm is not None:
            x = self.final_norm(x)
        
        # Language modeling head
        logits = self.lm_head(x)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
        
        return {'logits': logits, 'loss': loss}
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True
    ) -> torch.Tensor:
        """Generate text autoregressively."""
        for _ in range(max_new_tokens):
            # Get logits for last position
            outputs = self.forward(input_ids)
            logits = outputs['logits'][:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                values, _ = torch.topk(logits, top_k)
                min_value = values[:, -1].unsqueeze(-1)
                logits = torch.where(
                    logits < min_value,
                    torch.full_like(logits, float('-inf')),
                    logits
                )
            
            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits = logits.masked_fill(indices_to_remove, float('-inf'))
            
            # Sample or greedy
            if do_sample:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            input_ids = torch.cat([input_ids, next_token], dim=-1)
        
        return input_ids


# Example usage
if __name__ == "__main__":
    # GPT-style decoder configuration
    model = TransformerDecoder(
        vocab_size=50257,
        d_model=768,
        num_heads=12,
        num_layers=12,
        d_ff=3072,
        max_len=1024
    )
    
    # Sample input
    input_ids = torch.randint(0, 50257, (4, 128))
    labels = input_ids.clone()
    
    # Forward pass
    outputs = model(input_ids, labels=labels)
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Loss: {outputs['loss'].item():.4f}")
    
    # Generation
    prompt = torch.randint(0, 50257, (1, 10))
    generated = model.generate(prompt, max_new_tokens=20, temperature=0.8, top_k=50)
    print(f"Generated shape: {generated.shape}")
    
    # Parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
```

##### KV-Cache for Efficient Inference

During generation, recomputing attention for all previous tokens is wasteful. KV-caching stores key and value tensors from previous steps:

```python
class CausalSelfAttentionWithCache(nn.Module):
    """Causal self-attention with KV-cache for efficient inference."""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward with optional KV-cache.
        
        Args:
            x: Input [batch, seq_len, d_model] or [batch, 1, d_model] with cache
            past_kv: Cached (key, value) tensors from previous steps
            use_cache: Whether to return updated cache
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute Q, K, V for current input
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Concatenate with past KV if provided
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        
        # Store for next iteration
        present_kv = (k, v) if use_cache else None
        
        # Attention (no causal mask needed when using cache for single token)
        full_seq_len = k.size(2)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply causal mask only for initial context (seq_len > 1)
        if seq_len > 1:
            causal_mask = torch.triu(
                torch.ones(seq_len, full_seq_len, device=x.device),
                diagonal=full_seq_len - seq_len + 1
            ).bool()
            attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_proj(context)
        
        return output, present_kv
```

##### Computational Analysis

##### Without KV-Cache

For generating $T$ tokens with context $C$:

$$
\text{Complexity} = O(T \cdot (C + T)^2 \cdot d)
$$

##### With KV-Cache

$$
\text{Complexity} = O(T \cdot (C + T) \cdot d)
$$

This reduces quadratic to linear complexity in the generation length.

##### Decoder vs Encoder Comparison

| Aspect | Encoder | Decoder |
|--------|---------|---------|
| Attention Type | Bidirectional | Causal (masked) |
| Can see future | Yes | No |
| Primary use | Understanding | Generation |
| Examples | BERT, RoBERTa | GPT, LLaMA |

##### Summary

The Transformer decoder block enables autoregressive generation through:

1. **Causal Masking**: Prevents attending to future positions
2. **Cross-Attention**: Connects to encoder in seq2seq models
3. **KV-Caching**: Enables efficient incremental generation
4. **Pre-norm Architecture**: Improves training stability

##### References

1. Vaswani, A., et al. (2017). "Attention Is All You Need." NeurIPS.
2. Radford, A., et al. (2018). "Improving Language Understanding by Generative Pre-Training."
3. Brown, T., et al. (2020). "Language Models are Few-Shot Learners." NeurIPS.
