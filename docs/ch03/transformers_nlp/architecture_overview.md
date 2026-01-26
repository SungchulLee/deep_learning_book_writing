# Transformer Architecture Overview

## Introduction

The Transformer architecture, introduced in "Attention Is All You Need" (Vaswani et al., 2017), revolutionized sequence modeling by eliminating recurrence entirely in favor of self-attention mechanisms. This architectural paradigm shift enabled unprecedented parallelization during training and established the foundation for modern large language models.

## Historical Context

Before Transformers, sequence-to-sequence tasks relied on recurrent architectures (RNNs, LSTMs, GRUs) with attention mechanisms. While effective, these approaches suffered from sequential computation bottlenecks and difficulty capturing long-range dependencies. The Transformer addresses these limitations through its attention-only architecture.

## Architecture Overview

The Transformer follows an encoder-decoder structure, though many modern variants use only the encoder (BERT) or decoder (GPT) components.

### High-Level Structure

$$
\text{Transformer}(X) = \text{Decoder}(\text{Encoder}(X), Y_{\text{shifted}})
$$

The complete architecture consists of:

1. **Input Embedding Layer**: Converts tokens to dense vectors
2. **Positional Encoding**: Injects sequence order information
3. **Encoder Stack**: $N$ identical encoder layers
4. **Decoder Stack**: $N$ identical decoder layers  
5. **Output Linear Layer**: Projects to vocabulary size
6. **Softmax**: Produces probability distribution

### Dimensional Flow

For a model with:
- Vocabulary size $V$
- Model dimension $d_{\text{model}}$
- Sequence length $L$
- Batch size $B$

The dimensional transformations are:

$$
\begin{aligned}
\text{Input tokens} &: (B, L) \\
\text{After embedding} &: (B, L, d_{\text{model}}) \\
\text{After encoder} &: (B, L, d_{\text{model}}) \\
\text{After decoder} &: (B, L, d_{\text{model}}) \\
\text{After output projection} &: (B, L, V)
\end{aligned}
$$

## Core Components

### 1. Input Embeddings

The input embedding layer maps discrete tokens to continuous vectors:

$$
\mathbf{E} = \text{Embedding}(\mathbf{x}) \cdot \sqrt{d_{\text{model}}}
$$

The scaling factor $\sqrt{d_{\text{model}}}$ ensures the embedding magnitudes are comparable to positional encodings.

### 2. Positional Encoding

Since attention is permutation-invariant, positional information must be explicitly added:

$$
\text{PE}_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

$$
\text{PE}_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

The final input representation:

$$
\mathbf{X}_0 = \mathbf{E} + \text{PE}
$$

### 3. Encoder Layer

Each encoder layer contains two sub-layers:

$$
\begin{aligned}
\mathbf{Z} &= \text{LayerNorm}(\mathbf{X} + \text{MultiHeadAttention}(\mathbf{X}, \mathbf{X}, \mathbf{X})) \\
\mathbf{X}' &= \text{LayerNorm}(\mathbf{Z} + \text{FFN}(\mathbf{Z}))
\end{aligned}
$$

### 4. Decoder Layer

Each decoder layer contains three sub-layers:

$$
\begin{aligned}
\mathbf{Z}_1 &= \text{LayerNorm}(\mathbf{Y} + \text{MaskedMultiHeadAttention}(\mathbf{Y}, \mathbf{Y}, \mathbf{Y})) \\
\mathbf{Z}_2 &= \text{LayerNorm}(\mathbf{Z}_1 + \text{MultiHeadAttention}(\mathbf{Z}_1, \mathbf{X}_{\text{enc}}, \mathbf{X}_{\text{enc}})) \\
\mathbf{Y}' &= \text{LayerNorm}(\mathbf{Z}_2 + \text{FFN}(\mathbf{Z}_2))
\end{aligned}
$$

### 5. Feed-Forward Network

The position-wise feed-forward network:

$$
\text{FFN}(\mathbf{x}) = \max(0, \mathbf{x}\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2
$$

With dimensions:
- $\mathbf{W}_1 \in \mathbb{R}^{d_{\text{model}} \times d_{ff}}$
- $\mathbf{W}_2 \in \mathbb{R}^{d_{ff} \times d_{\text{model}}}$

Typically $d_{ff} = 4 \times d_{\text{model}}$.

## PyTorch Implementation

```python
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """Single Transformer encoder block."""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Multi-head self-attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        key_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Attention mask [seq_len, seq_len]
            key_padding_mask: Padding mask [batch_size, seq_len]
        """
        # Self-attention with residual connection
        attn_output, _ = self.self_attention(
            x, x, x,
            attn_mask=mask,
            key_padding_mask=key_padding_mask
        )
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class Transformer(nn.Module):
    """Complete Transformer model for sequence-to-sequence tasks."""
    
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        d_ff: int = 2048,
        max_len: int = 5000,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # Encoder
        self.encoder_layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        # Decoder (simplified - encoder-only shown here)
        self.decoder_layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def encode(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor = None,
        src_key_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """Encode source sequence."""
        # Embedding + positional encoding
        x = self.src_embedding(src) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        
        # Pass through encoder layers
        for layer in self.encoder_layers:
            x = layer(x, src_mask, src_key_padding_mask)
        
        return x
    
    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """Decode target sequence."""
        # Embedding + positional encoding
        x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        
        # Pass through decoder layers
        for layer in self.decoder_layers:
            x = layer(x, tgt_mask)
        
        return x
    
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None,
        src_key_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass through the Transformer.
        
        Args:
            src: Source sequence [batch_size, src_len]
            tgt: Target sequence [batch_size, tgt_len]
            src_mask: Source attention mask
            tgt_mask: Target attention mask (causal)
            src_key_padding_mask: Source padding mask
        
        Returns:
            Output logits [batch_size, tgt_len, tgt_vocab_size]
        """
        # Encode
        memory = self.encode(src, src_mask, src_key_padding_mask)
        
        # Decode
        output = self.decode(tgt, memory, tgt_mask)
        
        # Project to vocabulary
        return self.output_projection(output)
    
    @staticmethod
    def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
        """Generate causal mask for autoregressive decoding."""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask


# Usage example
if __name__ == "__main__":
    # Model configuration
    model = Transformer(
        src_vocab_size=10000,
        tgt_vocab_size=10000,
        d_model=512,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ff=2048
    )
    
    # Sample input
    batch_size, src_len, tgt_len = 32, 20, 15
    src = torch.randint(0, 10000, (batch_size, src_len))
    tgt = torch.randint(0, 10000, (batch_size, tgt_len))
    
    # Generate causal mask for decoder
    tgt_mask = Transformer.generate_square_subsequent_mask(tgt_len)
    
    # Forward pass
    output = model(src, tgt, tgt_mask=tgt_mask)
    print(f"Output shape: {output.shape}")  # [32, 15, 10000]
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
```

## Hyperparameters

The original Transformer paper used two configurations:

| Parameter | Base Model | Big Model |
|-----------|------------|-----------|
| $d_{\text{model}}$ | 512 | 1024 |
| $d_{ff}$ | 2048 | 4096 |
| $h$ (heads) | 8 | 16 |
| $N$ (layers) | 6 | 6 |
| $d_k = d_v$ | 64 | 64 |
| Parameters | 65M | 213M |

## Computational Complexity

### Self-Attention Complexity

For sequence length $n$ and model dimension $d$:

$$
\text{Time: } O(n^2 \cdot d) \qquad \text{Space: } O(n^2 + n \cdot d)
$$

The quadratic complexity in sequence length becomes a bottleneck for long sequences.

### Feed-Forward Complexity

$$
\text{Time: } O(n \cdot d \cdot d_{ff}) \qquad \text{Space: } O(n \cdot d_{ff})
$$

### Comparison with RNNs

| Aspect | Transformer | RNN |
|--------|-------------|-----|
| Sequential Operations | $O(1)$ | $O(n)$ |
| Complexity per Layer | $O(n^2 \cdot d)$ | $O(n \cdot d^2)$ |
| Maximum Path Length | $O(1)$ | $O(n)$ |
| Parallelization | High | Low |

## Key Design Decisions

### 1. Residual Connections

Residual connections enable gradient flow through deep networks:

$$
\mathbf{x}^{(l+1)} = \mathbf{x}^{(l)} + \text{SubLayer}(\text{LayerNorm}(\mathbf{x}^{(l)}))
$$

### 2. Layer Normalization

Layer normalization stabilizes training:

$$
\text{LayerNorm}(\mathbf{x}) = \gamma \odot \frac{\mathbf{x} - \mu}{\sigma + \epsilon} + \beta
$$

Where $\mu$ and $\sigma$ are computed across the feature dimension.

### 3. Pre-Norm vs Post-Norm

**Post-Norm (Original):**
$$
\mathbf{x}' = \text{LayerNorm}(\mathbf{x} + \text{SubLayer}(\mathbf{x}))
$$

**Pre-Norm (Common in modern variants):**
$$
\mathbf{x}' = \mathbf{x} + \text{SubLayer}(\text{LayerNorm}(\mathbf{x}))
$$

Pre-norm tends to be more stable for training very deep models.

## Variants and Extensions

### Encoder-Only (BERT-style)
- Bidirectional attention
- Used for understanding tasks
- Pre-trained with masked language modeling

### Decoder-Only (GPT-style)
- Causal (unidirectional) attention
- Used for generation tasks
- Pre-trained with next-token prediction

### Encoder-Decoder (T5-style)
- Full sequence-to-sequence capability
- Cross-attention between encoder and decoder
- Unified text-to-text framework

## Summary

The Transformer architecture represents a fundamental shift in sequence modeling:

1. **Parallelization**: Removes sequential dependencies during training
2. **Long-range dependencies**: Direct connections between any positions
3. **Scalability**: Foundation for billion-parameter models
4. **Versatility**: Applicable to text, images, audio, and multimodal data

The modular design (attention + FFN + normalization + residuals) has proven remarkably effective across domains, establishing Transformers as the dominant architecture in modern deep learning.

## References

1. Vaswani, A., et al. (2017). "Attention Is All You Need." NeurIPS.
2. Devlin, J., et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers." NAACL.
3. Brown, T., et al. (2020). "Language Models are Few-Shot Learners." NeurIPS.
4. Xiong, R., et al. (2020). "On Layer Normalization in the Transformer Architecture." ICML.
