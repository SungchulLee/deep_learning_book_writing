# Transformer Encoder Block

## Overview

The Transformer encoder block is the fundamental building unit of encoder-based architectures like BERT. Each block transforms its input through self-attention and feed-forward layers, maintaining the sequence length while refining representations.

## Architecture

Each encoder block consists of two sub-layers with residual connections and layer normalization:

$$
\begin{aligned}
\mathbf{Z} &= \text{LayerNorm}(\mathbf{X} + \text{MultiHeadAttention}(\mathbf{X}, \mathbf{X}, \mathbf{X})) \\
\mathbf{X}' &= \text{LayerNorm}(\mathbf{Z} + \text{FFN}(\mathbf{Z}))
\end{aligned}
$$

### Component Breakdown

1. **Multi-Head Self-Attention**: Captures dependencies between all positions
2. **Feed-Forward Network**: Position-wise transformation with expansion
3. **Residual Connections**: Enable gradient flow in deep networks
4. **Layer Normalization**: Stabilizes training

## Multi-Head Self-Attention in Encoder

The encoder uses bidirectional self-attention where each position can attend to all positions:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

For self-attention, $Q$, $K$, $V$ are all derived from the same input:

$$
Q = XW^Q, \quad K = XW^K, \quad V = XW^V
$$

### Multi-Head Formulation

$$
\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O
$$

Where each head:

$$
\text{head}_i = \text{Attention}(XW_i^Q, XW_i^K, XW_i^V)
$$

## Feed-Forward Network

The position-wise FFN applies the same transformation to each position independently:

$$
\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2
$$

Or with GELU activation (common in modern models):

$$
\text{FFN}(x) = \text{GELU}(xW_1 + b_1)W_2 + b_2
$$

### Dimension Expansion

The FFN typically expands the hidden dimension by a factor of 4:

- Input: $d_{\text{model}}$
- Hidden: $d_{ff} = 4 \times d_{\text{model}}$
- Output: $d_{\text{model}}$

## Pre-Norm vs Post-Norm

### Post-Norm (Original Transformer)

$$
\mathbf{X}' = \text{LayerNorm}(\mathbf{X} + \text{SubLayer}(\mathbf{X}))
$$

### Pre-Norm (Modern Standard)

$$
\mathbf{X}' = \mathbf{X} + \text{SubLayer}(\text{LayerNorm}(\mathbf{X}))
$$

Pre-norm is more stable for training deep models and doesn't require learning rate warmup.

## PyTorch Implementation

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

## Gradient Flow Analysis

### Residual Connection Benefits

Without residuals, gradients must flow through all sublayers:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{x}^{(0)}} = \prod_{l=0}^{L-1} \frac{\partial \text{SubLayer}^{(l)}}{\partial \mathbf{x}^{(l)}}
$$

With residuals, there's a direct gradient path:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{x}^{(0)}} = \frac{\partial \mathcal{L}}{\partial \mathbf{x}^{(L)}} + \text{other terms}
$$

### Layer Normalization Effects

Layer norm normalizes across the feature dimension:

$$
\text{LayerNorm}(\mathbf{x}) = \gamma \odot \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

Where:
- $\mu = \frac{1}{d} \sum_{i=1}^{d} x_i$
- $\sigma^2 = \frac{1}{d} \sum_{i=1}^{d} (x_i - \mu)^2$

This stabilizes the gradient magnitude across layers.

## Computational Complexity

### Per-Layer Complexity

For sequence length $n$, model dimension $d$, and FFN dimension $d_{ff}$:

| Component | Time Complexity | Space Complexity |
|-----------|-----------------|------------------|
| Self-Attention | $O(n^2 d)$ | $O(n^2 + nd)$ |
| FFN | $O(n d \cdot d_{ff})$ | $O(n d_{ff})$ |
| Layer Norm | $O(nd)$ | $O(nd)$ |

### Total Encoder Complexity

For $L$ layers:

$$
\text{Time: } O(L \cdot (n^2 d + n d \cdot d_{ff}))
$$

With typical $d_{ff} = 4d$:

$$
\text{Time: } O(L \cdot n \cdot d \cdot (n + 4d))
$$

## Summary

The Transformer encoder block is a modular, powerful building block that:

1. **Captures Global Dependencies**: Self-attention connects all positions
2. **Applies Non-linear Transformations**: FFN adds representational capacity
3. **Maintains Gradient Flow**: Residuals enable deep architectures
4. **Stabilizes Training**: Layer normalization controls activations

Understanding the encoder block is essential for implementing models like BERT, RoBERTa, and encoder-based architectures.

## References

1. Vaswani, A., et al. (2017). "Attention Is All You Need." NeurIPS.
2. Devlin, J., et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers." NAACL.
3. Xiong, R., et al. (2020). "On Layer Normalization in the Transformer Architecture." ICML.
4. He, K., et al. (2016). "Deep Residual Learning for Image Recognition." CVPR.
