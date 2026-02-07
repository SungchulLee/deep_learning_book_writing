# Attention Map Visualization

## Overview

Attention maps provide interpretable windows into how a Transformer processes information. By visualizing the attention weights, we can observe which tokens the model focuses on when computing representations, revealing learned linguistic patterns such as coreference, syntactic dependencies, and cross-lingual alignment.

The Transformer produces three distinct types of attention maps, each corresponding to a different attention mechanism in the architecture.

## Types of Attention Maps

### Encoder Self-Attention Maps

Encoder self-attention maps show how each token in the **source sequence** attends to every other token in the same sequence. These maps reveal the relationships and dependencies the encoder captures when building contextual representations.

**Shape**: $(\text{batch} \times \text{heads}, \; \text{src\_len}, \; \text{src\_len})$

- Rows represent query tokens (the token being processed)
- Columns represent key tokens (the token being attended to)

**What to look for**:

- Diagonal dominance: tokens attending primarily to themselves (common in early layers)
- Subject-verb agreement: verbs attending to their subjects
- Coreference: pronouns attending to their referents
- Phrase structure: tokens within a constituent attending to each other

**Example**: In "The cat sat on the mat," the encoder may show "sat" attending strongly to "cat" (subject-verb relationship) and "mat" (object relationship).

### Decoder Self-Attention Maps

Decoder self-attention maps show how each token in the **target sequence** attends to previously generated tokens. The causal mask ensures that position $t$ can only attend to positions $1, 2, \ldots, t$.

**Shape**: $(\text{batch} \times \text{heads}, \; \text{tgt\_len}, \; \text{tgt\_len})$

- The lower-triangular pattern reflects the causal constraint
- Shows how the decoder builds coherent output by referencing its own prior context

**What to look for**:

- Recent-token focus: strong attention to the immediately preceding token (bigram-like behavior)
- Long-range coherence: attention to earlier tokens that establish topic or grammatical structure
- Punctuation patterns: tokens attending to sentence boundaries

### Decoder Cross-Attention Maps

Cross-attention maps show how each token in the **target sequence** attends to tokens in the **source sequence** (encoder output). This is the alignment mechanism that connects source and target.

**Shape**: $(\text{batch} \times \text{heads}, \; \text{tgt\_len}, \; \text{src\_len})$

- Rows represent target query tokens
- Columns represent source key tokens

**What to look for**:

- Monotonic alignment: in translation, a roughly diagonal pattern indicates word-order-preserving alignment
- Reordering: off-diagonal attention reveals syntactic reordering between languages
- Many-to-one: multiple target tokens attending to a single source token (one source word translates to several target words)
- One-to-many: a single target token attending to multiple source tokens (several source words condense into one target word)

## Comparison of Attention Map Types

| Type | Query Source | Key/Value Source | Masking | Reveals |
|------|-------------|------------------|---------|---------|
| **Encoder self-attention** | Source tokens | Source tokens | Padding only | Intra-source dependencies |
| **Decoder self-attention** | Target tokens | Target tokens | Causal + padding | Autoregressive context usage |
| **Decoder cross-attention** | Target tokens | Encoder output | Padding only | Source-target alignment |

## PyTorch Implementation

### Transformer with Attention Extraction

```python
import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, List, Dict


class AttentionLayer(nn.Module):
    """Multi-head attention that stores attention weights for visualization."""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.attn_weights = None  # Stored after forward pass
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query, key, value: [batch_size, seq_len, d_model]
            attn_mask: [tgt_len, src_len] or [batch * heads, tgt_len, src_len]
            key_padding_mask: [batch_size, src_len]
        
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        attn_output, attn_weights = self.mha(
            query, key, value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            average_attn_weights=False          # Keep per-head weights
        )
        self.attn_weights = attn_weights.detach()  # [batch, heads, tgt_len, src_len]
        
        return self.norm(query + self.dropout(attn_output))


class FeedForwardBlock(nn.Module):
    """Position-wise feed-forward network with residual connection."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.dropout(self.ffn(x)))


class EncoderLayerWithAttn(nn.Module):
    """Encoder layer that exposes attention weights."""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = AttentionLayer(d_model, num_heads, dropout)
        self.ffn = FeedForwardBlock(d_model, d_ff, dropout)
    
    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        x = self.self_attn(x, x, x, attn_mask=src_mask,
                           key_padding_mask=src_key_padding_mask)
        return self.ffn(x)


class DecoderLayerWithAttn(nn.Module):
    """Decoder layer that exposes self-attention and cross-attention weights."""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = AttentionLayer(d_model, num_heads, dropout)
        self.cross_attn = AttentionLayer(d_model, num_heads, dropout)
        self.ffn = FeedForwardBlock(d_model, d_ff, dropout)
    
    def forward(self, x, memory, tgt_mask=None, memory_key_padding_mask=None):
        x = self.self_attn(x, x, x, attn_mask=tgt_mask)
        x = self.cross_attn(x, memory, memory, 
                            key_padding_mask=memory_key_padding_mask)
        return self.ffn(x)


class TransformerWithAttnMaps(nn.Module):
    """
    Full Transformer that stores all attention maps for visualization.
    
    After a forward pass, call get_attention_maps() to retrieve
    encoder, decoder self-attention, and cross-attention weights.
    """
    
    def __init__(
        self,
        src_vocab: int,
        tgt_vocab: int,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        num_layers: int = 6,
        dropout: float = 0.1,
        max_len: int = 5000
    ):
        super().__init__()
        self.d_model = d_model
        
        # Embeddings
        self.src_embed = nn.Embedding(src_vocab, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab, d_model)
        
        # Positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
        # Encoder and decoder layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayerWithAttn(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayerWithAttn(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.output_proj = nn.Linear(d_model, tgt_vocab)
        self.dropout = nn.Dropout(dropout)
    
    def _embed(self, tokens, embed_layer):
        x = embed_layer(tokens) * math.sqrt(self.d_model)
        return self.dropout(x + self.pe[:, :tokens.size(1), :])
    
    def forward(self, src, tgt, tgt_mask=None, src_key_padding_mask=None):
        """
        Forward pass storing all attention maps.
        
        Args:
            src: [batch_size, src_len]
            tgt: [batch_size, tgt_len]
            tgt_mask: [tgt_len, tgt_len] causal mask
            src_key_padding_mask: [batch_size, src_len]
        
        Returns:
            output: [batch_size, tgt_len, tgt_vocab]
        """
        # Encode
        enc = self._embed(src, self.src_embed)
        for layer in self.encoder_layers:
            enc = layer(enc, src_key_padding_mask=src_key_padding_mask)
        
        # Decode
        dec = self._embed(tgt, self.tgt_embed)
        for layer in self.decoder_layers:
            dec = layer(dec, enc, tgt_mask=tgt_mask,
                       memory_key_padding_mask=src_key_padding_mask)
        
        return self.output_proj(dec)
    
    def get_attention_maps(self) -> Dict[str, List[torch.Tensor]]:
        """
        Retrieve stored attention maps from the last forward pass.
        
        Returns:
            dict with keys:
                'encoder': List of [batch, heads, src_len, src_len]
                'decoder_self': List of [batch, heads, tgt_len, tgt_len]
                'decoder_cross': List of [batch, heads, tgt_len, src_len]
        """
        return {
            "encoder": [
                layer.self_attn.attn_weights
                for layer in self.encoder_layers
            ],
            "decoder_self": [
                layer.self_attn.attn_weights
                for layer in self.decoder_layers
            ],
            "decoder_cross": [
                layer.cross_attn.attn_weights
                for layer in self.decoder_layers
            ],
        }
```

### Visualization Functions

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_attention_map(
    attn_weights: torch.Tensor,
    x_labels: list = None,
    y_labels: list = None,
    title: str = "Attention Weights",
    figsize: tuple = (8, 6)
):
    """
    Plot a single attention map as a heatmap.
    
    Args:
        attn_weights: [tgt_len, src_len] attention weight matrix
        x_labels: Labels for source tokens (columns)
        y_labels: Labels for target tokens (rows)
        title: Plot title
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    attn = attn_weights.cpu().numpy()
    sns.heatmap(
        attn,
        xticklabels=x_labels if x_labels else "auto",
        yticklabels=y_labels if y_labels else "auto",
        cmap="viridis",
        vmin=0.0,
        vmax=attn.max(),
        ax=ax,
        square=True
    )
    ax.set_title(title)
    ax.set_xlabel("Key (attended to)")
    ax.set_ylabel("Query (attending)")
    plt.tight_layout()
    return fig


def plot_multihead_attention(
    attn_weights: torch.Tensor,
    num_heads: int,
    x_labels: list = None,
    y_labels: list = None,
    layer_name: str = "Layer",
    sample_idx: int = 0
):
    """
    Plot attention maps for all heads in a single layer.
    
    Args:
        attn_weights: [batch, heads, tgt_len, src_len]
        num_heads: Number of attention heads
        x_labels: Source token labels
        y_labels: Target token labels
        layer_name: Name for the plot title
        sample_idx: Batch index to visualize
    """
    cols = min(4, num_heads)
    rows = (num_heads + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if num_heads == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for head in range(num_heads):
        attn = attn_weights[sample_idx, head].cpu().numpy()
        
        sns.heatmap(
            attn,
            xticklabels=x_labels if x_labels else False,
            yticklabels=y_labels if y_labels and head % cols == 0 else False,
            cmap="viridis",
            vmin=0.0,
            ax=axes[head],
            square=True,
            cbar=False
        )
        axes[head].set_title(f"Head {head + 1}", fontsize=10)
    
    # Hide unused subplots
    for idx in range(num_heads, len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle(f"{layer_name} — All Heads", fontsize=14)
    plt.tight_layout()
    return fig


def plot_attention_across_layers(
    attention_maps: Dict[str, List[torch.Tensor]],
    map_type: str = "encoder",
    head: int = 0,
    sample_idx: int = 0,
    src_tokens: list = None,
    tgt_tokens: list = None
):
    """
    Plot attention for a specific head across all layers.
    
    Args:
        attention_maps: Output of model.get_attention_maps()
        map_type: 'encoder', 'decoder_self', or 'decoder_cross'
        head: Head index to visualize
        sample_idx: Batch sample index
        src_tokens: Source token strings
        tgt_tokens: Target token strings
    """
    maps = attention_maps[map_type]
    num_layers = len(maps)
    
    fig, axes = plt.subplots(1, num_layers, figsize=(5 * num_layers, 5))
    if num_layers == 1:
        axes = [axes]
    
    for layer_idx, attn in enumerate(maps):
        attn_data = attn[sample_idx, head].cpu().numpy()
        
        # Choose labels based on map type
        if map_type == "encoder":
            x_labels = src_tokens
            y_labels = src_tokens
        elif map_type == "decoder_self":
            x_labels = tgt_tokens
            y_labels = tgt_tokens
        else:  # decoder_cross
            x_labels = src_tokens
            y_labels = tgt_tokens
        
        sns.heatmap(
            attn_data,
            xticklabels=x_labels if x_labels else False,
            yticklabels=y_labels if y_labels and layer_idx == 0 else False,
            cmap="viridis",
            ax=axes[layer_idx],
            square=True,
            cbar=False
        )
        axes[layer_idx].set_title(f"Layer {layer_idx + 1}")
    
    type_names = {
        "encoder": "Encoder Self-Attention",
        "decoder_self": "Decoder Self-Attention",
        "decoder_cross": "Decoder Cross-Attention"
    }
    fig.suptitle(f"{type_names[map_type]} — Head {head + 1}", fontsize=14)
    plt.tight_layout()
    return fig
```

### Usage Example

```python
# Build model and run forward pass
model = TransformerWithAttnMaps(
    src_vocab=1000, tgt_vocab=1000,
    d_model=256, num_heads=8, d_ff=512,
    num_layers=3, dropout=0.1
)

# Example tokens
src = torch.tensor([[10, 42, 87, 3, 55, 91]])       # [1, 6]
tgt = torch.tensor([[1, 23, 67, 45, 12]])            # [1, 5]

# Create causal mask
tgt_len = tgt.size(1)
tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1).bool()

# Forward pass
output = model(src, tgt, tgt_mask=tgt_mask)

# Get attention maps
attn_maps = model.get_attention_maps()

# Token labels (for visualization)
src_tokens = ["The", "cat", "sat", "on", "the", "mat"]
tgt_tokens = ["Le", "chat", "était", "assis", "sur"]

# Visualize encoder self-attention across layers
plot_attention_across_layers(
    attn_maps, "encoder", head=0,
    src_tokens=src_tokens
)
plt.show()

# Visualize cross-attention across layers
plot_attention_across_layers(
    attn_maps, "decoder_cross", head=0,
    src_tokens=src_tokens, tgt_tokens=tgt_tokens
)
plt.show()

# Visualize all heads for encoder layer 1
plot_multihead_attention(
    attn_maps["encoder"][0], num_heads=8,
    x_labels=src_tokens, y_labels=src_tokens,
    layer_name="Encoder Layer 1"
)
plt.show()
```

## Interpreting Attention Patterns

### Common Encoder Patterns

Across different layers of a trained Transformer, encoder attention typically exhibits these patterns:

- **Layer 1–2**: Largely local attention; tokens attend to their immediate neighbors and to themselves. This captures surface-level patterns and local syntax.
- **Layer 3–4**: Broader syntactic patterns emerge; subjects attend to verbs, adjectives attend to the nouns they modify.
- **Layer 5–6**: Semantic and long-range dependencies; coreference resolution, discourse-level patterns, and more abstract relationships.

### Common Decoder Self-Attention Patterns

- **Strong recency bias**: The most recent tokens typically receive the highest attention, reflecting the sequential nature of generation.
- **Functional tokens**: Punctuation and structural tokens (commas, periods, conjunctions) often serve as attention anchors.
- **Repetition avoidance**: In well-trained models, attention patterns help the decoder track what has already been generated.

### Common Cross-Attention Patterns

- **Alignment**: In translation, cross-attention often mirrors word alignment, revealing which source words contribute to each target word.
- **Reordering**: Non-monotonic attention patterns reveal syntactic differences between source and target languages (e.g., verb-final languages).
- **Compression/expansion**: When one source word maps to multiple target words (or vice versa), the attention pattern broadens or narrows accordingly.

## Head Specialization

Individual attention heads within the same layer often specialize in different linguistic functions:

- **Positional heads**: Attend primarily to adjacent positions (previous or next token)
- **Syntactic heads**: Capture dependency relations (subject→verb, noun→determiner)
- **Rare-token heads**: Focus on low-frequency tokens that carry high information
- **Separator heads**: Attend to punctuation or special tokens

This specialization emerges naturally during training and is one reason multi-head attention outperforms single-head attention with the same total dimension.

## Limitations of Attention Visualization

While attention maps are informative, they have important caveats:

1. **Attention ≠ Attribution**: High attention weight does not necessarily mean a token is important for the final prediction. Attention weights indicate information flow, but the downstream layers may ignore or transform that information.
2. **Layer composition**: Attention in layer $l$ builds on representations from layer $l-1$, which themselves are influenced by attention in earlier layers. A token's effective influence on the output involves the composition of attention across all layers.
3. **Residual connections**: The residual stream can carry information that bypasses attention entirely, meaning the model can "remember" information without explicitly attending to it.
4. **Head averaging**: Averaging across heads can obscure individual head specialization. Always examine per-head patterns.

For more rigorous attribution, consider gradient-based methods (see Chapter 29: Model Interpretability).

## Summary

Attention maps provide three complementary views of Transformer processing:

1. **Encoder self-attention** reveals intra-source relationships and learned linguistic structure.
2. **Decoder self-attention** shows how the model uses its own generated context during autoregressive generation.
3. **Decoder cross-attention** exposes the alignment between source and target sequences.

Visualizing these maps across layers and heads reveals how the model progressively builds understanding from local patterns to global semantics.

## References

1. Vaswani, A., et al. (2017). "Attention Is All You Need." NeurIPS.
2. Clark, K., et al. (2019). "What Does BERT Look At? An Analysis of BERT's Attention." BlackboxNLP.
3. Vig, J. (2019). "A Multiscale Visualization of Attention in the Transformer Model." ACL Demo.
4. Jain, S., & Wallace, B. C. (2019). "Attention is Not Explanation." NAACL.
5. Wiegreffe, S., & Pinter, Y. (2019). "Attention is Not Not Explanation." EMNLP.
