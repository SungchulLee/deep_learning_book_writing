# Attention Fundamentals for Visualization

## Introduction

Attention visualization is a cornerstone technique for understanding how transformer models process and weigh information across sequences. By examining attention weights, we gain insights into which input tokens influence specific output tokens, revealing the internal reasoning patterns of neural networks.

This section establishes the mathematical and conceptual foundations necessary for effective attention visualization, bridging the gap between attention theory and practical interpretability techniques.

## Mathematical Foundations

### Scaled Dot-Product Attention

The attention mechanism computes a weighted sum of values based on the compatibility between queries and keys:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

where:

- $Q \in \mathbb{R}^{n \times d_k}$: Query matrix (what we're looking for)
- $K \in \mathbb{R}^{m \times d_k}$: Key matrix (what we're searching through)
- $V \in \mathbb{R}^{m \times d_v}$: Value matrix (what we retrieve)
- $d_k$: Key/query dimension (scaling factor)
- $n$: Query sequence length
- $m$: Key/value sequence length

### Attention Weight Matrix

The attention weight matrix $A$ captures the core visualization target:

$$A = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) \in \mathbb{R}^{n \times m}$$

Each element $A_{ij}$ represents the attention weight from query position $i$ to key position $j$, satisfying:

$$\sum_{j=1}^{m} A_{ij} = 1 \quad \forall i \in \{1, \ldots, n\}$$

This row-wise normalization property ensures that attention weights form a valid probability distribution over keys for each query.

### Interpretation of Attention Weights

The attention weight $A_{ij}$ can be interpreted as:

1. **Information flow**: The proportion of information flowing from position $j$ to position $i$
2. **Relevance score**: How relevant key $j$ is for computing the output at query position $i$
3. **Soft alignment**: A soft pointer from query $i$ to key $j$ in sequence-to-sequence tasks

## Extracting Attention Weights

### From PyTorch Models

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional

class AttentionExtractor:
    """
    Utility class for extracting attention weights from transformer models.
    
    Supports both custom implementations and Hugging Face models.
    """
    
    def __init__(self, model: nn.Module):
        """
        Initialize with a transformer model.
        
        Parameters
        ----------
        model : nn.Module
            Transformer model with attention layers
        """
        self.model = model
        self.attention_weights = []
        self._hooks = []
    
    def register_hooks(self, layer_names: Optional[List[str]] = None):
        """
        Register forward hooks to capture attention weights.
        
        Parameters
        ----------
        layer_names : list of str, optional
            Names of attention layers to hook. If None, hooks all attention layers.
        """
        def hook_fn(module, input, output):
            # Handle different output formats
            if isinstance(output, tuple):
                # Many implementations return (context, attention_weights)
                attn = output[1] if len(output) > 1 else output[0]
            else:
                attn = output
            self.attention_weights.append(attn.detach().cpu())
        
        for name, module in self.model.named_modules():
            if layer_names is None:
                # Auto-detect attention layers by common naming conventions
                if any(key in name.lower() for key in ['attention', 'attn', 'self_attn']):
                    hook = module.register_forward_hook(hook_fn)
                    self._hooks.append(hook)
            elif name in layer_names:
                hook = module.register_forward_hook(hook_fn)
                self._hooks.append(hook)
    
    def clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        self.attention_weights = []
    
    def get_attention(self, 
                     input_ids: torch.Tensor,
                     attention_mask: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        """
        Extract attention weights for given input.
        
        Parameters
        ----------
        input_ids : torch.Tensor
            Input token IDs, shape: (batch_size, seq_len)
        attention_mask : torch.Tensor, optional
            Attention mask for padding
        
        Returns
        -------
        list of torch.Tensor
            Attention weights from each layer
        """
        self.attention_weights = []
        
        self.model.eval()
        with torch.no_grad():
            if attention_mask is not None:
                _ = self.model(input_ids, attention_mask=attention_mask)
            else:
                _ = self.model(input_ids)
        
        return self.attention_weights
```

### From Hugging Face Transformers

Hugging Face models provide a convenient interface for attention extraction:

```python
from transformers import AutoModel, AutoTokenizer

def extract_huggingface_attention(
    model_name: str,
    text: str,
    layer_idx: int = -1,
    head_idx: Optional[int] = None
) -> Tuple[torch.Tensor, List[str]]:
    """
    Extract attention weights from a Hugging Face model.
    
    Parameters
    ----------
    model_name : str
        Name of the pretrained model (e.g., 'bert-base-uncased')
    text : str
        Input text to analyze
    layer_idx : int
        Layer index (-1 for last layer)
    head_idx : int, optional
        Head index (None for all heads)
    
    Returns
    -------
    attention : torch.Tensor
        Attention weights, shape depends on head_idx:
        - If head_idx is None: (num_heads, seq_len, seq_len)
        - If head_idx is int: (seq_len, seq_len)
    tokens : list of str
        Tokenized input including special tokens
    """
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_attentions=True)
    model.eval()
    
    # Tokenize input
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    
    # Forward pass with attention output
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract attention from specified layer
    # outputs.attentions is tuple of (batch, heads, seq, seq) per layer
    attentions = outputs.attentions
    
    if layer_idx < 0:
        layer_idx = len(attentions) + layer_idx
    
    # Shape: (batch=1, num_heads, seq_len, seq_len)
    layer_attention = attentions[layer_idx][0]  # Remove batch dimension
    
    # Optionally select specific head
    if head_idx is not None:
        layer_attention = layer_attention[head_idx]
    
    # Get tokens
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    return layer_attention, tokens
```

## Basic Visualization Techniques

### Attention Heatmaps

The most common visualization approach uses heatmaps to display the attention matrix:

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional

def plot_attention_heatmap(
    attention: torch.Tensor,
    x_tokens: List[str],
    y_tokens: Optional[List[str]] = None,
    title: str = "Attention Weights",
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = 'viridis',
    show_values: bool = False,
    save_path: Optional[str] = None
) -> None:
    """
    Create a heatmap visualization of attention weights.
    
    Parameters
    ----------
    attention : torch.Tensor
        Attention weight matrix, shape: (query_len, key_len)
    x_tokens : list of str
        Key tokens (x-axis labels)
    y_tokens : list of str, optional
        Query tokens (y-axis labels). If None, uses x_tokens.
    title : str
        Plot title
    figsize : tuple
        Figure size (width, height)
    cmap : str
        Matplotlib colormap
    show_values : bool
        Whether to display numerical values in cells
    save_path : str, optional
        Path to save the figure
    """
    if isinstance(attention, torch.Tensor):
        attention = attention.cpu().numpy()
    
    if y_tokens is None:
        y_tokens = x_tokens
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    im = sns.heatmap(
        attention,
        annot=show_values,
        fmt='.3f' if show_values else '',
        cmap=cmap,
        square=True,
        xticklabels=x_tokens,
        yticklabels=y_tokens,
        cbar_kws={'label': 'Attention Weight', 'shrink': 0.8},
        vmin=0,
        vmax=1,
        ax=ax
    )
    
    # Styling
    ax.set_xlabel('Key Tokens (attended to)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Query Tokens (attending from)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Rotate labels for readability
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()
```

### Attention Statistics

Understanding attention distributions requires quantitative metrics:

```python
def compute_attention_statistics(attention: torch.Tensor) -> dict:
    """
    Compute statistical measures of attention distribution.
    
    Parameters
    ----------
    attention : torch.Tensor
        Attention weight matrix, shape: (seq_len, seq_len)
    
    Returns
    -------
    dict
        Dictionary of statistical metrics:
        - mean: Average attention value
        - std: Standard deviation
        - max: Maximum attention value
        - diagonal_mean: Average self-attention
        - entropy: Mean entropy of attention distributions
        - sparsity: Gini coefficient measuring concentration
    """
    if isinstance(attention, torch.Tensor):
        attention = attention.cpu().numpy()
    
    stats = {}
    
    # Basic statistics
    stats['mean'] = float(np.mean(attention))
    stats['std'] = float(np.std(attention))
    stats['max'] = float(np.max(attention))
    stats['min'] = float(np.min(attention))
    
    # Diagonal statistics (self-attention strength)
    diagonal = np.diag(attention)
    stats['diagonal_mean'] = float(np.mean(diagonal))
    stats['diagonal_std'] = float(np.std(diagonal))
    
    # Entropy of each row's distribution
    # H = -sum(p * log(p))
    eps = 1e-10
    entropy_per_row = -np.sum(attention * np.log(attention + eps), axis=1)
    stats['mean_entropy'] = float(np.mean(entropy_per_row))
    stats['max_entropy'] = float(np.log(attention.shape[1]))  # Uniform distribution
    stats['normalized_entropy'] = stats['mean_entropy'] / stats['max_entropy']
    
    # Sparsity via Gini coefficient
    # Gini = 1 means all attention on one token
    # Gini = 0 means uniform attention
    def gini(x):
        sorted_x = np.sort(x.flatten())
        n = len(sorted_x)
        cumsum = np.cumsum(sorted_x)
        return (2 * np.sum((np.arange(1, n+1) * sorted_x)) / (n * np.sum(sorted_x))) - (n + 1) / n
    
    stats['sparsity_gini'] = float(gini(attention))
    
    # Effective number of attended tokens (perplexity)
    # exp(entropy) gives effective number of equally weighted tokens
    stats['effective_tokens'] = float(np.mean(np.exp(entropy_per_row)))
    
    return stats
```

## Understanding Attention Patterns

### Common Pattern Types

Transformers learn characteristic attention patterns that serve different functions:

| Pattern | Description | Visual Signature | Function |
|---------|-------------|------------------|----------|
| Diagonal | Strong self-attention | High values on diagonal | Token-level processing |
| Local | Attention to neighbors | Band around diagonal | Local context modeling |
| Vertical Stripe | All attend to one token | Single bright column | Information aggregation |
| BOS/CLS | Focus on first token | Bright first column | Global representation |
| Block | Segment-based attention | Rectangular blocks | Sentence-level grouping |
| Uniform | Equal attention | Flat heatmap | Distributed context |

```python
def identify_attention_pattern(attention: torch.Tensor) -> str:
    """
    Automatically identify the predominant attention pattern.
    
    Parameters
    ----------
    attention : torch.Tensor
        Attention weight matrix, shape: (seq_len, seq_len)
    
    Returns
    -------
    str
        Name of the detected pattern
    """
    if isinstance(attention, torch.Tensor):
        attention = attention.cpu().numpy()
    
    seq_len = attention.shape[0]
    
    # Compute diagnostic metrics
    diagonal_mean = np.mean(np.diag(attention))
    first_col_mean = np.mean(attention[:, 0])
    
    # Off-diagonal mean
    mask = ~np.eye(seq_len, dtype=bool)
    off_diagonal_mean = np.mean(attention[mask])
    
    # Local bandwidth (within 2 positions)
    local_mask = np.abs(np.arange(seq_len)[:, None] - np.arange(seq_len)) <= 2
    local_attention = np.sum(attention[local_mask]) / np.sum(attention)
    
    # Column concentration
    col_sums = np.sum(attention, axis=0)
    max_col_ratio = np.max(col_sums) / seq_len
    
    # Pattern detection logic
    if diagonal_mean > 0.5 and diagonal_mean > 3 * off_diagonal_mean:
        return "diagonal"
    elif first_col_mean > 0.4:
        return "beginning_of_sequence"
    elif max_col_ratio > 0.6:
        return "broadcast"
    elif local_attention > 0.8:
        return "local"
    elif np.std(attention) < 0.1:
        return "uniform"
    else:
        return "mixed"
```

### Creating Synthetic Patterns for Testing

```python
def create_synthetic_attention(
    seq_len: int,
    pattern: str = 'diagonal',
    noise_level: float = 0.05
) -> torch.Tensor:
    """
    Create synthetic attention patterns for testing and demonstration.
    
    Parameters
    ----------
    seq_len : int
        Sequence length
    pattern : str
        Pattern type: 'diagonal', 'local', 'broadcast', 'bos', 'uniform', 'block'
    noise_level : float
        Amount of random noise to add
    
    Returns
    -------
    torch.Tensor
        Synthetic attention matrix, shape: (seq_len, seq_len)
    """
    if pattern == 'diagonal':
        # Strong self-attention
        attention = torch.eye(seq_len) * 0.7
        attention += torch.rand(seq_len, seq_len) * noise_level
        
    elif pattern == 'local':
        # Attention to neighboring tokens
        attention = torch.zeros(seq_len, seq_len)
        for i in range(seq_len):
            for j in range(seq_len):
                distance = abs(i - j)
                if distance == 0:
                    attention[i, j] = 0.4
                elif distance == 1:
                    attention[i, j] = 0.25
                elif distance == 2:
                    attention[i, j] = 0.1
                else:
                    attention[i, j] = 0.01
        
    elif pattern == 'broadcast':
        # All tokens attend to a specific position
        attention = torch.zeros(seq_len, seq_len)
        broadcast_pos = seq_len // 3
        attention[:, broadcast_pos] = 0.7
        attention += torch.eye(seq_len) * 0.15
        attention += torch.rand(seq_len, seq_len) * noise_level
        
    elif pattern == 'bos':
        # Beginning-of-sequence attention
        attention = torch.zeros(seq_len, seq_len)
        for i in range(seq_len):
            attention[i, 0] = 0.6 - i * 0.03  # Decreasing attention to first
            attention[i, i] = 0.2  # Some self-attention
        attention += torch.rand(seq_len, seq_len) * noise_level
        
    elif pattern == 'uniform':
        # Equal attention to all tokens
        attention = torch.ones(seq_len, seq_len)
        attention += torch.rand(seq_len, seq_len) * noise_level
        
    elif pattern == 'block':
        # Segment-based attention
        attention = torch.zeros(seq_len, seq_len)
        mid = seq_len // 2
        attention[:mid, :mid] = 0.7 / mid
        attention[mid:, mid:] = 0.7 / (seq_len - mid)
        attention += torch.eye(seq_len) * 0.15
        
    else:
        raise ValueError(f"Unknown pattern: {pattern}")
    
    # Normalize rows to sum to 1
    attention = attention / attention.sum(dim=1, keepdim=True)
    
    return attention
```

## Best Practices

### Visualization Guidelines

1. **Use perceptually uniform colormaps**: `viridis`, `plasma`, or `inferno` are colorblind-friendly and perceptually uniform.

2. **Include clear axis labels**: Always label query (y-axis) and key (x-axis) positions with actual tokens.

3. **Set appropriate value ranges**: Use `vmin=0, vmax=1` for normalized attention, or adaptive scaling for comparisons.

4. **Consider sequence length**: For long sequences, use sampling or aggregation rather than displaying all positions.

5. **Provide context**: Include model name, layer number, head index, and input text in titles.

### Interpretation Caveats

1. **Attention ≠ Explanation**: High attention weights do not necessarily indicate causal importance for the output. Use gradient-based methods for attribution.

2. **Single layer limitation**: Examining one layer's attention misses the cumulative effect across the network. Use attention rollout or flow methods.

3. **Head specialization**: Different heads learn different patterns. Analyze multiple heads to get a complete picture.

4. **Position effects**: Some attention patterns reflect positional biases rather than semantic relationships.

## Complete Example

```python
def example_attention_fundamentals():
    """
    Complete example demonstrating attention extraction and visualization.
    """
    # Define sample text
    text = "The quick brown fox jumps over the lazy dog"
    tokens = text.split()
    seq_len = len(tokens)
    
    # Create different attention patterns
    patterns = ['diagonal', 'local', 'broadcast', 'bos']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for idx, pattern in enumerate(patterns):
        # Create synthetic attention
        attention = create_synthetic_attention(seq_len, pattern=pattern)
        
        # Compute statistics
        stats = compute_attention_statistics(attention)
        
        # Plot
        ax = axes[idx]
        im = ax.imshow(attention.numpy(), cmap='viridis', vmin=0, vmax=1)
        
        ax.set_xticks(range(seq_len))
        ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=9)
        ax.set_yticks(range(seq_len))
        ax.set_yticklabels(tokens, fontsize=9)
        
        ax.set_title(
            f'{pattern.capitalize()} Pattern\n'
            f'Entropy: {stats["normalized_entropy"]:.2f}, '
            f'Diag: {stats["diagonal_mean"]:.2f}',
            fontsize=11, fontweight='bold'
        )
        
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    plt.suptitle('Common Attention Patterns', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Print detected patterns
    print("\nPattern Detection Results:")
    print("-" * 40)
    for pattern in patterns:
        attention = create_synthetic_attention(seq_len, pattern=pattern)
        detected = identify_attention_pattern(attention)
        print(f"  Created: {pattern:12s} -> Detected: {detected}")

if __name__ == "__main__":
    example_attention_fundamentals()
```

## Summary

This section established the mathematical foundations for attention visualization:

- **Attention weights** form probability distributions over keys for each query position
- **Heatmaps** are the primary visualization tool, with rows representing queries and columns representing keys
- **Statistical metrics** (entropy, sparsity, diagonal strength) quantify attention characteristics
- **Pattern recognition** helps identify common attention behaviors (diagonal, local, broadcast, etc.)
- **Best practices** ensure accurate interpretation while avoiding common pitfalls

The following sections build on these fundamentals to explore multi-head attention analysis, layer-wise patterns, and advanced techniques like attention rollout and flow.

## References

1. Vaswani, A., et al. (2017). "Attention Is All You Need." NeurIPS.
2. Clark, K., et al. (2019). "What Does BERT Look At? An Analysis of BERT's Attention." BlackboxNLP.
3. Kovaleva, O., et al. (2019). "Revealing the Dark Secrets of BERT." EMNLP.
