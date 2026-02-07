# Attention Visualization

## Introduction

Visualizing attention weights is one of the most direct ways to interpret what Transformer models learn. By examining the attention matrices across layers and heads, we can gain insight into how models process information, debug architectural issues, and verify that attention patterns match linguistic expectations.

This section provides practical tools for visualizing attention in self-attention, cross-attention, and multi-head settings, along with guidance on interpreting the resulting patterns.

## Attention Matrix Basics

An attention matrix $\mathbf{A} \in \mathbb{R}^{n_q \times n_k}$ has entries $a_{ij}$ representing how much query position $i$ attends to key position $j$. Each row sums to 1 (a valid probability distribution).

For self-attention, the matrix is square ($n_q = n_k = n$). For cross-attention, it is rectangular ($n_q \neq n_k$ in general).

```
         Keys
         The  cat  sat
        ┌────┬────┬────┐
    The │ .6 │ .2 │ .2 │  ← Row sums to 1.0
        ├────┼────┼────┤
Q   cat │ .1 │ .7 │ .2 │  ← Row sums to 1.0
        ├────┼────┼────┤
    sat │ .2 │ .5 │ .3 │  ← Row sums to 1.0
        └────┴────┴────┘
```

## Visualizing Self-Attention

### Single-Head Heatmap

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math


def visualize_self_attention(
    weights: torch.Tensor,
    tokens: list,
    head_idx: int = 0,
    title: str = "Self-Attention"
):
    """
    Visualize self-attention weights as a heatmap.
    
    Args:
        weights: (batch, heads, seq, seq) or (seq, seq)
        tokens: List of token strings
        head_idx: Which head to visualize
        title: Plot title
    """
    if weights.dim() == 4:
        attn = weights[0, head_idx].detach().cpu().numpy()
    elif weights.dim() == 2:
        attn = weights.detach().cpu().numpy()
    else:
        raise ValueError(f"Expected 2D or 4D tensor, got {weights.dim()}D")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(attn, cmap='Blues', vmin=0, vmax=1, aspect='auto')
    
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha='right')
    ax.set_yticklabels(tokens)
    ax.set_xlabel('Key (attending to)')
    ax.set_ylabel('Query (attending from)')
    ax.set_title(title)
    
    # Annotate cells with weight values
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            color = 'white' if attn[i, j] > 0.5 else 'black'
            ax.text(j, i, f'{attn[i, j]:.2f}', ha='center', va='center',
                    color=color, fontsize=8)
    
    plt.colorbar(im, ax=ax, label='Attention Weight')
    plt.tight_layout()
    return fig
```

### Multi-Head Comparison

```python
def visualize_multi_head(
    weights: torch.Tensor,
    tokens: list,
    n_cols: int = 4,
    figsize_per_head: tuple = (3, 3)
):
    """
    Visualize all attention heads side by side.
    
    Args:
        weights: (batch, heads, seq, seq)
        tokens: List of token strings
        n_cols: Number of columns in the grid
    """
    n_heads = weights.shape[1]
    n_rows = (n_heads + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_head[0] * n_cols, figsize_per_head[1] * n_rows)
    )
    axes = axes.flatten() if n_heads > 1 else [axes]
    
    for h in range(n_heads):
        attn = weights[0, h].detach().cpu().numpy()
        axes[h].imshow(attn, cmap='Blues', vmin=0, vmax=1, aspect='auto')
        axes[h].set_title(f'Head {h}', fontsize=10)
        axes[h].set_xticks(range(len(tokens)))
        axes[h].set_yticks(range(len(tokens)))
        axes[h].set_xticklabels(tokens, rotation=45, ha='right', fontsize=7)
        axes[h].set_yticklabels(tokens, fontsize=7)
    
    # Hide unused axes
    for h in range(n_heads, len(axes)):
        axes[h].set_visible(False)
    
    plt.suptitle('Multi-Head Attention Patterns', fontsize=14)
    plt.tight_layout()
    return fig
```

## Visualizing Cross-Attention (Alignment)

```python
def visualize_cross_attention(
    weights: torch.Tensor,
    source_tokens: list,
    target_tokens: list,
    head_idx: int = 0,
    title: str = "Cross-Attention Alignment"
):
    """
    Visualize cross-attention as source-target alignment.
    
    Args:
        weights: (batch, heads, n_target, n_source) or (n_target, n_source)
        source_tokens: List of source (encoder) token strings
        target_tokens: List of target (decoder) token strings
        head_idx: Which head to visualize
    """
    if weights.dim() == 4:
        attn = weights[0, head_idx].detach().cpu().numpy()
    elif weights.dim() == 2:
        attn = weights.detach().cpu().numpy()
    else:
        raise ValueError(f"Expected 2D or 4D tensor, got {weights.dim()}D")
    
    fig, ax = plt.subplots(figsize=(max(8, len(source_tokens)), max(6, len(target_tokens) * 0.8)))
    im = ax.imshow(attn, cmap='Blues', aspect='auto', vmin=0, vmax=1)
    
    ax.set_xticks(range(len(source_tokens)))
    ax.set_yticks(range(len(target_tokens)))
    ax.set_xticklabels(source_tokens, rotation=45, ha='right')
    ax.set_yticklabels(target_tokens)
    ax.set_xlabel('Source (Encoder)')
    ax.set_ylabel('Target (Decoder)')
    ax.set_title(title)
    
    plt.colorbar(im, ax=ax, label='Attention Weight')
    plt.tight_layout()
    return fig
```

## Visualizing Attention as Dictionary Lookup

```python
def visualize_dictionary_lookup():
    """Visualize attention as soft dictionary lookup with three panels."""
    torch.manual_seed(42)
    
    num_queries = 3
    num_entries = 5
    dim = 4
    
    queries = torch.randn(num_queries, dim)
    keys = torch.randn(num_entries, dim)
    values = torch.arange(num_entries).float().view(-1, 1)
    
    # Compute attention
    scores = torch.matmul(queries, keys.T) / (dim ** 0.5)
    weights = F.softmax(scores, dim=-1)
    retrieved = torch.matmul(weights, values)
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # Panel 1: Raw similarity scores
    im1 = axes[0].imshow(scores.numpy(), cmap='RdBu', aspect='auto')
    axes[0].set_title('Query-Key Similarity\n(before softmax)')
    axes[0].set_xlabel('Dictionary Entry (Key)')
    axes[0].set_ylabel('Query')
    axes[0].set_xticks(range(num_entries))
    axes[0].set_yticks(range(num_queries))
    plt.colorbar(im1, ax=axes[0])
    
    # Panel 2: Attention weights
    im2 = axes[1].imshow(weights.numpy(), cmap='Blues', aspect='auto', vmin=0, vmax=1)
    axes[1].set_title('Retrieval Weights\n(after softmax)')
    axes[1].set_xlabel('Dictionary Entry')
    axes[1].set_ylabel('Query')
    axes[1].set_xticks(range(num_entries))
    axes[1].set_yticks(range(num_queries))
    plt.colorbar(im2, ax=axes[1])
    
    # Panel 3: Retrieved values
    axes[2].bar(range(num_queries), retrieved.squeeze().numpy(), color='steelblue')
    axes[2].set_title('Retrieved Values\n(weighted sum)')
    axes[2].set_xlabel('Query')
    axes[2].set_ylabel('Retrieved Value')
    axes[2].set_xticks(range(num_queries))
    axes[2].axhline(y=values.mean().item(), color='red', linestyle='--',
                     label=f'Mean value ({values.mean().item():.1f})')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('dictionary_lookup_visualization.png', dpi=150, bbox_inches='tight')
    return fig
```

## Visualizing the Effect of Scaling

```python
def visualize_scaling_effect():
    """Compare attention distributions with and without scaling."""
    torch.manual_seed(42)
    
    d_k = 64
    seq_len = 10
    
    Q = torch.randn(1, seq_len, d_k)
    K = torch.randn(1, seq_len, d_k)
    
    unscaled_scores = torch.matmul(Q, K.transpose(-2, -1)).squeeze(0)
    scaled_scores = unscaled_scores / math.sqrt(d_k)
    
    unscaled_weights = F.softmax(unscaled_scores, dim=-1)
    scaled_weights = F.softmax(scaled_scores, dim=-1)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Row 1: Unscaled
    im1 = axes[0, 0].imshow(unscaled_scores.numpy(), cmap='RdBu', vmin=-20, vmax=20)
    axes[0, 0].set_title(f'Unscaled Scores\n(std={unscaled_scores.std():.2f})')
    plt.colorbar(im1, ax=axes[0, 0])
    
    im2 = axes[0, 1].imshow(unscaled_weights.numpy(), cmap='Blues', vmin=0, vmax=1)
    axes[0, 1].set_title('Unscaled Attention Weights')
    plt.colorbar(im2, ax=axes[0, 1])
    
    unscaled_entropy = -(unscaled_weights * (unscaled_weights + 1e-10).log()).sum(dim=-1)
    axes[0, 2].bar(range(seq_len), unscaled_entropy.numpy())
    axes[0, 2].set_title(f'Entropy (mean={unscaled_entropy.mean():.2f})')
    axes[0, 2].axhline(y=np.log(seq_len), color='r', linestyle='--', label='Max')
    axes[0, 2].legend()
    
    # Row 2: Scaled
    im3 = axes[1, 0].imshow(scaled_scores.numpy(), cmap='RdBu', vmin=-3, vmax=3)
    axes[1, 0].set_title(f'Scaled Scores\n(std={scaled_scores.std():.2f})')
    plt.colorbar(im3, ax=axes[1, 0])
    
    im4 = axes[1, 1].imshow(scaled_weights.numpy(), cmap='Blues', vmin=0, vmax=1)
    axes[1, 1].set_title('Scaled Attention Weights')
    plt.colorbar(im4, ax=axes[1, 1])
    
    scaled_entropy = -(scaled_weights * (scaled_weights + 1e-10).log()).sum(dim=-1)
    axes[1, 2].bar(range(seq_len), scaled_entropy.numpy())
    axes[1, 2].set_title(f'Entropy (mean={scaled_entropy.mean():.2f})')
    axes[1, 2].axhline(y=np.log(seq_len), color='r', linestyle='--', label='Max')
    axes[1, 2].legend()
    
    for ax in axes.flat:
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
    
    plt.suptitle('Effect of Scaling on Attention Distributions', fontsize=14)
    plt.tight_layout()
    plt.savefig('scaling_effect.png', dpi=150, bbox_inches='tight')
    return fig
```

## Temperature Visualization

```python
def visualize_temperature_effect():
    """Show how temperature controls soft vs hard attention."""
    
    keys = torch.tensor([[1., 0., 0.],
                         [0., 1., 0.],
                         [0., 0., 1.]])
    values = torch.tensor([[1.], [2.], [3.]])
    query = torch.tensor([[0.3, 0.7, 0.0]])
    
    temperatures = [10.0, 1.0, 0.1, 0.01]
    
    fig, axes = plt.subplots(1, len(temperatures), figsize=(16, 3))
    
    for idx, temp in enumerate(temperatures):
        scores = torch.matmul(query, keys.T) / temp
        weights = F.softmax(scores, dim=-1).squeeze().numpy()
        
        axes[idx].bar(['Key 0', 'Key 1', 'Key 2'], weights, color='steelblue')
        axes[idx].set_ylim(0, 1.05)
        axes[idx].set_title(f'T = {temp}')
        axes[idx].set_ylabel('Weight')
        
        retrieved = (weights * values.squeeze().numpy()).sum()
        axes[idx].text(0.5, 0.95, f'Retrieved: {retrieved:.3f}',
                       transform=axes[idx].transAxes, ha='center', va='top',
                       fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat'))
    
    plt.suptitle('Temperature Effect: Soft → Hard Lookup', fontsize=14)
    plt.tight_layout()
    plt.savefig('temperature_effect.png', dpi=150, bbox_inches='tight')
    return fig
```

## Layer-wise Attention Evolution

```python
def visualize_layer_attention(
    all_layer_weights: list,
    tokens: list,
    head_idx: int = 0
):
    """
    Visualize how attention evolves across layers.
    
    Args:
        all_layer_weights: List of (batch, heads, seq, seq) tensors, one per layer
        tokens: List of token strings
        head_idx: Which head to track across layers
    """
    n_layers = len(all_layer_weights)
    fig, axes = plt.subplots(1, n_layers, figsize=(4 * n_layers, 4))
    
    if n_layers == 1:
        axes = [axes]
    
    for layer_idx, weights in enumerate(all_layer_weights):
        attn = weights[0, head_idx].detach().cpu().numpy()
        axes[layer_idx].imshow(attn, cmap='Blues', vmin=0, vmax=1, aspect='auto')
        axes[layer_idx].set_title(f'Layer {layer_idx}', fontsize=10)
        axes[layer_idx].set_xticks(range(len(tokens)))
        axes[layer_idx].set_yticks(range(len(tokens)))
        axes[layer_idx].set_xticklabels(tokens, rotation=45, ha='right', fontsize=7)
        axes[layer_idx].set_yticklabels(tokens, fontsize=7)
    
    plt.suptitle(f'Attention Evolution Across Layers (Head {head_idx})', fontsize=14)
    plt.tight_layout()
    return fig
```

## Attention Entropy Analysis

Entropy measures how distributed or concentrated attention is:

$$H(\alpha_i) = -\sum_j \alpha_{ij} \log \alpha_{ij}$$

Low entropy means concentrated attention (few positions dominate). High entropy means distributed attention (many positions contribute).

```python
def attention_entropy_analysis(weights: torch.Tensor):
    """
    Compute and visualize attention entropy across heads and layers.
    
    Args:
        weights: (batch, heads, seq_q, seq_k)
    """
    batch, n_heads, seq_q, seq_k = weights.shape
    max_entropy = np.log(seq_k)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Per-head mean entropy
    entropies = -(weights * (weights + 1e-10).log()).sum(-1)  # (batch, heads, seq_q)
    mean_entropy = entropies.mean(dim=(0, 2))  # (heads,)
    
    axes[0].bar(range(n_heads), mean_entropy.cpu().numpy(), color='steelblue')
    axes[0].axhline(y=max_entropy, color='r', linestyle='--', label=f'Max entropy ({max_entropy:.2f})')
    axes[0].set_xlabel('Head')
    axes[0].set_ylabel('Mean Entropy')
    axes[0].set_title('Attention Entropy per Head')
    axes[0].legend()
    
    # Entropy distribution across positions (for head 0)
    head_entropy = entropies[0, 0].cpu().numpy()  # (seq_q,)
    axes[1].bar(range(seq_q), head_entropy, color='steelblue')
    axes[1].axhline(y=max_entropy, color='r', linestyle='--', label='Max entropy')
    axes[1].set_xlabel('Query Position')
    axes[1].set_ylabel('Entropy')
    axes[1].set_title('Entropy per Position (Head 0)')
    axes[1].legend()
    
    plt.tight_layout()
    return fig
```

## Causal Attention Visualization

```python
def visualize_causal_pattern(seq_len=8, n_heads=4):
    """Demonstrate causal masking pattern with random attention weights."""
    torch.manual_seed(42)
    
    d_k = 32
    Q = torch.randn(1, n_heads, seq_len, d_k)
    K = torch.randn(1, n_heads, seq_len, d_k)
    
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    # Apply causal mask
    causal_mask = torch.tril(torch.ones(seq_len, seq_len))
    scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0) == 0, float('-inf'))
    weights = F.softmax(scores, dim=-1)
    
    fig, axes = plt.subplots(1, n_heads, figsize=(4 * n_heads, 4))
    tokens = [f'pos_{i}' for i in range(seq_len)]
    
    for h in range(n_heads):
        attn = weights[0, h].detach().cpu().numpy()
        axes[h].imshow(attn, cmap='Blues', vmin=0, vmax=0.5, aspect='auto')
        axes[h].set_title(f'Head {h}')
        axes[h].set_xticks(range(seq_len))
        axes[h].set_yticks(range(seq_len))
        axes[h].set_xticklabels(range(seq_len), fontsize=7)
        axes[h].set_yticklabels(range(seq_len), fontsize=7)
    
    plt.suptitle('Causal Attention (lower triangular)', fontsize=14)
    plt.tight_layout()
    return fig
```

## Complete Visualization Pipeline

```python
class AttentionVisualizer:
    """
    Unified attention visualization toolkit.
    
    Collects attention weights during forward passes and provides
    methods for comprehensive visualization.
    """
    
    def __init__(self):
        self.layer_weights = {}
        self.hooks = []
    
    def register_hooks(self, model: nn.Module, layer_names: list = None):
        """
        Register forward hooks to capture attention weights.
        
        Args:
            model: Transformer model
            layer_names: Specific layer names to hook (None = all attention layers)
        """
        for name, module in model.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                if layer_names is None or name in layer_names:
                    hook = module.register_forward_hook(
                        self._make_hook(name)
                    )
                    self.hooks.append(hook)
    
    def _make_hook(self, name):
        def hook_fn(module, input, output):
            # nn.MultiheadAttention returns (output, weights)
            if isinstance(output, tuple) and len(output) == 2:
                self.layer_weights[name] = output[1].detach()
        return hook_fn
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def plot_all_layers(self, tokens: list, head_idx: int = 0):
        """Visualize attention across all captured layers."""
        n_layers = len(self.layer_weights)
        if n_layers == 0:
            print("No attention weights captured. Run a forward pass first.")
            return
        
        fig, axes = plt.subplots(1, n_layers, figsize=(4 * n_layers, 4))
        if n_layers == 1:
            axes = [axes]
        
        for idx, (name, weights) in enumerate(self.layer_weights.items()):
            attn = weights[0, head_idx].cpu().numpy() if weights.dim() == 4 else weights[0].cpu().numpy()
            axes[idx].imshow(attn, cmap='Blues', vmin=0, vmax=1, aspect='auto')
            axes[idx].set_title(name, fontsize=9)
            axes[idx].set_xticks(range(len(tokens)))
            axes[idx].set_yticks(range(len(tokens)))
            axes[idx].set_xticklabels(tokens, rotation=45, ha='right', fontsize=6)
            axes[idx].set_yticklabels(tokens, fontsize=6)
        
        plt.tight_layout()
        return fig
    
    def summary_statistics(self) -> dict:
        """Compute summary statistics for all captured attention weights."""
        stats = {}
        for name, weights in self.layer_weights.items():
            entropy = -(weights * (weights + 1e-10).log()).sum(-1).mean().item()
            max_weight = weights.max().item()
            stats[name] = {
                'mean_entropy': entropy,
                'max_weight': max_weight,
                'shape': list(weights.shape),
            }
        return stats
```

## Interpretation Guidelines

### What to Look For

**Healthy attention patterns:**

- Varied patterns across heads (specialisation)
- Moderate entropy (neither uniform nor one-hot)
- Interpretable structure (diagonal, local, syntactic)

**Warning signs:**

- All heads look identical (redundancy, wasted capacity)
- Extreme entropy (either all uniform or all one-hot)
- No variation across layers (model may not be learning depth)

### Caveats

**Attention is not explanation.** High attention weight does not necessarily mean the attended token is "important" for the output. The value vectors, residual connections, and downstream layers all transform the attended information.

**Aggregation across heads.** Individual head patterns are meaningful, but the output projection $\mathbf{W}_O$ mixes heads, so the overall effect may differ from any single head's pattern.

**Training dynamics.** Attention patterns change during training and may not be stable until convergence.

## Summary

| Visualization | Purpose | Key Insight |
|---------------|---------|-------------|
| **Heatmap** | Show attention distribution | Which positions attend to which |
| **Multi-head grid** | Compare head specialisation | Different heads learn different patterns |
| **Cross-attention alignment** | Source-target correspondence | Translation/alignment quality |
| **Entropy plot** | Attention concentration | Distributed vs. focused attention |
| **Layer evolution** | Depth behaviour | Pattern progression from local to global |
| **Scaling comparison** | Effect of $\sqrt{d_k}$ | Saturation vs. smooth distributions |
| **Temperature sweep** | Soft-to-hard spectrum | Controlling retrieval sharpness |

## References

1. Clark, K., et al. (2019). "What Does BERT Look At? An Analysis of BERT's Attention." *BlackboxNLP*.
2. Vig, J. (2019). "A Multiscale Visualization of Attention in the Transformer Model." *ACL Demo*.
3. Jain, S., & Wallace, B. C. (2019). "Attention is not Explanation." *NAACL*.
4. Wiegreffe, S., & Pinter, Y. (2019). "Attention is not not Explanation." *EMNLP*.
5. Voita, E., et al. (2019). "Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting." *ACL*.
