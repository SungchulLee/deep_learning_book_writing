# Attention Pattern Analysis

## Introduction

Beyond basic attention weight visualization, understanding **how attention patterns organize across heads, layers, and modalities** is essential for interpreting transformer models. This section consolidates multi-head analysis, layer-wise progression, cross-attention interpretation, visualization tooling, and the critical distinction between attention and attribution.

## Multi-Head Attention Analysis

### Mathematical Background

Multi-head attention computes $H$ separate attention distributions in parallel:

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_H) W^O
$$

where each head computes:

$$
\text{head}_h = \text{Attention}(Q W_h^Q, K W_h^K, V W_h^V)
$$

Each head operates in a $d_k = d_{\text{model}} / H$ dimensional subspace, potentially specializing in different linguistic or structural patterns.

### Head Specialization Patterns

Research has identified several recurring specialization patterns:

| Pattern | Description | Typical Layer |
|---------|-------------|---------------|
| **Positional** | Attend to adjacent tokens | Early layers |
| **Syntactic** | Follow dependency structure | Middle layers |
| **Delimiter** | Attend to separators ([SEP], punctuation) | Various |
| **Rare token** | Focus on infrequent words | Middle layers |
| **Semantic** | Attend to related concepts | Later layers |
| **Broad/uniform** | Distribute attention evenly | Various |

### Multi-Head Visualization

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_all_heads(
    attention_weights: torch.Tensor,
    tokens: list,
    layer: int = 0,
    figsize: tuple = (20, 16)
) -> plt.Figure:
    """
    Visualize attention patterns for all heads in a layer.
    
    Args:
        attention_weights: [batch, num_heads, seq_len, seq_len]
        tokens: List of token strings
        layer: Layer index to visualize
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    attn = attention_weights[0].detach().cpu().numpy()
    num_heads = attn.shape[0]
    
    ncols = 4
    nrows = (num_heads + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()
    
    for head_idx in range(num_heads):
        ax = axes[head_idx]
        sns.heatmap(
            attn[head_idx],
            xticklabels=tokens,
            yticklabels=tokens,
            ax=ax,
            cmap='Blues',
            vmin=0, vmax=1,
            cbar=False
        )
        ax.set_title(f'Head {head_idx}', fontsize=10)
        ax.tick_params(labelsize=7)
    
    for idx in range(num_heads, len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle(f'Layer {layer} - All Attention Heads', fontsize=14)
    plt.tight_layout()
    return fig
```

### Head Diversity Analysis

Measuring how differently heads attend helps assess redundancy and specialization:

```python
def compute_head_diversity(
    attention_weights: torch.Tensor
) -> dict:
    """
    Analyze diversity among attention heads.
    
    Args:
        attention_weights: [batch, num_heads, seq_len, seq_len]
        
    Returns:
        Dictionary with diversity metrics
    """
    attn = attention_weights[0].detach().cpu()
    num_heads = attn.shape[0]
    seq_len = attn.shape[1]
    
    # Entropy per head
    entropies = []
    for h in range(num_heads):
        head_attn = attn[h]
        eps = 1e-10
        entropy = -(head_attn * torch.log(head_attn + eps)).sum(dim=-1).mean()
        entropies.append(entropy.item())
    
    # Pairwise cosine similarity between heads
    flat_heads = attn.reshape(num_heads, -1)
    flat_normed = flat_heads / flat_heads.norm(dim=1, keepdim=True)
    similarity_matrix = (flat_normed @ flat_normed.T).numpy()
    
    # Average off-diagonal similarity
    mask = ~np.eye(num_heads, dtype=bool)
    avg_similarity = similarity_matrix[mask].mean()
    
    return {
        'entropies': entropies,
        'similarity_matrix': similarity_matrix,
        'avg_pairwise_similarity': avg_similarity,
        'diversity_score': 1.0 - avg_similarity
    }
```

### Head Importance and Pruning

Not all heads contribute equally. Importance can be estimated by measuring the effect of removing each head:

```python
def compute_head_importance(
    model: nn.Module,
    data_loader,
    num_layers: int,
    num_heads: int,
    device: torch.device
) -> np.ndarray:
    """
    Compute head importance via gradient-based scoring.
    
    Uses the method from Michel et al. (2019):
    Importance = E[|grad(L) * attention|]
    
    Returns:
        importance_matrix: [num_layers, num_heads]
    """
    importance = torch.zeros(num_layers, num_heads).to(device)
    
    model.eval()
    for batch in data_loader:
        inputs = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(inputs, labels=labels, output_attentions=True)
        loss = outputs.loss
        attentions = outputs.attentions
        
        for layer_idx, attn in enumerate(attentions):
            attn.retain_grad()
            
        loss.backward()
        
        for layer_idx, attn in enumerate(attentions):
            if attn.grad is not None:
                head_importance = (attn * attn.grad).abs().sum(dim=(0, 2, 3))
                importance[layer_idx] += head_importance
    
    return importance.cpu().numpy()
```

## Layer-wise Attention Patterns

### Theoretical Background

Attention patterns evolve systematically as information flows through transformer layers:

- **Early layers** (1-3): Capture local, syntactic relationships—adjacent token attention, positional patterns
- **Middle layers** (4-8): Encode syntactic structure—dependency relations, coreference
- **Later layers** (9-12): Build abstract, semantic features—topic attention, long-range dependencies

This progression parallels the hierarchical feature learning observed in CNNs.

### Layer Comparison Implementation

```python
def compare_layers(
    attention_weights_by_layer: list,
    tokens: list,
    layers_to_compare: list = None
) -> plt.Figure:
    """
    Compare attention patterns across transformer layers.
    
    Args:
        attention_weights_by_layer: List of [batch, heads, seq, seq] per layer
        tokens: Token strings
        layers_to_compare: Which layers to show
    """
    if layers_to_compare is None:
        n_layers = len(attention_weights_by_layer)
        layers_to_compare = [0, n_layers // 3, 2 * n_layers // 3, n_layers - 1]
    
    fig, axes = plt.subplots(1, len(layers_to_compare), figsize=(5 * len(layers_to_compare), 5))
    
    for idx, layer in enumerate(layers_to_compare):
        attn = attention_weights_by_layer[layer][0].mean(dim=0).detach().cpu().numpy()
        
        sns.heatmap(
            attn,
            xticklabels=tokens,
            yticklabels=tokens,
            ax=axes[idx],
            cmap='Blues',
            vmin=0,
            cbar=idx == len(layers_to_compare) - 1
        )
        axes[idx].set_title(f'Layer {layer}')
    
    plt.tight_layout()
    return fig


def compute_locality_score(attention_weights: torch.Tensor, window: int = 3) -> float:
    """
    Measure how local vs global attention is.
    
    Locality = fraction of attention mass within ±window of diagonal.
    """
    attn = attention_weights[0].mean(dim=0).detach().cpu().numpy()
    seq_len = attn.shape[0]
    
    local_mass = 0.0
    total_mass = 0.0
    
    for i in range(seq_len):
        for j in range(seq_len):
            total_mass += attn[i, j]
            if abs(i - j) <= window:
                local_mass += attn[i, j]
    
    return local_mass / total_mass if total_mass > 0 else 0.0
```

### Key Findings

Research across multiple transformer architectures reveals consistent patterns:

1. **Locality decreases with depth**: Early layers attend locally (~80% within window of 3), later layers attend globally (~40%)
2. **Head pruning tolerance varies by layer**: Later layers are more robust to head removal
3. **Residual connections matter**: Information bypasses attention through skip connections, meaning attention alone doesn't capture full information flow

## Cross-Attention Interpretation

Cross-attention connects two different sequences, enabling information flow between encoder and decoder. Understanding cross-attention is essential for interpreting translation, summarization, and question-answering systems.

### Mathematical Foundation

Given encoder representations $K^e, V^e$ and decoder query $Q^d$:

$$
\text{CrossAttn}(Q^d, K^e, V^e) = \text{softmax}\left(\frac{Q^d (K^e)^\top}{\sqrt{d_k}}\right) V^e
$$

The attention weights form an alignment matrix $A \in \mathbb{R}^{T_d \times T_e}$, where $A_{ij}$ indicates how much decoder position $i$ attends to encoder position $j$.

### Implementation

```python
def visualize_cross_attention(
    cross_attention: torch.Tensor,
    source_tokens: list,
    target_tokens: list,
    head: int = None
) -> plt.Figure:
    """
    Visualize encoder-decoder cross-attention.
    
    Args:
        cross_attention: [batch, heads, target_len, source_len]
        source_tokens: Encoder input tokens
        target_tokens: Decoder output tokens
        head: Specific head (None = average all heads)
    """
    if head is not None:
        attn = cross_attention[0, head].detach().cpu().numpy()
        title = f'Cross-Attention (Head {head})'
    else:
        attn = cross_attention[0].mean(dim=0).detach().cpu().numpy()
        title = 'Cross-Attention (Averaged)'
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(
        attn,
        xticklabels=source_tokens,
        yticklabels=target_tokens,
        ax=ax,
        cmap='Blues',
        vmin=0, vmax=1
    )
    ax.set_xlabel('Source (Encoder)')
    ax.set_ylabel('Target (Decoder)')
    ax.set_title(title)
    
    plt.tight_layout()
    return fig


def analyze_cross_attention_alignment(
    cross_attention: torch.Tensor,
    source_tokens: list,
    target_tokens: list
) -> dict:
    """
    Analyze alignment quality from cross-attention.
    
    Returns metrics on alignment sharpness and coverage.
    """
    attn = cross_attention[0].mean(dim=0).detach().cpu()
    
    # Alignment entropy (lower = sharper alignment)
    eps = 1e-10
    entropy = -(attn * torch.log(attn + eps)).sum(dim=-1).mean().item()
    
    # Coverage: fraction of source tokens attended to above threshold
    max_attn_per_source = attn.max(dim=0)[0]
    coverage = (max_attn_per_source > 0.1).float().mean().item()
    
    # Monotonicity: how monotonic is the alignment
    argmax_positions = attn.argmax(dim=-1).float()
    diffs = argmax_positions[1:] - argmax_positions[:-1]
    monotonicity = (diffs >= 0).float().mean().item()
    
    return {
        'alignment_entropy': entropy,
        'source_coverage': coverage,
        'monotonicity': monotonicity
    }
```

## Visualization Tools: BertViz and Custom Solutions

### BertViz Overview

BertViz (Vig, 2019) provides three visualization modes:

| Mode | Shows | Use Case |
|------|-------|----------|
| **Attention Head View** | Single head attention pattern | Analyzing specific head behavior |
| **Model View** | All heads in all layers | Overview of attention distribution |
| **Neuron View** | Query-key decomposition | Understanding what drives attention |

### Using BertViz

```python
from bertviz import head_view, model_view
from transformers import AutoTokenizer, AutoModel

def interactive_attention_visualization(
    model_name: str,
    text: str,
    text_pair: str = None
):
    """
    Create interactive BertViz visualization.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_attentions=True)
    
    inputs = tokenizer(
        text, text_pair,
        return_tensors='pt',
        padding=True,
        truncation=True
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    attention = outputs.attentions
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    # Head view - detailed single-head analysis
    head_view(attention, tokens)
    
    # Model view - all heads, all layers
    model_view(attention, tokens)
```

### Building Custom Visualizations

```python
def create_attention_heatmap_grid(
    attention_by_layer: list,
    tokens: list,
    num_layers: int = 12,
    num_heads: int = 12,
    figsize: tuple = (24, 20)
) -> plt.Figure:
    """
    Create comprehensive layer × head attention grid.
    """
    fig, axes = plt.subplots(num_layers, num_heads, figsize=figsize)
    
    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            attn = attention_by_layer[layer_idx][0, head_idx].detach().cpu().numpy()
            ax = axes[layer_idx, head_idx]
            ax.imshow(attn, cmap='Blues', vmin=0, vmax=1, aspect='auto')
            ax.set_xticks([])
            ax.set_yticks([])
            
            if layer_idx == 0:
                ax.set_title(f'H{head_idx}', fontsize=8)
            if head_idx == 0:
                ax.set_ylabel(f'L{layer_idx}', fontsize=8)
    
    fig.suptitle('Attention Patterns: Layer (rows) × Head (columns)', fontsize=14)
    plt.tight_layout()
    return fig


def create_attention_flow_sankey(
    attention_weights: torch.Tensor,
    tokens: list,
    source_idx: int,
    top_k: int = 5
):
    """
    Create Sankey-style flow diagram showing where attention flows
    from a specific token.
    """
    attn = attention_weights[0].detach().cpu().numpy()
    num_heads = attn.shape[0]
    
    flows = []
    for head_idx in range(num_heads):
        head_attn = attn[head_idx, source_idx]
        top_targets = np.argsort(head_attn)[::-1][:top_k]
        
        for target_idx in top_targets:
            if head_attn[target_idx] > 0.05:
                flows.append({
                    'head': head_idx,
                    'source': tokens[source_idx],
                    'target': tokens[target_idx],
                    'weight': head_attn[target_idx]
                })
    
    return flows
```

## Attention vs Attribution: A Critical Distinction

### The Core Issue

A critical distinction in model interpretability is the difference between **attention weights** and **attribution scores**:

- **Attention** shows where a model "looks"—the distribution of weights over input tokens
- **Attribution** reveals what actually influences the output—the causal contribution of each input

These are **not the same thing**. Jain and Wallace (2019) demonstrated that attention weights often do not correlate with gradient-based attribution, and alternative attention distributions can produce identical predictions.

### Why Attention ≠ Explanation

Several factors weaken attention as an explanation method:

**Value transformation**: Attention weights determine which value vectors to combine, but the value vectors themselves undergo transformation. High attention to a token doesn't mean that token's information dominates the output:

$$
\text{output}_i = \sum_j \alpha_{ij} V_j W^V
$$

The contribution depends on both $\alpha_{ij}$ and the content of $V_j$.

**Residual connections**: Information bypasses attention through skip connections. The output representation includes both attended information and the original input, making attention weights an incomplete picture:

$$
\mathbf{h}_i^{(l+1)} = \mathbf{h}_i^{(l)} + \text{Attn}(\mathbf{h}^{(l)})
$$

**Multi-layer composition**: In multi-layer transformers, information from a given token may arrive at the output through many different attention paths. Single-layer attention cannot capture these indirect influences.

### Comparing Attention and Attribution

```python
def compare_attention_and_attribution(
    model,
    input_ids: torch.Tensor,
    target_class: int,
    layer: int = -1,
    device: torch.device = None
) -> dict:
    """
    Compare attention weights with gradient-based attribution.
    """
    model.eval()
    input_ids = input_ids.to(device)
    
    # Get attention weights
    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True)
        attention = outputs.attentions[layer][0].mean(dim=0)
    
    # Get gradient-based attribution
    embeddings = model.get_input_embeddings()(input_ids)
    embeddings.requires_grad_(True)
    
    outputs = model(inputs_embeds=embeddings)
    logits = outputs.logits
    logits[0, target_class].backward()
    
    gradient_attr = embeddings.grad.abs().sum(dim=-1)[0]
    
    # Attention from [CLS] token (common choice)
    attn_scores = attention[0].detach().cpu().numpy()
    grad_scores = gradient_attr.detach().cpu().numpy()
    
    # Normalize
    attn_scores = attn_scores / attn_scores.sum()
    grad_scores = grad_scores / grad_scores.sum()
    
    # Correlation
    from scipy.stats import spearmanr
    correlation, p_value = spearmanr(attn_scores, grad_scores)
    
    return {
        'attention_scores': attn_scores,
        'gradient_scores': grad_scores,
        'spearman_correlation': correlation,
        'p_value': p_value
    }
```

### When to Use Each

| Method | Best For | Limitations |
|--------|----------|------------|
| **Attention weights** | Exploring model behavior, understanding architecture, hypothesis generation | Not faithful attribution, affected by value content |
| **Attention rollout** | Tracking information flow across layers | Assumes attention ≈ information flow |
| **Attention flow** | More faithful multi-layer attribution | Computationally expensive |
| **Gradient attribution** | Faithful importance measurement | Misses attention-specific insights |
| **Combined** | Most comprehensive understanding | More complex to interpret |

## Complete Example

```python
def comprehensive_attention_analysis(
    model,
    tokenizer,
    text: str,
    device: torch.device
) -> dict:
    """
    Comprehensive attention analysis combining multiple techniques.
    """
    # Tokenize
    inputs = tokenizer(text, return_tensors='pt', padding=True)
    input_ids = inputs['input_ids'].to(device)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    # Forward pass with attention
    model.eval()
    with torch.no_grad():
        outputs = model(**{k: v.to(device) for k, v in inputs.items()},
                       output_attentions=True)
    
    attentions = outputs.attentions
    
    results = {
        'tokens': tokens,
        'num_layers': len(attentions),
        'num_heads': attentions[0].shape[1],
        'analyses': {}
    }
    
    # Per-layer analysis
    for layer_idx, attn in enumerate(attentions):
        diversity = compute_head_diversity(attn)
        locality = compute_locality_score(attn)
        
        results['analyses'][f'layer_{layer_idx}'] = {
            'diversity_score': diversity['diversity_score'],
            'head_entropies': diversity['entropies'],
            'locality_score': locality
        }
    
    # Cross-layer locality progression
    localities = [
        results['analyses'][f'layer_{i}']['locality_score']
        for i in range(len(attentions))
    ]
    results['locality_progression'] = localities
    
    return results
```

## Summary

Attention pattern analysis provides rich insights into transformer behavior, but requires careful interpretation:

1. **Multi-head diversity** reveals specialization—heads that are too similar may be prunable
2. **Layer-wise progression** shows hierarchical feature building from local to global patterns
3. **Cross-attention alignment** is informative for encoder-decoder models
4. **Attention ≠ attribution**: Always validate attention-based insights with gradient-based methods
5. **Interactive tools** like BertViz are valuable for exploratory analysis but should be supplemented with quantitative metrics

## References

1. Vig, J. (2019). "A Multiscale Visualization of Attention in the Transformer Model." *ACL Demo*.

2. Clark, K., et al. (2019). "What Does BERT Look At? An Analysis of BERT's Attention." *BlackboxNLP*.

3. Jain, S., & Wallace, B. C. (2019). "Attention is not Explanation." *NAACL*.

4. Wiegreffe, S., & Pinter, Y. (2019). "Attention is not not Explanation." *EMNLP*.

5. Michel, P., Levy, O., & Neubig, G. (2019). "Are Sixteen Heads Really Better than One?" *NeurIPS*.

6. Voita, E., et al. (2019). "Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting." *ACL*.

7. Abnar, S., & Zuidema, W. (2020). "Quantifying Attention Flow in Transformers." *ACL*.
