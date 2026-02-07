# Attention Flow

## Introduction

While attention weights show where a model looks, they don't necessarily indicate what information actually influences the output. **Attention flow** addresses this by combining attention weights with gradients, revealing which attention paths are most important for model predictions.

This technique provides more accurate attribution than raw attention weights, bridging the gap between attention visualization and true model interpretability.

## Theoretical Foundation

### The Problem with Raw Attention

Research has shown that attention weights can be misleading:

1. **Attention ≠ Explanation**: High attention doesn't guarantee importance for the prediction
2. **Value transformation**: Attention weights don't account for what information is actually passed through values
3. **Layer interactions**: Single-layer attention misses cumulative effects

### Attention Flow Formulation

Attention flow combines attention weights with gradient information:

$$\text{Flow}^{(l)} = A^{(l)} \odot |\nabla_{V^{(l)}} \mathcal{L}|$$

where:
- $A^{(l)}$ is the attention matrix at layer $l$
- $\nabla_{V^{(l)}} \mathcal{L}$ is the gradient of the loss with respect to values
- $\odot$ denotes element-wise multiplication

This formulation captures both where attention is directed AND which attention connections actually matter for the output.

### Gradient-Weighted Attention

For a more comprehensive flow analysis:

$$\text{Flow}_{ij} = \sum_k A_{ik} \cdot |g_{kj}|$$

where $g_{kj}$ is the gradient of output $k$ with respect to input $j$.

## Implementation

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple, Callable

class AttentionFlowAnalyzer:
    """
    Analyzer for computing attention flow using gradient information.
    
    Combines attention weights with gradients to determine which
    attention connections are most important for model predictions.
    """
    
    def __init__(self):
        """Initialize the attention flow analyzer."""
        pass
    
    def compute_attention_flow(
        self,
        attention: torch.Tensor,
        gradients: torch.Tensor,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Compute attention flow by combining attention and gradients.
        
        Parameters
        ----------
        attention : torch.Tensor
            Attention weights, shape: (seq_len, seq_len)
        gradients : torch.Tensor
            Gradients of output w.r.t. attention or values
        normalize : bool
            Whether to normalize the result
        
        Returns
        -------
        torch.Tensor
            Attention flow matrix
        """
        # Take absolute value of gradients (magnitude matters)
        grad_magnitude = torch.abs(gradients)
        
        # Element-wise product: attention * gradient magnitude
        flow = attention * grad_magnitude
        
        # Normalize rows to sum to 1
        if normalize:
            row_sums = flow.sum(dim=1, keepdim=True)
            flow = flow / (row_sums + 1e-10)
        
        return flow
    
    def compute_flow_from_model(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        target_idx: Optional[int] = None,
        layer_idx: int = -1,
        head_idx: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute attention flow from a model by extracting attention and gradients.
        
        Parameters
        ----------
        model : nn.Module
            Transformer model
        input_ids : torch.Tensor
            Input token IDs
        target_idx : int, optional
            Target position for gradient computation. If None, uses all positions.
        layer_idx : int
            Which layer to analyze (-1 for last)
        head_idx : int
            Which attention head to analyze
        
        Returns
        -------
        attention : torch.Tensor
            Raw attention weights
        gradients : torch.Tensor
            Attention gradients
        flow : torch.Tensor
            Computed attention flow
        """
        model.eval()
        
        # Enable gradient computation
        input_ids.requires_grad_(False)
        
        # Store attention for gradient computation
        attention_storage = {}
        
        def save_attention_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple) and len(output) > 1:
                    attn = output[1]
                else:
                    attn = output
                attn.retain_grad()
                attention_storage[name] = attn
            return hook
        
        # Register hooks
        hooks = []
        for name, module in model.named_modules():
            if 'attention' in name.lower() or 'attn' in name.lower():
                hooks.append(module.register_forward_hook(save_attention_hook(name)))
        
        # Forward pass
        outputs = model(input_ids, output_attentions=True)
        
        # Get attention from target layer
        attentions = outputs.attentions
        if layer_idx < 0:
            layer_idx = len(attentions) + layer_idx
        
        attention = attentions[layer_idx][0, head_idx]  # (seq_len, seq_len)
        attention.retain_grad()
        
        # Compute target for backward pass
        if target_idx is not None:
            # Gradient with respect to specific output position
            target = outputs.last_hidden_state[0, target_idx].sum()
        else:
            # Gradient with respect to all outputs
            target = outputs.last_hidden_state.sum()
        
        # Backward pass
        target.backward()
        
        # Get gradients
        gradients = attention.grad
        
        # Clean up hooks
        for hook in hooks:
            hook.remove()
        
        # Compute flow
        flow = self.compute_attention_flow(attention.detach(), gradients.detach())
        
        return attention.detach(), gradients.detach(), flow
    
    def visualize_flow(
        self,
        attention: torch.Tensor,
        flow: torch.Tensor,
        tokens: List[str],
        title: str = "Attention vs Flow",
        save_path: Optional[str] = None
    ) -> None:
        """
        Compare raw attention weights with attention flow.
        
        Parameters
        ----------
        attention : torch.Tensor
            Raw attention weights
        flow : torch.Tensor
            Computed attention flow
        tokens : list of str
            Token labels
        title : str
            Plot title
        save_path : str, optional
            Path to save figure
        """
        if isinstance(attention, torch.Tensor):
            attention = attention.cpu().numpy()
        if isinstance(flow, torch.Tensor):
            flow = flow.cpu().numpy()
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Plot attention
        sns.heatmap(
            attention,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap='viridis',
            square=True,
            cbar_kws={'label': 'Weight', 'shrink': 0.8},
            ax=axes[0],
            vmin=0,
            vmax=1
        )
        axes[0].set_title('Raw Attention Weights', fontsize=13, fontweight='bold')
        axes[0].set_xlabel('Key Tokens', fontsize=11)
        axes[0].set_ylabel('Query Tokens', fontsize=11)
        
        # Plot flow
        sns.heatmap(
            flow,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap='YlOrRd',
            square=True,
            cbar_kws={'label': 'Flow', 'shrink': 0.8},
            ax=axes[1],
            vmin=0
        )
        axes[1].set_title('Attention Flow (Gradient-Weighted)', fontsize=13, fontweight='bold')
        axes[1].set_xlabel('Key Tokens', fontsize=11)
        axes[1].set_ylabel('Query Tokens', fontsize=11)
        
        for ax in axes:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.suptitle(title, fontsize=15, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def visualize_difference(
        self,
        attention: torch.Tensor,
        flow: torch.Tensor,
        tokens: List[str]
    ) -> None:
        """
        Visualize the difference between attention and flow.
        
        Highlights where gradients modify the attention importance.
        
        Parameters
        ----------
        attention : torch.Tensor
            Raw attention weights
        flow : torch.Tensor
            Computed attention flow
        tokens : list of str
            Token labels
        """
        if isinstance(attention, torch.Tensor):
            attention = attention.cpu().numpy()
        if isinstance(flow, torch.Tensor):
            flow = flow.cpu().numpy()
        
        # Normalize for comparison
        attention_norm = attention / attention.max()
        flow_norm = flow / flow.max()
        
        # Difference: positive means flow > attention (more important than attention suggests)
        difference = flow_norm - attention_norm
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(
            difference,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap='RdBu_r',
            center=0,
            square=True,
            cbar_kws={'label': 'Flow - Attention'},
            ax=ax,
            vmin=-1,
            vmax=1
        )
        
        ax.set_xlabel('Key Tokens', fontsize=12, fontweight='bold')
        ax.set_ylabel('Query Tokens', fontsize=12, fontweight='bold')
        ax.set_title('Difference: Flow vs Attention\n(Red = More Important Than Attention Shows)',
                    fontsize=13, fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    
    def identify_critical_connections(
        self,
        flow: torch.Tensor,
        tokens: List[str],
        top_k: int = 5
    ) -> List[Tuple[str, str, float]]:
        """
        Identify the most critical attention connections based on flow.
        
        Parameters
        ----------
        flow : torch.Tensor
            Attention flow matrix
        tokens : list of str
            Token labels
        top_k : int
            Number of top connections to return
        
        Returns
        -------
        list of tuples
            (query_token, key_token, flow_value)
        """
        if isinstance(flow, torch.Tensor):
            flow = flow.cpu().numpy()
        
        # Flatten and get top indices
        flat_indices = np.argsort(flow.flatten())[-top_k:][::-1]
        
        critical = []
        for idx in flat_indices:
            i = idx // flow.shape[1]
            j = idx % flow.shape[1]
            critical.append((tokens[i], tokens[j], float(flow[i, j])))
        
        return critical
    
    def compute_token_importance(
        self,
        flow: torch.Tensor,
        tokens: List[str],
        direction: str = 'incoming'
    ) -> List[Tuple[str, float]]:
        """
        Compute importance score for each token based on flow.
        
        Parameters
        ----------
        flow : torch.Tensor
            Attention flow matrix
        tokens : list of str
            Token labels
        direction : str
            'incoming': importance based on how much attention flows TO the token
            'outgoing': importance based on how much attention flows FROM the token
        
        Returns
        -------
        list of tuples
            (token, importance_score)
        """
        if isinstance(flow, torch.Tensor):
            flow = flow.cpu().numpy()
        
        if direction == 'incoming':
            # Sum columns: how much attention flows to each token
            importance = flow.sum(axis=0)
        elif direction == 'outgoing':
            # Sum rows: how much attention flows from each token
            importance = flow.sum(axis=1)
        else:
            raise ValueError(f"Unknown direction: {direction}")
        
        # Normalize
        importance = importance / importance.sum()
        
        return [(tokens[i], float(importance[i])) for i in range(len(tokens))]


def plot_token_importance(
    importance: List[Tuple[str, float]],
    direction: str = 'incoming',
    figsize: Tuple[int, int] = (12, 5)
) -> None:
    """
    Visualize token importance scores.
    
    Parameters
    ----------
    importance : list of tuples
        (token, importance_score)
    direction : str
        'incoming' or 'outgoing'
    figsize : tuple
        Figure size
    """
    tokens = [t[0] for t in importance]
    scores = [t[1] for t in importance]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.viridis(np.array(scores) / max(scores))
    
    bars = ax.bar(range(len(tokens)), scores, color=colors, alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Tokens', fontsize=12, fontweight='bold')
    ax.set_ylabel('Importance Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Token Importance ({direction.capitalize()} Flow)',
                fontsize=14, fontweight='bold')
    
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha='right')
    
    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{score:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
```

## Comparing Attention and Flow

### When Do They Differ?

Attention and flow can differ significantly when:

1. **High attention, low gradient**: Token receives attention but doesn't affect output
2. **Low attention, high gradient**: Small attention has large effect (sensitivity)
3. **Intermediate layers**: Flow captures propagation effects attention misses

```python
def analyze_attention_flow_discrepancy(
    attention: torch.Tensor,
    flow: torch.Tensor,
    tokens: List[str],
    threshold: float = 0.1
) -> dict:
    """
    Analyze where attention and flow disagree.
    
    Parameters
    ----------
    attention : torch.Tensor
        Raw attention weights
    flow : torch.Tensor
        Attention flow
    tokens : list of str
        Token labels
    threshold : float
        Minimum difference to consider significant
    
    Returns
    -------
    dict
        Analysis results with high_attention_low_flow and low_attention_high_flow
    """
    if isinstance(attention, torch.Tensor):
        attention = attention.cpu().numpy()
    if isinstance(flow, torch.Tensor):
        flow = flow.cpu().numpy()
    
    # Normalize for comparison
    attn_norm = attention / attention.max()
    flow_norm = flow / flow.max()
    
    results = {
        'high_attention_low_flow': [],
        'low_attention_high_flow': []
    }
    
    seq_len = attention.shape[0]
    for i in range(seq_len):
        for j in range(seq_len):
            diff = attn_norm[i, j] - flow_norm[i, j]
            
            if diff > threshold:
                # High attention but low flow (attention misleading)
                results['high_attention_low_flow'].append({
                    'query': tokens[i],
                    'key': tokens[j],
                    'attention': attention[i, j],
                    'flow': flow[i, j],
                    'difference': diff
                })
            elif diff < -threshold:
                # Low attention but high flow (hidden importance)
                results['low_attention_high_flow'].append({
                    'query': tokens[i],
                    'key': tokens[j],
                    'attention': attention[i, j],
                    'flow': flow[i, j],
                    'difference': -diff
                })
    
    # Sort by magnitude of difference
    for key in results:
        results[key] = sorted(results[key], key=lambda x: x['difference'], reverse=True)
    
    return results
```

## Complete Example

```python
def example_attention_flow():
    """Comprehensive attention flow analysis example."""
    
    # Create synthetic data
    tokens = ["The", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"]
    seq_len = len(tokens)
    
    # Create attention pattern
    attention = torch.zeros(seq_len, seq_len)
    for i in range(seq_len):
        for j in range(seq_len):
            distance = abs(i - j)
            attention[i, j] = np.exp(-distance * 0.5)
    attention = attention / attention.sum(dim=1, keepdim=True)
    
    # Create gradient pattern (simulating importance)
    gradients = torch.rand(seq_len, seq_len) * 0.2
    # Make certain connections important
    gradients[4, 0] = 2.0  # "jumps" <- "The" important
    gradients[4, 3] = 1.5  # "jumps" <- "fox" important
    gradients[7, 6] = 1.8  # "dog" <- "lazy" important
    gradients[2, 1] = 0.1  # "brown" <- "quick" NOT important
    
    print("=" * 70)
    print("Attention Flow Analysis")
    print("=" * 70)
    
    # Initialize analyzer
    analyzer = AttentionFlowAnalyzer()
    
    # Compute flow
    flow = analyzer.compute_attention_flow(attention, gradients)
    
    # 1. Visualize attention vs flow
    print("\n1. Attention vs Flow Comparison:")
    analyzer.visualize_flow(attention, flow, tokens)
    
    # 2. Visualize difference
    print("\n2. Difference Analysis:")
    analyzer.visualize_difference(attention, flow, tokens)
    
    # 3. Find critical connections
    print("\n3. Most Critical Connections (by Flow):")
    print("-" * 50)
    critical = analyzer.identify_critical_connections(flow, tokens, top_k=5)
    for query, key, flow_val in critical:
        print(f"   {query:10s} <- {key:10s} : {flow_val:.4f}")
    
    # 4. Token importance
    print("\n4. Token Importance (Incoming Flow):")
    importance = analyzer.compute_token_importance(flow, tokens, direction='incoming')
    plot_token_importance(importance, direction='incoming')
    
    # 5. Discrepancy analysis
    print("\n5. Attention vs Flow Discrepancies:")
    discrepancy = analyze_attention_flow_discrepancy(attention, flow, tokens)
    
    print("\n   High Attention, Low Flow (Attention Misleading):")
    for item in discrepancy['high_attention_low_flow'][:3]:
        print(f"      {item['query']} <- {item['key']}: "
              f"attn={item['attention']:.3f}, flow={item['flow']:.3f}")
    
    print("\n   Low Attention, High Flow (Hidden Importance):")
    for item in discrepancy['low_attention_high_flow'][:3]:
        print(f"      {item['query']} <- {item['key']}: "
              f"attn={item['attention']:.3f}, flow={item['flow']:.3f}")
    
    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)


if __name__ == "__main__":
    torch.manual_seed(42)
    example_attention_flow()
```

## Summary

Attention flow provides more accurate attribution than raw attention:

- **Gradient weighting** reveals which attention connections actually matter
- **Flow vs attention comparison** identifies misleading attention patterns
- **Critical connection analysis** finds the most important information pathways
- **Token importance** summarizes which tokens are most influential

Key insights:
1. High attention ≠ high importance (gradients reveal true impact)
2. Flow can reveal hidden important connections with low attention weights
3. Combining attention and gradients provides more faithful explanations
4. Discrepancy analysis helps understand when attention is misleading

## References

1. Jain, S., & Wallace, B. C. (2019). "Attention is not Explanation." NAACL.
2. Wiegreffe, S., & Pinter, Y. (2019). "Attention is not not Explanation." EMNLP.
3. Serrano, S., & Smith, N. A. (2019). "Is Attention Interpretable?" ACL.
