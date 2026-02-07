# Attention Rollout

## Introduction

Single-layer attention provides only a partial view of how information flows through transformers. **Attention rollout** addresses this limitation by computing cumulative attention across all layers, revealing the total influence of input tokens on output representations.

This technique is essential for understanding deep transformer behavior and provides more interpretable attribution than examining individual layers.

## Mathematical Foundation

### The Rollout Algorithm

Attention rollout computes cumulative attention through matrix multiplication:

$$\tilde{A}^{(1)} = A^{(1)}$$

$$\tilde{A}^{(l)} = A^{(l)} \times \tilde{A}^{(l-1)} \quad \text{for } l = 2, \ldots, L$$

where:
- $A^{(l)}$ is the attention matrix at layer $l$
- $\tilde{A}^{(l)}$ is the rolled-out attention up to layer $l$
- $\times$ denotes matrix multiplication

The final rollout $\tilde{A}^{(L)}_{ij}$ represents the total attention from output position $i$ to input position $j$ accumulated across all layers.

### Including Residual Connections

Transformers have residual connections that allow information to bypass attention layers. To account for this:

$$\bar{A}^{(l)} = \frac{1}{2}(A^{(l)} + I)$$

where $I$ is the identity matrix. This modification models the residual stream that preserves information from earlier layers.

### Multi-Head Fusion

For multi-head attention, heads must be combined before rollout. Common fusion strategies:

| Method | Formula | Use Case |
|--------|---------|----------|
| Mean | $\bar{A} = \frac{1}{h}\sum_{i=1}^h A_i$ | General analysis |
| Max | $\bar{A}_{jk} = \max_i A_{i,jk}$ | Finding any strong connection |
| Min | $\bar{A}_{jk} = \min_i A_{i,jk}$ | Conservative estimates |

## Implementation

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple

class AttentionRollout:
    """
    Implementation of attention rollout for transformers.
    
    Reference: "Quantifying Attention Flow in Transformers" (Abnar & Zuidema, 2020)
    """
    
    def __init__(
        self,
        discard_ratio: float = 0.1,
        head_fusion: str = 'mean',
        include_residual: bool = True
    ):
        """
        Initialize attention rollout.
        
        Parameters
        ----------
        discard_ratio : float
            Fraction of lowest attention values to discard (reduces noise)
        head_fusion : str
            Method to combine multi-head attention: 'mean', 'max', 'min'
        include_residual : bool
            Whether to model residual connections
        """
        self.discard_ratio = discard_ratio
        self.head_fusion = head_fusion
        self.include_residual = include_residual
    
    def _fuse_heads(self, attention_heads: torch.Tensor) -> torch.Tensor:
        """
        Combine attention from multiple heads.
        
        Parameters
        ----------
        attention_heads : torch.Tensor
            Shape: (num_heads, seq_len, seq_len)
        
        Returns
        -------
        torch.Tensor
            Fused attention, shape: (seq_len, seq_len)
        """
        if self.head_fusion == 'mean':
            return attention_heads.mean(dim=0)
        elif self.head_fusion == 'max':
            return attention_heads.max(dim=0)[0]
        elif self.head_fusion == 'min':
            return attention_heads.min(dim=0)[0]
        else:
            raise ValueError(f"Unknown head_fusion: {self.head_fusion}")
    
    def _apply_discard_ratio(self, attention: torch.Tensor) -> torch.Tensor:
        """
        Discard lowest attention values to reduce noise.
        
        Parameters
        ----------
        attention : torch.Tensor
            Attention matrix
        
        Returns
        -------
        torch.Tensor
            Filtered attention matrix
        """
        if self.discard_ratio <= 0:
            return attention
        
        # Find threshold
        flat = attention.flatten()
        threshold_idx = int(len(flat) * self.discard_ratio)
        threshold = torch.sort(flat)[0][threshold_idx]
        
        # Zero out below threshold
        attention = attention.clone()
        attention[attention < threshold] = 0
        
        # Renormalize rows
        row_sums = attention.sum(dim=1, keepdim=True)
        attention = attention / (row_sums + 1e-10)
        
        return attention
    
    def _add_residual(self, attention: torch.Tensor) -> torch.Tensor:
        """
        Add residual connection to attention matrix.
        
        Models: output = attention(input) + input
        
        Parameters
        ----------
        attention : torch.Tensor
            Attention matrix
        
        Returns
        -------
        torch.Tensor
            Attention with residual
        """
        seq_len = attention.shape[0]
        identity = torch.eye(seq_len, device=attention.device)
        
        # Average attention and identity
        attention_with_residual = (attention + identity) / 2.0
        
        # Renormalize
        row_sums = attention_with_residual.sum(dim=1, keepdim=True)
        attention_with_residual = attention_with_residual / (row_sums + 1e-10)
        
        return attention_with_residual
    
    def compute_rollout(
        self,
        attention_layers: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute attention rollout across all layers.
        
        Parameters
        ----------
        attention_layers : list of torch.Tensor
            Attention from each layer. Each tensor can be:
            - (seq_len, seq_len) for single-head
            - (num_heads, seq_len, seq_len) for multi-head
            - (batch, num_heads, seq_len, seq_len) for batched
        
        Returns
        -------
        torch.Tensor
            Rolled-out attention, shape: (seq_len, seq_len)
        """
        rollout = None
        
        for layer_idx, layer_attention in enumerate(attention_layers):
            # Handle batched input
            if layer_attention.dim() == 4:
                layer_attention = layer_attention[0]  # Take first batch
            
            # Fuse multi-head attention
            if layer_attention.dim() == 3:
                attention = self._fuse_heads(layer_attention)
            else:
                attention = layer_attention
            
            # Add residual connection
            if self.include_residual:
                attention = self._add_residual(attention)
            
            # Apply discard ratio
            attention = self._apply_discard_ratio(attention)
            
            # Compute rollout
            if rollout is None:
                rollout = attention
            else:
                rollout = torch.matmul(attention, rollout)
        
        return rollout
    
    def compute_rollout_by_layer(
        self,
        attention_layers: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Compute rollout incrementally, returning result at each layer.
        
        Useful for visualizing how attention evolves through the network.
        
        Parameters
        ----------
        attention_layers : list of torch.Tensor
            Attention from each layer
        
        Returns
        -------
        list of torch.Tensor
            Rollout at each layer
        """
        rollouts = []
        rollout = None
        
        for layer_attention in attention_layers:
            if layer_attention.dim() == 4:
                layer_attention = layer_attention[0]
            
            if layer_attention.dim() == 3:
                attention = self._fuse_heads(layer_attention)
            else:
                attention = layer_attention
            
            if self.include_residual:
                attention = self._add_residual(attention)
            
            attention = self._apply_discard_ratio(attention)
            
            if rollout is None:
                rollout = attention
            else:
                rollout = torch.matmul(attention, rollout)
            
            rollouts.append(rollout.clone())
        
        return rollouts
    
    def visualize(
        self,
        rollout: torch.Tensor,
        tokens: List[str],
        title: str = "Attention Rollout",
        save_path: Optional[str] = None
    ) -> None:
        """
        Visualize rolled-out attention.
        
        Parameters
        ----------
        rollout : torch.Tensor
            Rollout matrix, shape: (seq_len, seq_len)
        tokens : list of str
            Token labels
        title : str
            Plot title
        save_path : str, optional
            Path to save figure
        """
        if isinstance(rollout, torch.Tensor):
            rollout = rollout.cpu().numpy()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(
            rollout,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap='viridis',
            square=True,
            cbar_kws={'label': 'Cumulative Attention'},
            ax=ax,
            vmin=0
        )
        
        ax.set_xlabel('Input Tokens', fontsize=12, fontweight='bold')
        ax.set_ylabel('Output Tokens', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def compare_with_single_layer(
        self,
        attention_layers: List[torch.Tensor],
        tokens: List[str],
        layer_idx: int = -1
    ) -> None:
        """
        Compare rollout with single-layer attention.
        
        Parameters
        ----------
        attention_layers : list of torch.Tensor
            Attention from each layer
        tokens : list of str
            Token labels
        layer_idx : int
            Which layer to compare against (-1 for last)
        """
        # Compute rollout
        rollout = self.compute_rollout(attention_layers)
        
        # Get single layer attention
        if layer_idx < 0:
            layer_idx = len(attention_layers) + layer_idx
        
        single_layer = attention_layers[layer_idx]
        if single_layer.dim() == 4:
            single_layer = single_layer[0]
        if single_layer.dim() == 3:
            single_layer = self._fuse_heads(single_layer)
        
        # Convert to numpy
        rollout_np = rollout.cpu().numpy()
        single_np = single_layer.cpu().numpy()
        
        # Plot comparison
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        sns.heatmap(
            single_np,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap='viridis',
            square=True,
            cbar_kws={'label': 'Attention', 'shrink': 0.8},
            ax=axes[0],
            vmin=0,
            vmax=max(single_np.max(), rollout_np.max())
        )
        axes[0].set_title(f'Single Layer ({layer_idx})', fontsize=13, fontweight='bold')
        axes[0].set_xlabel('Key Tokens', fontsize=11)
        axes[0].set_ylabel('Query Tokens', fontsize=11)
        
        sns.heatmap(
            rollout_np,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap='viridis',
            square=True,
            cbar_kws={'label': 'Cumulative Attention', 'shrink': 0.8},
            ax=axes[1],
            vmin=0,
            vmax=max(single_np.max(), rollout_np.max())
        )
        axes[1].set_title('Attention Rollout (All Layers)', fontsize=13, fontweight='bold')
        axes[1].set_xlabel('Input Tokens', fontsize=11)
        axes[1].set_ylabel('Output Tokens', fontsize=11)
        
        for ax in axes:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.suptitle('Single Layer vs. Attention Rollout', fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def visualize_evolution(
        self,
        attention_layers: List[torch.Tensor],
        tokens: List[str],
        layers_to_show: Optional[List[int]] = None
    ) -> None:
        """
        Visualize how rollout evolves through layers.
        
        Parameters
        ----------
        attention_layers : list of torch.Tensor
            Attention from each layer
        tokens : list of str
            Token labels
        layers_to_show : list of int, optional
            Which layers to display
        """
        rollouts = self.compute_rollout_by_layer(attention_layers)
        num_layers = len(rollouts)
        
        if layers_to_show is None:
            if num_layers <= 6:
                layers_to_show = list(range(num_layers))
            else:
                step = num_layers // 4
                layers_to_show = [0, step, 2*step, 3*step, num_layers-1]
        
        n_plots = len(layers_to_show)
        ncols = min(4, n_plots)
        nrows = (n_plots + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
        if n_plots == 1:
            axes = np.array([[axes]])
        elif nrows == 1:
            axes = axes.reshape(1, -1)
        
        # Find global max for consistent colorbar
        vmax = max(r.cpu().numpy().max() for r in rollouts)
        
        for idx, layer_idx in enumerate(layers_to_show):
            row, col = idx // ncols, idx % ncols
            ax = axes[row, col]
            
            rollout = rollouts[layer_idx].cpu().numpy()
            
            sns.heatmap(
                rollout,
                xticklabels=tokens if row == nrows - 1 else [],
                yticklabels=tokens if col == 0 else [],
                cmap='viridis',
                square=True,
                cbar=True,
                ax=ax,
                vmin=0,
                vmax=vmax,
                cbar_kws={'shrink': 0.7}
            )
            
            ax.set_title(f'After Layer {layer_idx}', fontsize=11, fontweight='bold')
        
        # Hide unused subplots
        for idx in range(n_plots, nrows * ncols):
            row, col = idx // ncols, idx % ncols
            axes[row, col].axis('off')
        
        plt.suptitle('Attention Rollout Evolution', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
```

## Effect of Hyperparameters

### Discard Ratio

The discard ratio controls noise filtering:

```python
def compare_discard_ratios(
    attention_layers: List[torch.Tensor],
    tokens: List[str],
    ratios: List[float] = [0.0, 0.1, 0.3, 0.5]
) -> None:
    """Compare rollout with different discard ratios."""
    
    fig, axes = plt.subplots(1, len(ratios), figsize=(5*len(ratios), 5))
    
    for idx, ratio in enumerate(ratios):
        rollout_obj = AttentionRollout(discard_ratio=ratio)
        result = rollout_obj.compute_rollout(attention_layers)
        result_np = result.cpu().numpy()
        
        ax = axes[idx]
        sns.heatmap(
            result_np,
            xticklabels=tokens,
            yticklabels=tokens if idx == 0 else [],
            cmap='viridis',
            square=True,
            cbar=True,
            ax=ax,
            vmin=0,
            cbar_kws={'shrink': 0.8}
        )
        
        ax.set_title(f'Discard Ratio: {ratio}', fontsize=12, fontweight='bold')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.suptitle('Effect of Discard Ratio', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
```

| Discard Ratio | Effect | Use Case |
|--------------|--------|----------|
| 0.0 | Keep all connections | Raw analysis |
| 0.1 | Light filtering | General use (recommended) |
| 0.3 | Moderate filtering | Noisy attention |
| 0.5+ | Heavy filtering | Finding strongest paths |

### Head Fusion Methods

```python
def compare_head_fusion(
    attention_layers: List[torch.Tensor],
    tokens: List[str]
) -> None:
    """Compare different head fusion methods."""
    methods = ['mean', 'max', 'min']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, method in enumerate(methods):
        rollout_obj = AttentionRollout(head_fusion=method)
        result = rollout_obj.compute_rollout(attention_layers)
        result_np = result.cpu().numpy()
        
        ax = axes[idx]
        sns.heatmap(
            result_np,
            xticklabels=tokens,
            yticklabels=tokens if idx == 0 else [],
            cmap='viridis',
            square=True,
            ax=ax,
            vmin=0
        )
        ax.set_title(f'Head Fusion: {method}', fontsize=12, fontweight='bold')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.suptitle('Effect of Head Fusion Method', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
```

## Complete Example

```python
def example_attention_rollout():
    """Comprehensive attention rollout example."""
    
    # Create synthetic transformer attention
    num_layers = 6
    num_heads = 8
    seq_len = 8
    tokens = ["[CLS]", "The", "quick", "brown", "fox", "jumps", "over", "[SEP]"]
    
    # Generate evolving attention patterns
    attention_layers = []
    for layer_idx in range(num_layers):
        attention_heads = torch.zeros(num_heads, seq_len, seq_len)
        
        for head_idx in range(num_heads):
            # Early layers: local; Late layers: global
            locality = 1.0 - layer_idx / num_layers * 0.5
            
            attention = torch.zeros(seq_len, seq_len)
            for i in range(seq_len):
                for j in range(seq_len):
                    distance = abs(i - j)
                    attention[i, j] = np.exp(-distance * locality)
            
            # Add special token attention in later layers
            attention[:, 0] += layer_idx / num_layers * 0.3
            attention[:, -1] += layer_idx / num_layers * 0.2
            
            attention = attention / attention.sum(dim=1, keepdim=True)
            attention_heads[head_idx] = attention
        
        attention_layers.append(attention_heads)
    
    print("=" * 70)
    print("Attention Rollout Analysis")
    print("=" * 70)
    
    # Initialize rollout
    rollout = AttentionRollout(discard_ratio=0.1, head_fusion='mean')
    
    # 1. Basic rollout
    print("\n1. Basic Rollout:")
    result = rollout.compute_rollout(attention_layers)
    rollout.visualize(result, tokens, "Attention Rollout")
    
    # 2. Compare with single layer
    print("\n2. Rollout vs Single Layer:")
    rollout.compare_with_single_layer(attention_layers, tokens, layer_idx=-1)
    
    # 3. Evolution through layers
    print("\n3. Rollout Evolution:")
    rollout.visualize_evolution(attention_layers, tokens)
    
    # 4. Discard ratio comparison
    print("\n4. Discard Ratio Comparison:")
    compare_discard_ratios(attention_layers, tokens)
    
    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)

if __name__ == "__main__":
    torch.manual_seed(42)
    example_attention_rollout()
```

## Summary

Attention rollout is a powerful technique for understanding transformer attention:

- **Cumulative attention** captures information flow across all layers
- **Residual modeling** accounts for the transformer's skip connections
- **Discard ratio** helps filter noise for cleaner visualizations
- **Head fusion** combines multi-head attention appropriately
- **Evolution visualization** shows how attention accumulates layer by layer

Key insights:
1. Rollout often reveals long-range dependencies missed by single-layer analysis
2. Special tokens ([CLS], [SEP]) typically accumulate attention through layers
3. Local patterns in early layers contribute to global patterns via rollout
4. Discard ratio of 0.1 provides good balance between signal and noise

## References

1. Abnar, S., & Zuidema, W. (2020). "Quantifying Attention Flow in Transformers." ACL.
2. Chefer, H., et al. (2021). "Transformer Interpretability Beyond Attention Visualization." CVPR.
