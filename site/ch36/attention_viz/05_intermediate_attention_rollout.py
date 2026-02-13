"""
Intermediate Level: Attention Rollout
=====================================

This module implements attention rollout, a technique for computing attention
across multiple layers of a transformer. Single-layer attention often doesn't
capture the full picture of how information flows through deep networks.

Learning Goals:
--------------
1. Understand the attention rollout algorithm
2. Implement attention flow across layers
3. Visualize cumulative attention
4. Compare rollout vs single-layer attention
5. Use rollout for model interpretability

Mathematical Background:
-----------------------
Attention rollout computes cumulative attention through layers by
matrix multiplication:

    Ã^(1) = A^(1)
    Ã^(2) = A^(2) × Ã^(1)
    Ã^(3) = A^(3) × Ã^(2)
    ...
    Ã^(L) = A^(L) × Ã^(L-1)

Where:
- A^(l) is the attention matrix at layer l
- Ã^(l) is the rolled-out attention up to layer l
- × represents matrix multiplication

Key insight: Ã^(L)[i,j] represents the total attention from token i
in the output to token j in the input, accumulated across all layers.

Reference:
---------
"Quantifying Attention Flow in Transformers" (Abnar & Zuidema, 2020)

Author: Deep Learning Curriculum
Date: November 2025
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class AttentionRollout:
    """
    Implementation of attention rollout for transformers.
    
    This class provides tools to:
    - Compute attention rollout across layers
    - Handle multi-head attention averaging
    - Visualize cumulative attention
    - Compare rollout with single-layer attention
    - Analyze attention flow patterns
    
    Attributes:
    ----------
    discard_ratio : float
        Ratio of lowest attention to discard (helps reduce noise)
    head_fusion : str
        Method to combine multi-head attention ('mean', 'max', 'min')
    """
    
    def __init__(self, discard_ratio: float = 0.1, head_fusion: str = 'mean'):
        """
        Initialize attention rollout.
        
        Parameters:
        ----------
        discard_ratio : float, default=0.1
            Fraction of lowest attention values to set to zero before rollout.
            This helps reduce noise and focus on strong attention paths.
            Value between 0 (keep all) and 1 (discard all)
        head_fusion : str, default='mean'
            How to combine attention from multiple heads:
            - 'mean': Average across heads
            - 'max': Take maximum across heads
            - 'min': Take minimum across heads
            
        Notes:
        -----
        The discard_ratio helps handle weak attention connections that
        might be noise rather than meaningful attention.
        """
        self.discard_ratio = discard_ratio
        self.head_fusion = head_fusion
    
    def _fuse_heads(self, attention_heads: torch.Tensor) -> torch.Tensor:
        """
        Combine attention from multiple heads into single matrix.
        
        Parameters:
        ----------
        attention_heads : torch.Tensor
            Multi-head attention, shape: (num_heads, seq_len, seq_len)
        
        Returns:
        -------
        torch.Tensor
            Fused attention matrix, shape: (seq_len, seq_len)
        """
        if self.head_fusion == 'mean':
            # Average across heads
            fused = attention_heads.mean(dim=0)
        elif self.head_fusion == 'max':
            # Take maximum across heads (most attended connection)
            fused = attention_heads.max(dim=0)[0]
        elif self.head_fusion == 'min':
            # Take minimum across heads (least attended connection)
            fused = attention_heads.min(dim=0)[0]
        else:
            raise ValueError(f"Unknown head_fusion: {self.head_fusion}")
        
        return fused
    
    def _apply_discard_ratio(self, attention: torch.Tensor) -> torch.Tensor:
        """
        Apply discard ratio to attention matrix.
        
        This sets the lowest discard_ratio fraction of attention values to zero,
        helping to focus on strong attention connections.
        
        Parameters:
        ----------
        attention : torch.Tensor
            Attention matrix, shape: (seq_len, seq_len)
        
        Returns:
        -------
        torch.Tensor
            Attention with lowest values set to zero
        """
        if self.discard_ratio <= 0:
            return attention
        
        # Flatten attention to find threshold
        flat_attention = attention.flatten()
        
        # Find threshold value at discard_ratio percentile
        threshold_idx = int(len(flat_attention) * self.discard_ratio)
        sorted_attention = torch.sort(flat_attention)[0]
        threshold = sorted_attention[threshold_idx]
        
        # Zero out values below threshold
        attention = attention.clone()
        attention[attention < threshold] = 0
        
        # Renormalize rows to sum to 1
        row_sums = attention.sum(dim=1, keepdim=True)
        attention = attention / (row_sums + 1e-10)
        
        return attention
    
    def _add_residual_connection(self, attention: torch.Tensor) -> torch.Tensor:
        """
        Add residual connection to attention matrix.
        
        Transformers have residual connections, so information can flow
        directly from input to output. We model this by adding identity
        to the attention matrix.
        
        Formula:
        -------
        Ã = (A + I) / 2
        
        Where:
        - A is the attention matrix
        - I is the identity matrix
        - Division by 2 maintains normalization
        
        Parameters:
        ----------
        attention : torch.Tensor
            Attention matrix, shape: (seq_len, seq_len)
        
        Returns:
        -------
        torch.Tensor
            Attention with residual connection
        """
        seq_len = attention.shape[0]
        identity = torch.eye(seq_len, device=attention.device)
        
        # Combine attention and residual
        attention_with_residual = (attention + identity) / 2.0
        
        # Renormalize
        row_sums = attention_with_residual.sum(dim=1, keepdim=True)
        attention_with_residual = attention_with_residual / (row_sums + 1e-10)
        
        return attention_with_residual
    
    def compute_rollout(self,
                       attention_layers: List[torch.Tensor],
                       include_residual: bool = True) -> torch.Tensor:
        """
        Compute attention rollout across all layers.
        
        This is the main function that implements the rollout algorithm.
        It processes each layer sequentially, multiplying attention matrices
        to track cumulative attention flow.
        
        Algorithm:
        ---------
        1. For each layer:
           a. Fuse multi-head attention (if multiple heads)
           b. Add residual connection (optional)
           c. Apply discard ratio (remove weak connections)
           d. Multiply with previous rollout result
        2. Return final rolled-out attention
        
        Parameters:
        ----------
        attention_layers : list of torch.Tensor
            List of attention matrices, one per layer.
            Each tensor can be either:
            - (seq_len, seq_len) for single-head
            - (num_heads, seq_len, seq_len) for multi-head
        include_residual : bool, default=True
            Whether to include residual connections in the computation
        
        Returns:
        -------
        torch.Tensor
            Rolled-out attention matrix, shape: (seq_len, seq_len)
            Element [i,j] shows total attention from output token i
            to input token j
            
        Example:
        -------
        >>> layers = [layer1_attention, layer2_attention, layer3_attention]
        >>> rollout = AttentionRollout()
        >>> result = rollout.compute_rollout(layers)
        >>> print(result[0, :])  # How first output token attends to all input tokens
        """
        # Initialize with identity (each token attends to itself initially)
        seq_len = attention_layers[0].shape[-1]
        rollout_attention = torch.eye(seq_len, device=attention_layers[0].device)
        
        # Process each layer
        for layer_idx, layer_attention in enumerate(attention_layers):
            # Step 1: Fuse heads if multi-head
            if layer_attention.dim() == 3:  # (num_heads, seq_len, seq_len)
                layer_attention = self._fuse_heads(layer_attention)
            
            # Step 2: Add residual connection
            if include_residual:
                layer_attention = self._add_residual_connection(layer_attention)
            
            # Step 3: Apply discard ratio
            layer_attention = self._apply_discard_ratio(layer_attention)
            
            # Step 4: Multiply with previous rollout
            # This accumulates attention flow through layers
            rollout_attention = torch.matmul(layer_attention, rollout_attention)
        
        return rollout_attention
    
    def compute_rollout_by_layer(self,
                                attention_layers: List[torch.Tensor],
                                include_residual: bool = True) -> List[torch.Tensor]:
        """
        Compute rollout at each layer (not just final).
        
        This is useful for:
        - Visualizing how attention evolves
        - Understanding layer-wise information flow
        - Debugging attention patterns
        
        Parameters:
        ----------
        attention_layers : list of torch.Tensor
            List of attention matrices
        include_residual : bool, default=True
            Whether to include residual connections
        
        Returns:
        -------
        list of torch.Tensor
            List of rollout attention at each layer
        """
        seq_len = attention_layers[0].shape[-1]
        rollout_attention = torch.eye(seq_len, device=attention_layers[0].device)
        
        rollout_by_layer = []
        
        for layer_attention in attention_layers:
            # Fuse heads
            if layer_attention.dim() == 3:
                layer_attention = self._fuse_heads(layer_attention)
            
            # Add residual
            if include_residual:
                layer_attention = self._add_residual_connection(layer_attention)
            
            # Apply discard ratio
            layer_attention = self._apply_discard_ratio(layer_attention)
            
            # Multiply with previous
            rollout_attention = torch.matmul(layer_attention, rollout_attention)
            
            # Save this layer's rollout
            rollout_by_layer.append(rollout_attention.clone())
        
        return rollout_by_layer
    
    def visualize(self,
                 rollout_attention: torch.Tensor,
                 tokens: List[str],
                 title: str = "Attention Rollout",
                 save_path: Optional[str] = None) -> None:
        """
        Visualize rolled-out attention as heatmap.
        
        Parameters:
        ----------
        rollout_attention : torch.Tensor
            Rolled-out attention matrix
        tokens : list of str
            Token strings
        title : str
            Plot title
        save_path : str, optional
            Path to save figure
        """
        if isinstance(rollout_attention, torch.Tensor):
            rollout_attention = rollout_attention.cpu().numpy()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(
            rollout_attention,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap='viridis',
            square=True,
            cbar_kws={'label': 'Rolled-out Attention'},
            ax=ax,
            vmin=0,
            vmax=rollout_attention.max()
        )
        
        ax.set_xlabel('Input Tokens', fontsize=12, fontweight='bold')
        ax.set_ylabel('Output Tokens', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"Saved to: {save_path}")
        
        plt.show()
    
    def compare_with_single_layer(self,
                                 attention_layers: List[torch.Tensor],
                                 tokens: List[str],
                                 layer_idx: int = -1,
                                 save_path: Optional[str] = None) -> None:
        """
        Compare rollout attention with single-layer attention.
        
        This visualization shows why rollout is useful:
        - Single layer: Only immediate attention
        - Rollout: Cumulative attention through all layers
        
        Parameters:
        ----------
        attention_layers : list of torch.Tensor
            List of attention matrices
        tokens : list of str
            Token strings
        layer_idx : int, default=-1
            Which single layer to compare with (default: last layer)
        save_path : str, optional
            Path to save figure
        """
        # Compute rollout
        rollout = self.compute_rollout(attention_layers)
        
        # Get single layer
        single_layer = attention_layers[layer_idx]
        if single_layer.dim() == 3:
            single_layer = self._fuse_heads(single_layer)
        
        # Convert to numpy
        rollout_np = rollout.cpu().numpy()
        single_np = single_layer.cpu().numpy()
        
        # Create comparison plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Plot single layer
        sns.heatmap(
            single_np,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap='viridis',
            square=True,
            cbar_kws={'label': 'Attention'},
            ax=axes[0],
            vmin=0,
            vmax=1
        )
        axes[0].set_title(f'Single Layer (Layer {layer_idx})', 
                         fontsize=13, fontweight='bold')
        axes[0].set_xlabel('Key Tokens', fontsize=11)
        axes[0].set_ylabel('Query Tokens', fontsize=11)
        
        # Plot rollout
        sns.heatmap(
            rollout_np,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap='viridis',
            square=True,
            cbar_kws={'label': 'Rolled-out Attention'},
            ax=axes[1],
            vmin=0,
            vmax=rollout_np.max()
        )
        axes[1].set_title('Attention Rollout (All Layers)', 
                         fontsize=13, fontweight='bold')
        axes[1].set_xlabel('Input Tokens', fontsize=11)
        axes[1].set_ylabel('Output Tokens', fontsize=11)
        
        # Rotate labels
        for ax in axes:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.suptitle('Single-Layer vs Rollout Attention', 
                    fontsize=15, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"Saved to: {save_path}")
        
        plt.show()
    
    def visualize_evolution(self,
                          attention_layers: List[torch.Tensor],
                          tokens: List[str],
                          layers_to_show: Optional[List[int]] = None,
                          save_path: Optional[str] = None) -> None:
        """
        Visualize how rollout attention evolves through layers.
        
        This shows the progressive accumulation of attention information
        as we move through the network.
        
        Parameters:
        ----------
        attention_layers : list of torch.Tensor
            List of attention matrices
        tokens : list of str
            Token strings
        layers_to_show : list of int, optional
            Which layers to visualize (default: evenly spaced)
        save_path : str, optional
            Path to save figure
        """
        # Compute rollout at each layer
        rollout_by_layer = self.compute_rollout_by_layer(attention_layers)
        
        # Determine which layers to show
        num_layers = len(rollout_by_layer)
        if layers_to_show is None:
            # Show first, middle, and last layers
            if num_layers <= 4:
                layers_to_show = list(range(num_layers))
            else:
                step = num_layers // 3
                layers_to_show = [0, step, 2*step, num_layers-1]
        
        num_plots = len(layers_to_show)
        
        # Create subplots
        ncols = min(4, num_plots)
        nrows = (num_plots + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
        if num_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        # Plot each layer
        for idx, layer_idx in enumerate(layers_to_show):
            ax = axes[idx]
            rollout = rollout_by_layer[layer_idx].cpu().numpy()
            
            sns.heatmap(
                rollout,
                xticklabels=tokens if idx % ncols == 0 else [],
                yticklabels=tokens,
                cmap='viridis',
                square=True,
                cbar=True,
                ax=ax,
                vmin=0,
                vmax=np.max([r.cpu().numpy().max() for r in rollout_by_layer]),
                cbar_kws={'shrink': 0.8}
            )
            
            ax.set_title(f'After Layer {layer_idx}', fontsize=11, fontweight='bold')
            
            if idx % ncols == 0:
                ax.set_ylabel('Output', fontsize=10)
            if idx >= num_plots - ncols:
                ax.set_xlabel('Input', fontsize=10)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Hide extra subplots
        for idx in range(num_plots, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Attention Rollout Evolution Through Layers', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"Saved to: {save_path}")
        
        plt.show()


# ============================================================================
# Example Usage and Demonstrations
# ============================================================================

def create_synthetic_transformer(num_layers: int, seq_len: int, num_heads: int = 8):
    """Create synthetic transformer attention for testing."""
    attention_layers = []
    
    for layer_idx in range(num_layers):
        # Create different patterns for different layers
        attention_heads = torch.zeros(num_heads, seq_len, seq_len)
        
        for head_idx in range(num_heads):
            # Early layers: more local attention
            # Late layers: more global attention
            if layer_idx < num_layers // 2:
                # Local pattern
                attn = torch.zeros(seq_len, seq_len)
                bandwidth = 2
                for i in range(seq_len):
                    for j in range(max(0, i-bandwidth), min(seq_len, i+bandwidth+1)):
                        attn[i, j] = 1.0 / (abs(i-j) + 1)
            else:
                # More global pattern
                attn = torch.rand(seq_len, seq_len)
            
            # Normalize
            attn = attn / attn.sum(dim=1, keepdim=True)
            attention_heads[head_idx] = attn
        
        attention_layers.append(attention_heads)
    
    return attention_layers


def example_basic_rollout():
    """Example 1: Basic attention rollout."""
    print("=" * 70)
    print("Example 1: Basic Attention Rollout")
    print("=" * 70)
    
    # Create synthetic attention
    num_layers = 6
    seq_len = 8
    tokens = ["The", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"]
    
    attention_layers = create_synthetic_transformer(num_layers, seq_len)
    
    # Compute rollout
    rollout = AttentionRollout(discard_ratio=0.1)
    result = rollout.compute_rollout(attention_layers)
    
    print(f"\nRollout computed for {num_layers} layers")
    print(f"Shape: {result.shape}")
    print(f"Sum of first row: {result[0].sum():.4f}")  # Should be ~1.0
    
    # Visualize
    rollout.visualize(result, tokens, "Attention Rollout Example")


def example_comparison():
    """Example 2: Compare rollout with single layer."""
    print("\n" + "=" * 70)
    print("Example 2: Rollout vs Single-Layer Attention")
    print("=" * 70)
    
    num_layers = 6
    seq_len = 8
    tokens = ["I", "love", "deep", "learning", "and", "neural", "networks", "!"]
    
    attention_layers = create_synthetic_transformer(num_layers, seq_len)
    
    rollout = AttentionRollout(discard_ratio=0.1)
    
    # Compare
    rollout.compare_with_single_layer(attention_layers, tokens, layer_idx=-1)
    
    print("\nKey Observations:")
    print("  - Single layer shows immediate attention only")
    print("  - Rollout captures cumulative attention through all layers")
    print("  - Rollout often reveals long-range dependencies better")


def example_evolution():
    """Example 3: Visualize rollout evolution."""
    print("\n" + "=" * 70)
    print("Example 3: Attention Rollout Evolution")
    print("=" * 70)
    
    num_layers = 8
    seq_len = 8
    tokens = ["CLS", "This", "is", "a", "test", "sentence", "here", "SEP"]
    
    attention_layers = create_synthetic_transformer(num_layers, seq_len)
    
    rollout = AttentionRollout(discard_ratio=0.1)
    
    # Show evolution
    rollout.visualize_evolution(attention_layers, tokens, 
                               layers_to_show=[0, 2, 4, 7])


def example_hyperparameters():
    """Example 4: Effect of hyperparameters."""
    print("\n" + "=" * 70)
    print("Example 4: Hyperparameter Effects")
    print("=" * 70)
    
    num_layers = 6
    seq_len = 6
    tokens = ["The", "cat", "sat", "on", "mat", "."]
    
    attention_layers = create_synthetic_transformer(num_layers, seq_len, num_heads=4)
    
    # Try different discard ratios
    discard_ratios = [0.0, 0.1, 0.3]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, ratio in enumerate(discard_ratios):
        rollout = AttentionRollout(discard_ratio=ratio)
        result = rollout.compute_rollout(attention_layers)
        result_np = result.cpu().numpy()
        
        sns.heatmap(
            result_np,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap='viridis',
            square=True,
            cbar=True,
            ax=axes[idx],
            vmin=0,
            vmax=result_np.max(),
            cbar_kws={'shrink': 0.8}
        )
        
        axes[idx].set_title(f'Discard Ratio: {ratio}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Input Tokens', fontsize=10)
        axes[idx].set_ylabel('Output Tokens' if idx == 0 else '', fontsize=10)
    
    plt.suptitle('Effect of Discard Ratio on Attention Rollout',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print("\nObservations:")
    print("  - Discard ratio 0.0: Keeps all attention (noisier)")
    print("  - Discard ratio 0.1: Removes weak connections (cleaner)")
    print("  - Discard ratio 0.3: More aggressive filtering")


if __name__ == "__main__":
    """
    Main execution demonstrating attention rollout.
    
    Run this script to learn about attention rollout:
        python attention_rollout.py
    """
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("\n" + "=" * 70)
    print(" ATTENTION ROLLOUT ")
    print(" Module 59: Computing Cumulative Attention ")
    print("=" * 70)
    
    # Run examples
    example_basic_rollout()
    example_comparison()
    example_evolution()
    example_hyperparameters()
    
    print("\n" + "=" * 70)
    print(" Examples completed! ")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  1. Rollout tracks attention flow through all layers")
    print("  2. Provides better interpretability than single-layer attention")
    print("  3. Captures long-range dependencies")
    print("  4. Discard ratio helps reduce noise")
    print("  5. Useful for understanding deep transformer behavior")
