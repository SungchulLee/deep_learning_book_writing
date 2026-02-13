"""
Beginner Level: Attention Basics
==================================

This module provides foundational tools for extracting and visualizing attention weights
from transformer models. It covers the basics of attention weight extraction, simple
heatmap creation, and understanding attention score distributions.

Learning Goals:
--------------
1. Extract attention weights from transformer layers
2. Create basic attention heatmaps using matplotlib
3. Understand attention weight normalization
4. Visualize token-to-token attention relationships

Mathematical Background:
-----------------------
Attention weights come from the softmax operation in the attention mechanism:
    
    attention_weights = softmax(Q @ K^T / sqrt(d_k))
    
Where:
- Q: Query matrix (seq_len x d_model)
- K: Key matrix (seq_len x d_model)
- d_k: Key dimension (for scaling)
- attention_weights: Matrix of shape (seq_len x seq_len)

Each element attention_weights[i, j] represents how much token i attends to token j.

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


class AttentionVisualizer:
    """
    A comprehensive class for basic attention weight visualization.
    
    This class provides methods to extract, process, and visualize attention weights
    from transformer models. It's designed for educational purposes with extensive
    comments explaining each step.
    
    Attributes:
    ----------
    figsize : tuple
        Default figure size for plots (width, height)
    cmap : str
        Default colormap for heatmaps
    dpi : int
        Dots per inch for saved figures
    """
    
    def __init__(self, figsize: Tuple[int, int] = (10, 8), cmap: str = 'viridis', dpi: int = 100):
        """
        Initialize the AttentionVisualizer.
        
        Parameters:
        ----------
        figsize : tuple, default=(10, 8)
            Figure size (width, height) in inches
        cmap : str, default='viridis'
            Matplotlib colormap name. 'viridis' is perceptually uniform and colorblind-friendly
        dpi : int, default=100
            Resolution for saved figures
        """
        self.figsize = figsize
        self.cmap = cmap
        self.dpi = dpi
        
        # Set style for better-looking plots
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def extract_attention_from_model(self, 
                                    model: nn.Module, 
                                    input_ids: torch.Tensor,
                                    layer_idx: int = 0,
                                    head_idx: int = 0) -> torch.Tensor:
        """
        Extract attention weights from a specific layer and head of a transformer model.
        
        This method assumes the model returns attention weights when called with
        output_attentions=True. This is standard for Hugging Face transformers.
        
        Parameters:
        ----------
        model : nn.Module
            The transformer model (should support output_attentions=True)
        input_ids : torch.Tensor
            Input token IDs, shape: (batch_size, seq_len)
        layer_idx : int, default=0
            Which transformer layer to extract attention from (0-indexed)
        head_idx : int, default=0
            Which attention head to extract (0-indexed)
        
        Returns:
        -------
        torch.Tensor
            Attention weights matrix of shape (seq_len, seq_len)
            
        Notes:
        -----
        - Attention weights are always between 0 and 1 (due to softmax)
        - Each row sums to 1 (property of softmax)
        - Diagonal elements often have high values (tokens attending to themselves)
        """
        # Set model to evaluation mode
        model.eval()
        
        # Forward pass with attention output enabled
        with torch.no_grad():
            # Most Hugging Face models return a tuple where the second element is attentions
            outputs = model(input_ids, output_attentions=True)
            
            # Extract attention weights
            # attentions is a tuple of length num_layers
            # Each element has shape: (batch_size, num_heads, seq_len, seq_len)
            attentions = outputs.attentions
            
            # Get specific layer and head
            # Shape: (batch_size, num_heads, seq_len, seq_len)
            layer_attention = attentions[layer_idx]
            
            # Extract specific head and first batch element
            # Shape: (seq_len, seq_len)
            attention_weights = layer_attention[0, head_idx, :, :]
        
        return attention_weights
    
    def plot_attention_heatmap(self,
                              attention_weights: torch.Tensor,
                              tokens: List[str],
                              title: str = "Attention Weights",
                              save_path: Optional[str] = None,
                              show_values: bool = False) -> None:
        """
        Create a heatmap visualization of attention weights.
        
        A heatmap is the most common way to visualize attention. Each cell (i,j)
        shows how much token i attends to token j. Brighter colors indicate stronger
        attention.
        
        Parameters:
        ----------
        attention_weights : torch.Tensor
            Attention weight matrix, shape: (seq_len, seq_len)
        tokens : list of str
            List of token strings corresponding to the sequence
        title : str, default="Attention Weights"
            Title for the plot
        save_path : str, optional
            If provided, save the figure to this path
        show_values : bool, default=False
            If True, display numerical values in each cell
            
        Notes:
        -----
        - Rows represent query tokens (what's attending)
        - Columns represent key tokens (what's being attended to)
        - Values are normalized (row-wise sum = 1)
        """
        # Convert to numpy for plotting
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.cpu().numpy()
        
        # Verify dimensions match
        assert attention_weights.shape[0] == len(tokens), \
            f"Attention shape {attention_weights.shape[0]} doesn't match tokens {len(tokens)}"
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create heatmap using seaborn
        # annot=show_values adds numbers to cells
        # fmt='.3f' formats numbers to 3 decimal places
        # square=True makes cells square-shaped
        # cbar_kws adds label to colorbar
        sns.heatmap(
            attention_weights,
            annot=show_values,
            fmt='.3f' if show_values else '',
            cmap=self.cmap,
            square=True,
            xticklabels=tokens,
            yticklabels=tokens,
            cbar_kws={'label': 'Attention Weight'},
            vmin=0,  # Minimum value (attention weights are 0 to 1)
            vmax=1,  # Maximum value
            ax=ax
        )
        
        # Set labels and title
        ax.set_xlabel('Key Tokens (attended to)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Query Tokens (attending from)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        plt.show()
    
    def compute_attention_statistics(self, attention_weights: torch.Tensor) -> Dict[str, float]:
        """
        Compute statistical measures of attention distribution.
        
        These statistics help understand the nature of attention patterns:
        - Mean: Average attention value
        - Std: How spread out attention values are
        - Max: Strongest attention connection
        - Diagonal mean: How much tokens attend to themselves
        - Entropy: Measure of attention concentration
        
        Parameters:
        ----------
        attention_weights : torch.Tensor
            Attention weight matrix, shape: (seq_len, seq_len)
        
        Returns:
        -------
        dict
            Dictionary containing various statistics
        """
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.cpu().numpy()
        
        # Basic statistics
        stats = {
            'mean': float(np.mean(attention_weights)),
            'std': float(np.std(attention_weights)),
            'max': float(np.max(attention_weights)),
            'min': float(np.min(attention_weights)),
        }
        
        # Diagonal statistics (self-attention)
        diagonal = np.diag(attention_weights)
        stats['diagonal_mean'] = float(np.mean(diagonal))
        stats['diagonal_std'] = float(np.std(diagonal))
        
        # Off-diagonal statistics (attention to other tokens)
        seq_len = attention_weights.shape[0]
        off_diag_mask = ~np.eye(seq_len, dtype=bool)
        off_diagonal = attention_weights[off_diag_mask]
        stats['off_diagonal_mean'] = float(np.mean(off_diagonal))
        
        # Entropy calculation
        # Entropy measures how "spread out" the attention is
        # Low entropy = focused attention, High entropy = diffuse attention
        # H = -sum(p * log(p)) for each row
        entropies = []
        for row in attention_weights:
            # Add small epsilon to avoid log(0)
            row_safe = row + 1e-10
            entropy = -np.sum(row_safe * np.log(row_safe))
            entropies.append(entropy)
        stats['mean_entropy'] = float(np.mean(entropies))
        stats['std_entropy'] = float(np.std(entropies))
        
        return stats
    
    def plot_attention_distribution(self,
                                   attention_weights: torch.Tensor,
                                   title: str = "Attention Weight Distribution",
                                   save_path: Optional[str] = None) -> None:
        """
        Plot the distribution of attention weights as a histogram.
        
        This helps understand the overall pattern of attention:
        - Uniform distribution: Model attends equally to all tokens
        - Peaked distribution: Model focuses on specific tokens
        - Bimodal distribution: Model has two attention modes
        
        Parameters:
        ----------
        attention_weights : torch.Tensor
            Attention weight matrix, shape: (seq_len, seq_len)
        title : str
            Title for the plot
        save_path : str, optional
            If provided, save figure to this path
        """
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.cpu().numpy()
        
        # Flatten attention weights
        weights_flat = attention_weights.flatten()
        
        # Create figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Subplot 1: Histogram
        axes[0].hist(weights_flat, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Attention Weight', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Histogram of Attention Weights', fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Add vertical line for mean
        mean_weight = np.mean(weights_flat)
        axes[0].axvline(mean_weight, color='red', linestyle='--', 
                       label=f'Mean: {mean_weight:.4f}', linewidth=2)
        axes[0].legend()
        
        # Subplot 2: Box plot comparing diagonal vs off-diagonal
        diagonal = np.diag(attention_weights)
        seq_len = attention_weights.shape[0]
        off_diag_mask = ~np.eye(seq_len, dtype=bool)
        off_diagonal = attention_weights[off_diag_mask]
        
        box_data = [diagonal, off_diagonal]
        axes[1].boxplot(box_data, labels=['Self-Attention\n(Diagonal)', 'Other Tokens\n(Off-diagonal)'])
        axes[1].set_ylabel('Attention Weight', fontsize=12)
        axes[1].set_title('Self-Attention vs Other Tokens', fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # Overall title
        fig.suptitle(title, fontsize=15, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        plt.show()
    
    def plot_token_attention(self,
                            attention_weights: torch.Tensor,
                            tokens: List[str],
                            focus_token_idx: int,
                            title: Optional[str] = None,
                            save_path: Optional[str] = None) -> None:
        """
        Plot attention weights for a specific token to all other tokens.
        
        This visualization shows how one specific token (query) attends to all
        other tokens (keys). It's useful for understanding what a particular
        token is "looking at" in the sequence.
        
        Parameters:
        ----------
        attention_weights : torch.Tensor
            Attention weight matrix, shape: (seq_len, seq_len)
        tokens : list of str
            List of token strings
        focus_token_idx : int
            Index of the token to focus on (0-indexed)
        title : str, optional
            Custom title for the plot
        save_path : str, optional
            If provided, save figure to this path
        """
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.cpu().numpy()
        
        # Extract attention weights for the focus token
        # This is row focus_token_idx of the attention matrix
        token_attention = attention_weights[focus_token_idx, :]
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create positions for bars
        positions = np.arange(len(tokens))
        
        # Create bars with different colors for the focus token
        colors = ['steelblue'] * len(tokens)
        colors[focus_token_idx] = 'coral'  # Highlight the focus token itself
        
        bars = ax.bar(positions, token_attention, color=colors, alpha=0.7, edgecolor='black')
        
        # Add value labels on top of bars
        for i, (pos, val) in enumerate(zip(positions, token_attention)):
            ax.text(pos, val + 0.01, f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Customize plot
        ax.set_xlabel('Tokens', fontsize=12, fontweight='bold')
        ax.set_ylabel('Attention Weight', fontsize=12, fontweight='bold')
        
        # Set title
        if title is None:
            title = f"Attention from '{tokens[focus_token_idx]}' to all tokens"
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Set x-axis labels
        ax.set_xticks(positions)
        ax.set_xticklabels(tokens, rotation=45, ha='right')
        
        # Add horizontal line at 1/seq_len (uniform attention baseline)
        uniform_attention = 1.0 / len(tokens)
        ax.axhline(uniform_attention, color='red', linestyle='--', 
                  label=f'Uniform attention: {uniform_attention:.3f}', linewidth=1.5)
        
        # Add grid
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add legend
        ax.legend(loc='upper right')
        
        # Set y-axis limits
        ax.set_ylim(0, max(token_attention) * 1.15)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        plt.show()


def create_synthetic_attention(seq_len: int, pattern: str = 'diagonal') -> torch.Tensor:
    """
    Create synthetic attention weights with specific patterns for testing/demonstration.
    
    This function is useful for:
    - Testing visualization code without running full models
    - Demonstrating different attention patterns
    - Educational purposes
    
    Parameters:
    ----------
    seq_len : int
        Length of the sequence
    pattern : str, default='diagonal'
        Type of pattern to create. Options:
        - 'diagonal': Strong self-attention
        - 'uniform': Equal attention to all tokens
        - 'local': Attention to nearby tokens
        - 'broadcast': One token receives all attention
        - 'random': Random attention pattern
    
    Returns:
    -------
    torch.Tensor
        Synthetic attention weight matrix, shape: (seq_len, seq_len)
        
    Examples:
    --------
    >>> # Create diagonal pattern (self-attention)
    >>> attn = create_synthetic_attention(5, 'diagonal')
    >>> print(attn)
    tensor([[0.7, 0.1, 0.1, 0.05, 0.05],
            [0.1, 0.7, 0.1, 0.05, 0.05],
            ...])
    """
    if pattern == 'diagonal':
        # Strong self-attention pattern
        # Tokens primarily attend to themselves
        attention = torch.eye(seq_len) * 0.7
        # Add small attention to other tokens
        attention += torch.rand(seq_len, seq_len) * 0.05
        
    elif pattern == 'uniform':
        # Uniform attention to all tokens
        attention = torch.ones(seq_len, seq_len) / seq_len
        
    elif pattern == 'local':
        # Local attention pattern (attending to neighbors)
        attention = torch.zeros(seq_len, seq_len)
        for i in range(seq_len):
            for j in range(seq_len):
                # Distance-based attention
                distance = abs(i - j)
                if distance == 0:
                    attention[i, j] = 0.5
                elif distance == 1:
                    attention[i, j] = 0.3
                elif distance == 2:
                    attention[i, j] = 0.1
                else:
                    attention[i, j] = 0.02
                    
    elif pattern == 'broadcast':
        # One token (position 1) receives attention from all others
        attention = torch.zeros(seq_len, seq_len)
        attention[:, 1] = 0.7  # All attend to token 1
        # Add self-attention
        attention += torch.eye(seq_len) * 0.2
        
    elif pattern == 'random':
        # Random attention pattern
        attention = torch.rand(seq_len, seq_len)
        
    else:
        raise ValueError(f"Unknown pattern: {pattern}")
    
    # Normalize rows to sum to 1 (softmax property)
    attention = attention / attention.sum(dim=1, keepdim=True)
    
    return attention


# ============================================================================
# Example Usage and Demonstrations
# ============================================================================

def example_basic_visualization():
    """
    Example 1: Basic attention heatmap visualization with synthetic data.
    
    This example demonstrates:
    - Creating synthetic attention weights
    - Basic heatmap visualization
    - Computing attention statistics
    """
    print("=" * 70)
    print("Example 1: Basic Attention Heatmap")
    print("=" * 70)
    
    # Define a sample sentence
    tokens = ["The", "cat", "sat", "on", "the", "mat"]
    seq_len = len(tokens)
    
    # Create synthetic attention weights with diagonal pattern
    attention = create_synthetic_attention(seq_len, pattern='diagonal')
    
    # Initialize visualizer
    viz = AttentionVisualizer(figsize=(10, 8))
    
    # Plot heatmap
    viz.plot_attention_heatmap(
        attention,
        tokens,
        title="Self-Attention Pattern (Diagonal)",
        show_values=True
    )
    
    # Compute and print statistics
    stats = viz.compute_attention_statistics(attention)
    print("\nAttention Statistics:")
    print("-" * 40)
    for key, value in stats.items():
        print(f"{key:20s}: {value:.4f}")


def example_pattern_comparison():
    """
    Example 2: Compare different attention patterns.
    
    This example demonstrates:
    - Multiple attention pattern types
    - Side-by-side comparison
    - Understanding different attention behaviors
    """
    print("\n" + "=" * 70)
    print("Example 2: Comparing Different Attention Patterns")
    print("=" * 70)
    
    tokens = ["The", "quick", "brown", "fox", "jumps"]
    seq_len = len(tokens)
    
    patterns = ['diagonal', 'uniform', 'local', 'broadcast']
    
    # Create visualizer
    viz = AttentionVisualizer(figsize=(8, 6))
    
    # Visualize each pattern
    for pattern in patterns:
        attention = create_synthetic_attention(seq_len, pattern=pattern)
        
        viz.plot_attention_heatmap(
            attention,
            tokens,
            title=f"Attention Pattern: {pattern.capitalize()}",
            show_values=False
        )
        
        # Show statistics
        stats = viz.compute_attention_statistics(attention)
        print(f"\n{pattern.upper()} Pattern Statistics:")
        print(f"  Mean: {stats['mean']:.4f}")
        print(f"  Entropy: {stats['mean_entropy']:.4f}")
        print(f"  Diagonal mean: {stats['diagonal_mean']:.4f}")


def example_token_focus():
    """
    Example 3: Focus on individual token attention.
    
    This example demonstrates:
    - Token-specific attention visualization
    - Understanding what each token attends to
    - Bar plot representation
    """
    print("\n" + "=" * 70)
    print("Example 3: Token-Specific Attention Analysis")
    print("=" * 70)
    
    tokens = ["I", "love", "deep", "learning", "very", "much"]
    seq_len = len(tokens)
    
    # Create attention with some structure
    attention = create_synthetic_attention(seq_len, pattern='local')
    
    # Initialize visualizer
    viz = AttentionVisualizer()
    
    # Focus on different tokens
    focus_tokens = [0, 2, 4]  # "I", "deep", "very"
    
    for idx in focus_tokens:
        viz.plot_token_attention(
            attention,
            tokens,
            focus_token_idx=idx,
            title=f"How '{tokens[idx]}' attends to other tokens"
        )


def example_attention_distribution():
    """
    Example 4: Analyze attention weight distributions.
    
    This example demonstrates:
    - Distribution analysis
    - Comparing self-attention vs attention to others
    - Statistical visualization
    """
    print("\n" + "=" * 70)
    print("Example 4: Attention Distribution Analysis")
    print("=" * 70)
    
    tokens = ["This", "is", "a", "sample", "sentence", "for", "analysis"]
    seq_len = len(tokens)
    
    # Create visualizer
    viz = AttentionVisualizer()
    
    # Compare different patterns
    patterns = ['diagonal', 'uniform', 'local']
    
    for pattern in patterns:
        attention = create_synthetic_attention(seq_len, pattern=pattern)
        
        viz.plot_attention_distribution(
            attention,
            title=f"Attention Distribution: {pattern.capitalize()} Pattern"
        )


if __name__ == "__main__":
    """
    Main execution block with multiple examples.
    
    Run this script to see all examples in action:
        python attention_basics.py
    
    Comment out examples you don't want to run.
    """
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("\n" + "=" * 70)
    print(" ATTENTION VISUALIZATION - BEGINNER LEVEL ")
    print(" Module 59: Understanding Attention Basics ")
    print("=" * 70)
    
    # Run all examples
    example_basic_visualization()
    example_pattern_comparison()
    example_token_focus()
    example_attention_distribution()
    
    print("\n" + "=" * 70)
    print(" All examples completed successfully! ")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  1. Attention weights show token-to-token relationships")
    print("  2. Heatmaps are the primary visualization tool")
    print("  3. Different patterns reveal different model behaviors")
    print("  4. Statistics help quantify attention characteristics")
    print("  5. Focus visualizations help understand individual tokens")
