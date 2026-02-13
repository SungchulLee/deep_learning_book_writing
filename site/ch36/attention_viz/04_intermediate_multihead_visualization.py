"""
Intermediate Level: Multi-Head Attention Visualization
======================================================

This module provides comprehensive tools for visualizing and analyzing multi-head
attention in transformer models. Different attention heads often learn to capture
different types of relationships, and this module helps understand that diversity.

Learning Goals:
--------------
1. Extract and visualize all attention heads simultaneously
2. Understand head specialization and diversity
3. Analyze head importance and redundancy
4. Compare heads across layers
5. Identify and prune less important heads

Mathematical Background:
-----------------------
Multi-head attention computes h parallel attention operations:
    
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
    
    where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)

Each head has its own learned projection matrices W_i^Q, W_i^K, W_i^V.
This allows the model to jointly attend to information from different
representation subspaces at different positions.

Author: Deep Learning Curriculum
Date: November 2025
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple, Dict
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
import warnings
warnings.filterwarnings('ignore')

try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class MultiHeadVisualizer:
    """
    Comprehensive multi-head attention visualizer.
    
    This class provides tools to:
    - Visualize all heads in a grid
    - Analyze head diversity and specialization
    - Compute head importance scores
    - Cluster similar heads
    - Compare heads across layers
    
    Attributes:
    ----------
    num_heads : int
        Number of attention heads
    figsize : tuple
        Default figure size
    """
    
    def __init__(self, num_heads: int = 12, figsize: Tuple[int, int] = (15, 10)):
        """
        Initialize multi-head visualizer.
        
        Parameters:
        ----------
        num_heads : int, default=12
            Expected number of attention heads (BERT-base has 12)
        figsize : tuple, default=(15, 10)
            Default figure size for plots
        """
        self.num_heads = num_heads
        self.figsize = figsize
    
    def plot_all_heads(self,
                      attention_heads: torch.Tensor,
                      tokens: List[str],
                      layer_name: str = "Layer",
                      save_path: Optional[str] = None,
                      show_values: bool = False) -> None:
        """
        Plot all attention heads in a grid layout.
        
        This provides an overview of all heads at once, making it easy to:
        - Compare different head behaviors
        - Identify similar heads
        - Spot degenerate or redundant heads
        - Understand head specialization
        
        Parameters:
        ----------
        attention_heads : torch.Tensor
            Attention from all heads, shape: (num_heads, seq_len, seq_len)
        tokens : list of str
            Token strings
        layer_name : str, default="Layer"
            Name/identifier for the layer
        save_path : str, optional
            Path to save figure
        show_values : bool, default=False
            Whether to show numerical values in cells
            
        Layout:
        ------
        Creates a grid where each subplot is one attention head.
        Optimal layout is calculated automatically based on number of heads.
        """
        if isinstance(attention_heads, torch.Tensor):
            attention_heads = attention_heads.cpu().numpy()
        
        num_heads = attention_heads.shape[0]
        
        # Calculate grid dimensions
        # Try to make it roughly square
        ncols = min(4, num_heads)  # Max 4 columns for readability
        nrows = (num_heads + ncols - 1) // ncols
        
        # Create figure
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes = axes.flatten() if num_heads > 1 else [axes]
        
        # Plot each head
        for head_idx in range(num_heads):
            ax = axes[head_idx]
            attention = attention_heads[head_idx]
            
            # Create heatmap
            im = ax.imshow(attention, cmap='viridis', aspect='auto', vmin=0, vmax=1)
            
            # Add colorbar to each subplot
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Attention', rotation=270, labelpad=15, fontsize=8)
            
            # Set title
            ax.set_title(f'Head {head_idx}', fontsize=10, fontweight='bold')
            
            # Show token labels only on left column and bottom row
            if head_idx % ncols == 0:  # Left column
                ax.set_yticks(range(len(tokens)))
                ax.set_yticklabels(tokens, fontsize=7)
                ax.set_ylabel('Query', fontsize=9)
            else:
                ax.set_yticks([])
            
            if head_idx >= num_heads - ncols:  # Bottom row
                ax.set_xticks(range(len(tokens)))
                ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=7)
                ax.set_xlabel('Key', fontsize=9)
            else:
                ax.set_xticks([])
            
            # Optionally show values
            if show_values and attention.shape[0] <= 10:  # Only for small matrices
                for i in range(attention.shape[0]):
                    for j in range(attention.shape[1]):
                        text = ax.text(j, i, f'{attention[i, j]:.2f}',
                                     ha="center", va="center", color="white",
                                     fontsize=6)
        
        # Hide extra subplots
        for idx in range(num_heads, len(axes)):
            axes[idx].axis('off')
        
        # Overall title
        fig.suptitle(f'Multi-Head Attention - {layer_name}', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"Saved to: {save_path}")
        
        plt.show()
    
    def compute_head_diversity(self, attention_heads: torch.Tensor) -> Dict[str, float]:
        """
        Compute metrics quantifying diversity among attention heads.
        
        Head diversity is important because:
        - High diversity: Heads capture different aspects
        - Low diversity: Redundancy, potential for pruning
        
        Metrics:
        -------
        1. Mean pairwise distance: Average dissimilarity between heads
        2. Entropy of head similarities: Distribution of similarities
        3. Effective number of heads: How many "unique" heads exist
        
        Parameters:
        ----------
        attention_heads : torch.Tensor
            Attention from all heads, shape: (num_heads, seq_len, seq_len)
        
        Returns:
        -------
        dict
            Dictionary of diversity metrics
        """
        if isinstance(attention_heads, torch.Tensor):
            attention_heads = attention_heads.cpu().numpy()
        
        num_heads = attention_heads.shape[0]
        
        # Flatten each head to vector
        heads_flat = attention_heads.reshape(num_heads, -1)
        
        # 1. Compute pairwise distances (Euclidean)
        # Shape: (num_heads, num_heads)
        distances = np.zeros((num_heads, num_heads))
        for i in range(num_heads):
            for j in range(i+1, num_heads):
                dist = np.linalg.norm(heads_flat[i] - heads_flat[j])
                distances[i, j] = dist
                distances[j, i] = dist
        
        mean_distance = np.mean(distances[np.triu_indices(num_heads, k=1)])
        std_distance = np.std(distances[np.triu_indices(num_heads, k=1)])
        
        # 2. Compute pairwise correlations
        correlations = np.corrcoef(heads_flat)
        mean_correlation = np.mean(correlations[np.triu_indices(num_heads, k=1)])
        
        # 3. Effective number of heads
        # Based on participation ratio from physics
        # If all heads are identical: effective_heads ≈ 1
        # If all heads are orthogonal: effective_heads ≈ num_heads
        
        # Normalize attention patterns
        heads_normalized = heads_flat / (np.linalg.norm(heads_flat, axis=1, keepdims=True) + 1e-10)
        
        # Compute overlap matrix
        overlap = heads_normalized @ heads_normalized.T
        eigenvalues = np.linalg.eigvalsh(overlap)
        eigenvalues = np.maximum(eigenvalues, 0)  # Remove small negative values
        
        # Participation ratio
        effective_heads = (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)
        
        # 4. Entropy of correlation distribution
        # Measures how spread out the correlations are
        corr_flat = correlations[np.triu_indices(num_heads, k=1)]
        # Create histogram
        hist, _ = np.histogram(corr_flat, bins=20, range=(0, 1), density=True)
        hist = hist / np.sum(hist)  # Normalize
        hist = hist[hist > 0]  # Remove zeros
        entropy = -np.sum(hist * np.log(hist))
        
        diversity_metrics = {
            'mean_pairwise_distance': float(mean_distance),
            'std_pairwise_distance': float(std_distance),
            'mean_pairwise_correlation': float(mean_correlation),
            'effective_num_heads': float(effective_heads),
            'correlation_entropy': float(entropy),
        }
        
        return diversity_metrics
    
    def plot_head_similarity_matrix(self,
                                   attention_heads: torch.Tensor,
                                   layer_name: str = "Layer",
                                   metric: str = 'correlation',
                                   save_path: Optional[str] = None) -> None:
        """
        Plot similarity matrix showing relationships between heads.
        
        This visualization helps identify:
        - Which heads are similar (potential redundancy)
        - Which heads are unique (important for diversity)
        - Clusters of similar heads
        
        Parameters:
        ----------
        attention_heads : torch.Tensor
            Attention from all heads, shape: (num_heads, seq_len, seq_len)
        layer_name : str, default="Layer"
            Layer identifier
        metric : str, default='correlation'
            Similarity metric ('correlation' or 'distance')
        save_path : str, optional
            Path to save figure
        """
        if isinstance(attention_heads, torch.Tensor):
            attention_heads = attention_heads.cpu().numpy()
        
        num_heads = attention_heads.shape[0]
        heads_flat = attention_heads.reshape(num_heads, -1)
        
        # Compute similarity matrix
        if metric == 'correlation':
            similarity = np.corrcoef(heads_flat)
            cmap = 'RdBu_r'
            vmin, vmax = -1, 1
            label = 'Correlation'
        else:  # distance
            # Compute pairwise distances and convert to similarity
            from sklearn.metrics import pairwise_distances
            distances = pairwise_distances(heads_flat, metric='euclidean')
            # Convert to similarity (closer = more similar)
            similarity = 1 / (1 + distances)
            cmap = 'viridis'
            vmin, vmax = 0, 1
            label = 'Similarity'
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot heatmap
        im = ax.imshow(similarity, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(label, rotation=270, labelpad=20, fontsize=12)
        
        # Set ticks and labels
        ax.set_xticks(range(num_heads))
        ax.set_yticks(range(num_heads))
        ax.set_xticklabels([f'H{i}' for i in range(num_heads)])
        ax.set_yticklabels([f'H{i}' for i in range(num_heads)])
        
        # Labels
        ax.set_xlabel('Attention Head', fontsize=12, fontweight='bold')
        ax.set_ylabel('Attention Head', fontsize=12, fontweight='bold')
        ax.set_title(f'Head Similarity Matrix - {layer_name}\n({metric.capitalize()})',
                    fontsize=14, fontweight='bold', pad=20)
        
        # Add text annotations for correlations
        for i in range(num_heads):
            for j in range(num_heads):
                if i != j and abs(similarity[i, j]) > 0.7:  # Highlight strong relationships
                    text = ax.text(j, i, f'{similarity[i, j]:.2f}',
                                 ha="center", va="center", 
                                 color="yellow" if similarity[i, j] > 0 else "black",
                                 fontsize=8, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"Saved to: {save_path}")
        
        plt.show()
    
    def cluster_heads(self,
                     attention_heads: torch.Tensor,
                     layer_name: str = "Layer",
                     method: str = 'ward',
                     save_path: Optional[str] = None) -> np.ndarray:
        """
        Perform hierarchical clustering on attention heads.
        
        Clustering helps:
        - Identify groups of similar heads
        - Understand head organization
        - Guide model pruning decisions
        
        Parameters:
        ----------
        attention_heads : torch.Tensor
            Attention from all heads, shape: (num_heads, seq_len, seq_len)
        layer_name : str, default="Layer"
            Layer identifier
        method : str, default='ward'
            Linkage method for hierarchical clustering
        save_path : str, optional
            Path to save figure
        
        Returns:
        -------
        np.ndarray
            Linkage matrix from hierarchical clustering
        """
        if isinstance(attention_heads, torch.Tensor):
            attention_heads = attention_heads.cpu().numpy()
        
        num_heads = attention_heads.shape[0]
        heads_flat = attention_heads.reshape(num_heads, -1)
        
        # Perform hierarchical clustering
        linkage_matrix = linkage(heads_flat, method=method, metric='euclidean')
        
        # Create dendrogram
        fig, ax = plt.subplots(figsize=(12, 6))
        
        dendrogram(
            linkage_matrix,
            labels=[f'Head {i}' for i in range(num_heads)],
            ax=ax,
            color_threshold=0.7 * max(linkage_matrix[:, 2]),
        )
        
        ax.set_xlabel('Attention Head', fontsize=12, fontweight='bold')
        ax.set_ylabel('Distance', fontsize=12, fontweight='bold')
        ax.set_title(f'Hierarchical Clustering of Attention Heads - {layer_name}',
                    fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"Saved to: {save_path}")
        
        plt.show()
        
        return linkage_matrix
    
    def compute_head_importance(self,
                               attention_heads: torch.Tensor,
                               method: str = 'entropy') -> np.ndarray:
        """
        Compute importance score for each attention head.
        
        Head importance helps:
        - Identify which heads contribute most to model behavior
        - Guide model pruning/compression
        - Understand model efficiency
        
        Methods:
        -------
        1. 'entropy': Heads with lower entropy are more focused (more important)
        2. 'max_attention': Heads with higher max attention are more decisive
        3. 'variance': Heads with higher variance show more discrimination
        
        Parameters:
        ----------
        attention_heads : torch.Tensor
            Attention from all heads, shape: (num_heads, seq_len, seq_len)
        method : str, default='entropy'
            Importance metric to use
        
        Returns:
        -------
        np.ndarray
            Importance scores for each head, shape: (num_heads,)
        """
        if isinstance(attention_heads, torch.Tensor):
            attention_heads = attention_heads.cpu().numpy()
        
        num_heads = attention_heads.shape[0]
        importance_scores = np.zeros(num_heads)
        
        for head_idx in range(num_heads):
            attention = attention_heads[head_idx]
            
            if method == 'entropy':
                # Lower entropy = more focused = more important
                # We'll use negative entropy so higher is more important
                entropies = []
                for row in attention:
                    row_safe = row + 1e-10
                    entropy = -np.sum(row_safe * np.log(row_safe))
                    entropies.append(entropy)
                # Negative mean entropy (higher = more focused)
                importance_scores[head_idx] = -np.mean(entropies)
                
            elif method == 'max_attention':
                # Higher max attention = more decisive = more important
                importance_scores[head_idx] = np.max(attention)
                
            elif method == 'variance':
                # Higher variance = more discrimination = more important
                importance_scores[head_idx] = np.var(attention)
            
            else:
                raise ValueError(f"Unknown method: {method}")
        
        # Normalize to [0, 1]
        importance_scores = (importance_scores - importance_scores.min()) / \
                          (importance_scores.max() - importance_scores.min() + 1e-10)
        
        return importance_scores
    
    def plot_head_importance(self,
                            attention_heads: torch.Tensor,
                            layer_name: str = "Layer",
                            method: str = 'entropy',
                            save_path: Optional[str] = None) -> None:
        """
        Visualize importance scores for all heads.
        
        Parameters:
        ----------
        attention_heads : torch.Tensor
            Attention from all heads
        layer_name : str, default="Layer"
            Layer identifier
        method : str, default='entropy'
            Importance metric
        save_path : str, optional
            Path to save figure
        """
        importance = self.compute_head_importance(attention_heads, method=method)
        num_heads = len(importance)
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        positions = np.arange(num_heads)
        colors = plt.cm.viridis(importance)
        
        bars = ax.bar(positions, importance, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels
        for pos, val in zip(positions, importance):
            ax.text(pos, val + 0.02, f'{val:.3f}', 
                   ha='center', va='bottom', fontsize=9)
        
        # Customize
        ax.set_xlabel('Attention Head', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'Importance Score ({method})', fontsize=12, fontweight='bold')
        ax.set_title(f'Head Importance - {layer_name}',
                    fontsize=14, fontweight='bold', pad=20)
        
        ax.set_xticks(positions)
        ax.set_xticklabels([f'H{i}' for i in range(num_heads)])
        
        ax.set_ylim(0, max(importance) * 1.15)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add horizontal line at mean
        mean_importance = np.mean(importance)
        ax.axhline(mean_importance, color='red', linestyle='--', 
                  label=f'Mean: {mean_importance:.3f}', linewidth=2)
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"Saved to: {save_path}")
        
        plt.show()


# ============================================================================
# Example Usage and Demonstrations
# ============================================================================

def example_basic_multihead():
    """Example 1: Basic multi-head visualization with synthetic data."""
    print("=" * 70)
    print("Example 1: Multi-Head Attention Overview")
    print("=" * 70)
    
    # Create synthetic multi-head attention
    num_heads = 8
    seq_len = 6
    tokens = ["The", "cat", "sat", "on", "the", "mat"]
    
    # Generate different patterns for different heads
    attention_heads = torch.zeros(num_heads, seq_len, seq_len)
    
    patterns = ['diagonal', 'local', 'broadcast', 'uniform', 
               'diagonal', 'local', 'broadcast', 'uniform']
    
    for head_idx, pattern in enumerate(patterns):
        if pattern == 'diagonal':
            attn = torch.eye(seq_len) * 0.7 + torch.rand(seq_len, seq_len) * 0.05
        elif pattern == 'local':
            attn = torch.zeros(seq_len, seq_len)
            for i in range(seq_len):
                for j in range(max(0, i-1), min(seq_len, i+2)):
                    attn[i, j] = 0.4 if i == j else 0.3
        elif pattern == 'broadcast':
            attn = torch.zeros(seq_len, seq_len)
            attn[:, 1] = 0.6
            attn += torch.eye(seq_len) * 0.2
        else:  # uniform
            attn = torch.ones(seq_len, seq_len) / seq_len
        
        attn = attn / attn.sum(dim=1, keepdim=True)
        attention_heads[head_idx] = attn
    
    # Visualize
    viz = MultiHeadVisualizer(num_heads=num_heads)
    viz.plot_all_heads(attention_heads, tokens, layer_name="Synthetic Layer")


def example_head_diversity():
    """Example 2: Analyze head diversity."""
    print("\n" + "=" * 70)
    print("Example 2: Head Diversity Analysis")
    print("=" * 70)
    
    # Create diverse vs redundant attention patterns
    num_heads = 12
    seq_len = 8
    
    # Scenario 1: Diverse heads
    diverse_heads = torch.rand(num_heads, seq_len, seq_len)
    diverse_heads = diverse_heads / diverse_heads.sum(dim=2, keepdim=True)
    
    # Scenario 2: Redundant heads (all similar)
    base_pattern = torch.rand(seq_len, seq_len)
    base_pattern = base_pattern / base_pattern.sum(dim=1, keepdim=True)
    redundant_heads = base_pattern.unsqueeze(0).repeat(num_heads, 1, 1)
    redundant_heads += torch.rand(num_heads, seq_len, seq_len) * 0.05  # Small noise
    redundant_heads = redundant_heads / redundant_heads.sum(dim=2, keepdim=True)
    
    # Analyze diversity
    viz = MultiHeadVisualizer(num_heads=num_heads)
    
    print("\nDiverse Heads:")
    diverse_metrics = viz.compute_head_diversity(diverse_heads)
    for key, value in diverse_metrics.items():
        print(f"  {key:30s}: {value:.4f}")
    
    print("\nRedundant Heads:")
    redundant_metrics = viz.compute_head_diversity(redundant_heads)
    for key, value in redundant_metrics.items():
        print(f"  {key:30s}: {value:.4f}")
    
    print("\nKey Observations:")
    print(f"  - Diverse effective heads: {diverse_metrics['effective_num_heads']:.2f}")
    print(f"  - Redundant effective heads: {redundant_metrics['effective_num_heads']:.2f}")
    print(f"  - Diverse correlation: {diverse_metrics['mean_pairwise_correlation']:.3f}")
    print(f"  - Redundant correlation: {redundant_metrics['mean_pairwise_correlation']:.3f}")


def example_similarity_and_clustering():
    """Example 3: Head similarity matrix and clustering."""
    print("\n" + "=" * 70)
    print("Example 3: Head Similarity and Clustering")
    print("=" * 70)
    
    # Create heads with some clusters
    num_heads = 12
    seq_len = 8
    
    attention_heads = torch.zeros(num_heads, seq_len, seq_len)
    
    # Create 3 clusters of 4 heads each
    for cluster in range(3):
        base_pattern = torch.rand(seq_len, seq_len)
        base_pattern = base_pattern / base_pattern.sum(dim=1, keepdim=True)
        
        for i in range(4):
            head_idx = cluster * 4 + i
            # Add small variations
            attention_heads[head_idx] = base_pattern + torch.rand(seq_len, seq_len) * 0.1
            attention_heads[head_idx] = attention_heads[head_idx] / \
                                       attention_heads[head_idx].sum(dim=1, keepdim=True)
    
    viz = MultiHeadVisualizer(num_heads=num_heads)
    
    # Plot similarity matrix
    viz.plot_head_similarity_matrix(attention_heads, layer_name="Clustered Layer")
    
    # Perform clustering
    viz.cluster_heads(attention_heads, layer_name="Clustered Layer")


def example_head_importance():
    """Example 4: Compute and visualize head importance."""
    print("\n" + "=" * 70)
    print("Example 4: Head Importance Ranking")
    print("=" * 70)
    
    num_heads = 12
    seq_len = 8
    
    # Create heads with varying "quality"
    attention_heads = torch.zeros(num_heads, seq_len, seq_len)
    
    for head_idx in range(num_heads):
        if head_idx < 3:  # First 3: focused (important)
            attn = torch.eye(seq_len) * 0.8 + torch.rand(seq_len, seq_len) * 0.05
        elif head_idx < 6:  # Next 3: moderately focused
            attn = torch.eye(seq_len) * 0.5 + torch.rand(seq_len, seq_len) * 0.1
        elif head_idx < 9:  # Next 3: diffuse
            attn = torch.rand(seq_len, seq_len)
        else:  # Last 3: very diffuse (less important)
            attn = torch.ones(seq_len, seq_len) / seq_len
            attn += torch.rand(seq_len, seq_len) * 0.05
        
        attn = attn / attn.sum(dim=1, keepdim=True)
        attention_heads[head_idx] = attn
    
    viz = MultiHeadVisualizer(num_heads=num_heads)
    
    # Plot importance for different methods
    for method in ['entropy', 'max_attention', 'variance']:
        print(f"\nImportance by {method}:")
        viz.plot_head_importance(attention_heads, 
                               layer_name="Varying Quality Layer",
                               method=method)


if __name__ == "__main__":
    """
    Main execution demonstrating multi-head attention analysis.
    
    Run this script to learn about multi-head attention:
        python multihead_visualization.py
    """
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("\n" + "=" * 70)
    print(" MULTI-HEAD ATTENTION VISUALIZATION ")
    print(" Module 59: Analyzing Head Diversity and Specialization ")
    print("=" * 70)
    
    # Run examples
    example_basic_multihead()
    example_head_diversity()
    example_similarity_and_clustering()
    example_head_importance()
    
    print("\n" + "=" * 70)
    print(" Examples completed! ")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  1. Different heads learn different patterns")
    print("  2. Head diversity is important for model capacity")
    print("  3. Redundant heads can be pruned for efficiency")
    print("  4. Head importance guides compression decisions")
    print("  5. Clustering reveals head organization")
