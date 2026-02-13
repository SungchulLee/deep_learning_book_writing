"""
Beginner Level: Common Attention Patterns
==========================================

This module teaches students to recognize and understand common attention patterns
that emerge in transformer models. Understanding these patterns is crucial for
model interpretation and debugging.

Learning Goals:
--------------
1. Identify common attention patterns (diagonal, broadcast, block, etc.)
2. Understand what different patterns mean
3. Detect problematic patterns (attention collapse, degenerate patterns)
4. Quantify pattern characteristics programmatically
5. Compare patterns across models and tasks

Pattern Types Covered:
--------------------
1. **Diagonal Pattern**: Strong self-attention
2. **Vertical Stripes**: Broadcasting pattern
3. **Block Pattern**: Segment-based attention
4. **Beginning-of-Sequence (BOS)**: Focus on first token
5. **Local Pattern**: Neighboring token attention
6. **Attention Collapse**: Degenerate pattern (problem!)

Author: Deep Learning Curriculum
Date: November 2025
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional
from scipy import stats
from sklearn.metrics import pairwise_distances
import warnings
warnings.filterwarnings('ignore')


class AttentionPatternAnalyzer:
    """
    Analyzer for identifying and characterizing attention patterns.
    
    This class provides tools to:
    - Detect specific attention patterns
    - Quantify pattern characteristics
    - Compare patterns across heads/layers
    - Identify problematic patterns
    
    Attributes:
    ----------
    pattern_thresholds : dict
        Thresholds for pattern detection
    """
    
    def __init__(self):
        """Initialize the pattern analyzer with default thresholds."""
        # Thresholds for pattern detection
        # These can be tuned based on your specific use case
        self.pattern_thresholds = {
            'diagonal_strength': 0.5,      # Minimum diagonal mean for diagonal pattern
            'broadcast_concentration': 0.6, # Min max column sum for broadcast
            'local_bandwidth': 3,           # Window size for local pattern
            'bos_threshold': 0.4,          # Min attention to first token
            'collapse_threshold': 0.95,     # Similarity threshold for collapse
        }
    
    def identify_pattern_type(self, attention: torch.Tensor) -> str:
        """
        Automatically identify the predominant attention pattern.
        
        This function analyzes the attention matrix and returns the most
        prominent pattern type. It checks patterns in order of specificity.
        
        Parameters:
        ----------
        attention : torch.Tensor
            Attention weight matrix, shape: (seq_len, seq_len)
        
        Returns:
        -------
        str
            Name of the detected pattern
            
        Pattern Detection Logic:
        -----------------------
        1. Check for attention collapse (most specific/problematic)
        2. Check for diagonal pattern (self-attention)
        3. Check for broadcast pattern (vertical stripes)
        4. Check for BOS pattern (first token attention)
        5. Check for local pattern (neighboring tokens)
        6. Check for block pattern (segmented attention)
        7. Default to "mixed" if no clear pattern
        """
        if isinstance(attention, torch.Tensor):
            attention = attention.cpu().numpy()
        
        # 1. Check for attention collapse
        # This happens when all heads/rows attend to the same tokens
        # It's a sign of poor training or model issues
        if self._detect_attention_collapse(attention):
            return "attention_collapse"
        
        # 2. Check for diagonal pattern
        # High values on the diagonal indicate strong self-attention
        if self._detect_diagonal_pattern(attention):
            return "diagonal"
        
        # 3. Check for broadcast pattern
        # One or few columns receive most attention
        if self._detect_broadcast_pattern(attention):
            return "broadcast"
        
        # 4. Check for BOS (beginning-of-sequence) pattern
        # All tokens attend strongly to the first token
        if self._detect_bos_pattern(attention):
            return "beginning_of_sequence"
        
        # 5. Check for local pattern
        # Attention concentrated around the diagonal (neighboring tokens)
        if self._detect_local_pattern(attention):
            return "local"
        
        # 6. Check for block pattern
        # Attention within segments/blocks
        if self._detect_block_pattern(attention):
            return "block"
        
        # Default: Mixed or unclear pattern
        return "mixed"
    
    def _detect_diagonal_pattern(self, attention: np.ndarray) -> bool:
        """
        Detect if attention follows a diagonal pattern.
        
        Diagonal pattern characteristics:
        - High values on the diagonal (tokens attend to themselves)
        - Diagonal mean significantly higher than off-diagonal mean
        
        Mathematical Check:
        ------------------
        mean(diagonal) > threshold AND
        mean(diagonal) > 2 * mean(off_diagonal)
        """
        # Extract diagonal
        diagonal = np.diag(attention)
        diagonal_mean = np.mean(diagonal)
        
        # Extract off-diagonal
        seq_len = attention.shape[0]
        mask = ~np.eye(seq_len, dtype=bool)
        off_diagonal = attention[mask]
        off_diagonal_mean = np.mean(off_diagonal)
        
        # Check thresholds
        is_diagonal = (
            diagonal_mean > self.pattern_thresholds['diagonal_strength'] and
            diagonal_mean > 2 * off_diagonal_mean
        )
        
        return is_diagonal
    
    def _detect_broadcast_pattern(self, attention: np.ndarray) -> bool:
        """
        Detect if attention follows a broadcast pattern.
        
        Broadcast pattern characteristics:
        - One or few columns have very high values (vertical stripes)
        - These columns receive attention from most/all query tokens
        
        Mathematical Check:
        ------------------
        max(column_sums) > threshold * seq_len
        (One column receives > threshold fraction of total attention)
        """
        # Sum each column (how much attention each key receives)
        column_sums = np.sum(attention, axis=0)
        max_column_sum = np.max(column_sums)
        
        seq_len = attention.shape[0]
        
        # In a broadcast pattern, one column gets attention from most tokens
        # Since each row sums to 1, and we have seq_len rows,
        # perfect broadcast would give one column a sum of seq_len
        threshold = self.pattern_thresholds['broadcast_concentration']
        
        is_broadcast = max_column_sum > threshold * seq_len
        
        return is_broadcast
    
    def _detect_bos_pattern(self, attention: np.ndarray) -> bool:
        """
        Detect if attention focuses on beginning-of-sequence.
        
        BOS pattern characteristics:
        - First column has high values
        - All or most tokens attend to the first token
        - Common in BERT due to [CLS] token
        
        Mathematical Check:
        ------------------
        mean(attention[:, 0]) > threshold
        """
        # Get first column (attention to first token)
        first_column = attention[:, 0]
        first_column_mean = np.mean(first_column)
        
        is_bos = first_column_mean > self.pattern_thresholds['bos_threshold']
        
        return is_bos
    
    def _detect_local_pattern(self, attention: np.ndarray) -> bool:
        """
        Detect if attention is local (focuses on nearby tokens).
        
        Local pattern characteristics:
        - High values near the diagonal (band pattern)
        - Tokens attend to immediate neighbors
        - Values decay with distance from diagonal
        
        Mathematical Check:
        ------------------
        attention within bandwidth > threshold * total_attention
        """
        seq_len = attention.shape[0]
        bandwidth = self.pattern_thresholds['local_bandwidth']
        
        # Create a band mask around diagonal
        band_mask = np.abs(np.arange(seq_len)[:, None] - np.arange(seq_len)) <= bandwidth
        
        # Calculate attention within band
        band_attention = np.sum(attention[band_mask])
        total_attention = np.sum(attention)
        
        # In local pattern, most attention is within the band
        local_ratio = band_attention / total_attention
        
        is_local = local_ratio > 0.7  # 70% of attention within band
        
        return is_local
    
    def _detect_block_pattern(self, attention: np.ndarray) -> bool:
        """
        Detect if attention has a block structure.
        
        Block pattern characteristics:
        - Attention is high within blocks (segments)
        - Low attention between blocks
        - Useful for modeling structured text (paragraphs, sentences)
        
        Mathematical Check:
        ------------------
        Uses hierarchical clustering to detect block structure
        """
        seq_len = attention.shape[0]
        
        # Need sufficient length to detect blocks
        if seq_len < 6:
            return False
        
        # Compute dissimilarity matrix
        # Similar rows indicate block structure
        dissimilarity = pairwise_distances(attention, metric='euclidean')
        
        # Check if there's clear block structure
        # This is a simplified check - more sophisticated methods exist
        # We look for off-diagonal blocks with low dissimilarity
        
        # Divide into potential blocks and check internal vs external similarity
        mid = seq_len // 2
        
        # Internal dissimilarity (within blocks)
        block1_dissim = np.mean(dissimilarity[:mid, :mid])
        block2_dissim = np.mean(dissimilarity[mid:, mid:])
        internal_dissim = (block1_dissim + block2_dissim) / 2
        
        # External dissimilarity (between blocks)
        external_dissim = np.mean(dissimilarity[:mid, mid:])
        
        # Block pattern exists if internal dissimilarity << external dissimilarity
        is_block = external_dissim > 2 * internal_dissim
        
        return is_block
    
    def _detect_attention_collapse(self, attention: np.ndarray) -> bool:
        """
        Detect attention collapse (degenerate pattern).
        
        Attention collapse characteristics:
        - All rows are very similar (all tokens attend the same way)
        - Loss of diversity in attention patterns
        - Often indicates training issues
        
        Mathematical Check:
        ------------------
        pairwise_correlation(rows) > threshold for most pairs
        """
        # Compute correlation between all pairs of rows
        correlations = np.corrcoef(attention)
        
        # Get upper triangle (excluding diagonal)
        upper_tri = correlations[np.triu_indices_from(correlations, k=1)]
        
        # Check if most correlations are very high
        high_correlation_ratio = np.mean(upper_tri > self.pattern_thresholds['collapse_threshold'])
        
        # Collapse if > 80% of row pairs have correlation > threshold
        is_collapsed = high_correlation_ratio > 0.8
        
        return is_collapsed
    
    def compute_pattern_metrics(self, attention: torch.Tensor) -> Dict[str, float]:
        """
        Compute comprehensive metrics for attention pattern characterization.
        
        These metrics provide quantitative measures of pattern properties
        and can be used for:
        - Comparing attention across layers/heads
        - Tracking attention during training
        - Identifying problematic patterns
        
        Parameters:
        ----------
        attention : torch.Tensor
            Attention weight matrix, shape: (seq_len, seq_len)
        
        Returns:
        -------
        dict
            Dictionary of pattern metrics
        """
        if isinstance(attention, torch.Tensor):
            attention = attention.cpu().numpy()
        
        seq_len = attention.shape[0]
        
        metrics = {}
        
        # 1. Diagonal strength
        # How much tokens attend to themselves
        diagonal = np.diag(attention)
        metrics['diagonal_mean'] = float(np.mean(diagonal))
        metrics['diagonal_std'] = float(np.std(diagonal))
        
        # 2. Sparsity
        # How concentrated is the attention
        # Low sparsity = uniform attention, High sparsity = focused attention
        # Using Gini coefficient as sparsity measure
        attention_flat = attention.flatten()
        attention_sorted = np.sort(attention_flat)
        n = len(attention_sorted)
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * attention_sorted)) / (n * np.sum(attention_sorted)) - (n + 1) / n
        metrics['sparsity_gini'] = float(gini)
        
        # 3. Entropy
        # Measure of attention distribution
        # Low entropy = focused, High entropy = diffuse
        row_entropies = []
        for row in attention:
            row_safe = row + 1e-10
            entropy = -np.sum(row_safe * np.log(row_safe))
            row_entropies.append(entropy)
        metrics['mean_entropy'] = float(np.mean(row_entropies))
        metrics['std_entropy'] = float(np.std(row_entropies))
        
        # Normalize entropy by max possible (log(seq_len))
        max_entropy = np.log(seq_len)
        metrics['normalized_entropy'] = float(np.mean(row_entropies) / max_entropy)
        
        # 4. Bandwidth
        # Average distance of attention from diagonal
        # Small bandwidth = local attention
        distances = []
        for i in range(seq_len):
            for j in range(seq_len):
                distance = abs(i - j)
                # Weight by attention strength
                distances.extend([distance] * int(attention[i, j] * 1000))
        metrics['mean_bandwidth'] = float(np.mean(distances))
        
        # 5. Max attention
        # Strongest single attention connection
        metrics['max_attention'] = float(np.max(attention))
        metrics['min_attention'] = float(np.min(attention))
        
        # 6. First token attention
        # How much attention goes to first token (BOS)
        metrics['first_token_attention'] = float(np.mean(attention[:, 0]))
        
        # 7. Last token attention
        # How much attention goes to last token (EOS)
        metrics['last_token_attention'] = float(np.mean(attention[:, -1]))
        
        # 8. Row similarity
        # How similar are attention patterns across tokens
        # High similarity might indicate collapse
        correlations = np.corrcoef(attention)
        upper_tri = correlations[np.triu_indices_from(correlations, k=1)]
        metrics['mean_row_correlation'] = float(np.mean(upper_tri))
        metrics['std_row_correlation'] = float(np.std(upper_tri))
        
        # 9. Column concentration
        # How concentrated is attention on specific keys
        column_sums = np.sum(attention, axis=0)
        metrics['max_column_sum'] = float(np.max(column_sums))
        metrics['std_column_sum'] = float(np.std(column_sums))
        
        return metrics
    
    def visualize_pattern_comparison(self,
                                    attentions: List[torch.Tensor],
                                    labels: List[str],
                                    tokens: List[str],
                                    save_path: Optional[str] = None) -> None:
        """
        Visualize and compare multiple attention patterns side by side.
        
        Parameters:
        ----------
        attentions : list of torch.Tensor
            List of attention matrices to compare
        labels : list of str
            Labels for each attention matrix
        tokens : list of str
            Token labels
        save_path : str, optional
            Path to save figure
        """
        num_patterns = len(attentions)
        
        fig, axes = plt.subplots(1, num_patterns, figsize=(6 * num_patterns, 5))
        if num_patterns == 1:
            axes = [axes]
        
        for idx, (attention, label) in enumerate(zip(attentions, labels)):
            if isinstance(attention, torch.Tensor):
                attention = attention.cpu().numpy()
            
            # Detect pattern type
            pattern_type = self.identify_pattern_type(torch.tensor(attention))
            
            # Create heatmap
            sns.heatmap(
                attention,
                xticklabels=tokens if idx == 0 else [],
                yticklabels=tokens,
                cmap='viridis',
                square=True,
                cbar=True,
                ax=axes[idx],
                vmin=0,
                vmax=1,
                cbar_kws={'label': 'Attention'}
            )
            
            # Title with detected pattern
            axes[idx].set_title(
                f'{label}\nDetected: {pattern_type}',
                fontsize=11,
                fontweight='bold'
            )
            
            if idx == 0:
                axes[idx].set_ylabel('Query Tokens', fontsize=10)
        
        plt.suptitle('Attention Pattern Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        
        plt.show()
    
    def plot_pattern_metrics_radar(self,
                                  attention: torch.Tensor,
                                  title: str = "Pattern Characteristics",
                                  save_path: Optional[str] = None) -> None:
        """
        Create a radar plot showing pattern characteristics.
        
        Radar plots are useful for:
        - Visualizing multiple metrics simultaneously
        - Comparing pattern profiles
        - Quick pattern identification
        
        Parameters:
        ----------
        attention : torch.Tensor
            Attention matrix to analyze
        title : str
            Plot title
        save_path : str, optional
            Path to save figure
        """
        # Compute metrics
        metrics = self.compute_pattern_metrics(attention)
        
        # Select key metrics for radar plot (normalized to 0-1)
        selected_metrics = {
            'Diagonal\nStrength': metrics['diagonal_mean'],
            'Sparsity': metrics['sparsity_gini'],
            'Normalized\nEntropy': metrics['normalized_entropy'],
            'First Token\nAttention': metrics['first_token_attention'],
            'Max Column\nConcentration': metrics['max_column_sum'] / attention.shape[0],
            'Row\nSimilarity': metrics['mean_row_correlation'],
        }
        
        # Prepare data for radar plot
        categories = list(selected_metrics.keys())
        values = list(selected_metrics.values())
        
        # Number of variables
        num_vars = len(categories)
        
        # Compute angle for each axis
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        
        # Close the plot
        values += values[:1]
        angles += angles[:1]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        
        # Plot data
        ax.plot(angles, values, 'o-', linewidth=2, label='Pattern', color='steelblue')
        ax.fill(angles, values, alpha=0.25, color='steelblue')
        
        # Fix axis to go from 0 to 1
        ax.set_ylim(0, 1)
        
        # Add labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=10)
        
        # Add title
        ax.set_title(title, size=14, fontweight='bold', pad=20)
        
        # Add grid
        ax.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        
        plt.show()


def create_pattern_library() -> Dict[str, torch.Tensor]:
    """
    Create a library of canonical attention patterns for reference.
    
    This function generates synthetic examples of each major pattern type.
    Useful for:
    - Teaching and demonstration
    - Testing pattern detection algorithms
    - Understanding pattern characteristics
    
    Returns:
    -------
    dict
        Dictionary mapping pattern names to attention matrices
    """
    seq_len = 8
    patterns = {}
    
    # 1. Diagonal Pattern
    diagonal = torch.eye(seq_len) * 0.7
    diagonal += torch.rand(seq_len, seq_len) * 0.05
    diagonal = diagonal / diagonal.sum(dim=1, keepdim=True)
    patterns['diagonal'] = diagonal
    
    # 2. Uniform Pattern
    uniform = torch.ones(seq_len, seq_len) / seq_len
    patterns['uniform'] = uniform
    
    # 3. Local Pattern
    local = torch.zeros(seq_len, seq_len)
    for i in range(seq_len):
        for j in range(seq_len):
            distance = abs(i - j)
            if distance == 0:
                local[i, j] = 0.5
            elif distance == 1:
                local[i, j] = 0.3
            elif distance == 2:
                local[i, j] = 0.1
            else:
                local[i, j] = 0.02
    local = local / local.sum(dim=1, keepdim=True)
    patterns['local'] = local
    
    # 4. Broadcast Pattern
    broadcast = torch.zeros(seq_len, seq_len)
    broadcast[:, 2] = 0.7  # All attend to position 2
    broadcast += torch.eye(seq_len) * 0.2
    broadcast = broadcast / broadcast.sum(dim=1, keepdim=True)
    patterns['broadcast'] = broadcast
    
    # 5. BOS Pattern
    bos = torch.zeros(seq_len, seq_len)
    for i in range(seq_len):
        bos[i, 0] = 0.7 - i * 0.05  # Decreasing attention to first token
        bos[i, i] = 0.2
    bos = bos / bos.sum(dim=1, keepdim=True)
    patterns['bos'] = bos
    
    # 6. Block Pattern
    block = torch.zeros(seq_len, seq_len)
    mid = seq_len // 2
    block[:mid, :mid] = 0.8 / mid
    block[mid:, mid:] = 0.8 / (seq_len - mid)
    for i in range(seq_len):
        block[i, i] += 0.1
    block = block / block.sum(dim=1, keepdim=True)
    patterns['block'] = block
    
    # 7. Attention Collapse (problematic)
    collapse = torch.ones(seq_len, seq_len) / seq_len
    collapse += torch.rand(seq_len, seq_len) * 0.01  # Tiny random noise
    collapse = collapse / collapse.sum(dim=1, keepdim=True)
    patterns['collapse'] = collapse
    
    return patterns


# ============================================================================
# Examples and Demonstrations
# ============================================================================

def example_pattern_detection():
    """
    Example 1: Automatic pattern detection.
    
    Demonstrates:
    - Creating different pattern types
    - Automatic pattern identification
    - Pattern visualization
    """
    print("=" * 70)
    print("Example 1: Automatic Pattern Detection")
    print("=" * 70)
    
    # Create pattern library
    patterns = create_pattern_library()
    
    # Initialize analyzer
    analyzer = AttentionPatternAnalyzer()
    
    tokens = ["tok" + str(i) for i in range(8)]
    
    # Test pattern detection
    for name, pattern in patterns.items():
        detected = analyzer.identify_pattern_type(pattern)
        
        print(f"\nPattern: {name:15s} | Detected: {detected}")
        
        # Visualize
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(
            pattern.numpy(),
            xticklabels=tokens,
            yticklabels=tokens,
            cmap='viridis',
            square=True,
            cbar_kws={'label': 'Attention'},
            ax=ax,
            vmin=0,
            vmax=1
        )
        ax.set_title(f'{name.capitalize()} Pattern (Detected: {detected})', 
                    fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.show()


def example_pattern_metrics():
    """
    Example 2: Compute and compare pattern metrics.
    
    Demonstrates:
    - Quantitative pattern characterization
    - Metric computation
    - Comparing patterns numerically
    """
    print("\n" + "=" * 70)
    print("Example 2: Pattern Metrics Comparison")
    print("=" * 70)
    
    patterns = create_pattern_library()
    analyzer = AttentionPatternAnalyzer()
    
    print("\nPattern Metrics Summary:")
    print("-" * 70)
    
    for name, pattern in patterns.items():
        metrics = analyzer.compute_pattern_metrics(pattern)
        
        print(f"\n{name.upper()}:")
        print(f"  Diagonal mean:       {metrics['diagonal_mean']:.3f}")
        print(f"  Sparsity (Gini):     {metrics['sparsity_gini']:.3f}")
        print(f"  Normalized entropy:  {metrics['normalized_entropy']:.3f}")
        print(f"  Mean bandwidth:      {metrics['mean_bandwidth']:.2f}")
        print(f"  First token attn:    {metrics['first_token_attention']:.3f}")
        print(f"  Row correlation:     {metrics['mean_row_correlation']:.3f}")


def example_radar_visualization():
    """
    Example 3: Radar plot visualization of pattern characteristics.
    
    Demonstrates:
    - Multi-dimensional pattern visualization
    - Radar/spider plots
    - Quick pattern comparison
    """
    print("\n" + "=" * 70)
    print("Example 3: Radar Plot Visualization")
    print("=" * 70)
    
    patterns = create_pattern_library()
    analyzer = AttentionPatternAnalyzer()
    
    # Select a few patterns for radar visualization
    selected = ['diagonal', 'local', 'broadcast', 'uniform']
    
    for name in selected:
        print(f"\nVisualizing pattern: {name}")
        analyzer.plot_pattern_metrics_radar(
            patterns[name],
            title=f'{name.capitalize()} Pattern Characteristics'
        )


def example_pattern_comparison():
    """
    Example 4: Side-by-side pattern comparison.
    
    Demonstrates:
    - Comparing multiple patterns
    - Side-by-side visualization
    - Pattern diversity
    """
    print("\n" + "=" * 70)
    print("Example 4: Side-by-Side Pattern Comparison")
    print("=" * 70)
    
    patterns = create_pattern_library()
    analyzer = AttentionPatternAnalyzer()
    
    # Compare three different patterns
    attention_list = [patterns['diagonal'], patterns['local'], patterns['broadcast']]
    labels = ['Diagonal', 'Local', 'Broadcast']
    tokens = ["T" + str(i) for i in range(8)]
    
    analyzer.visualize_pattern_comparison(attention_list, labels, tokens)


if __name__ == "__main__":
    """
    Main execution demonstrating all pattern analysis capabilities.
    
    Run this script to learn about attention patterns:
        python attention_patterns.py
    """
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("\n" + "=" * 70)
    print(" ATTENTION PATTERN RECOGNITION AND ANALYSIS ")
    print(" Module 59: Understanding Common Patterns ")
    print("=" * 70)
    
    # Run examples
    example_pattern_detection()
    example_pattern_metrics()
    # example_radar_visualization()  # Uncomment to run
    example_pattern_comparison()
    
    print("\n" + "=" * 70)
    print(" Examples completed successfully! ")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  1. Different patterns indicate different attention behaviors")
    print("  2. Diagonal pattern: strong self-attention")
    print("  3. Broadcast pattern: focusing on specific tokens")
    print("  4. Local pattern: attention to nearby tokens")
    print("  5. Attention collapse: problematic degenerate pattern")
    print("  6. Metrics help quantify and compare patterns")
