"""
Intermediate Level: Cross-Attention Visualization for Seq2Seq Models

This module focuses on visualizing cross-attention in encoder-decoder architectures,
particularly useful for machine translation, summarization, and seq2seq tasks.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple

class CrossAttentionVisualizer:
    """Visualizer for encoder-decoder cross-attention."""
    
    def __init__(self, figsize=(12, 8)):
        self.figsize = figsize
    
    def plot_cross_attention(self, 
                            cross_attention: torch.Tensor,
                            source_tokens: List[str],
                            target_tokens: List[str],
                            title: str = "Cross-Attention",
                            save_path: Optional[str] = None):
        """
        Visualize cross-attention between source and target sequences.
        
        Parameters:
        ----------
        cross_attention : torch.Tensor
            Cross-attention weights, shape: (target_len, source_len)
        source_tokens : list
            Source sequence tokens
        target_tokens : list
            Target sequence tokens
        """
        if isinstance(cross_attention, torch.Tensor):
            cross_attention = cross_attention.cpu().numpy()
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        sns.heatmap(
            cross_attention,
            xticklabels=source_tokens,
            yticklabels=target_tokens,
            cmap='YlOrRd',
            square=False,
            cbar_kws={'label': 'Attention Weight'},
            ax=ax,
            vmin=0,
            vmax=1
        )
        
        ax.set_xlabel('Source (Encoder)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Target (Decoder)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.show()
    
    def plot_alignment_matrix(self,
                             cross_attention: torch.Tensor,
                             source_tokens: List[str],
                             target_tokens: List[str]):
        """
        Create alignment matrix visualization common in machine translation.
        
        Shows which source words align with which target words.
        """
        if isinstance(cross_attention, torch.Tensor):
            cross_attention = cross_attention.cpu().numpy()
        
        # Find maximum attention for each target token
        max_alignments = np.argmax(cross_attention, axis=1)
        
        print("\nWord Alignments:")
        print("-" * 50)
        for target_idx, source_idx in enumerate(max_alignments):
            target_word = target_tokens[target_idx]
            source_word = source_tokens[source_idx]
            attention_weight = cross_attention[target_idx, source_idx]
            print(f"{target_word:15s} <- {source_word:15s} (weight: {attention_weight:.3f})")
        
        # Visualize
        self.plot_cross_attention(cross_attention, source_tokens, target_tokens,
                                 "Word Alignment Matrix")

def example_translation_attention():
    """Example: Machine translation cross-attention."""
    print("=" * 70)
    print("Cross-Attention Visualization Example")
    print("=" * 70)
    
    # English to French translation example
    source = ["I", "love", "machine", "learning"]
    target = ["J'", "adore", "l'", "apprentissage", "automatique"]
    
    # Create synthetic cross-attention
    # Target length x Source length
    cross_attn = torch.zeros(len(target), len(source))
    
    # Simulate reasonable alignments
    cross_attn[0, 0] = 0.8  # J' <- I
    cross_attn[1, 1] = 0.7  # adore <- love
    cross_attn[2, 2] = 0.3  # l' <- machine (article)
    cross_attn[3, 2] = 0.6  # apprentissage <- machine
    cross_attn[3, 3] = 0.3  # apprentissage <- learning
    cross_attn[4, 3] = 0.7  # automatique <- learning
    
    # Add some background attention
    cross_attn += torch.rand(len(target), len(source)) * 0.05
    
    # Normalize
    cross_attn = cross_attn / cross_attn.sum(dim=1, keepdim=True)
    
    # Visualize
    viz = CrossAttentionVisualizer()
    viz.plot_alignment_matrix(cross_attn, source, target)

if __name__ == "__main__":
    torch.manual_seed(42)
    example_translation_attention()
    
    print("\nKey Insights:")
    print("  - Cross-attention shows source-target relationships")
    print("  - Useful for understanding translation/generation")
    print("  - Reveals word alignment patterns")
