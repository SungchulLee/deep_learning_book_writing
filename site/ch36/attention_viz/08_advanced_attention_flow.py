"""
Advanced Level: Attention Flow Analysis

Combines attention weights with gradient information to understand
which attention connections are most important for predictions.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple

class AttentionFlowAnalyzer:
    """
    Analyzer for attention flow using gradients.
    
    Attention weights alone don't tell us which connections are important.
    By combining attention with gradients, we can identify critical paths.
    
    Formula:
    -------
    Flow = Attention × |Gradient|
    
    Where gradient is with respect to the output.
    """
    
    def __init__(self):
        pass
    
    def compute_attention_flow(self,
                              attention: torch.Tensor,
                              gradients: torch.Tensor) -> torch.Tensor:
        """
        Compute attention flow by combining attention and gradients.
        
        Parameters:
        ----------
        attention : torch.Tensor
            Attention weights, shape: (seq_len, seq_len)
        gradients : torch.Tensor
            Gradients of output w.r.t. attention, same shape
        
        Returns:
        -------
        torch.Tensor
            Attention flow matrix
        """
        # Take absolute value of gradients (we care about magnitude)
        grad_magnitude = torch.abs(gradients)
        
        # Multiply attention by gradient magnitude
        flow = attention * grad_magnitude
        
        # Normalize
        flow = flow / (flow.sum(dim=1, keepdim=True) + 1e-10)
        
        return flow
    
    def visualize_flow(self,
                      attention: torch.Tensor,
                      flow: torch.Tensor,
                      tokens: List[str],
                      save_path: Optional[str] = None):
        """
        Compare attention weights vs attention flow.
        
        This shows which attention connections actually matter for predictions.
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
            cbar_kws={'label': 'Weight'},
            ax=axes[0],
            vmin=0,
            vmax=1
        )
        axes[0].set_title('Attention Weights', fontsize=13, fontweight='bold')
        axes[0].set_xlabel('Key Tokens')
        axes[0].set_ylabel('Query Tokens')
        
        # Plot flow
        sns.heatmap(
            flow,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap='YlOrRd',
            square=True,
            cbar_kws={'label': 'Flow'},
            ax=axes[1],
            vmin=0,
            vmax=flow.max()
        )
        axes[1].set_title('Attention Flow (with Gradients)', fontsize=13, fontweight='bold')
        axes[1].set_xlabel('Key Tokens')
        axes[1].set_ylabel('Query Tokens')
        
        for ax in axes:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.suptitle('Attention vs Flow: Which Connections Matter?',
                    fontsize=15, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        
        plt.show()
    
    def identify_critical_connections(self,
                                     flow: torch.Tensor,
                                     tokens: List[str],
                                     top_k: int = 5) -> List[Tuple]:
        """
        Identify the most critical attention connections based on flow.
        
        Returns:
        -------
        list of tuples
            (query_token, key_token, flow_value)
        """
        if isinstance(flow, torch.Tensor):
            flow = flow.cpu().numpy()
        
        # Get top-k connections
        flat_indices = np.argsort(flow.flatten())[-top_k:][::-1]
        
        critical = []
        for idx in flat_indices:
            i = idx // flow.shape[1]
            j = idx % flow.shape[1]
            critical.append((tokens[i], tokens[j], flow[i, j]))
        
        return critical

def example_attention_flow():
    """Example: Attention flow computation."""
    print("=" * 70)
    print("Attention Flow Analysis")
    print("=" * 70)
    
    # Create example
    tokens = ["The", "cat", "sat", "on", "the", "mat"]
    seq_len = len(tokens)
    
    # Synthetic attention
    attention = torch.softmax(torch.randn(seq_len, seq_len), dim=1)
    
    # Synthetic gradients (simulating importance)
    # Make some connections have high gradients
    gradients = torch.rand(seq_len, seq_len) * 0.1
    gradients[1, 0] = 2.0  # "cat" <- "The" is important
    gradients[2, 1] = 1.5  # "sat" <- "cat" is important
    gradients[5, 3] = 1.8  # "mat" <- "on" is important
    
    # Compute flow
    analyzer = AttentionFlowAnalyzer()
    flow = analyzer.compute_attention_flow(attention, gradients)
    
    # Visualize
    analyzer.visualize_flow(attention, flow, tokens)
    
    # Find critical connections
    print("\nTop 5 Critical Attention Connections:")
    print("-" * 50)
    critical = analyzer.identify_critical_connections(flow, tokens, top_k=5)
    for query, key, flow_val in critical:
        print(f"  {query:10s} <- {key:10s} : {flow_val:.4f}")

if __name__ == "__main__":
    torch.manual_seed(42)
    example_attention_flow()
    
    print("\nKey Insights:")
    print("  - Attention weights show all connections")
    print("  - Gradients show which connections affect output")
    print("  - Flow combines both for true importance")
    print("  - Critical for attribution and interpretability")
