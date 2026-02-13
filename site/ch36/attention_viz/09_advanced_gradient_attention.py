"""
Advanced Level: Gradient-Based Attention Attribution

Implements integrated gradients and other gradient-based methods
for attention attribution and model interpretability.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Callable

class GradientAttentionAnalyzer:
    """
    Gradient-based attention attribution methods.
    
    These methods use gradients to determine which attention weights
    are most responsible for model predictions.
    """
    
    def __init__(self, n_steps: int = 50):
        """
        Parameters:
        ----------
        n_steps : int
            Number of interpolation steps for integrated gradients
        """
        self.n_steps = n_steps
    
    def integrated_gradients_attention(self,
                                      model: Callable,
                                      input_ids: torch.Tensor,
                                      baseline_ids: Optional[torch.Tensor] = None,
                                      target_idx: Optional[int] = None) -> torch.Tensor:
        """
        Compute integrated gradients for attention.
        
        Integrated Gradients:
        --------------------
        IG(x) = (x - x') × ∫[α=0 to 1] ∂F(x' + α(x - x'))/∂x dα
        
        Where:
        - x is the input
        - x' is the baseline
        - α is the interpolation coefficient
        - Integral is approximated by Riemann sum
        
        Parameters:
        ----------
        model : callable
            Model that returns attention weights
        input_ids : torch.Tensor
            Input token IDs
        baseline_ids : torch.Tensor, optional
            Baseline input (default: zeros)
        target_idx : int, optional
            Target output position
        
        Returns:
        -------
        torch.Tensor
            Attribution scores for attention
        """
        if baseline_ids is None:
            baseline_ids = torch.zeros_like(input_ids)
        
        # Create interpolated inputs
        alphas = torch.linspace(0, 1, self.n_steps)
        
        attributions = []
        
        for alpha in alphas:
            # Interpolate
            interpolated = baseline_ids + alpha * (input_ids - baseline_ids)
            interpolated = interpolated.long()
            
            # Enable gradients
            interpolated.requires_grad = True
            
            # Forward pass
            outputs = model(interpolated, output_attentions=True)
            
            # Get attention and compute gradient
            attention = outputs.attentions[-1][0, 0]  # Last layer, first head
            
            if target_idx is not None:
                # Gradient w.r.t. specific position
                target_attn = attention[target_idx].sum()
            else:
                # Gradient w.r.t. all positions
                target_attn = attention.sum()
            
            # Backward
            if interpolated.grad is not None:
                interpolated.grad.zero_()
            
            target_attn.backward()
            
            # Store gradients
            if interpolated.grad is not None:
                attributions.append(interpolated.grad.detach())
        
        # Approximate integral using Riemann sum
        attributions = torch.stack(attributions)
        integrated_grads = attributions.mean(dim=0)
        
        # Multiply by (x - x')
        final_attribution = (input_ids - baseline_ids).float() * integrated_grads
        
        return final_attribution
    
    def gradient_x_input(self,
                        model: Callable,
                        input_ids: torch.Tensor,
                        target_idx: Optional[int] = None) -> torch.Tensor:
        """
        Simple gradient × input attribution.
        
        Attribution = ∂output/∂input × input
        
        This is a simpler alternative to integrated gradients.
        """
        input_ids.requires_grad = True
        
        # Forward
        outputs = model(input_ids, output_attentions=True)
        attention = outputs.attentions[-1][0, 0]
        
        if target_idx is not None:
            target = attention[target_idx].sum()
        else:
            target = attention.sum()
        
        # Backward
        if input_ids.grad is not None:
            input_ids.grad.zero_()
        
        target.backward()
        
        # Attribution
        attribution = input_ids.grad * input_ids.float()
        
        return attribution.detach()
    
    def visualize_attribution(self,
                             attribution: torch.Tensor,
                             tokens: List[str],
                             title: str = "Attention Attribution",
                             save_path: Optional[str] = None):
        """Visualize attribution scores."""
        if isinstance(attribution, torch.Tensor):
            attribution = attribution.cpu().numpy()
        
        if attribution.ndim > 1:
            attribution = attribution.squeeze()
        
        fig, ax = plt.subplots(figsize=(12, 5))
        
        positions = np.arange(len(tokens))
        colors = ['red' if a < 0 else 'green' for a in attribution]
        
        bars = ax.bar(positions, np.abs(attribution), color=colors, alpha=0.7, edgecolor='black')
        
        ax.set_xlabel('Tokens', fontsize=12, fontweight='bold')
        ax.set_ylabel('Attribution Score (magnitude)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        ax.set_xticks(positions)
        ax.set_xticklabels(tokens, rotation=45, ha='right')
        
        # Add value labels
        for pos, val in zip(positions, attribution):
            ax.text(pos, abs(val) + max(abs(attribution)) * 0.02, 
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(0, color='black', linewidth=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        
        plt.show()

def example_gradient_attribution():
    """Example: Gradient-based attribution (simplified)."""
    print("=" * 70)
    print("Gradient-Based Attention Attribution")
    print("=" * 70)
    
    print("\nNote: This example uses synthetic data.")
    print("For real models, use with actual transformer models.")
    
    tokens = ["The", "cat", "chased", "the", "mouse"]
    seq_len = len(tokens)
    
    # Simulate attribution scores
    attribution = torch.tensor([0.2, 0.8, 0.6, 0.1, 0.9])
    
    analyzer = GradientAttentionAnalyzer()
    analyzer.visualize_attribution(
        attribution,
        tokens,
        title="Token Attribution Scores (Synthetic)"
    )
    
    print("\nInterpretation:")
    print("  - Higher scores = more important for predictions")
    print("  - 'cat' and 'mouse' have high attribution")
    print("  - Articles ('the') have low attribution")

if __name__ == "__main__":
    example_gradient_attribution()
    
    print("\nKey Concepts:")
    print("  - Integrated gradients: robust attribution method")
    print("  - Gradient × Input: simpler alternative")
    print("  - Reveals which tokens drive predictions")
    print("  - Essential for model interpretability")
