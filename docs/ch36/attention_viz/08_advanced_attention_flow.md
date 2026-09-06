# Advanced Level

Advanced Level: Attention Flow Analysis Combines attention weights with gradient information to understand

Understanding what neural networks learn is crucial for building trust and debugging models. This module demonstrates attention visualization techniques that reveal how models process inputs and make decisions, providing visual and quantitative insight into network behavior.

## Code

```python
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

# ========================================================================
# Main
# ========================================================================

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
    print("  - Critical for attribution and interpretability")```

## Discussion

Visualization plays an important role in understanding model behavior and diagnosing training issues. The plotting code provides insight into the learned representations, convergence dynamics, or evaluation metrics, making abstract computations tangible.

The patterns demonstrated here extend naturally to more complex scenarios. Experimenting with hyperparameters, architectural variations, and different datasets deepens understanding and builds practical intuition for model interpretability tasks.

## Exercises

**Exercise 1.**
Read through the code and identify the key design decisions. List three specific implementation choices and explain why each is appropriate for attention visualization.

??? success "Solution to Exercise 1"
    Design decisions vary by implementation but commonly include: (1) choice of activation functions -- ReLU variants provide non-saturating gradients for faster training; (2) normalization strategy -- batch normalization stabilizes training by reducing internal covariate shift; (3) residual connections -- when present, they enable gradient flow in deep networks by providing skip paths. Each choice reflects a trade-off between expressiveness, computational cost, and training stability.

---

**Exercise 2.**
Add a dropout layer after the attention weights (before multiplying with values). Use a dropout rate of 0.1 during training. Explain why attention dropout helps with regularization.

??? success "Solution to Exercise 2"
    Add `self.attn_dropout = nn.Dropout(0.1)` in `__init__` and apply it after the softmax: `attn_weights = self.attn_dropout(F.softmax(scores, dim=-1))`. Attention dropout randomly zeroes some attention weights during training, preventing the model from relying too heavily on specific token-to-token relationships. This encourages the model to distribute attention more broadly and learn more robust representations, similar to how standard dropout prevents co-adaptation of neurons.

---

**Exercise 3.**
Explain the computational complexity of self-attention as a function of sequence length $n$ and model dimension $d$. Why does this motivate architectures like Longformer or Linformer for long sequences?

??? success "Solution to Exercise 3"
    Standard self-attention computes an $n \times n$ attention matrix, giving $O(n^2 d)$ time complexity and $O(n^2)$ memory for the attention weights. For long sequences (e.g., $n = 4096$), this becomes prohibitive. Longformer uses a combination of local sliding-window attention ($O(n \cdot w \cdot d)$ where $w$ is window size) and sparse global attention for selected tokens. Linformer projects keys and values to a lower dimension $k \ll n$, reducing complexity to $O(n \cdot k \cdot d)$. Both trade some expressiveness for practical efficiency on long inputs.

---

**Exercise 4.**
Write a comprehensive test function that validates the Advanced Level implementation. Test edge cases including empty inputs, single-element inputs, very large inputs, and inputs with extreme values (zeros, very large numbers).

??? success "Solution to Exercise 4"
    Create a test function that exercises boundary conditions:
    ```python
    def test_attentionflowanalyzer():
        model = AttentionFlowAnalyzer(...)
        # Normal input
        assert model(normal_input).shape == expected_shape
        # Single element batch
        assert model(single_input).shape == (1, ...)
        # Large values (check for overflow)
        out = model(torch.ones(...) * 1000)
        assert torch.isfinite(out).all()
        # Gradient flow
        out = model(normal_input)
        out.sum().backward()
        for p in model.parameters():
            assert p.grad is not None
    ```
    Testing gradient flow is especially important to ensure the architecture supports end-to-end training.
