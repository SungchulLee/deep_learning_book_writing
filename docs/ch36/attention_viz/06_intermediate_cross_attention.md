# Intermediate Level

Intermediate Level: Cross-Attention Visualization for Seq2Seq Models This module focuses on visualizing cross-attention in encoder-decoder architectures,

Understanding what neural networks learn is crucial for building trust and debugging models. This module demonstrates attention visualization techniques that reveal how models process inputs and make decisions, providing visual and quantitative insight into network behavior.

## Code

```python
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

# ========================================================================
# Main
# ========================================================================

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
    print("  - Reveals word alignment patterns")```

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
Write a comprehensive test function that validates the Intermediate Level implementation. Test edge cases including empty inputs, single-element inputs, very large inputs, and inputs with extreme values (zeros, very large numbers).

??? success "Solution to Exercise 4"
    Create a test function that exercises boundary conditions:
    ```python
    def test_crossattentionvisualizer():
        model = CrossAttentionVisualizer(...)
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
