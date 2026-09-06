# Attention Visualization

Attention Visualization for Transformer Models Visualizes self-attention patterns in transformer architectures

Understanding what neural networks learn is crucial for building trust and debugging models. This module demonstrates gradient-based explanation techniques that reveal how models process inputs and make decisions, providing visual and quantitative insight into network behavior.

## Code

```python
"""
Attention Visualization for Transformer Models
Visualizes self-attention patterns in transformer architectures
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple, Dict
import warnings

# ========================================================================
# Main
# ========================================================================


class AttentionVisualizer:
    """
    Visualizes attention weights from transformer models.
    """
    
    def __init__(self, model: nn.Module):
        """
        Initialize attention visualizer.
        
        Args:
            model: Transformer model (e.g., BERT, GPT, ViT)
        """
        self.model = model
        self.attention_maps = {}
        self.hooks = []
        
    def register_hooks(self, layer_names: Optional[List[str]] = None):
        """
        Register forward hooks to capture attention weights.
        
        Args:
            layer_names: List of layer names to hook. If None, hooks all attention layers.
        """
        def hook_fn(name):
            def hook(module, input, output):
                # Store attention weights
                # Output format depends on the model architecture
                if isinstance(output, tuple) and len(output) > 1:
                    # Usually (output, attention_weights)
                    self.attention_maps[name] = output[1].detach()
                else:
                    self.attention_maps[name] = output.detach()
            return hook
        
        # Find and hook attention modules
        for name, module in self.model.named_modules():
            # Common attention module names
            if any(x in name.lower() for x in ['attention', 'attn', 'self_attn']):
                if layer_names is None or name in layer_names:
                    handle = module.register_forward_hook(hook_fn(name))
                    self.hooks.append(handle)
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def get_attention_maps(self) -> Dict[str, torch.Tensor]:
        """
        Get captured attention maps.
        
        Returns:
            Dictionary mapping layer names to attention tensors
        """
        return self.attention_maps
    
    def visualize_attention_head(self, attention_weights: torch.Tensor, 
                                tokens: Optional[List[str]] = None,
                                head_idx: int = 0,
                                figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Visualize attention weights for a specific head.
        
        Args:
            attention_weights: Attention tensor of shape (batch, heads, seq_len, seq_len)
            tokens: List of token strings for labeling
            head_idx: Index of attention head to visualize
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Extract single head
        if attention_weights.dim() == 4:
            attn = attention_weights[0, head_idx].cpu().numpy()
        elif attention_weights.dim() == 3:
            attn = attention_weights[head_idx].cpu().numpy()
        else:
            attn = attention_weights.cpu().numpy()
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot heatmap
        sns.heatmap(attn, annot=False, cmap='viridis', square=True, 
                   cbar_kws={'label': 'Attention Weight'}, ax=ax)
        
        # Add labels if tokens provided
        if tokens:
            ax.set_xticklabels(tokens, rotation=90)
            ax.set_yticklabels(tokens, rotation=0)
        
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        ax.set_title(f'Attention Head {head_idx}')
        
        plt.tight_layout()
        return fig
    
    def visualize_all_heads(self, attention_weights: torch.Tensor,
                           tokens: Optional[List[str]] = None,
                           max_heads: int = 8,
                           figsize: Tuple[int, int] = (16, 12)) -> plt.Figure:
        """
        Visualize multiple attention heads in a grid.
        
        Args:
            attention_weights: Attention tensor of shape (batch, heads, seq_len, seq_len)
            tokens: List of token strings
            max_heads: Maximum number of heads to display
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if attention_weights.dim() == 4:
            attn = attention_weights[0].cpu().numpy()
        else:
            attn = attention_weights.cpu().numpy()
        
        num_heads = min(attn.shape[0], max_heads)
        ncols = 4
        nrows = (num_heads + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = axes.flatten() if num_heads > 1 else [axes]
        
        for i in range(num_heads):
            ax = axes[i]
            sns.heatmap(attn[i], annot=False, cmap='viridis', 
                       square=True, cbar=False, ax=ax)
            ax.set_title(f'Head {i}')
            
            if tokens and len(tokens) <= 20:
                ax.set_xticklabels(tokens, rotation=90, fontsize=8)
                ax.set_yticklabels(tokens, rotation=0, fontsize=8)
            else:
                ax.set_xticks([])
                ax.set_yticks([])
        
        # Hide unused subplots
        for i in range(num_heads, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('Multi-Head Attention Patterns', fontsize=16)
        plt.tight_layout()
        return fig
    
    def visualize_layer_attention(self, layer_name: str, 
                                  tokens: Optional[List[str]] = None,
                                  average_heads: bool = True,
                                  figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Visualize attention for a specific layer.
        
        Args:
            layer_name: Name of the layer
            tokens: List of token strings
            average_heads: Whether to average across heads
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if layer_name not in self.attention_maps:
            raise ValueError(f"Layer {layer_name} not found in attention maps")
        
        attn = self.attention_maps[layer_name]
        
        if average_heads and attn.dim() >= 3:
            # Average across heads
            if attn.dim() == 4:
                attn = attn[0].mean(dim=0).cpu().numpy()
            else:
                attn = attn.mean(dim=0).cpu().numpy()
        else:
            return self.visualize_all_heads(attn, tokens, figsize=figsize)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(attn, annot=False, cmap='viridis', square=True,
                   cbar_kws={'label': 'Average Attention'}, ax=ax)
        
        if tokens:
            ax.set_xticklabels(tokens, rotation=90)
            ax.set_yticklabels(tokens, rotation=0)
        
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        ax.set_title(f'Layer: {layer_name} (Average Attention)')
        
        plt.tight_layout()
        return fig
    
    def plot_attention_flow(self, attention_weights: torch.Tensor,
                           tokens: List[str],
                           query_idx: int,
                           figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Plot attention flow from a specific query token.
        
        Args:
            attention_weights: Attention tensor
            tokens: List of token strings
            query_idx: Index of query token
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if attention_weights.dim() == 4:
            # Average across batch and heads
            attn = attention_weights[0].mean(dim=0)[query_idx].cpu().numpy()
        elif attention_weights.dim() == 3:
            attn = attention_weights.mean(dim=0)[query_idx].cpu().numpy()
        else:
            attn = attention_weights[query_idx].cpu().numpy()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        positions = np.arange(len(tokens))
        ax.bar(positions, attn, color='steelblue', alpha=0.7)
        ax.set_xticks(positions)
        ax.set_xticklabels(tokens, rotation=45, ha='right')
        ax.set_ylabel('Attention Weight')
        ax.set_title(f'Attention from "{tokens[query_idx]}" to other tokens')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        return fig


class BERTAttentionVisualizer(AttentionVisualizer):
    """
    Specialized visualizer for BERT-like models.
    """
    
    def extract_attention_from_output(self, outputs):
        """
        Extract attention weights from BERT model outputs.
        
        Args:
            outputs: Model outputs from BERT (transformers library)
            
        Returns:
            List of attention tensors, one per layer
        """
        if hasattr(outputs, 'attentions') and outputs.attentions is not None:
            return outputs.attentions
        else:
            warnings.warn("Model outputs don't contain attention weights. "
                        "Make sure to call model with output_attentions=True")
            return []


def visualize_token_attention(attention_weights: torch.Tensor,
                              tokens: List[str],
                              layer_idx: int = -1,
                              save_path: Optional[str] = None) -> plt.Figure:
    """
    Quick utility function to visualize attention for tokens.
    
    Args:
        attention_weights: Attention tensor (layers, batch, heads, seq, seq)
        tokens: List of tokens
        layer_idx: Layer index to visualize (-1 for last layer)
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    # Select layer
    if isinstance(attention_weights, (list, tuple)):
        attn = attention_weights[layer_idx]
    else:
        attn = attention_weights
    
    # Average across batch and heads
    if attn.dim() == 4:
        attn = attn[0].mean(dim=0)
    elif attn.dim() == 3:
        attn = attn.mean(dim=0)
    
    attn = attn.cpu().numpy()
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(attn, xticklabels=tokens, yticklabels=tokens,
               cmap='viridis', square=True, 
               cbar_kws={'label': 'Attention Weight'})
    
    ax.set_xlabel('Key Position')
    ax.set_ylabel('Query Position')
    ax.set_title(f'Attention Pattern (Layer {layer_idx})')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


if __name__ == "__main__":
    pass```

## Discussion

Visualization plays an important role in understanding model behavior and diagnosing training issues. The plotting code provides insight into the learned representations, convergence dynamics, or evaluation metrics, making abstract computations tangible.

The patterns demonstrated here extend naturally to more complex scenarios. Experimenting with hyperparameters, architectural variations, and different datasets deepens understanding and builds practical intuition for model interpretability tasks.

## Exercises

**Exercise 1.**
Read through the code and identify the key design decisions. List three specific implementation choices and explain why each is appropriate for gradient-based explanation.

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
Write a comprehensive test function that validates the Attention Visualization implementation. Test edge cases including empty inputs, single-element inputs, very large inputs, and inputs with extreme values (zeros, very large numbers).

??? success "Solution to Exercise 4"
    Create a test function that exercises boundary conditions:
    ```python
    def test_attentionvisualizer():
        model = AttentionVisualizer(...)
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
