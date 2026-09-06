# Advanced Level

Advanced Level: Vision Transformer (ViT) Attention Visualization Specialized visualization for Vision Transformers, showing spatial attention

Understanding what neural networks learn is crucial for building trust and debugging models. This module demonstrates attention visualization techniques that reveal how models process inputs and make decisions, providing visual and quantitative insight into network behavior.

## Code

```python
"""
Advanced Level: Vision Transformer (ViT) Attention Visualization

Specialized visualization for Vision Transformers, showing spatial attention
patterns across image patches.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
import warnings

# ========================================================================
# Main
# ========================================================================
warnings.filterwarnings('ignore')

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

class ViTAttentionVisualizer:
    """
    Visualizer for Vision Transformer attention patterns.
    
    ViTs divide images into patches and apply transformers. This class
    helps visualize how the model attends to different spatial regions.
    """
    
    def __init__(self, image_size: int = 224, patch_size: int = 16):
        """
        Parameters:
        ----------
        image_size : int
            Input image size (assumed square)
        patch_size : int
            Size of each patch
        """
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
    
    def visualize_patch_attention(self,
                                  attention: torch.Tensor,
                                  image: Optional[torch.Tensor] = None,
                                  focus_patch: int = 0,
                                  save_path: Optional[str] = None):
        """
        Visualize how one patch attends to all other patches.
        
        Parameters:
        ----------
        attention : torch.Tensor
            Attention matrix, shape: (num_patches+1, num_patches+1)
            (+1 for CLS token)
        image : torch.Tensor, optional
            Original image tensor, shape: (3, H, W)
        focus_patch : int
            Which patch to focus on (0 = CLS token)
        """
        if isinstance(attention, torch.Tensor):
            attention = attention.cpu().numpy()
        
        # Extract attention from focus patch
        patch_attention = attention[focus_patch, :]
        
        # Reshape to 2D grid (excluding CLS token)
        grid_size = int(np.sqrt(self.num_patches))
        
        if focus_patch == 0:  # CLS token
            # Skip CLS token in visualization
            spatial_attention = patch_attention[1:].reshape(grid_size, grid_size)
            title = "CLS Token Attention to Patches"
        else:
            spatial_attention = patch_attention[1:].reshape(grid_size, grid_size)
            title = f"Patch {focus_patch-1} Attention"
        
        # Create figure
        if image is not None:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Show original image
            if isinstance(image, torch.Tensor):
                img_np = image.cpu().numpy()
                if img_np.shape[0] == 3:  # CHW format
                    img_np = np.transpose(img_np, (1, 2, 0))
                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
            else:
                img_np = image
            
            axes[0].imshow(img_np)
            axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
            axes[0].axis('off')
            
            # Show attention map overlaid
            axes[1].imshow(img_np, alpha=0.5)
            
            # Resize attention map to image size
            attn_resized = F.interpolate(
                torch.tensor(spatial_attention).unsqueeze(0).unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False
            )[0, 0].numpy()
            
            im = axes[1].imshow(attn_resized, cmap='jet', alpha=0.5, vmin=0, vmax=spatial_attention.max())
            axes[1].set_title(title, fontsize=12, fontweight='bold')
            axes[1].axis('off')
            
            plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
            
        else:
            fig, ax = plt.subplots(figsize=(8, 7))
            im = ax.imshow(spatial_attention, cmap='viridis', aspect='auto')
            ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
            ax.set_xlabel('Patch Column', fontsize=11)
            ax.set_ylabel('Patch Row', fontsize=11)
            plt.colorbar(im, ax=ax, label='Attention Weight')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        
        plt.show()
    
    def visualize_attention_map(self,
                               attention: torch.Tensor,
                               layer_idx: int = -1,
                               head_idx: int = 0):
        """
        Visualize full attention map for a specific layer and head.
        
        Creates a grid showing spatial attention patterns.
        """
        if attention.dim() == 4:  # (batch, heads, seq, seq)
            attention = attention[0, head_idx]  # Get specific head
        
        attention = attention.cpu().numpy()
        
        # Visualize attention matrix
        fig, ax = plt.subplots(figsize=(10, 9))
        
        im = ax.imshow(attention, cmap='viridis', aspect='auto')
        ax.set_xlabel('Key Patches', fontsize=12, fontweight='bold')
        ax.set_ylabel('Query Patches', fontsize=12, fontweight='bold')
        ax.set_title(f'ViT Attention - Layer {layer_idx}, Head {head_idx}',
                    fontsize=14, fontweight='bold', pad=20)
        
        plt.colorbar(im, ax=ax, label='Attention Weight')
        plt.tight_layout()
        plt.show()

def example_vit_attention():
    """Example: ViT spatial attention visualization."""
    print("=" * 70)
    print("Vision Transformer Attention Visualization")
    print("=" * 70)
    
    # Create synthetic ViT attention
    image_size = 224
    patch_size = 16
    num_patches = (image_size // patch_size) ** 2
    
    # Create synthetic attention (num_patches+1 for CLS token)
    seq_len = num_patches + 1
    attention = torch.zeros(seq_len, seq_len)
    
    # CLS token attends to all patches
    attention[0, 1:] = torch.softmax(torch.randn(num_patches), dim=0)
    attention[0, 0] = 0.1
    
    # Other patches have local attention
    for i in range(1, seq_len):
        # Create local attention pattern
        distances = torch.abs(torch.arange(1, seq_len) - i)
        attn_logits = -distances.float() * 0.5
        attention[i, 1:] = torch.softmax(attn_logits, dim=0) * 0.9
        attention[i, 0] = 0.05  # Some attention to CLS
        attention[i, i] = 0.05  # Self attention
    
    # Visualize
    viz = ViTAttentionVisualizer(image_size=image_size, patch_size=patch_size)
    
    print("\nVisualizing CLS token attention (what the model focuses on):")
    viz.visualize_patch_attention(attention, focus_patch=0)
    
    print("\nVisualizing center patch attention:")
    center_patch = num_patches // 2
    viz.visualize_patch_attention(attention, focus_patch=center_patch)

if __name__ == "__main__":
    torch.manual_seed(42)
    example_vit_attention()
    
    print("\nKey Insights:")
    print("  - CLS token aggregates information from all patches")
    print("  - Spatial attention reveals which image regions are important")
    print("  - Local patches often attend to nearby regions")
    print("  - Attention maps can highlight salient objects")```

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
    def test_vitattentionvisualizer():
        model = ViTAttentionVisualizer(...)
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
