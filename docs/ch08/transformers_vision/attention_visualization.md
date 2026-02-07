# Attention Visualization in Vision Transformers

## Introduction

One of the key advantages of Vision Transformers is their interpretability through attention maps. Unlike CNNs where understanding what the model "looks at" requires gradient-based methods, ViT's attention weights directly show which parts of the image influence each prediction.

## Understanding Attention Maps

### What Attention Maps Show

In ViT, attention maps reveal:
1. **CLS Token Attention**: What the classification token focuses on
2. **Patch-to-Patch Attention**: Relationships between image regions
3. **Head-Specific Patterns**: Different heads capture different patterns
4. **Layer-wise Evolution**: How attention evolves through the network

### Mathematical Background

For each attention head, the attention weights are computed as:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

The attention matrix $A = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)$ has shape $(N+1) \times (N+1)$ where $N$ is the number of patches.

For visualization, we typically extract $A[0, 1:]$ — the attention from the CLS token to all patch tokens.

## Implementation

### Attention Extraction

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple

class AttentionExtractor:
    """
    Extract attention maps from Vision Transformer.
    Uses forward hooks to capture attention weights during inference.
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.attention_maps = []
        self._register_hooks()
        
    def _register_hooks(self):
        """Register forward hooks on attention layers."""
        def create_hook(layer_idx):
            def hook_fn(module, input, output):
                # Get attention weights before softmax
                B, N, C = input[0].shape
                qkv = module.qkv(input[0])
                qkv = qkv.reshape(B, N, 3, module.n_heads, module.head_dim)
                qkv = qkv.permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]
                
                # Compute attention weights
                attn = (q @ k.transpose(-2, -1)) * module.scale
                attn = attn.softmax(dim=-1)
                
                self.attention_maps.append(attn.detach().cpu())
            return hook_fn
        
        for idx, block in enumerate(self.model.blocks):
            block.attn.register_forward_hook(create_hook(idx))
            
    def get_attention(self, image: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract attention maps for an image.
        
        Args:
            image: Input image tensor (1, C, H, W)
            
        Returns:
            List of attention tensors, one per layer
            Each tensor has shape (batch, heads, seq_len, seq_len)
        """
        self.attention_maps = []
        self.model.eval()
        
        with torch.no_grad():
            _ = self.model(image)
            
        return self.attention_maps
```

### Attention Visualization

```python
class AttentionVisualizer:
    """Visualize attention maps from Vision Transformer."""
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
        self.extractor = AttentionExtractor(model)
        
    def visualize_cls_attention(self, 
                                image: torch.Tensor,
                                layer_idx: int = -1,
                                head_idx: Optional[int] = None,
                                save_path: Optional[str] = None):
        """
        Visualize attention from CLS token to image patches.
        
        Args:
            image: Input image (1, 3, H, W)
            layer_idx: Which layer to visualize (-1 for last)
            head_idx: Which head (None for average of all heads)
            save_path: Optional path to save figure
        """
        # Extract attention maps
        attn_maps = self.extractor.get_attention(image.to(self.device))
        
        # Select layer
        attn = attn_maps[layer_idx]  # (1, n_heads, N+1, N+1)
        
        # Average over heads or select specific head
        if head_idx is None:
            attn = attn.mean(dim=1)  # (1, N+1, N+1)
        else:
            attn = attn[:, head_idx]  # (1, N+1, N+1)
            
        # Get CLS token attention to patches (exclude CLS itself)
        cls_attn = attn[0, 0, 1:]  # (N,)
        
        # Reshape to spatial grid
        n_patches = int(np.sqrt(len(cls_attn)))
        attn_map = cls_attn.reshape(n_patches, n_patches).numpy()
        
        # Prepare original image
        img = self._prepare_image(image)
        
        # Resize attention map to image size
        import cv2
        attn_resized = cv2.resize(attn_map, (img.shape[1], img.shape[0]))
        attn_resized = (attn_resized - attn_resized.min()) / \
                       (attn_resized.max() - attn_resized.min() + 1e-8)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(attn_resized, cmap='viridis')
        axes[1].set_title(f'Attention Map (Layer {layer_idx})')
        axes[1].axis('off')
        
        axes[2].imshow(img)
        axes[2].imshow(attn_resized, cmap='jet', alpha=0.5)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.show()
        
    def visualize_all_heads(self, 
                           image: torch.Tensor,
                           layer_idx: int = -1,
                           save_path: Optional[str] = None):
        """
        Visualize attention from all heads in a layer.
        """
        attn_maps = self.extractor.get_attention(image.to(self.device))
        attn = attn_maps[layer_idx]  # (1, n_heads, N+1, N+1)
        
        n_heads = attn.shape[1]
        n_cols = 4
        n_rows = (n_heads + n_cols - 1) // n_cols
        
        img = self._prepare_image(image)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
        axes = axes.flatten()
        
        for head_idx in range(n_heads):
            cls_attn = attn[0, head_idx, 0, 1:].numpy()
            n_patches = int(np.sqrt(len(cls_attn)))
            attn_map = cls_attn.reshape(n_patches, n_patches)
            
            import cv2
            attn_resized = cv2.resize(attn_map, (img.shape[1], img.shape[0]))
            
            axes[head_idx].imshow(img)
            axes[head_idx].imshow(attn_resized, cmap='jet', alpha=0.5)
            axes[head_idx].set_title(f'Head {head_idx}')
            axes[head_idx].axis('off')
            
        # Hide unused subplots
        for idx in range(n_heads, len(axes)):
            axes[idx].axis('off')
            
        plt.suptitle(f'Attention Heads (Layer {layer_idx})')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.show()
        
    def visualize_layer_progression(self,
                                   image: torch.Tensor,
                                   layers: List[int] = None,
                                   save_path: Optional[str] = None):
        """
        Show how attention evolves through layers.
        """
        attn_maps = self.extractor.get_attention(image.to(self.device))
        
        if layers is None:
            n_layers = len(attn_maps)
            layers = [0, n_layers//3, 2*n_layers//3, n_layers-1]
            
        img = self._prepare_image(image)
        
        fig, axes = plt.subplots(1, len(layers) + 1, figsize=(5*(len(layers)+1), 5))
        
        # Original image
        axes[0].imshow(img)
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        for idx, layer_idx in enumerate(layers):
            attn = attn_maps[layer_idx].mean(dim=1)  # Average over heads
            cls_attn = attn[0, 0, 1:].numpy()
            n_patches = int(np.sqrt(len(cls_attn)))
            attn_map = cls_attn.reshape(n_patches, n_patches)
            
            import cv2
            attn_resized = cv2.resize(attn_map, (img.shape[1], img.shape[0]))
            
            axes[idx+1].imshow(img)
            axes[idx+1].imshow(attn_resized, cmap='jet', alpha=0.5)
            axes[idx+1].set_title(f'Layer {layer_idx}')
            axes[idx+1].axis('off')
            
        plt.suptitle('Attention Evolution Through Layers')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.show()
        
    def _prepare_image(self, image: torch.Tensor) -> np.ndarray:
        """Convert tensor to displayable numpy array."""
        img = image[0].permute(1, 2, 0).cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min())
        return img
```

## Attention Rollout

Attention rollout recursively combines attention maps to show cumulative attention flow:

```python
def attention_rollout(attention_maps: List[torch.Tensor], 
                     discard_ratio: float = 0.1) -> torch.Tensor:
    """
    Compute attention rollout across all layers.
    
    Recursively multiplies attention matrices to get the 
    total attention flow from input to output.
    
    Args:
        attention_maps: List of attention tensors from each layer
        discard_ratio: Ratio of lowest attention values to discard
        
    Returns:
        Rolled-out attention from CLS to patches
    """
    result = None
    
    for attn in attention_maps:
        # Average over heads
        attn = attn.mean(dim=1)  # (B, N, N)
        
        # Add identity (residual connection)
        eye = torch.eye(attn.size(-1), device=attn.device)
        attn = attn + eye
        
        # Normalize
        attn = attn / attn.sum(dim=-1, keepdim=True)
        
        # Optional: discard low attention
        if discard_ratio > 0:
            flat = attn.view(-1)
            threshold = torch.quantile(flat, discard_ratio)
            attn = attn.masked_fill(attn < threshold, 0)
            attn = attn / attn.sum(dim=-1, keepdim=True)
        
        if result is None:
            result = attn
        else:
            result = attn @ result
            
    return result[0, 0, 1:]  # CLS to patches


def attention_flow(attention_maps: List[torch.Tensor]) -> torch.Tensor:
    """
    Compute attention flow using gradient-like propagation.
    
    Alternative to rollout that better captures information flow.
    """
    # Start with uniform distribution
    num_tokens = attention_maps[0].shape[-1]
    R = torch.eye(num_tokens, device=attention_maps[0].device)
    
    for attn in attention_maps:
        attn = attn.mean(dim=1)  # Average heads
        
        # Add residual
        attn = attn + torch.eye(num_tokens, device=attn.device)
        attn = attn / attn.sum(dim=-1, keepdim=True)
        
        R = R + attn @ R
        
    return R[0, 0, 1:]  # CLS attention to patches
```

## Interpreting Attention Patterns

### Common Patterns

```python
def analyze_attention_patterns(model: nn.Module, 
                               image: torch.Tensor) -> dict:
    """
    Analyze common attention patterns.
    
    Returns:
        Dictionary with pattern analysis
    """
    extractor = AttentionExtractor(model)
    attn_maps = extractor.get_attention(image)
    
    analysis = {}
    
    # 1. Attention entropy (how focused vs spread)
    entropies = []
    for attn in attn_maps:
        avg_attn = attn.mean(dim=1)  # Average heads
        cls_attn = avg_attn[0, 0, 1:]
        entropy = -(cls_attn * torch.log(cls_attn + 1e-9)).sum()
        entropies.append(entropy.item())
    analysis['entropy_per_layer'] = entropies
    
    # 2. Head diversity (do different heads attend differently?)
    head_diversity = []
    for attn in attn_maps:
        heads = attn[0, :, 0, 1:]  # (n_heads, N)
        diversity = torch.pdist(heads).mean()
        head_diversity.append(diversity.item())
    analysis['head_diversity'] = head_diversity
    
    # 3. Spatial concentration (does attention focus on specific regions?)
    for layer_idx, attn in enumerate(attn_maps):
        cls_attn = attn.mean(dim=1)[0, 0, 1:]
        n_patches = int(np.sqrt(len(cls_attn)))
        attn_2d = cls_attn.reshape(n_patches, n_patches)
        
        # Find peak attention location
        max_idx = attn_2d.argmax()
        peak_row, peak_col = max_idx // n_patches, max_idx % n_patches
        analysis[f'layer_{layer_idx}_peak'] = (peak_row.item(), peak_col.item())
        
    return analysis
```

### Visualization of Head Specialization

```python
def visualize_head_specialization(model: nn.Module,
                                 images: List[torch.Tensor],
                                 layer_idx: int = -1):
    """
    Show how different attention heads specialize on different features.
    """
    extractor = AttentionExtractor(model)
    
    # Collect attention patterns across multiple images
    head_patterns = []
    
    for image in images:
        attn = extractor.get_attention(image)[layer_idx]
        head_patterns.append(attn[0, :, 0, 1:])  # (n_heads, N)
        
    head_patterns = torch.stack(head_patterns)  # (n_images, n_heads, N)
    
    # Compute head similarity matrix
    n_heads = head_patterns.shape[1]
    similarity = torch.zeros(n_heads, n_heads)
    
    for i in range(n_heads):
        for j in range(n_heads):
            # Correlation across images
            corr = torch.corrcoef(torch.stack([
                head_patterns[:, i].flatten(),
                head_patterns[:, j].flatten()
            ]))[0, 1]
            similarity[i, j] = corr
            
    # Visualize
    plt.figure(figsize=(8, 6))
    plt.imshow(similarity.numpy(), cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label='Correlation')
    plt.xlabel('Head')
    plt.ylabel('Head')
    plt.title(f'Attention Head Similarity (Layer {layer_idx})')
    plt.show()
```

## Practical Applications

### Debugging Model Predictions

```python
def debug_prediction(model: nn.Module, 
                    image: torch.Tensor,
                    true_label: int,
                    class_names: List[str]):
    """
    Debug a model prediction using attention visualization.
    """
    model.eval()
    visualizer = AttentionVisualizer(model)
    
    # Get prediction
    with torch.no_grad():
        output = model(image)
        pred_class = output.argmax(dim=1).item()
        confidence = torch.softmax(output, dim=1)[0, pred_class].item()
    
    print(f"True label: {class_names[true_label]}")
    print(f"Prediction: {class_names[pred_class]} ({confidence:.2%})")
    
    # Visualize attention
    if pred_class != true_label:
        print("\nIncorrect prediction - analyzing attention...")
    
    visualizer.visualize_cls_attention(image, layer_idx=-1)
    visualizer.visualize_all_heads(image, layer_idx=-1)
```

## Summary

Attention visualization is a powerful tool for understanding Vision Transformers:

1. **Direct interpretability**: No gradient computation needed
2. **Multi-scale analysis**: View attention at different layers
3. **Head specialization**: Understand what different heads learn
4. **Debugging**: Diagnose model failures

Key techniques:
- CLS token attention maps
- Attention rollout for cumulative attention
- Head diversity analysis
- Layer progression visualization

## References

1. Abnar, S., Zuidema, W. "Quantifying Attention Flow in Transformers." ACL 2020.
2. Chefer, H., et al. "Transformer Interpretability Beyond Attention Visualization." CVPR 2021.
3. Dosovitskiy, A., et al. "An Image is Worth 16x16 Words." ICLR 2021.
