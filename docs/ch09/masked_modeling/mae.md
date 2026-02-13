# Masked Autoencoders (MAE)

## Learning Objectives

By the end of this section, you will be able to:

- Understand the masked image modeling paradigm for self-supervised learning
- Implement MAE architecture with Vision Transformer backbone
- Explain the design choices: high masking ratio, asymmetric encoder-decoder
- Train MAE models and use them for downstream tasks
- Compare MAE with contrastive learning approaches

## Introduction

Masked Autoencoders (MAE) represent a departure from contrastive learning, drawing inspiration from masked language modeling in NLP (e.g., BERT). The key insight is that **images contain significant redundancy**, allowing models to learn meaningful representations by reconstructing heavily masked images.

### Key Insight

Unlike language where each word carries significant meaning, images have high spatial redundancy. This motivates:

1. **High masking ratio (75%)**: Forces the model to learn holistic understanding
2. **Asymmetric architecture**: Lightweight decoder, heavy encoder
3. **Reconstruction in pixel space**: Direct supervision signal

## MAE Architecture

```
Input Image (224×224)
        │
        ▼
┌───────────────────────────────────────────┐
│        Patch Embedding (16×16 patches)     │
│            196 patches total               │
└───────────────────┬───────────────────────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
┌───────────────┐       ┌───────────────┐
│ Visible (25%) │       │ Masked (75%)  │
│  ~49 patches  │       │  ~147 patches │
└───────┬───────┘       └───────────────┘
        │
        ▼
┌───────────────────────────────────────────┐
│           Encoder (ViT-Large)              │
│      Only processes visible patches!       │
│           (Saves computation)              │
└───────────────────┬───────────────────────┘
        │
        ▼
┌───────────────────────────────────────────┐
│    Add mask tokens + positional embed      │
│        Full sequence: 196 tokens           │
└───────────────────┬───────────────────────┘
        │
        ▼
┌───────────────────────────────────────────┐
│         Decoder (Lightweight ViT)          │
│              8 blocks                      │
└───────────────────┬───────────────────────┘
        │
        ▼
┌───────────────────────────────────────────┐
│      Reconstruct masked patches            │
│        MSE loss on masked only             │
└───────────────────────────────────────────┘
```

## Complete Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms


# =============================================================================
# Vision Transformer Components
# =============================================================================

class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    
    Converts image into a sequence of patch embeddings using convolution.
    
    Args:
        img_size: Input image size
        patch_size: Size of each patch
        in_chans: Number of input channels
        embed_dim: Embedding dimension
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Use convolution for efficient patch embedding
        self.proj = nn.Conv2d(
            in_chans, embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, num_patches, embed_dim)
        """
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class Attention(nn.Module):
    """Multi-Head Self-Attention."""
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """Feedforward network with GELU activation."""
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer Block with pre-norm."""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, 
                 drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop
        )
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden, drop=drop)
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# =============================================================================
# MAE Model
# =============================================================================

class MAE(nn.Module):
    """
    Masked Autoencoder for Self-Supervised Learning.
    
    Key design choices:
    1. High masking ratio (75%): Images are redundant
    2. Asymmetric encoder-decoder: Heavy encoder, light decoder
    3. Encoder only sees visible patches: Computational efficiency
    4. Reconstruction loss only on masked patches
    
    Args:
        img_size: Input image size
        patch_size: Patch size
        in_chans: Number of input channels
        embed_dim: Encoder embedding dimension
        depth: Number of encoder transformer blocks
        num_heads: Number of attention heads in encoder
        decoder_embed_dim: Decoder embedding dimension
        decoder_depth: Number of decoder transformer blocks
        decoder_num_heads: Number of attention heads in decoder
        mlp_ratio: MLP hidden dimension ratio
        mask_ratio: Ratio of patches to mask (default: 0.75)
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.,
        mask_ratio=0.75
    ):
        super().__init__()
        
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        
        # ---------- Encoder ----------
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, qkv_bias=True)
            for _ in range(depth)
        ])
        self.encoder_norm = nn.LayerNorm(embed_dim)
        
        # ---------- Decoder ----------
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim)
        )
        
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        
        # Prediction head: predict pixel values
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size**2 * in_chans, bias=True
        )
        
        self.initialize_weights()
    
    def initialize_weights(self):
        """Initialize positional embeddings and other parameters."""
        # Initialize positional embeddings with sin-cos
        pos_embed = self._get_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches ** 0.5),
            cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        decoder_pos_embed = self._get_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.patch_embed.num_patches ** 0.5),
            cls_token=True
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        )
        
        # Initialize patch_embed like nn.Linear
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        
        # Initialize tokens
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        
        # Initialize linear layers
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def _get_sincos_pos_embed(self, embed_dim, grid_size, cls_token=False):
        """Generate 2D sin-cos positional embeddings."""
        grid_h = np.arange(grid_size, dtype=np.float32)
        grid_w = np.arange(grid_size, dtype=np.float32)
        grid = np.meshgrid(grid_w, grid_h)
        grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
        
        pos_embed = self._get_1d_sincos_from_grid(embed_dim, grid)
        if cls_token:
            pos_embed = np.concatenate(
                [np.zeros([1, embed_dim]), pos_embed], axis=0
            )
        return pos_embed
    
    def _get_1d_sincos_from_grid(self, embed_dim, grid):
        """Get 1D sin-cos positional embedding from grid."""
        omega = np.arange(embed_dim // 4, dtype=np.float32)
        omega /= embed_dim / 4.
        omega = 1. / 10000**omega
        
        pos_h = grid[0].reshape(-1)
        pos_w = grid[1].reshape(-1)
        
        out_h = np.einsum('m,d->md', pos_h, omega)
        out_w = np.einsum('m,d->md', pos_w, omega)
        
        emb_h = np.concatenate([np.sin(out_h), np.cos(out_h)], axis=1)
        emb_w = np.concatenate([np.sin(out_w), np.cos(out_w)], axis=1)
        
        return np.concatenate([emb_h, emb_w], axis=1)
    
    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by shuffling.
        
        Args:
            x: (B, N, D) patch embeddings
            mask_ratio: Ratio of patches to mask
        
        Returns:
            x_masked: (B, N_visible, D) visible patches
            mask: (B, N) binary mask, 0=keep, 1=remove
            ids_restore: (B, N) indices to restore original order
        """
        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))
        
        # Random noise for shuffling
        noise = torch.rand(B, N, device=x.device)
        
        # Sort noise to get shuffled indices
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep first len_keep patches
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1, 
            index=ids_keep.unsqueeze(-1).expand(-1, -1, D)
        )
        
        # Generate binary mask: 0=keep, 1=remove
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
    
    def forward_encoder(self, x, mask_ratio):
        """
        Encoder forward pass.
        
        Only processes VISIBLE patches for efficiency!
        """
        # Embed patches
        x = self.patch_embed(x)  # (B, N, D)
        
        # Add positional embedding (without cls token)
        x = x + self.pos_embed[:, 1:, :]
        
        # Masking: remove masked patches
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        
        # Append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Apply encoder blocks
        for blk in self.encoder_blocks:
            x = blk(x)
        x = self.encoder_norm(x)
        
        return x, mask, ids_restore
    
    def forward_decoder(self, x, ids_restore):
        """
        Decoder forward pass.
        
        Processes full sequence (visible + mask tokens).
        """
        # Embed tokens
        x = self.decoder_embed(x)
        
        # Append mask tokens
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # No cls token
        
        # Unshuffle to original order
        x_ = torch.gather(
            x_, dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[2])
        )
        x = torch.cat([x[:, :1, :], x_], dim=1)  # Append cls token
        
        # Add positional embedding
        x = x + self.decoder_pos_embed
        
        # Apply decoder blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        
        # Predict pixel values
        x = self.decoder_pred(x)
        
        # Remove cls token
        x = x[:, 1:, :]
        
        return x
    
    def patchify(self, imgs):
        """
        Convert images to patches.
        
        Args:
            imgs: (B, 3, H, W)
        Returns:
            x: (B, N, patch_size**2 * 3)
        """
        p = self.patch_size
        h = w = imgs.shape[2] // p
        
        x = imgs.reshape(imgs.shape[0], 3, h, p, w, p)
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(imgs.shape[0], h * w, p**2 * 3)
        return x
    
    def unpatchify(self, x):
        """
        Convert patches back to images.
        
        Args:
            x: (B, N, patch_size**2 * 3)
        Returns:
            imgs: (B, 3, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        
        x = x.reshape(x.shape[0], h, w, p, p, 3)
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(x.shape[0], 3, h * p, w * p)
        return imgs
    
    def forward_loss(self, imgs, pred, mask):
        """
        Compute reconstruction loss on masked patches only.
        
        Args:
            imgs: (B, 3, H, W) original images
            pred: (B, N, p*p*3) predicted patches
            mask: (B, N) binary mask, 1=masked
        """
        target = self.patchify(imgs)
        
        # Normalize target by patch (important for stable training)
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1e-6) ** 0.5
        
        # MSE loss
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # (B, N) mean per patch
        
        # Loss only on masked patches
        loss = (loss * mask).sum() / mask.sum()
        
        return loss
    
    def forward(self, imgs, mask_ratio=None):
        """
        Full forward pass.
        
        Args:
            imgs: (B, 3, H, W)
            mask_ratio: Override default mask ratio
        
        Returns:
            loss: Reconstruction loss
            pred: Predicted patches
            mask: Binary mask
        """
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        
        return loss, pred, mask
    
    def get_representation(self, imgs):
        """
        Get encoder representation for downstream tasks.
        
        For downstream tasks, we use the encoder without masking.
        """
        x = self.patch_embed(imgs)
        x = x + self.pos_embed[:, 1:, :]
        
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        for blk in self.encoder_blocks:
            x = blk(x)
        x = self.encoder_norm(x)
        
        return x[:, 0]  # Return CLS token


# =============================================================================
# Training
# =============================================================================

class MAEAugmentation:
    """
    Simple augmentation for MAE.
    
    MAE uses minimal augmentation compared to contrastive methods:
    - Random resized crop
    - Random horizontal flip
    - No color augmentation!
    """
    def __init__(self, size=224):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def __call__(self, x):
        return self.transform(x)


def train_mae_epoch(model, dataloader, optimizer, device, epoch):
    """Train MAE for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (imgs, _) in enumerate(dataloader):
        imgs = imgs.to(device)
        
        loss, _, _ = model(imgs)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % 100 == 0:
            print(f"  Batch {batch_idx}: Loss = {loss.item():.4f}")
    
    return total_loss / num_batches


# =============================================================================
# Visualization
# =============================================================================

def visualize_reconstruction(model, img, device):
    """
    Visualize MAE reconstruction.
    
    Returns:
        original: Original image
        masked: Image with masked patches
        reconstruction: Reconstructed image
    """
    model.eval()
    
    with torch.no_grad():
        img = img.unsqueeze(0).to(device)
        loss, pred, mask = model(img)
        
        # Unpatchify prediction
        pred_img = model.unpatchify(pred)
        
        # Create masked image visualization
        mask_vis = mask.unsqueeze(-1).repeat(1, 1, model.patch_size**2 * 3)
        mask_vis = model.unpatchify(mask_vis)
        masked_img = img * (1 - mask_vis)
        
        return img[0], masked_img[0], pred_img[0], loss.item()


# =============================================================================
# Model Variants
# =============================================================================

def mae_vit_base(**kwargs):
    """MAE with ViT-Base encoder."""
    return MAE(
        embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        **kwargs
    )

def mae_vit_large(**kwargs):
    """MAE with ViT-Large encoder."""
    return MAE(
        embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        **kwargs
    )

def mae_vit_huge(**kwargs):
    """MAE with ViT-Huge encoder."""
    return MAE(
        embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        **kwargs
    )


# =============================================================================
# Usage Example
# =============================================================================

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = mae_vit_base(img_size=224, patch_size=16, mask_ratio=0.75)
    print(f"\nMAE Model:")
    print(f"  Patches: {model.patch_embed.num_patches}")
    print(f"  Mask ratio: {model.mask_ratio}")
    print(f"  Visible patches: {int(model.patch_embed.num_patches * (1 - model.mask_ratio))}")
    
    # Test forward pass
    imgs = torch.randn(4, 3, 224, 224)
    
    model.eval()
    loss, pred, mask = model(imgs)
    
    print(f"\nForward pass:")
    print(f"  Input: {imgs.shape}")
    print(f"  Prediction: {pred.shape}")
    print(f"  Mask: {mask.shape}")
    print(f"  Masked patches: {mask.sum(dim=1).mean().item():.0f}")
    print(f"  Loss: {loss.item():.4f}")
```

## Key Design Choices

### 1. High Masking Ratio (75%)

| Mask Ratio | ImageNet Acc | Reasoning |
|------------|--------------|-----------|
| 25% | 80.1% | Too easy |
| 50% | 82.4% | Better |
| **75%** | **83.6%** | Optimal |
| 90% | 82.5% | Too hard |

The 75% masking ratio forces the model to learn semantic understanding rather than relying on local texture interpolation.

### 2. Asymmetric Encoder-Decoder

- **Encoder**: Heavy (ViT-Large), only processes visible patches
- **Decoder**: Light (8 blocks), reconstructs full image

This is computationally efficient: encoder processes only 25% of patches.

### 3. Reconstruction Target

MAE reconstructs **normalized pixel values**, not features:
- Simple and interpretable
- No additional pretrained model needed
- Works well empirically

## MAE vs Contrastive Learning

| Aspect | MAE | Contrastive (SimCLR/MoCo) |
|--------|-----|---------------------------|
| Supervision | Pixel reconstruction | Instance discrimination |
| Negatives | Not needed | Required |
| Augmentation | Minimal | Heavy |
| Batch size | Any | Large preferred |
| Architecture | ViT | CNN or ViT |
| Pre-training | Faster | Slower |

## Summary

MAE demonstrates that masked image modeling is a powerful alternative to contrastive learning:

1. **Simplicity**: No negatives, minimal augmentation
2. **Efficiency**: Only encode visible patches
3. **Scalability**: Benefits from larger models
4. **Versatility**: Works well for various vision tasks

## References

1. He, K., Chen, X., Xie, S., Li, Y., Dollár, P., & Girshick, R. (2022). Masked Autoencoders Are Scalable Vision Learners. CVPR.
2. Dosovitskiy, A., et al. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. ICLR.
3. Bao, H., Dong, L., & Wei, F. (2022). BEiT: BERT Pre-Training of Image Transformers. ICLR.
