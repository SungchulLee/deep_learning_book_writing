# Video Transformers

## Learning Objectives

By the end of this section, you will be able to:

- Understand how attention mechanisms apply to video understanding
- Implement space-time self-attention for video
- Compare different video transformer architectures (TimeSformer, ViViT)
- Design efficient attention patterns for video (factorized, sparse)
- Evaluate trade-offs between 3D CNNs and video transformers

## From Image to Video Transformers

### Vision Transformer (ViT) Recap

For images, ViT divides an image into patches and applies self-attention:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

For an image with $N$ patches, self-attention has complexity $O(N^2)$.

### The Video Challenge

Videos add a temporal dimension:
- Image: $N = \frac{H \times W}{P^2}$ patches (e.g., 196 for 224×224 with 16×16 patches)
- Video with $T$ frames: $N = T \times \frac{H \times W}{P^2}$ tokens (e.g., 1,568 for 8 frames)

Full space-time attention: $O((T \times N)^2)$ — **prohibitively expensive!**

### Solution: Factorized Attention

Decompose space-time attention into manageable components:

1. **Spatial attention**: Attend within each frame
2. **Temporal attention**: Attend across frames at each spatial location
3. **Mixed strategies**: Various combinations

## Attention Strategies for Video

### Full Space-Time Attention

Every token attends to every other token across all frames:

$$\text{Complexity: } O((T \cdot N)^2 \cdot d)$$

```python
def full_spacetime_attention(x, T, N):
    """
    Full space-time attention.
    x: (B, T*N, D) - all tokens from all frames
    
    Each token attends to ALL other tokens (space + time).
    """
    B, L, D = x.shape  # L = T * N
    
    # Standard self-attention
    qkv = linear(x)  # (B, L, 3*D)
    q, k, v = qkv.chunk(3, dim=-1)
    
    attn = (q @ k.transpose(-2, -1)) / math.sqrt(D)
    attn = attn.softmax(dim=-1)
    out = attn @ v
    
    return out
```

### Divided Space-Time Attention (TimeSformer)

Separate spatial and temporal attention:

$$\text{Complexity: } O(T \cdot N^2 \cdot d + N \cdot T^2 \cdot d)$$

```python
def divided_spacetime_attention(x, T, N):
    """
    Divided space-time attention (TimeSformer style).
    
    1. Temporal attention: each spatial position across frames
    2. Spatial attention: within each frame
    """
    B, L, D = x.shape
    
    # Temporal attention: (B*N, T, D)
    x_temporal = rearrange(x, 'b (t n) d -> (b n) t d', t=T, n=N)
    x_temporal = temporal_attention(x_temporal)
    x = rearrange(x_temporal, '(b n) t d -> b (t n) d', b=B, n=N)
    
    # Spatial attention: (B*T, N, D)
    x_spatial = rearrange(x, 'b (t n) d -> (b t) n d', t=T, n=N)
    x_spatial = spatial_attention(x_spatial)
    x = rearrange(x_spatial, '(b t) n d -> b (t n) d', b=B, t=T)
    
    return x
```

### Joint Space-Time Attention

Attend to local spatial-temporal neighborhoods:

```python
def joint_spacetime_attention(x, T, N, window_size=(2, 7, 7)):
    """
    Joint space-time attention within local windows.
    
    Only attends within a window_size neighborhood.
    Reduces complexity significantly for long videos.
    """
    t_win, h_win, w_win = window_size
    
    # Reshape to 3D grid
    x = rearrange(x, 'b (t h w) d -> b t h w d', t=T, h=H, w=W)
    
    # Window partition
    x = window_partition_3d(x, window_size)  # (B*num_windows, t*h*w, D)
    
    # Attention within windows
    x = window_attention(x)
    
    # Reverse window partition
    x = window_reverse_3d(x, window_size, T, H, W)
    
    return x
```

## TimeSformer Architecture

```python
import torch
import torch.nn as nn
from einops import rearrange

class TimeSformer(nn.Module):
    """
    TimeSformer: Is Space-Time Attention All You Need?
    (Bertasius et al., 2021)
    
    Factorizes attention into temporal and spatial components.
    """
    
    def __init__(self,
                 num_classes: int = 400,
                 num_frames: int = 8,
                 img_size: int = 224,
                 patch_size: int = 16,
                 embed_dim: int = 768,
                 depth: int = 12,
                 num_heads: int = 12):
        super().__init__()
        
        self.num_frames = num_frames
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            3, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        
        # Positional embeddings
        self.pos_embed_spatial = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )
        self.pos_embed_temporal = nn.Parameter(
            torch.zeros(1, num_frames, embed_dim)
        )
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            DividedSpaceTimeBlock(embed_dim, num_heads, num_frames, self.num_patches)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Video (B, C, T, H, W)
        Returns:
            Logits (B, num_classes)
        """
        B, C, T, H, W = x.shape
        
        # Patch embed each frame
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        
        # Add spatial position embedding
        x = x + self.pos_embed_spatial
        
        # Reshape to (B, T, N, D)
        x = rearrange(x, '(b t) n d -> b t n d', b=B, t=T)
        
        # Add temporal position embedding
        x = x + self.pos_embed_temporal.unsqueeze(2)
        
        # Flatten to sequence
        x = rearrange(x, 'b t n d -> b (t n) d')
        
        # Add CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, T, self.num_patches)
        
        # Classification from CLS token
        x = self.norm(x[:, 0])
        return self.head(x)


class DividedSpaceTimeBlock(nn.Module):
    """Transformer block with divided space-time attention."""
    
    def __init__(self, dim, num_heads, num_frames, num_patches):
        super().__init__()
        
        self.num_frames = num_frames
        self.num_patches = num_patches
        
        # Temporal attention components
        self.norm_temporal = nn.LayerNorm(dim)
        self.attn_temporal = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        
        # Spatial attention components
        self.norm_spatial = nn.LayerNorm(dim)
        self.attn_spatial = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        
        # MLP
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 4, dim),
            nn.Dropout(0.1)
        )
    
    def forward(self, x, T, N):
        B = x.shape[0]
        
        # Split CLS and patch tokens
        cls_token, patch_tokens = x[:, :1], x[:, 1:]
        
        # Temporal attention
        # Reshape: each spatial location across time
        xt = rearrange(patch_tokens, 'b (t n) d -> (b n) t d', t=T, n=N)
        xt = self.norm_temporal(xt)
        xt_attn, _ = self.attn_temporal(xt, xt, xt)
        patch_tokens = patch_tokens + rearrange(xt_attn, '(b n) t d -> b (t n) d', b=B)
        
        # Spatial attention
        # Reshape: each frame independently
        xs = rearrange(patch_tokens, 'b (t n) d -> (b t) n d', t=T, n=N)
        xs = self.norm_spatial(xs)
        xs_attn, _ = self.attn_spatial(xs, xs, xs)
        patch_tokens = patch_tokens + rearrange(xs_attn, '(b t) n d -> b (t n) d', b=B)
        
        # Recombine with CLS
        x = torch.cat([cls_token, patch_tokens], dim=1)
        
        # MLP
        x = x + self.mlp(self.norm_mlp(x))
        
        return x
```

## ViViT: Video Vision Transformer

```python
class ViViT(nn.Module):
    """
    ViViT: A Video Vision Transformer (Arnab et al., 2021)
    
    Multiple variants:
    1. Spatio-temporal attention (like full attention)
    2. Factorized encoder
    3. Factorized self-attention
    4. Factorized dot-product attention
    """
    
    def __init__(self,
                 num_classes: int = 400,
                 num_frames: int = 16,
                 img_size: int = 224,
                 patch_size: int = 16,
                 tubelet_size: int = 2,  # Temporal size of 3D patches
                 embed_dim: int = 768,
                 depth: int = 12,
                 num_heads: int = 12,
                 variant: str = 'factorized_encoder'):
        super().__init__()
        
        self.variant = variant
        self.num_frames = num_frames // tubelet_size  # After tubelet embedding
        self.num_patches = (img_size // patch_size) ** 2
        
        # 3D patch embedding (tubelets)
        # Embeds spatio-temporal volumes
        self.patch_embed = nn.Conv3d(
            3, embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size)
        )
        
        # Positional embedding
        total_tokens = self.num_frames * self.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, total_tokens + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        if variant == 'factorized_encoder':
            # Separate spatial and temporal transformers
            self.spatial_transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(embed_dim, num_heads, embed_dim * 4, 
                                          batch_first=True),
                num_layers=depth // 2
            )
            self.temporal_transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(embed_dim, num_heads, embed_dim * 4,
                                          batch_first=True),
                num_layers=depth // 2
            )
        else:
            # Single transformer with full attention
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(embed_dim, num_heads, embed_dim * 4,
                                          batch_first=True),
                num_layers=depth
            )
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        
        # 3D patch embedding (tubelets)
        x = self.patch_embed(x)  # (B, D, T', H', W')
        x = rearrange(x, 'b d t h w -> b (t h w) d')
        
        # Add CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        if self.variant == 'factorized_encoder':
            return self._forward_factorized(x, B)
        else:
            return self._forward_full(x)
    
    def _forward_factorized(self, x, B):
        """Factorized encoder: spatial then temporal."""
        
        cls_token, patch_tokens = x[:, :1], x[:, 1:]
        
        # Spatial transformer (per frame)
        xs = rearrange(patch_tokens, 'b (t n) d -> (b t) n d', 
                      t=self.num_frames, n=self.num_patches)
        xs = self.spatial_transformer(xs)
        
        # Pool spatial tokens per frame
        xs = xs.mean(dim=1)  # (B*T, D)
        xs = rearrange(xs, '(b t) d -> b t d', b=B)
        
        # Temporal transformer
        xt = self.temporal_transformer(xs)
        
        # Global average
        out = xt.mean(dim=1)
        out = self.norm(out)
        
        return self.head(out)
    
    def _forward_full(self, x):
        """Full space-time attention."""
        x = self.transformer(x)
        x = self.norm(x[:, 0])  # CLS token
        return self.head(x)
```

## Efficient Video Transformers

### Sparse Attention Patterns

```python
class SparseVideoAttention(nn.Module):
    """
    Sparse attention for efficient video processing.
    
    Patterns:
    - Local temporal: attend to nearby frames only
    - Global spatial: attend to all patches in current frame
    - Strided temporal: attend to every k-th frame
    """
    
    def __init__(self, embed_dim, num_heads, 
                 temporal_window: int = 3,
                 spatial_stride: int = 2):
        super().__init__()
        
        self.temporal_window = temporal_window
        self.spatial_stride = spatial_stride
        
        self.local_temporal = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )
        self.global_spatial = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )
    
    def forward(self, x, T, N):
        B, L, D = x.shape
        
        # Local temporal attention
        x_temporal = rearrange(x, 'b (t n) d -> (b n) t d', t=T)
        
        # Create local attention mask
        mask = self._create_local_mask(T, self.temporal_window)
        
        x_temporal, _ = self.local_temporal(
            x_temporal, x_temporal, x_temporal, 
            attn_mask=mask
        )
        
        x = rearrange(x_temporal, '(b n) t d -> b (t n) d', b=B)
        
        # Strided spatial attention
        x_spatial = rearrange(x, 'b (t n) d -> (b t) n d', t=T)
        
        # Subsample for efficiency
        x_strided = x_spatial[:, ::self.spatial_stride]
        x_spatial, _ = self.global_spatial(x_spatial, x_strided, x_strided)
        
        x = rearrange(x_spatial, '(b t) n d -> b (t n) d', b=B)
        
        return x
    
    def _create_local_mask(self, seq_len, window_size):
        """Create mask for local attention window."""
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool)
        for i in range(seq_len):
            start = max(0, i - window_size // 2)
            end = min(seq_len, i + window_size // 2 + 1)
            mask[i, start:end] = False
        return mask
```

### Multi-Scale Video Transformer

```python
class MultiScaleViT(nn.Module):
    """
    Multi-Scale Vision Transformer for video.
    
    Processes video at multiple temporal scales:
    - Fast pathway: High temporal resolution, fewer channels
    - Slow pathway: Low temporal resolution, more channels
    
    Similar in spirit to SlowFast but with transformers.
    """
    
    def __init__(self, num_classes: int = 400):
        super().__init__()
        
        # Slow pathway (4 frames, 768 dim)
        self.slow_embed = PatchEmbed3D(tubelet_size=4, embed_dim=768)
        self.slow_blocks = nn.ModuleList([
            TransformerBlock(768, 12) for _ in range(12)
        ])
        
        # Fast pathway (16 frames, 192 dim)
        self.fast_embed = PatchEmbed3D(tubelet_size=1, embed_dim=192)
        self.fast_blocks = nn.ModuleList([
            TransformerBlock(192, 4) for _ in range(12)
        ])
        
        # Cross-pathway fusion
        self.fusion = nn.ModuleList([
            CrossAttention(768, 192) for _ in range(4)
        ])
        
        self.head = nn.Linear(768 + 192, num_classes)
    
    def forward(self, x):
        # Slow pathway (subsample temporally)
        x_slow = x[:, :, ::4]  # Every 4th frame
        x_slow = self.slow_embed(x_slow)
        
        # Fast pathway (full temporal)
        x_fast = self.fast_embed(x)
        
        # Process with periodic fusion
        for i, (slow_block, fast_block) in enumerate(
            zip(self.slow_blocks, self.fast_blocks)
        ):
            x_slow = slow_block(x_slow)
            x_fast = fast_block(x_fast)
            
            # Fuse every 3 blocks
            if i % 3 == 2 and i // 3 < len(self.fusion):
                x_slow = self.fusion[i // 3](x_slow, x_fast)
        
        # Global pooling
        slow_out = x_slow.mean(dim=1)
        fast_out = x_fast.mean(dim=1)
        
        # Concatenate and classify
        out = torch.cat([slow_out, fast_out], dim=-1)
        return self.head(out)
```

## Comparison: 3D CNNs vs Video Transformers

| Aspect | 3D CNNs | Video Transformers |
|--------|---------|-------------------|
| Inductive bias | Local receptive field | Global attention |
| Long-range | Multiple layers needed | Direct attention |
| Parameters | Fixed kernel size | Flexible attention |
| Data efficiency | Better with less data | Needs more data |
| Computation | O(T·H·W·K³) | O((T·N)²) or factorized |
| Pretrained weights | Sports-1M, Kinetics | ImageNet ViT initialization |

### When to Use What

**3D CNNs:**
- Limited training data
- Real-time requirements
- Local motion patterns important
- Transfer from action recognition datasets

**Video Transformers:**
- Large-scale training data
- Long-range temporal dependencies
- Fine-grained temporal reasoning
- Transfer from image transformers

## Training Video Transformers

### Data Augmentation

```python
class VideoAugmentation:
    """Augmentation for video transformer training."""
    
    def __init__(self):
        self.spatial_transforms = [
            RandomResizedCrop((224, 224), scale=(0.5, 1.0)),
            RandomHorizontalFlip(),
            ColorJitter(0.4, 0.4, 0.4, 0.1),
        ]
    
    def __call__(self, video):
        # Temporal augmentation
        video = self.temporal_subsample(video)
        
        # Apply same spatial transform to all frames
        for transform in self.spatial_transforms:
            video = transform(video)
        
        return video
    
    def temporal_subsample(self, video, target_frames=8):
        T = video.shape[2]
        if T > target_frames:
            # Random start position
            start = torch.randint(0, T - target_frames + 1, (1,)).item()
            video = video[:, :, start:start + target_frames]
        return video
```

### Training Recipe

```python
def train_video_transformer(model, train_loader, val_loader):
    """Training configuration for video transformers."""
    
    # Optimizer: AdamW with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.05
    )
    
    # Scheduler: Cosine annealing with warmup
    scheduler = CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)
    
    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(30):
        model.train()
        for videos, labels in train_loader:
            videos, labels = videos.cuda(), labels.cuda()
            
            with torch.cuda.amp.autocast():
                outputs = model(videos)
                loss = F.cross_entropy(outputs, labels)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        scheduler.step()
```

## Summary

| Model | Attention Type | Complexity | Key Feature |
|-------|---------------|------------|-------------|
| TimeSformer | Divided space-time | O(T·N² + N·T²) | Factorized attention |
| ViViT | Multiple variants | Varies | Tubelet embedding |
| Video Swin | Window attention | O(T·N·W²) | Shifted windows |
| MViT | Multi-scale | Progressive | Cross-scale attention |

### Key Takeaways

1. **Full space-time attention** is expensive but captures all dependencies
2. **Factorized attention** dramatically reduces computation
3. **Positional embeddings** must encode both space and time
4. **Initialization from image transformers** accelerates convergence
5. **Large-scale pretraining** is essential for best performance

## Next Steps

- **SlowFast Networks**: Dual-pathway CNN design
- **Action Detection**: Temporal localization with transformers
- **Video-Language Models**: CLIP for video, VideoBERT
