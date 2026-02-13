# Temporal Modeling for Video Understanding

## Learning Objectives

By the end of this section, you will be able to:

- Understand why temporal modeling is essential for video understanding
- Distinguish between different approaches to temporal feature extraction
- Implement temporal pooling and aggregation strategies
- Design architectures that capture both short-term and long-term temporal dependencies
- Evaluate trade-offs between different temporal modeling approaches

## Why Temporal Modeling Matters

### The Limitation of Per-Frame Analysis

Consider the challenge of recognizing actions like "running" vs "walking". Individual frames may look nearly identical—the key discriminating information lies in **how** the scene changes over time:

- **Velocity**: How fast are limbs moving?
- **Periodicity**: What is the stride frequency?
- **Trajectory**: What path do objects follow?

### Motion as the Key Signal

Temporal modeling captures **motion patterns** that single-frame analysis cannot:

$$\text{Motion}_{t \to t+1} = I_{t+1} - I_t \approx \frac{\partial I}{\partial t}$$

More sophisticated motion representations include:

1. **Optical flow**: Dense motion vectors between frames
2. **Temporal gradients**: Frame-to-frame differences
3. **Learned motion features**: Implicitly learned by temporal networks

## Approaches to Temporal Modeling

### Taxonomy of Methods

| Approach | Temporal Scope | Computation | Key Advantage |
|----------|---------------|-------------|---------------|
| Late Fusion | Global | Low | Simple, modular |
| 3D Convolution | Local (kernel size) | High | End-to-end learning |
| RNN/LSTM | Flexible | Medium | Variable-length sequences |
| Temporal Attention | Global | Medium-High | Long-range dependencies |
| Factorized (2D+1D) | Local | Medium | Parameter efficient |

### Late Fusion

Process each frame independently with a 2D CNN, then aggregate predictions:

```python
import torch
import torch.nn as nn
import torchvision.models as models

class LateFusionModel(nn.Module):
    """
    Late fusion: aggregate predictions from individual frames.
    
    Mathematical formulation:
        f_t = CNN(I_t)           # Per-frame features
        p = aggregate({f_1, ..., f_T})  # Temporal aggregation
    """
    
    def __init__(self, num_classes: int, aggregation: str = 'mean'):
        super().__init__()
        self.aggregation = aggregation
        
        # Pre-trained 2D CNN backbone
        backbone = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        
        # Classifier
        self.classifier = nn.Linear(2048, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Video tensor (B, T, C, H, W)
        Returns:
            Class logits (B, num_classes)
        """
        B, T, C, H, W = x.shape
        
        # Reshape for batch processing: (B*T, C, H, W)
        x = x.view(B * T, C, H, W)
        
        # Extract features for each frame
        features = self.features(x)  # (B*T, 2048, 1, 1)
        features = features.flatten(1)  # (B*T, 2048)
        
        # Reshape back: (B, T, 2048)
        features = features.view(B, T, -1)
        
        # Temporal aggregation
        if self.aggregation == 'mean':
            aggregated = features.mean(dim=1)
        elif self.aggregation == 'max':
            aggregated, _ = features.max(dim=1)
        elif self.aggregation == 'last':
            aggregated = features[:, -1, :]
        
        # Classify
        return self.classifier(aggregated)
```

### Early Fusion

Concatenate frames along the channel dimension:

```python
class EarlyFusionModel(nn.Module):
    """
    Early fusion: stack frames as input channels.
    
    Mathematical formulation:
        Input: concat(I_1, ..., I_T) ∈ R^(T*C, H, W)
        Output: CNN(Input) → p
    """
    
    def __init__(self, num_frames: int, num_classes: int):
        super().__init__()
        
        # Modified first conv layer for T*C input channels
        self.conv1 = nn.Conv2d(
            num_frames * 3,  # T RGB frames stacked
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        
        # Rest of ResNet (except first conv)
        resnet = models.resnet50(pretrained=False)
        self.features = nn.Sequential(
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool
        )
        
        self.classifier = nn.Linear(2048, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Video tensor (B, T, C, H, W)
        Returns:
            Class logits (B, num_classes)
        """
        B, T, C, H, W = x.shape
        
        # Stack along channel dimension: (B, T*C, H, W)
        x = x.view(B, T * C, H, W)
        
        # Forward through network
        x = self.conv1(x)
        x = self.features(x)
        x = x.flatten(1)
        
        return self.classifier(x)
```

### Factorized Spatiotemporal Convolution (2D+1D)

Separate spatial and temporal processing for efficiency:

```python
class Factorized2D1DConv(nn.Module):
    """
    Factorized (2+1)D convolution.
    
    Decomposes 3D convolution into:
        1. 2D spatial convolution: (1, k, k)
        2. 1D temporal convolution: (t, 1, 1)
    
    Benefits:
        - Fewer parameters than full 3D conv
        - Doubles the number of nonlinearities
        - Easier optimization
    """
    
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int,
                 spatial_kernel: int = 3,
                 temporal_kernel: int = 3):
        super().__init__()
        
        # Calculate intermediate channels for same parameter count
        # as equivalent 3D conv: c_i * c_o * t * k * k = c_i * M * k * k + M * c_o * t
        # M ≈ (t * k * k * c_i * c_o) / (k * k * c_i + t * c_o)
        M = (temporal_kernel * spatial_kernel * spatial_kernel * 
             in_channels * out_channels) // \
            (spatial_kernel * spatial_kernel * in_channels + 
             temporal_kernel * out_channels)
        
        # Spatial convolution: process each frame
        self.spatial_conv = nn.Conv3d(
            in_channels, M,
            kernel_size=(1, spatial_kernel, spatial_kernel),
            padding=(0, spatial_kernel // 2, spatial_kernel // 2),
            bias=False
        )
        self.spatial_bn = nn.BatchNorm3d(M)
        
        # Temporal convolution: aggregate across time
        self.temporal_conv = nn.Conv3d(
            M, out_channels,
            kernel_size=(temporal_kernel, 1, 1),
            padding=(temporal_kernel // 2, 0, 0),
            bias=False
        )
        self.temporal_bn = nn.BatchNorm3d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, C, T, H, W)
        Returns:
            Output tensor (B, out_C, T, H, W)
        """
        # Spatial processing
        x = self.spatial_conv(x)
        x = self.spatial_bn(x)
        x = self.relu(x)
        
        # Temporal processing
        x = self.temporal_conv(x)
        x = self.temporal_bn(x)
        x = self.relu(x)
        
        return x
```

## Temporal Pooling Strategies

### Average Pooling

Simple but effective baseline:

$$f_{avg} = \frac{1}{T} \sum_{t=1}^{T} f_t$$

### Max Pooling

Captures strongest activation across time:

$$f_{max} = \max_{t \in \{1, \ldots, T\}} f_t$$

### Learned Temporal Weights

Attention-based weighting:

```python
class TemporalAttentionPooling(nn.Module):
    """
    Soft attention pooling over temporal dimension.
    
    Mathematical formulation:
        α_t = softmax(W · f_t)     # Attention weights
        f = Σ α_t · f_t            # Weighted aggregation
    """
    
    def __init__(self, feature_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Temporal features (B, T, D)
        Returns:
            Aggregated features (B, D)
        """
        # Compute attention scores
        scores = self.attention(features)  # (B, T, 1)
        weights = torch.softmax(scores, dim=1)  # (B, T, 1)
        
        # Weighted aggregation
        aggregated = (weights * features).sum(dim=1)  # (B, D)
        
        return aggregated
```

### NetVLAD

Cluster-based aggregation for discriminative representation:

```python
class NetVLAD(nn.Module):
    """
    NetVLAD: CNN architecture for weakly supervised place recognition.
    
    Clusters features and aggregates residuals to cluster centers.
    
    Mathematical formulation:
        V(j, k) = Σ α_k(f_t) · (f_t - c_k)
        
    where:
        - α_k(f_t) = soft assignment to cluster k
        - c_k = cluster center k
        - V = VLAD descriptor matrix
    """
    
    def __init__(self, 
                 feature_dim: int, 
                 num_clusters: int = 64,
                 normalize: bool = True):
        super().__init__()
        
        self.num_clusters = num_clusters
        self.feature_dim = feature_dim
        self.normalize = normalize
        
        # Soft assignment parameters
        self.conv = nn.Conv2d(feature_dim, num_clusters, kernel_size=1)
        
        # Cluster centers
        self.centroids = nn.Parameter(
            torch.randn(num_clusters, feature_dim)
        )
        
        nn.init.normal_(self.centroids, std=1 / np.sqrt(feature_dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Features (B, C, T, H, W) or (B, C, N)
        Returns:
            VLAD descriptor (B, num_clusters * C)
        """
        B, C = x.shape[:2]
        
        # Flatten spatial/temporal dimensions
        if x.dim() == 5:
            x = x.view(B, C, -1)  # (B, C, T*H*W)
        
        N = x.shape[2]
        
        # Soft assignment
        soft_assign = self.conv(x.unsqueeze(-1)).squeeze(-1)  # (B, K, N)
        soft_assign = torch.softmax(soft_assign, dim=1)
        
        # Compute residuals
        x_expand = x.unsqueeze(1)  # (B, 1, C, N)
        c_expand = self.centroids.unsqueeze(0).unsqueeze(-1)  # (1, K, C, 1)
        
        residuals = x_expand - c_expand  # (B, K, C, N)
        
        # Weighted residuals
        soft_assign_expand = soft_assign.unsqueeze(2)  # (B, K, 1, N)
        vlad = (soft_assign_expand * residuals).sum(dim=-1)  # (B, K, C)
        
        # Normalize
        vlad = vlad.view(B, -1)  # (B, K*C)
        
        if self.normalize:
            vlad = F.normalize(vlad, p=2, dim=1)
        
        return vlad
```

## Multi-Scale Temporal Modeling

### Temporal Pyramid

Capture information at multiple temporal granularities:

```python
class TemporalPyramidPooling(nn.Module):
    """
    Temporal pyramid pooling for multi-scale temporal features.
    
    Aggregates features at multiple temporal scales:
        - Level 0: Global pooling (1 bin)
        - Level 1: 2 temporal bins
        - Level 2: 4 temporal bins
        - etc.
    """
    
    def __init__(self, levels: list = [1, 2, 4, 8]):
        super().__init__()
        self.levels = levels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Temporal features (B, T, D)
        Returns:
            Multi-scale features (B, sum(levels) * D)
        """
        B, T, D = x.shape
        pyramids = []
        
        for num_bins in self.levels:
            # Compute bin size
            bin_size = T // num_bins
            
            for i in range(num_bins):
                start = i * bin_size
                end = (i + 1) * bin_size if i < num_bins - 1 else T
                
                # Pool within bin
                bin_features = x[:, start:end, :].mean(dim=1)  # (B, D)
                pyramids.append(bin_features)
        
        # Concatenate all pyramid levels
        return torch.cat(pyramids, dim=1)  # (B, sum(levels) * D)
```

### SlowFast Dual-Pathway

Process video at different temporal resolutions:

```python
class SlowFastPathways(nn.Module):
    """
    SlowFast network dual-pathway design.
    
    Two pathways operating at different temporal speeds:
    
    Slow Pathway:
        - Low frame rate (e.g., 4 FPS)
        - More spatial capacity
        - Captures spatial semantics
    
    Fast Pathway:
        - High frame rate (e.g., 32 FPS)
        - Lightweight (fewer channels)
        - Captures fine temporal patterns
    
    Lateral connections fuse information between pathways.
    """
    
    def __init__(self, 
                 alpha: int = 8,      # Frame rate ratio (fast/slow)
                 beta: float = 1/8):  # Channel ratio (fast/slow)
        super().__init__()
        
        self.alpha = alpha
        self.beta = beta
        
        # Slow pathway (heavier, fewer frames)
        self.slow_conv1 = nn.Conv3d(3, 64, kernel_size=(1, 7, 7),
                                    stride=(1, 2, 2), padding=(0, 3, 3))
        
        # Fast pathway (lighter, more frames)
        fast_channels = int(64 * beta)
        self.fast_conv1 = nn.Conv3d(3, fast_channels, kernel_size=(5, 7, 7),
                                    stride=(1, 2, 2), padding=(2, 3, 3))
        
        # Lateral connection: fast → slow
        self.lateral = nn.Conv3d(
            fast_channels, 
            fast_channels * 2,  # Doubles for concatenation
            kernel_size=(5, 1, 1),
            stride=(alpha, 1, 1),
            padding=(2, 0, 0)
        )
    
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Video tensor (B, C, T, H, W)
        Returns:
            slow_features, fast_features
        """
        B, C, T, H, W = x.shape
        
        # Subsample for slow pathway
        slow_input = x[:, :, ::self.alpha, :, :]  # Every alpha-th frame
        
        # Full resolution for fast pathway
        fast_input = x
        
        # Process through pathways
        slow_features = self.slow_conv1(slow_input)
        fast_features = self.fast_conv1(fast_input)
        
        # Lateral connection (fuse fast → slow)
        lateral_features = self.lateral(fast_features)
        slow_features = torch.cat([slow_features, lateral_features], dim=1)
        
        return slow_features, fast_features
```

## Comparison and Trade-offs

| Method | Pros | Cons | Best For |
|--------|------|------|----------|
| Late Fusion | Simple, modular, uses pretrained 2D CNNs | No low-level temporal features | Quick baseline |
| Early Fusion | Captures temporal patterns | Limited temporal range, no pretrained weights | Short clips |
| 3D Conv | End-to-end learning | Computationally expensive | Action recognition |
| 2D+1D | Parameter efficient | May miss joint spatiotemporal patterns | Resource-constrained |
| RNN/LSTM | Variable-length sequences | Sequential processing (slow) | Long videos |
| Attention | Long-range dependencies | Quadratic complexity | Fine-grained temporal |

## PyTorch Implementation Tips

### Efficient Batch Processing

```python
def process_video_batch(videos: torch.Tensor, 
                       frame_model: nn.Module) -> torch.Tensor:
    """
    Efficiently process video batch through 2D CNN.
    
    Trick: Reshape (B, T, C, H, W) → (B*T, C, H, W) for batch processing.
    """
    B, T, C, H, W = videos.shape
    
    # Flatten batch and time
    flat = videos.view(B * T, C, H, W)
    
    # Process all frames in parallel
    features = frame_model(flat)  # (B*T, D)
    
    # Reshape back
    return features.view(B, T, -1)  # (B, T, D)
```

### Memory-Efficient Training

```python
# Use gradient checkpointing for long videos
from torch.utils.checkpoint import checkpoint

class MemoryEfficientTemporal(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x):
        for layer in self.layers:
            x = checkpoint(layer, x)
        return x
```

## Summary

Temporal modeling is crucial for video understanding, enabling networks to capture motion patterns and dynamic information. The choice of approach depends on:

1. **Computational budget**: Late fusion is cheapest, 3D conv most expensive
2. **Temporal range**: Attention for long-range, local conv for short-range
3. **Dataset characteristics**: Some tasks need fine-grained motion, others coarse semantics
4. **Deployment constraints**: Real-time applications favor efficient factorized approaches

## Next Steps

- **3D Convolutions**: Deep dive into spatiotemporal feature learning
- **Two-Stream Networks**: Combining appearance and motion streams
- **Video Transformers**: Attention-based temporal modeling at scale
