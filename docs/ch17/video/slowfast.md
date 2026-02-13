# SlowFast Networks

## Learning Objectives

By the end of this section, you will be able to:

- Understand the dual-pathway design philosophy
- Implement Slow and Fast pathways with lateral connections
- Configure the α (frame rate) and β (channel) ratios
- Train SlowFast networks for action recognition
- Apply SlowFast to various video understanding tasks

## Design Philosophy

### Biological Inspiration

The human visual system processes information at different temporal scales:
- **Magnocellular pathway**: Fast, low spatial resolution, motion-sensitive
- **Parvocellular pathway**: Slow, high spatial resolution, detail-oriented

SlowFast mimics this with two network pathways:

### Two Pathways

| Pathway | Frame Rate | Channels | Focus |
|---------|-----------|----------|-------|
| **Slow** | Low (e.g., 4 fps) | High (e.g., 64) | Spatial semantics |
| **Fast** | High (e.g., 32 fps) | Low (e.g., 8) | Temporal dynamics |

### Key Parameters

**α (alpha)**: Frame rate ratio between pathways
$$\text{Fast frames} = \alpha \times \text{Slow frames}$$

Typically α = 8, meaning Fast processes 8× more frames.

**β (beta)**: Channel ratio
$$\text{Fast channels} = \beta \times \text{Slow channels}$$

Typically β = 1/8, meaning Fast has 8× fewer channels.

## Architecture

```python
import torch
import torch.nn as nn

class SlowFast(nn.Module):
    """
    SlowFast Networks for Video Recognition
    (Feichtenhofer et al., 2019)
    
    Two-pathway architecture:
    - Slow pathway: Low frame rate, high channel capacity
    - Fast pathway: High frame rate, low channel capacity
    - Lateral connections fuse information between pathways
    """
    
    def __init__(self,
                 num_classes: int = 400,
                 alpha: int = 8,      # Frame rate ratio
                 beta: float = 1/8,   # Channel ratio
                 slow_channels: int = 64,
                 num_frames: int = 32):
        super().__init__()
        
        self.alpha = alpha
        self.beta = beta
        self.num_frames = num_frames
        
        fast_channels = int(slow_channels * beta)
        
        # Slow pathway
        self.slow_pathway = SlowPathway(
            in_channels=3,
            base_channels=slow_channels
        )
        
        # Fast pathway
        self.fast_pathway = FastPathway(
            in_channels=3,
            base_channels=fast_channels,
            alpha=alpha
        )
        
        # Lateral connections (Fast → Slow)
        self.lateral_connections = nn.ModuleList([
            LateralConnection(fast_channels * mult, slow_channels * mult, alpha)
            for mult in [1, 2, 4, 8]  # At each stage
        ])
        
        # Final classification
        slow_out = slow_channels * 8
        fast_out = fast_channels * 8
        self.head = nn.Linear(slow_out + fast_out, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Video tensor (B, C, T, H, W)
               T should be divisible by alpha
        
        Returns:
            Class logits (B, num_classes)
        """
        B, C, T, H, W = x.shape
        
        # Sample frames for each pathway
        # Slow: every alpha-th frame
        x_slow = x[:, :, ::self.alpha, :, :]  # (B, C, T/α, H, W)
        
        # Fast: all frames
        x_fast = x  # (B, C, T, H, W)
        
        # Process through pathways with lateral connections
        slow_features = []
        fast_features = []
        
        x_slow, x_fast = self._forward_stem(x_slow, x_fast)
        
        for i, (slow_block, fast_block, lateral) in enumerate(
            zip(self.slow_pathway.stages, 
                self.fast_pathway.stages,
                self.lateral_connections)
        ):
            x_fast = fast_block(x_fast)
            lateral_out = lateral(x_fast)
            
            # Fuse lateral connection with slow pathway
            x_slow = torch.cat([x_slow, lateral_out], dim=1)
            x_slow = slow_block(x_slow)
        
        # Global average pooling
        x_slow = x_slow.mean(dim=[2, 3, 4])  # (B, C_slow)
        x_fast = x_fast.mean(dim=[2, 3, 4])  # (B, C_fast)
        
        # Concatenate and classify
        x = torch.cat([x_slow, x_fast], dim=1)
        
        return self.head(x)


class SlowPathway(nn.Module):
    """
    Slow pathway: processes fewer frames with more channels.
    
    Focuses on spatial semantics and object recognition.
    """
    
    def __init__(self, in_channels: int, base_channels: int):
        super().__init__()
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, 
                     kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3)),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )
        
        # Residual stages (channels: 64 → 128 → 256 → 512)
        self.stages = nn.ModuleList([
            self._make_stage(base_channels, base_channels, blocks=3, temporal_stride=1),
            self._make_stage(base_channels, base_channels * 2, blocks=4, temporal_stride=1),
            self._make_stage(base_channels * 2, base_channels * 4, blocks=6, temporal_stride=2),
            self._make_stage(base_channels * 4, base_channels * 8, blocks=3, temporal_stride=2),
        ])
    
    def _make_stage(self, in_ch, out_ch, blocks, temporal_stride):
        layers = [ResBlock3D(in_ch, out_ch, temporal_stride=temporal_stride)]
        for _ in range(1, blocks):
            layers.append(ResBlock3D(out_ch, out_ch))
        return nn.Sequential(*layers)


class FastPathway(nn.Module):
    """
    Fast pathway: processes more frames with fewer channels.
    
    Focuses on motion and temporal dynamics.
    """
    
    def __init__(self, in_channels: int, base_channels: int, alpha: int):
        super().__init__()
        
        self.alpha = alpha
        
        # Stem with larger temporal kernel to capture motion
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, base_channels,
                     kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3)),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )
        
        # Stages with temporal downsampling
        self.stages = nn.ModuleList([
            self._make_stage(base_channels, base_channels, blocks=3, temporal_stride=1),
            self._make_stage(base_channels, base_channels * 2, blocks=4, temporal_stride=2),
            self._make_stage(base_channels * 2, base_channels * 4, blocks=6, temporal_stride=2),
            self._make_stage(base_channels * 4, base_channels * 8, blocks=3, temporal_stride=2),
        ])
    
    def _make_stage(self, in_ch, out_ch, blocks, temporal_stride):
        layers = [ResBlock3D(in_ch, out_ch, temporal_stride=temporal_stride)]
        for _ in range(1, blocks):
            layers.append(ResBlock3D(out_ch, out_ch))
        return nn.Sequential(*layers)


class LateralConnection(nn.Module):
    """
    Lateral connection from Fast to Slow pathway.
    
    Performs temporal downsampling to match Slow pathway's frame rate,
    then channel transformation for fusion.
    """
    
    def __init__(self, fast_channels: int, slow_channels: int, alpha: int):
        super().__init__()
        
        self.alpha = alpha
        
        # Temporal downsampling: Fast frames → Slow frames
        # Using strided 3D conv
        self.transform = nn.Sequential(
            nn.Conv3d(fast_channels, fast_channels * 2,
                     kernel_size=(5, 1, 1), 
                     stride=(alpha, 1, 1),
                     padding=(2, 0, 0)),
            nn.BatchNorm3d(fast_channels * 2),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x_fast: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_fast: Fast pathway features (B, C_fast, T_fast, H, W)
        
        Returns:
            Lateral features aligned with Slow pathway (B, C_out, T_slow, H, W)
        """
        return self.transform(x_fast)


class ResBlock3D(nn.Module):
    """3D Residual block."""
    
    def __init__(self, in_channels, out_channels, temporal_stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, 
                               kernel_size=3, stride=(temporal_stride, 1, 1), padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Identity()
        if in_channels != out_channels or temporal_stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, 
                         stride=(temporal_stride, 1, 1)),
                nn.BatchNorm3d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(identity)
        return self.relu(out)
```

## Configuration Variants

```python
def slowfast_4x16_r50():
    """
    SlowFast 4×16 R50:
    - Slow: 4 frames, ResNet-50 backbone
    - Fast: 32 frames (α=8), 1/8 channels (β=1/8)
    """
    return SlowFast(
        num_classes=400,
        alpha=8,
        beta=1/8,
        slow_channels=64,
        num_frames=32
    )


def slowfast_8x8_r101():
    """
    SlowFast 8×8 R101:
    - Slow: 8 frames
    - Fast: 32 frames (α=4)
    - ResNet-101 backbone
    """
    return SlowFast(
        num_classes=400,
        alpha=4,
        beta=1/8,
        slow_channels=64,
        num_frames=32
    )
```

## Training

```python
def train_slowfast(model, train_loader, epochs=196):
    """
    Training recipe from the paper.
    """
    
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=1e-4
    )
    
    # Half-period cosine schedule
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )
    
    # Warmup for first 34 epochs
    warmup_epochs = 34
    
    for epoch in range(epochs):
        # Linear warmup
        if epoch < warmup_epochs:
            lr = 0.1 * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
        model.train()
        for videos, labels in train_loader:
            # Videos should be (B, C, T, H, W) with T=32 frames
            videos = videos.cuda()
            labels = labels.cuda()
            
            optimizer.zero_grad()
            outputs = model(videos)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
        
        if epoch >= warmup_epochs:
            scheduler.step()
```

## Results

### Kinetics-400 Performance

| Model | Pretrain | Top-1 | Top-5 |
|-------|----------|-------|-------|
| SlowFast 4×16 R50 | - | 75.6% | 92.1% |
| SlowFast 8×8 R101 | - | 77.9% | 93.2% |
| SlowFast 16×8 R101+NL | - | 79.8% | 93.9% |

### Comparison with Other Methods

| Method | Pretrain | GFLOPs | Top-1 |
|--------|----------|--------|-------|
| I3D | ImageNet | 108 | 71.1% |
| R(2+1)D | - | 152 | 72.0% |
| SlowFast 4×16 | - | 36.1 | 75.6% |
| SlowFast 8×8 | - | 65.7 | 77.0% |

Key finding: SlowFast achieves better accuracy with fewer FLOPs.

## Summary

SlowFast's key innovations:
1. **Dual pathways** capture both spatial semantics and temporal dynamics
2. **Asymmetric design** (α, β parameters) efficiently allocates computation
3. **Lateral connections** enable cross-pathway information flow
4. **Strong performance** with reasonable computational cost

Best suited for:
- Action recognition requiring both appearance and motion
- Scenarios where temporal dynamics are important
- Balance between accuracy and efficiency
