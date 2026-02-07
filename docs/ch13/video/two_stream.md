# Two-Stream Networks for Action Recognition

## Learning Objectives

By the end of this section, you will be able to:

- Understand the motivation for two-stream architectures
- Implement spatial and temporal stream networks
- Compute and preprocess optical flow for the temporal stream
- Design effective fusion strategies for combining stream predictions
- Evaluate the complementary nature of appearance and motion

## Motivation: Appearance and Motion

### The Two-Stream Hypothesis

Human visual processing involves two distinct pathways:

1. **Ventral stream** (what): Object recognition, shape, color
2. **Dorsal stream** (where/how): Motion, spatial relationships

Two-stream networks mimic this by separately processing:

- **Spatial stream**: RGB frames for appearance (objects, scenes, poses)
- **Temporal stream**: Optical flow for motion (velocity, direction, dynamics)

### Why Separate Streams?

Single-frame models miss critical temporal information:

| Action | Appearance | Motion Required? |
|--------|-----------|------------------|
| "Standing" vs "Walking" | Same pose | Yes - leg movement |
| "Drinking" vs "Pouring" | Same objects | Yes - hand trajectory |
| "Opening door" vs "Closing door" | Same scene | Yes - door direction |

### Mathematical Framework

Given video $V$ with frames $\{I_1, \ldots, I_T\}$ and optical flow $\{F_1, \ldots, F_{T-1}\}$:

**Spatial stream:**
$$p_{spatial} = f_s(I_t) \in \mathbb{R}^K$$

**Temporal stream:**
$$p_{temporal} = f_t(\{F_{t}, F_{t+1}, \ldots, F_{t+L-1}\}) \in \mathbb{R}^K$$

**Fusion:**
$$p_{final} = \alpha \cdot p_{spatial} + (1-\alpha) \cdot p_{temporal}$$

where $K$ is the number of classes and $\alpha$ is the fusion weight.

## Spatial Stream

### Architecture

The spatial stream processes single RGB frames using standard 2D CNNs:

```python
import torch
import torch.nn as nn
import torchvision.models as models

class SpatialStream(nn.Module):
    """
    Spatial stream for appearance-based recognition.
    
    Uses pretrained ImageNet models for transfer learning.
    Key insight: Actions often correlate with objects and scenes.
    """
    
    def __init__(self, 
                 num_classes: int = 101,
                 backbone: str = 'resnet50',
                 pretrained: bool = True,
                 dropout: float = 0.5):
        super().__init__()
        
        # Load pretrained backbone
        if backbone == 'resnet50':
            base = models.resnet50(pretrained=pretrained)
            self.feature_dim = 2048
        elif backbone == 'resnet101':
            base = models.resnet101(pretrained=pretrained)
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Remove final FC layer
        self.features = nn.Sequential(*list(base.children())[:-1])
        
        # Action classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.feature_dim, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: RGB frames (B, 3, H, W) or video (B, T, 3, H, W)
        Returns:
            Class logits (B, num_classes)
        """
        # Handle video input
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
            is_video = True
        else:
            is_video = False
            B = x.shape[0]
        
        # Extract features
        features = self.features(x)
        features = features.flatten(1)  # (B*T, feature_dim)
        
        # Classify
        logits = self.classifier(features)
        
        # Average predictions over time for video
        if is_video:
            logits = logits.view(B, T, -1).mean(dim=1)
        
        return logits
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification."""
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
            features = self.features(x).flatten(1)
            features = features.view(B, T, -1)
        else:
            features = self.features(x).flatten(1)
        return features
```

### Multi-Frame Sampling

For video-level prediction, sample multiple frames:

```python
def sample_frames_spatial(video: torch.Tensor, 
                          num_samples: int = 25) -> torch.Tensor:
    """
    Sample frames for spatial stream testing.
    
    Strategy: 25 frames uniformly sampled across video.
    Prediction: Average of per-frame softmax scores.
    """
    T = video.shape[0]
    indices = torch.linspace(0, T - 1, num_samples).long()
    return video[indices]
```

## Temporal Stream

### Optical Flow Input

The temporal stream processes stacked optical flow:

```python
class TemporalStream(nn.Module):
    """
    Temporal stream for motion-based recognition.
    
    Input: Stack of L consecutive optical flow fields.
    Each flow field has 2 channels (horizontal u, vertical v).
    Total input channels: 2L
    
    Typical configuration: L = 10 (10 flow fields = 20 channels)
    """
    
    def __init__(self,
                 num_classes: int = 101,
                 flow_length: int = 10,
                 dropout: float = 0.5):
        super().__init__()
        
        self.flow_length = flow_length
        input_channels = 2 * flow_length  # u and v for each flow
        
        # Modified ResNet for flow input
        # Cannot use pretrained weights (different input channels)
        resnet = models.resnet50(pretrained=False)
        
        # Replace first conv layer
        self.conv1 = nn.Conv2d(
            input_channels, 64,
            kernel_size=7, stride=2, padding=3, bias=False
        )
        
        # Initialize with mean of pretrained RGB weights
        # This helps transfer some spatial structure knowledge
        if True:  # Optional weight initialization
            pretrained = models.resnet50(pretrained=True)
            pretrained_weight = pretrained.conv1.weight.data
            # Average over RGB channels, repeat for flow channels
            mean_weight = pretrained_weight.mean(dim=1, keepdim=True)
            self.conv1.weight.data = mean_weight.repeat(1, input_channels, 1, 1)
            self.conv1.weight.data /= input_channels  # Scale appropriately
        
        # Copy remaining layers
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(2048, num_classes)
        )
    
    def forward(self, flow: torch.Tensor) -> torch.Tensor:
        """
        Args:
            flow: Stacked optical flow (B, 2*L, H, W)
        Returns:
            Class logits (B, num_classes)
        """
        x = self.conv1(flow)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.flatten(1)
        
        return self.classifier(x)
```

## Optical Flow Computation

### Mathematical Background

Optical flow estimates motion between frames based on brightness constancy:

$$I(x, y, t) = I(x + u, y + v, t + \Delta t)$$

where $(u, v)$ is the flow vector (displacement per unit time).

Taylor expansion and simplification yield:

$$I_x u + I_y v + I_t = 0$$

where $I_x$, $I_y$, $I_t$ are image gradients.

### Dense Optical Flow with OpenCV

```python
import cv2
import numpy as np

def compute_optical_flow(frame1: np.ndarray, 
                         frame2: np.ndarray,
                         method: str = 'farneback') -> np.ndarray:
    """
    Compute dense optical flow between two frames.
    
    Args:
        frame1: Previous frame (H, W, 3) RGB, values [0, 1]
        frame2: Current frame (H, W, 3) RGB, values [0, 1]
        method: Flow algorithm ('farneback' or 'tvl1')
    
    Returns:
        flow: Optical flow (H, W, 2) with (u, v) components
        
    The flow vectors indicate pixel displacement:
        - u: horizontal motion (positive = right)
        - v: vertical motion (positive = down)
    """
    # Convert to grayscale uint8
    gray1 = cv2.cvtColor((frame1 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor((frame2 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    
    if method == 'farneback':
        # Gunnar Farneback's algorithm
        # Uses polynomial expansion to approximate neighborhoods
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None,
            pyr_scale=0.5,    # Pyramid scale (0.5 = classical pyramid)
            levels=3,          # Number of pyramid layers
            winsize=15,        # Averaging window size
            iterations=3,      # Iterations at each pyramid level
            poly_n=5,          # Size of pixel neighborhood
            poly_sigma=1.2,    # Gaussian std for polynomial expansion
            flags=0
        )
    elif method == 'tvl1':
        # TV-L1 algorithm (more accurate, slower)
        tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
        flow = tvl1.calc(gray1, gray2, None)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return flow  # (H, W, 2)


def extract_flow_stack(video: torch.Tensor, 
                       flow_length: int = 10,
                       normalize: bool = True) -> torch.Tensor:
    """
    Extract stacked optical flow from video for temporal stream.
    
    Args:
        video: Video tensor (T, C, H, W) with values [0, 1]
        flow_length: Number of flow fields to stack (L)
        normalize: Whether to normalize flow values
    
    Returns:
        flow_stack: Stacked flows (2*L, H, W)
    """
    T, C, H, W = video.shape
    
    if T < flow_length + 1:
        raise ValueError(f"Need at least {flow_length + 1} frames")
    
    flows = []
    for t in range(flow_length):
        # Convert to numpy for OpenCV
        frame1 = video[t].permute(1, 2, 0).numpy()  # (H, W, C)
        frame2 = video[t + 1].permute(1, 2, 0).numpy()
        
        # Compute flow
        flow = compute_optical_flow(frame1, frame2)  # (H, W, 2)
        
        # Normalize flow values
        if normalize:
            # Typical flow values are in [-20, 20] pixels
            # Normalize to [-1, 1]
            flow = np.clip(flow / 20.0, -1, 1)
        
        # Convert to tensor: (2, H, W)
        flow_tensor = torch.from_numpy(flow).permute(2, 0, 1).float()
        flows.append(flow_tensor)
    
    # Stack: (L, 2, H, W) → (2*L, H, W)
    flow_stack = torch.cat(flows, dim=0)
    
    return flow_stack
```

### Optical Flow Visualization

```python
def visualize_flow(flow: np.ndarray) -> np.ndarray:
    """
    Convert optical flow to RGB visualization.
    
    Encoding:
        - Hue: Flow direction (angle)
        - Saturation: Maximum (constant)
        - Value: Flow magnitude
    
    Args:
        flow: Optical flow (H, W, 2)
    Returns:
        RGB visualization (H, W, 3)
    """
    h, w = flow.shape[:2]
    
    # Compute magnitude and angle
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # Create HSV image
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2  # Hue: direction
    hsv[..., 1] = 255  # Saturation: maximum
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # Value: magnitude
    
    # Convert to RGB
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    return rgb
```

## Fusion Strategies

### Late Fusion (Score Averaging)

```python
class TwoStreamNetwork(nn.Module):
    """
    Complete two-stream network with fusion.
    
    Fusion options:
    1. Average: Simple average of softmax scores
    2. Weighted: Learnable or fixed weights
    3. Learned: MLP combines feature vectors
    """
    
    def __init__(self,
                 num_classes: int = 101,
                 flow_length: int = 10,
                 fusion: str = 'average',
                 spatial_weight: float = 0.4):
        super().__init__()
        
        self.spatial = SpatialStream(num_classes)
        self.temporal = TemporalStream(num_classes, flow_length)
        self.fusion = fusion
        
        if fusion == 'weighted':
            # Fixed or learnable weight
            self.alpha = nn.Parameter(torch.tensor(spatial_weight))
        elif fusion == 'learned':
            # Concatenate features, learn combination
            self.fusion_net = nn.Sequential(
                nn.Linear(num_classes * 2, num_classes * 2),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(num_classes * 2, num_classes)
            )
    
    def forward(self, 
                rgb: torch.Tensor, 
                flow: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb: RGB frames (B, 3, H, W) or (B, T, 3, H, W)
            flow: Optical flow stack (B, 2*L, H, W)
        Returns:
            Fused class logits (B, num_classes)
        """
        # Get stream predictions
        spatial_logits = self.spatial(rgb)
        temporal_logits = self.temporal(flow)
        
        # Fusion
        if self.fusion == 'average':
            # Simple average (empirically: temporal slightly better)
            fused = (spatial_logits + temporal_logits) / 2
            
        elif self.fusion == 'weighted':
            # Weighted combination
            alpha = torch.sigmoid(self.alpha)  # Keep in [0, 1]
            fused = alpha * spatial_logits + (1 - alpha) * temporal_logits
            
        elif self.fusion == 'learned':
            # Concatenate and learn
            combined = torch.cat([spatial_logits, temporal_logits], dim=1)
            fused = self.fusion_net(combined)
        
        return fused
```

### Feature-Level Fusion

Combine features before classification:

```python
class FeatureFusionTwoStream(nn.Module):
    """
    Early fusion: Combine features from both streams.
    
    More expressive than score fusion but requires
    matching feature dimensions.
    """
    
    def __init__(self, num_classes: int = 101, flow_length: int = 10):
        super().__init__()
        
        self.spatial = SpatialStream(num_classes)
        self.temporal = TemporalStream(num_classes, flow_length)
        
        # Feature dimension: 2048 from ResNet-50
        self.fusion_layer = nn.Sequential(
            nn.Linear(2048 * 2, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, num_classes)
        )
    
    def forward(self, rgb, flow):
        # Extract features (not logits)
        spatial_feat = self.spatial.extract_features(rgb)  # (B, 2048)
        temporal_feat = self.temporal.features(flow).flatten(1)  # (B, 2048)
        
        # Concatenate and fuse
        combined = torch.cat([spatial_feat, temporal_feat], dim=1)
        return self.fusion_layer(combined)
```

## Training Procedure

### Separate Pre-training

The original two-stream paper trains streams separately:

```python
def train_two_stream_separate():
    """
    Training procedure for two-stream networks.
    
    1. Train spatial stream on RGB frames (can use ImageNet pretrained)
    2. Train temporal stream on optical flow stacks
    3. Fuse predictions at test time
    """
    
    # Spatial stream training
    spatial_stream = SpatialStream(num_classes=101)
    spatial_optimizer = torch.optim.SGD(
        spatial_stream.parameters(),
        lr=0.01, momentum=0.9, weight_decay=5e-4
    )
    
    for epoch in range(epochs):
        for frames, labels in spatial_loader:
            # Sample one frame per video
            idx = torch.randint(0, frames.shape[1], (1,)).item()
            single_frame = frames[:, idx]
            
            output = spatial_stream(single_frame)
            loss = F.cross_entropy(output, labels)
            
            spatial_optimizer.zero_grad()
            loss.backward()
            spatial_optimizer.step()
    
    # Temporal stream training (similarly)
    temporal_stream = TemporalStream(num_classes=101)
    temporal_optimizer = torch.optim.SGD(
        temporal_stream.parameters(),
        lr=0.01, momentum=0.9, weight_decay=5e-4
    )
    
    for epoch in range(epochs):
        for flow_stacks, labels in temporal_loader:
            output = temporal_stream(flow_stacks)
            loss = F.cross_entropy(output, labels)
            
            temporal_optimizer.zero_grad()
            loss.backward()
            temporal_optimizer.step()
```

### End-to-End Training

Joint training with both streams:

```python
def train_two_stream_e2e(model, train_loader, optimizer, epochs):
    """End-to-end training of fused two-stream network."""
    
    for epoch in range(epochs):
        for rgb, flow, labels in train_loader:
            rgb, flow, labels = rgb.cuda(), flow.cuda(), labels.cuda()
            
            output = model(rgb, flow)
            loss = F.cross_entropy(output, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

## Performance Analysis

### Complementary Information

The two streams capture different aspects:

| Stream | Strengths | Weaknesses |
|--------|-----------|------------|
| Spatial | Objects, scenes, poses | Cannot distinguish motion-dependent actions |
| Temporal | Motion patterns, speed | Ignores object identity |
| Fused | Both appearance and motion | More computation |

### Typical Accuracy (UCF-101)

| Method | Spatial | Temporal | Fused |
|--------|---------|----------|-------|
| Original Two-Stream | 73.0% | 83.7% | 88.0% |
| With VGG-16 | 78.4% | 85.1% | 91.4% |
| With ResNet | 82.3% | 87.2% | 93.6% |

Key observation: **Temporal stream alone outperforms spatial**, showing the importance of motion for action recognition.

## Summary

Two-stream networks demonstrate that:

1. **Appearance and motion** are complementary cues for action recognition
2. **Optical flow** provides explicit motion representation
3. **Late fusion** is simple but effective
4. **Temporal information** is often more discriminative than appearance alone
5. **Separate pretraining** allows leveraging ImageNet weights for spatial stream

### Limitations

- **Optical flow computation** is expensive (often precomputed offline)
- **Two separate networks** double parameters and inference time
- **No long-range temporal modeling** (limited to flow stack length)

### Modern Alternatives

- **3D CNNs** (C3D, I3D): Learn motion implicitly from RGB
- **SlowFast**: Dual-pathway with different temporal resolutions
- **Video Transformers**: Attention-based spatiotemporal modeling

## Next Steps

- **Optical Flow Details**: Horn-Schunck, Lucas-Kanade, learned flow
- **CNN-LSTM**: Recurrent temporal modeling
- **Video Transformers**: Attention for video understanding
