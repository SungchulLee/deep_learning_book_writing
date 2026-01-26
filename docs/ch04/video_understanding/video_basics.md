# Video Basics: Loading and Processing

## Learning Objectives

By the end of this section, you will be able to:

- Understand video data representation as temporal sequences of frames
- Load and process videos using multiple backends (torchvision, OpenCV)
- Convert between different video tensor formats
- Implement frame extraction and sampling strategies
- Apply appropriate preprocessing for video neural networks

## Mathematical Foundation

### Video as Temporal Sequence

A video $V$ is fundamentally a sequence of $T$ frames:

$$V = \{I_1, I_2, \ldots, I_T\}$$

where each frame $I_t \in \mathbb{R}^{H \times W \times C}$ represents an image at time $t$ with height $H$, width $W$, and $C$ channels (typically 3 for RGB).

### Tensor Representation

In PyTorch, videos are represented as 4D or 5D tensors:

**Single video (4D):**
$$V \in \mathbb{R}^{T \times C \times H \times W}$$

**Batch of videos (5D):**
$$V \in \mathbb{R}^{B \times T \times C \times H \times W}$$

where $B$ is the batch size.

### Frame Rate and Duration

The relationship between temporal properties:

$$\text{Duration (seconds)} = \frac{T}{\text{FPS}}$$

$$T = \text{Duration} \times \text{FPS}$$

where FPS (frames per second) determines temporal resolution.

## Video Loading Backends

### OpenCV Backend

OpenCV provides flexible, production-ready video loading:

```python
import cv2
import numpy as np
import torch

class VideoLoader:
    """Comprehensive video loading utility."""
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        
    def load_video_opencv(self):
        """
        Load video using OpenCV backend.
        
        Returns:
            frames: NumPy array of shape (T, H, W, C)
            info: Dictionary with video metadata
        """
        cap = cv2.VideoCapture(self.video_path)
        
        # Extract video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR (OpenCV default) to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()
        
        frames = np.array(frames)  # Shape: (T, H, W, 3)
        
        info = {
            'fps': fps,
            'frame_count': frame_count,
            'width': width,
            'height': height,
            'duration': frame_count / fps
        }
        
        return frames, info
```

### Torchvision Backend

Native PyTorch integration with torchvision:

```python
from torchvision.io import read_video, write_video

def load_video_torchvision(video_path: str):
    """
    Load video using torchvision backend.
    
    Returns:
        video: Tensor of shape (T, H, W, C) with values in [0, 255]
        audio: Audio tensor if present
        info: Dictionary with video metadata
    """
    video, audio, info = read_video(
        video_path,
        pts_unit='sec'  # Use seconds for timestamps
    )
    
    return video, audio, info
```

### Format Conversion

Converting between formats is essential for different frameworks:

```python
def convert_to_pytorch_format(frames: np.ndarray) -> torch.Tensor:
    """
    Convert NumPy video to PyTorch tensor format.
    
    Args:
        frames: NumPy array (T, H, W, C) with values [0, 255]
        
    Returns:
        video_tensor: PyTorch tensor (T, C, H, W) with values [0, 1]
    """
    # Convert to tensor
    video_tensor = torch.from_numpy(frames).float()
    
    # Rearrange dimensions: (T, H, W, C) → (T, C, H, W)
    video_tensor = video_tensor.permute(0, 3, 1, 2)
    
    # Normalize to [0, 1]
    video_tensor = video_tensor / 255.0
    
    return video_tensor
```

## Frame Sampling Strategies

Sampling strategies determine which frames to process, balancing computational cost with temporal coverage.

### Uniform Sampling

Sample frames evenly distributed across the video:

$$i_k = \left\lfloor \frac{k \cdot T}{n} \right\rfloor \quad \text{for } k = 0, 1, \ldots, n-1$$

where $n$ is the number of frames to sample.

```python
def uniform_sampling(video: torch.Tensor, num_frames: int) -> torch.Tensor:
    """
    Sample frames uniformly across the video.
    
    Args:
        video: Input tensor (T, C, H, W)
        num_frames: Number of frames to sample
        
    Returns:
        Sampled tensor (num_frames, C, H, W)
    """
    T = video.shape[0]
    indices = torch.linspace(0, T - 1, num_frames).long()
    return video[indices]
```

### Temporal Stride Sampling

Sample every $s$-th frame:

```python
def stride_sampling(video: torch.Tensor, stride: int) -> torch.Tensor:
    """
    Sample frames with fixed temporal stride.
    
    Args:
        video: Input tensor (T, C, H, W)
        stride: Temporal stride (skip every stride-1 frames)
        
    Returns:
        Sampled tensor with reduced temporal dimension
    """
    return video[::stride]
```

### Dense Sampling

Extract multiple overlapping clips for temporal networks:

```python
def dense_sampling(video: torch.Tensor, 
                   clip_length: int, 
                   num_clips: int) -> list:
    """
    Extract multiple clips from video for dense predictions.
    
    Args:
        video: Input tensor (T, C, H, W)
        clip_length: Number of frames per clip
        num_clips: Number of clips to extract
        
    Returns:
        List of clip tensors, each (clip_length, C, H, W)
    """
    T = video.shape[0]
    
    if T < clip_length:
        # Pad if video is shorter than clip length
        padding = torch.zeros(clip_length - T, *video.shape[1:])
        video = torch.cat([video, padding], dim=0)
        T = clip_length
    
    # Calculate clip start positions
    max_start = T - clip_length
    if num_clips == 1:
        starts = [max_start // 2]
    else:
        starts = torch.linspace(0, max_start, num_clips).long().tolist()
    
    clips = [video[start:start + clip_length] for start in starts]
    return clips
```

### Random Sampling

Useful for data augmentation during training:

```python
def random_sampling(video: torch.Tensor, num_frames: int) -> torch.Tensor:
    """
    Randomly sample frames (for data augmentation).
    
    Args:
        video: Input tensor (T, C, H, W)
        num_frames: Number of frames to sample
        
    Returns:
        Sampled tensor (num_frames, C, H, W)
    """
    T = video.shape[0]
    
    # Random indices, then sort to maintain temporal order
    indices = torch.randint(0, T, (num_frames,))
    indices, _ = torch.sort(indices)
    
    return video[indices]
```

## Video Preprocessing

### Normalization

Apply ImageNet normalization for transfer learning:

```python
class VideoPreprocessor:
    """Preprocessing utilities for video data."""
    
    # ImageNet statistics
    MEAN = torch.tensor([0.485, 0.456, 0.406])
    STD = torch.tensor([0.229, 0.224, 0.225])
    
    def normalize(self, video: torch.Tensor) -> torch.Tensor:
        """
        Apply ImageNet normalization to video.
        
        Args:
            video: Input tensor (T, C, H, W) with values in [0, 1]
            
        Returns:
            Normalized tensor with zero mean and unit variance
        """
        # Reshape for broadcasting: (1, C, 1, 1)
        mean = self.MEAN.view(1, -1, 1, 1)
        std = self.STD.view(1, -1, 1, 1)
        
        return (video - mean) / std
    
    def denormalize(self, video: torch.Tensor) -> torch.Tensor:
        """Reverse normalization for visualization."""
        mean = self.MEAN.view(1, -1, 1, 1)
        std = self.STD.view(1, -1, 1, 1)
        
        return video * std + mean
```

### Spatial Cropping

Apply consistent spatial crops across all frames:

```python
def spatial_crop(video: torch.Tensor, 
                 crop_size: tuple,
                 position: str = 'center') -> torch.Tensor:
    """
    Perform spatial crop on all frames.
    
    Args:
        video: Input tensor (T, C, H, W)
        crop_size: (crop_h, crop_w)
        position: 'center', 'random', or 'top_left'
        
    Returns:
        Cropped tensor (T, C, crop_h, crop_w)
    """
    T, C, H, W = video.shape
    crop_h, crop_w = crop_size
    
    if position == 'center':
        top = (H - crop_h) // 2
        left = (W - crop_w) // 2
    elif position == 'random':
        top = torch.randint(0, H - crop_h + 1, (1,)).item()
        left = torch.randint(0, W - crop_w + 1, (1,)).item()
    else:  # top_left
        top, left = 0, 0
    
    return video[:, :, top:top+crop_h, left:left+crop_w]
```

### Resizing

Resize frames to target resolution:

```python
import torch.nn.functional as F

def resize_video(video: torch.Tensor, 
                 target_size: tuple) -> torch.Tensor:
    """
    Resize all frames to target size.
    
    Args:
        video: Input tensor (T, C, H, W)
        target_size: (target_h, target_w)
        
    Returns:
        Resized tensor (T, C, target_h, target_w)
    """
    T, C, H, W = video.shape
    target_h, target_w = target_size
    
    # Reshape for batch processing
    video_flat = video.view(T, C, H, W)
    
    # Use bilinear interpolation
    resized = F.interpolate(
        video_flat,
        size=(target_h, target_w),
        mode='bilinear',
        align_corners=False
    )
    
    return resized
```

## Visualization

### Frame Display

```python
import matplotlib.pyplot as plt

def visualize_frames(video: torch.Tensor, num_frames: int = 8):
    """
    Display sampled frames from video.
    
    Args:
        video: Video tensor (T, C, H, W)
        num_frames: Number of frames to display
    """
    T = video.shape[0]
    indices = torch.linspace(0, T - 1, num_frames).long()
    
    fig, axes = plt.subplots(1, num_frames, figsize=(16, 4))
    
    for i, ax in enumerate(axes):
        frame = video[indices[i]].permute(1, 2, 0)  # (C, H, W) → (H, W, C)
        frame = torch.clamp(frame, 0, 1)
        
        ax.imshow(frame.cpu().numpy())
        ax.axis('off')
        ax.set_title(f'Frame {indices[i].item()}')
    
    plt.tight_layout()
    plt.show()
```

## Summary

| Aspect | Key Points |
|--------|------------|
| **Video Format** | $V \in \mathbb{R}^{T \times C \times H \times W}$ in PyTorch |
| **Loading** | OpenCV for flexibility, torchvision for native PyTorch |
| **Sampling** | Uniform for even coverage, random for augmentation |
| **Preprocessing** | ImageNet normalization for transfer learning |
| **Considerations** | Maintain temporal consistency across all frames |

## Next Steps

With video basics established, we can now explore:

1. **3D Convolutions** - Spatiotemporal feature extraction
2. **Temporal Modeling** - Understanding motion and dynamics
3. **Two-Stream Networks** - Combining appearance and motion
