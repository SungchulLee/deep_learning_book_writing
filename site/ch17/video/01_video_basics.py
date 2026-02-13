"""
Module 34: Video Understanding - Beginner Level
File 01: Video Basics - Loading and Processing Videos

This file covers the fundamentals of working with video data:
- Understanding video as temporal sequences of frames
- Loading videos using various methods
- Basic video preprocessing operations
- Video data representation in PyTorch
- Frame extraction and sampling strategies

Mathematical Foundation:
- A video V is a sequence of T frames: V = {I_1, I_2, ..., I_T}
- Each frame I_t ∈ ℝ^(H×W×C) where H=height, W=width, C=channels
- Video tensor: V ∈ ℝ^(T×C×H×W) in PyTorch (T, C, H, W format)
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.io import read_video, write_video
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')


#=============================================================================
# PART 1: VIDEO REPRESENTATION AND LOADING
#=============================================================================

class VideoLoader:
    """
    Comprehensive video loading utility supporting multiple backends.
    
    Attributes:
        video_path: Path to video file
        backend: Loading backend ('torchvision', 'opencv', 'decord')
    """
    
    def __init__(self, video_path: str, backend: str = 'opencv'):
        """
        Initialize video loader.
        
        Args:
            video_path: Path to video file
            backend: Backend to use ('torchvision', 'opencv', 'decord')
        """
        self.video_path = video_path
        self.backend = backend
        
    def load_video_torchvision(self) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Load video using torchvision backend.
        
        Returns:
            video: Video tensor of shape (T, H, W, C) with values in [0, 255]
            audio: Audio tensor if present
            info: Dictionary with video metadata (fps, duration, etc.)
            
        Mathematical representation:
            V ∈ ℝ^(T×H×W×3) where T = number of frames
        """
        # Load video with torchvision
        # Note: torchvision returns (T, H, W, C) format
        video, audio, info = read_video(
            self.video_path,
            pts_unit='sec'  # Use seconds for timestamps
        )
        
        print(f"Video shape: {video.shape}")  # (T, H, W, C)
        print(f"FPS: {info['video_fps']}")
        print(f"Duration: {video.shape[0] / info['video_fps']:.2f} seconds")
        
        return video, audio, info
    
    def load_video_opencv(self) -> Tuple[np.ndarray, dict]:
        """
        Load video using OpenCV backend (more flexible, widely supported).
        
        Returns:
            frames: NumPy array of shape (T, H, W, C)
            info: Dictionary with video metadata
            
        OpenCV is efficient and supports many formats, good for production.
        """
        # Open video file
        cap = cv2.VideoCapture(self.video_path)
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Read all frames
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB (OpenCV loads in BGR format)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()
        
        # Convert to numpy array
        frames = np.array(frames)  # Shape: (T, H, W, 3)
        
        info = {
            'fps': fps,
            'frame_count': frame_count,
            'width': width,
            'height': height,
            'duration': frame_count / fps
        }
        
        print(f"Loaded {len(frames)} frames")
        print(f"Shape: {frames.shape}, FPS: {fps:.2f}")
        
        return frames, info
    
    def load_video_to_tensor(self, 
                            num_frames: Optional[int] = None,
                            target_size: Tuple[int, int] = (224, 224)) -> torch.Tensor:
        """
        Load video and convert to PyTorch tensor format.
        
        Args:
            num_frames: Number of frames to sample (None = all frames)
            target_size: Resize frames to (H, W)
            
        Returns:
            video_tensor: Shape (T, C, H, W) - PyTorch standard format
            
        Format conversion:
            OpenCV: (T, H, W, C) with values [0, 255]
            → PyTorch: (T, C, H, W) with values [0, 1]
        """
        frames, info = self.load_video_opencv()
        
        # Sample frames if specified
        if num_frames is not None and num_frames < len(frames):
            indices = np.linspace(0, len(frames) - 1, num_frames).astype(int)
            frames = frames[indices]
        
        # Resize frames
        resized_frames = []
        for frame in frames:
            # cv2.resize expects (W, H) not (H, W)
            resized = cv2.resize(frame, (target_size[1], target_size[0]))
            resized_frames.append(resized)
        
        # Convert to tensor and normalize
        # From: (T, H, W, C) numpy array with values [0, 255]
        # To: (T, C, H, W) tensor with values [0, 1]
        video_tensor = torch.from_numpy(np.array(resized_frames)).float()
        video_tensor = video_tensor.permute(0, 3, 1, 2)  # (T, H, W, C) → (T, C, H, W)
        video_tensor = video_tensor / 255.0  # Normalize to [0, 1]
        
        return video_tensor


#=============================================================================
# PART 2: FRAME SAMPLING STRATEGIES
#=============================================================================

class FrameSampler:
    """
    Implements various frame sampling strategies for video understanding.
    
    Sampling is crucial for:
    1. Managing computational cost (videos have many frames)
    2. Capturing temporal dynamics at different scales
    3. Handling variable-length videos
    """
    
    @staticmethod
    def uniform_sampling(video: torch.Tensor, num_frames: int) -> torch.Tensor:
        """
        Sample frames uniformly across the video.
        
        Args:
            video: Input video tensor (T, C, H, W)
            num_frames: Number of frames to sample
            
        Returns:
            Sampled video tensor (num_frames, C, H, W)
            
        Mathematical formulation:
            Sample indices: i_k = floor(k * T / num_frames) for k = 0, 1, ..., num_frames-1
            This ensures even spacing across the entire video duration
        """
        T = video.shape[0]
        
        # Generate uniformly spaced indices
        # linspace gives num_frames evenly spaced points in [0, T-1]
        indices = torch.linspace(0, T - 1, num_frames).long()
        
        # Index into video tensor
        sampled_video = video[indices]
        
        print(f"Uniform sampling: {T} → {num_frames} frames")
        print(f"Sampled indices: {indices.tolist()[:10]}...")
        
        return sampled_video
    
    @staticmethod
    def random_sampling(video: torch.Tensor, num_frames: int) -> torch.Tensor:
        """
        Randomly sample frames (useful for data augmentation).
        
        Args:
            video: Input video tensor (T, C, H, W)
            num_frames: Number of frames to sample
            
        Returns:
            Sampled video tensor (num_frames, C, H, W)
            
        Random sampling helps model learn temporal invariance
        """
        T = video.shape[0]
        
        # Sample random indices with replacement
        indices = torch.randint(0, T, (num_frames,))
        
        # Sort to maintain temporal order
        indices, _ = torch.sort(indices)
        
        sampled_video = video[indices]
        
        return sampled_video
    
    @staticmethod
    def temporal_stride_sampling(video: torch.Tensor, 
                                 stride: int) -> torch.Tensor:
        """
        Sample every stride-th frame.
        
        Args:
            video: Input video tensor (T, C, H, W)
            stride: Temporal stride (e.g., stride=2 takes every 2nd frame)
            
        Returns:
            Sampled video tensor
            
        Example:
            stride=1: all frames [0, 1, 2, 3, 4, 5, 6, 7, 8]
            stride=2: every 2nd  [0, 2, 4, 6, 8]
            stride=4: every 4th  [0, 4, 8]
        """
        # Use slicing with stride
        sampled_video = video[::stride]
        
        print(f"Stride sampling (stride={stride}): "
              f"{video.shape[0]} → {sampled_video.shape[0]} frames")
        
        return sampled_video
    
    @staticmethod
    def dense_sampling(video: torch.Tensor, 
                      clip_length: int,
                      num_clips: int = 1) -> List[torch.Tensor]:
        """
        Sample dense temporal clips from video.
        
        Args:
            video: Input video tensor (T, C, H, W)
            clip_length: Number of consecutive frames per clip
            num_clips: Number of clips to sample
            
        Returns:
            List of video clips, each of shape (clip_length, C, H, W)
            
        Used in TSN (Temporal Segment Networks) and similar architectures
        Divides video into segments and samples from each
        """
        T = video.shape[0]
        
        if T < clip_length:
            # If video is shorter than clip_length, pad or return original
            return [video]
        
        clips = []
        
        # Calculate segment length
        segment_length = (T - clip_length) // num_clips
        
        for i in range(num_clips):
            # Start of segment
            start_idx = i * segment_length
            
            # Randomly sample within segment for data augmentation
            random_offset = torch.randint(0, segment_length + 1, (1,)).item()
            clip_start = start_idx + random_offset
            clip_end = clip_start + clip_length
            
            # Extract clip
            if clip_end <= T:
                clip = video[clip_start:clip_end]
                clips.append(clip)
        
        return clips


#=============================================================================
# PART 3: VIDEO PREPROCESSING AND AUGMENTATION
#=============================================================================

class VideoPreprocessor:
    """
    Video-specific preprocessing operations.
    
    Preprocessing is crucial for:
    1. Normalizing inputs (mean, std)
    2. Handling variable resolutions
    3. Data augmentation for robustness
    """
    
    def __init__(self, 
                 mean: List[float] = [0.485, 0.456, 0.406],
                 std: List[float] = [0.229, 0.224, 0.225]):
        """
        Initialize preprocessor with normalization parameters.
        
        Args:
            mean: Channel-wise mean (ImageNet defaults)
            std: Channel-wise standard deviation
            
        Normalization formula:
            x_normalized = (x - mean) / std
        """
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)
    
    def normalize(self, video: torch.Tensor) -> torch.Tensor:
        """
        Normalize video tensor.
        
        Args:
            video: Input tensor (T, C, H, W) with values in [0, 1]
            
        Returns:
            Normalized video tensor
            
        Normalization ensures stable training and faster convergence
        """
        # Expand dimensions for broadcasting: (1, 3, 1, 1) → (T, 3, H, W)
        mean = self.mean.to(video.device)
        std = self.std.to(video.device)
        
        normalized = (video - mean) / std
        
        return normalized
    
    def denormalize(self, video: torch.Tensor) -> torch.Tensor:
        """
        Reverse normalization for visualization.
        
        Args:
            video: Normalized tensor (T, C, H, W)
            
        Returns:
            Denormalized video tensor in [0, 1]
        """
        mean = self.mean.to(video.device)
        std = self.std.to(video.device)
        
        denormalized = video * std + mean
        denormalized = torch.clamp(denormalized, 0, 1)
        
        return denormalized
    
    def temporal_crop(self, 
                     video: torch.Tensor,
                     start_frame: int,
                     num_frames: int) -> torch.Tensor:
        """
        Extract temporal crop (contiguous sequence of frames).
        
        Args:
            video: Input tensor (T, C, H, W)
            start_frame: Starting frame index
            num_frames: Number of frames to extract
            
        Returns:
            Cropped video tensor (num_frames, C, H, W)
        """
        end_frame = start_frame + num_frames
        cropped = video[start_frame:end_frame]
        
        return cropped
    
    def spatial_crop(self,
                    video: torch.Tensor,
                    crop_size: Tuple[int, int],
                    position: str = 'center') -> torch.Tensor:
        """
        Perform spatial crop on all frames.
        
        Args:
            video: Input tensor (T, C, H, W)
            crop_size: (crop_h, crop_w)
            position: 'center', 'random', or 'top_left'
            
        Returns:
            Spatially cropped video
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
        
        cropped = video[:, :, top:top+crop_h, left:left+crop_w]
        
        return cropped


#=============================================================================
# PART 4: VIDEO VISUALIZATION
#=============================================================================

def visualize_frames(video: torch.Tensor,
                    num_frames: int = 8,
                    figsize: Tuple[int, int] = (16, 4)):
    """
    Visualize sampled frames from video.
    
    Args:
        video: Video tensor (T, C, H, W)
        num_frames: Number of frames to display
        figsize: Figure size
    """
    T = video.shape[0]
    
    # Sample frames uniformly
    if T > num_frames:
        indices = torch.linspace(0, T - 1, num_frames).long()
        frames_to_show = video[indices]
    else:
        frames_to_show = video
        num_frames = T
    
    # Create subplot
    fig, axes = plt.subplots(1, num_frames, figsize=figsize)
    if num_frames == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        # Convert to displayable format
        frame = frames_to_show[i].permute(1, 2, 0)  # (C, H, W) → (H, W, C)
        frame = torch.clamp(frame, 0, 1)  # Ensure [0, 1] range
        
        ax.imshow(frame.cpu().numpy())
        ax.axis('off')
        ax.set_title(f'Frame {indices[i].item() if T > num_frames else i}')
    
    plt.tight_layout()
    plt.savefig('/home/claude/34_video_understanding/01_video_frames.png', 
                dpi=150, bbox_inches='tight')
    print(f"Visualization saved to 01_video_frames.png")
    plt.close()


def visualize_optical_flow(flow: np.ndarray,
                          figsize: Tuple[int, int] = (12, 4)):
    """
    Visualize optical flow as RGB (preview for later module).
    
    Args:
        flow: Optical flow array (H, W, 2) with (u, v) components
        figsize: Figure size
    """
    # Convert flow to HSV for visualization
    # Angle → Hue, Magnitude → Value
    h, w = flow.shape[:2]
    
    # Calculate magnitude and angle
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # Create HSV image
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2  # Angle → Hue
    hsv[..., 1] = 255  # Full saturation
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # Magnitude → Value
    
    # Convert to RGB
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    plt.figure(figsize=figsize)
    plt.imshow(rgb)
    plt.title('Optical Flow Visualization')
    plt.axis('off')
    plt.savefig('/home/claude/34_video_understanding/01_optical_flow.png',
                dpi=150, bbox_inches='tight')
    print(f"Flow visualization saved to 01_optical_flow.png")
    plt.close()


#=============================================================================
# PART 5: EXAMPLE USAGE AND DEMONSTRATIONS
#=============================================================================

def demonstrate_video_loading():
    """
    Demonstrate various video loading and processing techniques.
    """
    print("="*80)
    print("VIDEO BASICS DEMONSTRATION")
    print("="*80)
    
    # Create synthetic video for demonstration
    print("\n1. Creating synthetic video...")
    # Shape: (T, C, H, W) = (30, 3, 128, 128)
    # 30 frames of 128x128 RGB video
    T, C, H, W = 30, 3, 128, 128
    
    # Create video with moving pattern
    video = torch.zeros(T, C, H, W)
    for t in range(T):
        # Create a moving vertical bar
        bar_position = int((t / T) * W)
        video[t, :, :, max(0, bar_position-5):min(W, bar_position+5)] = 1.0
        
        # Add some noise for realism
        video[t] += torch.randn(C, H, W) * 0.1
    
    video = torch.clamp(video, 0, 1)
    
    print(f"Created synthetic video: {video.shape}")
    print(f"Min value: {video.min():.3f}, Max value: {video.max():.3f}")
    
    # Demonstrate sampling strategies
    print("\n2. Testing sampling strategies...")
    sampler = FrameSampler()
    
    # Uniform sampling
    uniform_sampled = sampler.uniform_sampling(video, num_frames=10)
    print(f"After uniform sampling: {uniform_sampled.shape}")
    
    # Stride sampling
    stride_sampled = sampler.temporal_stride_sampling(video, stride=3)
    print(f"After stride sampling: {stride_sampled.shape}")
    
    # Dense sampling
    clips = sampler.dense_sampling(video, clip_length=8, num_clips=3)
    print(f"Dense sampling produced {len(clips)} clips")
    print(f"Each clip shape: {clips[0].shape}")
    
    # Demonstrate preprocessing
    print("\n3. Testing preprocessing...")
    preprocessor = VideoPreprocessor()
    
    normalized = preprocessor.normalize(video)
    print(f"After normalization - Mean: {normalized.mean():.3f}, Std: {normalized.std():.3f}")
    
    denormalized = preprocessor.denormalize(normalized)
    print(f"After denormalization - Mean: {denormalized.mean():.3f}")
    
    # Spatial crop
    cropped = preprocessor.spatial_crop(video, crop_size=(96, 96), position='center')
    print(f"After spatial crop: {cropped.shape}")
    
    # Visualize frames
    print("\n4. Visualizing frames...")
    visualize_frames(video, num_frames=8)
    
    # Create and visualize synthetic optical flow
    print("\n5. Creating synthetic optical flow...")
    flow = np.zeros((H, W, 2), dtype=np.float32)
    # Create horizontal flow pattern
    flow[:, :, 0] = 5.0  # u component (horizontal)
    flow[:, :, 1] = 0.0  # v component (vertical)
    visualize_optical_flow(flow)
    
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80)


def main():
    """
    Main execution function demonstrating video basics.
    """
    print(__doc__)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run demonstrations
    demonstrate_video_loading()
    
    # Summary
    print("\n" + "="*80)
    print("KEY TAKEAWAYS")
    print("="*80)
    print("""
    1. Video Format: Videos are sequences of frames V = {I_1, I_2, ..., I_T}
       - PyTorch format: (T, C, H, W) where T=time, C=channels, H=height, W=width
    
    2. Loading Methods:
       - torchvision: Native PyTorch, easy to use
       - OpenCV: Flexible, widely supported, production-ready
       - Choose based on your needs and existing infrastructure
    
    3. Frame Sampling:
       - Uniform: Even spacing across video (most common)
       - Random: For data augmentation
       - Stride: Skip frames for efficiency
       - Dense: Multiple clips for temporal networks
    
    4. Preprocessing:
       - Normalization: Stabilizes training with mean/std
       - Cropping: Temporal (frames) and spatial (regions)
       - Augmentation: Improves generalization
    
    5. Video vs Images:
       - Videos have temporal dimension (extra complexity)
       - Sampling strategy affects what model learns
       - Preprocessing must maintain temporal consistency
    
    Next: We'll explore 3D convolutions for spatiotemporal feature learning!
    """)


if __name__ == "__main__":
    main()
