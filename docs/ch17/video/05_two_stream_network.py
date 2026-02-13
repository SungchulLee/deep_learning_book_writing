"""
Module 34: Video Understanding - Intermediate Level
File 05: Two-Stream Network - Spatial and Temporal Stream Fusion

This file covers two-stream architectures for action recognition:
- Spatial stream for appearance information (RGB frames)
- Temporal stream for motion information (optical flow)
- Late fusion strategies
- Implementation of original two-stream CNN

Mathematical Foundation:
Two-Stream Architecture:
    Given video V with frames {I_1, ..., I_T} and optical flow {F_1, ..., F_{T-1}}:
    
    Spatial Stream: f_s(I_t) → p_spatial
    Temporal Stream: f_t(F_t) → p_temporal
    
    Final prediction: p = α·p_spatial + (1-α)·p_temporal
    
    where α is fusion weight (typically 0.5)

Key Insight: Appearance and motion are complementary cues!
    - Spatial: What objects/scenes are present
    - Temporal: How things are moving
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict
import warnings
warnings.filterwarnings('ignore')


#=============================================================================
# PART 1: SPATIAL STREAM (APPEARANCE)
#=============================================================================

class SpatialStream(nn.Module):
    """
    Spatial stream CNN for appearance information.
    
    Processes single RGB frames to capture:
    - Object identity
    - Scene information  
    - Poses and appearances
    
    Uses standard 2D CNN (e.g., ResNet, VGG)
    """
    
    def __init__(self, 
                 num_classes: int = 400,
                 pretrained: bool = True,
                 dropout: float = 0.5):
        """
        Initialize spatial stream.
        
        Args:
            num_classes: Number of action classes
            pretrained: Whether to use ImageNet pretrained weights
            dropout: Dropout probability
        """
        super().__init__()
        
        # Use ResNet-50 as backbone (standard choice)
        # Input: single RGB frame (B, 3, H, W)
        resnet = models.resnet50(pretrained=pretrained)
        
        # Remove final fc layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Custom classifier for actions
        # ResNet-50 outputs 2048-dim features
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(2048, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through spatial stream.
        
        Args:
            x: RGB frames (B, 3, H, W) - single frames or (B, T, 3, H, W) batch
            
        Returns:
            Class scores (B, num_classes)
            
        Note: For videos, can sample multiple frames and average predictions
        """
        # Handle both single frame and video input
        if x.dim() == 5:  # (B, T, 3, H, W)
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)  # Process all frames
            batch_mode = True
        else:
            batch_mode = False
            B = x.shape[0]
        
        # Extract features
        features = self.features(x)  # (B*T, 2048, 1, 1) or (B, 2048, 1, 1)
        features = features.flatten(start_dim=1)  # (B*T, 2048)
        
        # Classify
        logits = self.classifier(features)  # (B*T, num_classes)
        
        # Average over time if video input
        if batch_mode:
            logits = logits.view(B, T, -1).mean(dim=1)  # (B, num_classes)
        
        return logits


#=============================================================================
# PART 2: TEMPORAL STREAM (MOTION)
#=============================================================================

class TemporalStream(nn.Module):
    """
    Temporal stream CNN for motion information.
    
    Processes optical flow to capture:
    - Motion patterns
    - Velocity and direction
    - Temporal dynamics
    
    Input: Stack of L consecutive optical flow fields
    """
    
    def __init__(self,
                 num_classes: int = 400,
                 flow_length: int = 10,
                 dropout: float = 0.5):
        """
        Initialize temporal stream.
        
        Args:
            num_classes: Number of action classes
            flow_length: Number of flow fields to stack (L)
            dropout: Dropout probability
        """
        super().__init__()
        
        self.flow_length = flow_length
        
        # Input channels: 2*L (u and v components for L flows)
        # Stack L consecutive optical flows as input
        input_channels = 2 * flow_length
        
        # Modified ResNet with different input channels
        resnet = models.resnet50(pretrained=False)
        
        # Replace first conv layer to accept 2*L channels
        self.conv1 = nn.Conv2d(
            input_channels,  # 2*L channels instead of 3
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        
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
        Forward pass through temporal stream.
        
        Args:
            flow: Optical flow stack (B, 2*L, H, W)
                  Each flow field has 2 channels (u, v)
                  Stack L consecutive flows
                  
        Returns:
            Class scores (B, num_classes)
        """
        # Pass through network
        x = self.conv1(flow)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.flatten(start_dim=1)
        
        # Classify
        logits = self.classifier(x)
        
        return logits


#=============================================================================
# PART 3: TWO-STREAM FUSION
#=============================================================================

class TwoStreamNetwork(nn.Module):
    """
    Complete two-stream network with fusion.
    
    Combines spatial and temporal streams for action recognition.
    
    Fusion strategies:
        1. Late fusion (average): p = (p_spatial + p_temporal) / 2
        2. Weighted fusion: p = α·p_spatial + (1-α)·p_temporal
        3. Learned fusion: p = MLP([p_spatial, p_temporal])
    """
    
    def __init__(self,
                 num_classes: int = 400,
                 flow_length: int = 10,
                 fusion_type: str = 'average'):
        """
        Initialize two-stream network.
        
        Args:
            num_classes: Number of action classes
            flow_length: Number of optical flow fields
            fusion_type: 'average', 'weighted', or 'learned'
        """
        super().__init__()
        
        self.fusion_type = fusion_type
        
        # Spatial and temporal streams
        self.spatial_stream = SpatialStream(num_classes=num_classes)
        self.temporal_stream = TemporalStream(
            num_classes=num_classes,
            flow_length=flow_length
        )
        
        # Weighted fusion parameter
        if fusion_type == 'weighted':
            self.fusion_weight = nn.Parameter(torch.tensor(0.5))
        
        # Learned fusion network
        elif fusion_type == 'learned':
            self.fusion_net = nn.Sequential(
                nn.Linear(num_classes * 2, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
    
    def forward(self,
                rgb: torch.Tensor,
                flow: torch.Tensor,
                return_separate: bool = False) -> torch.Tensor:
        """
        Forward pass through two-stream network.
        
        Args:
            rgb: RGB frames (B, 3, H, W) or (B, T, 3, H, W)
            flow: Optical flow stack (B, 2*L, H, W)
            return_separate: Whether to return individual stream outputs
            
        Returns:
            Fused class scores (B, num_classes)
            Or tuple of (fused, spatial, temporal) if return_separate=True
        """
        # Get predictions from both streams
        spatial_logits = self.spatial_stream(rgb)
        temporal_logits = self.temporal_stream(flow)
        
        # Fusion strategy
        if self.fusion_type == 'average':
            # Simple average fusion
            fused_logits = (spatial_logits + temporal_logits) / 2
            
        elif self.fusion_type == 'weighted':
            # Weighted fusion with learnable weight
            alpha = torch.sigmoid(self.fusion_weight)
            fused_logits = alpha * spatial_logits + (1 - alpha) * temporal_logits
            
        elif self.fusion_type == 'learned':
            # Learned fusion with MLP
            combined = torch.cat([spatial_logits, temporal_logits], dim=1)
            fused_logits = self.fusion_net(combined)
        
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")
        
        if return_separate:
            return fused_logits, spatial_logits, temporal_logits
        return fused_logits


#=============================================================================
# PART 4: OPTICAL FLOW GENERATION (SIMPLIFIED)
#=============================================================================

def compute_dense_optical_flow(prev_frame: np.ndarray,
                               next_frame: np.ndarray,
                               method: str = 'farneback') -> np.ndarray:
    """
    Compute dense optical flow between two frames.
    
    Args:
        prev_frame: Previous frame (H, W, 3) or (H, W)
        next_frame: Next frame (H, W, 3) or (H, W)
        method: Flow computation method
        
    Returns:
        flow: Optical flow (H, W, 2) with (u, v) components
        
    Mathematical Background:
        Optical flow equation (brightness constancy):
        I(x, y, t) = I(x+u, y+v, t+1)
        
        Linearization:
        I_x·u + I_y·v + I_t = 0
        
        where (u,v) is the flow vector
    """
    # Convert to grayscale if needed
    if prev_frame.ndim == 3:
        prev_gray = cv2.cvtColor((prev_frame * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        next_gray = cv2.cvtColor((next_frame * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        prev_gray = (prev_frame * 255).astype(np.uint8)
        next_gray = (next_frame * 255).astype(np.uint8)
    
    if method == 'farneback':
        # Farnebäck method: polynomial expansion
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            next_gray,
            None,
            pyr_scale=0.5,  # Pyramid scale
            levels=3,        # Number of pyramid layers
            winsize=15,      # Window size
            iterations=3,    # Iterations at each pyramid level
            poly_n=5,        # Polynomial expansion degree
            poly_sigma=1.2,  # Gaussian std for polynomial expansion
            flags=0
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return flow


def extract_flow_stack(video: torch.Tensor,
                       flow_length: int = 10) -> torch.Tensor:
    """
    Extract stacked optical flow from video.
    
    Args:
        video: Video tensor (T, C, H, W) or (B, T, C, H, W)
        flow_length: Number of flows to stack (L)
        
    Returns:
        flow_stack: Stacked flows (B, 2*L, H, W)
    """
    if video.dim() == 4:
        video = video.unsqueeze(0)  # Add batch dim
    
    B, T, C, H, W = video.shape
    
    # We need at least flow_length+1 frames
    if T < flow_length + 1:
        raise ValueError(f"Need at least {flow_length+1} frames, got {T}")
    
    flow_stacks = []
    
    for b in range(B):
        flows = []
        
        # Compute optical flow between consecutive frames
        for t in range(flow_length):
            prev_frame = video[b, t].permute(1, 2, 0).cpu().numpy()  # (H, W, C)
            next_frame = video[b, t+1].permute(1, 2, 0).cpu().numpy()
            
            # Compute flow
            flow = compute_dense_optical_flow(prev_frame, next_frame)
            
            # Convert to tensor and normalize
            flow_tensor = torch.from_numpy(flow).permute(2, 0, 1).float()  # (2, H, W)
            flows.append(flow_tensor)
        
        # Stack flows: (L, 2, H, W) → (2*L, H, W)
        flow_stack = torch.cat(flows, dim=0)
        flow_stacks.append(flow_stack)
    
    flow_stacks = torch.stack(flow_stacks, dim=0)  # (B, 2*L, H, W)
    
    return flow_stacks


#=============================================================================
# PART 5: DEMONSTRATION
#=============================================================================

def demonstrate_two_stream():
    """
    Demonstrate two-stream network architecture and fusion.
    """
    print("\n" + "="*80)
    print("TWO-STREAM NETWORK DEMONSTRATION")
    print("="*80)
    
    # Configuration
    num_classes = 400  # Kinetics-400
    batch_size = 4
    num_frames = 16
    flow_length = 10
    height, width = 224, 224
    
    print(f"\nConfiguration:")
    print(f"  Number of classes: {num_classes}")
    print(f"  Batch size: {batch_size}")
    print(f"  Video frames: {num_frames}")
    print(f"  Flow stack length: {flow_length}")
    print(f"  Frame size: {height}x{width}")
    
    # Create models
    print("\n1. Creating two-stream network...")
    
    # Test different fusion strategies
    fusion_types = ['average', 'weighted', 'learned']
    
    for fusion_type in fusion_types:
        print(f"\n   Testing {fusion_type} fusion:")
        model = TwoStreamNetwork(
            num_classes=num_classes,
            flow_length=flow_length,
            fusion_type=fusion_type
        )
        
        # Count parameters
        params = sum(p.numel() for p in model.parameters())
        print(f"   Total parameters: {params:,}")
        
        # Create sample inputs
        rgb = torch.randn(batch_size, 3, height, width)
        flow = torch.randn(batch_size, 2 * flow_length, height, width)
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            output, spatial_out, temporal_out = model(
                rgb, flow, return_separate=True
            )
        
        print(f"   RGB input: {rgb.shape}")
        print(f"   Flow input: {flow.shape}")
        print(f"   Spatial output: {spatial_out.shape}")
        print(f"   Temporal output: {temporal_out.shape}")
        print(f"   Fused output: {output.shape}")
        
        # Analyze fusion
        spatial_probs = F.softmax(spatial_out[0], dim=0)
        temporal_probs = F.softmax(temporal_out[0], dim=0)
        fused_probs = F.softmax(output[0], dim=0)
        
        top5_spatial = torch.topk(spatial_probs, 3)
        top5_temporal = torch.topk(temporal_probs, 3)
        top5_fused = torch.topk(fused_probs, 3)
        
        print(f"\n   Top-3 predictions (first sample):")
        print(f"   Spatial:  {top5_spatial.indices.tolist()} "
              f"({top5_spatial.values[0].item():.3f})")
        print(f"   Temporal: {top5_temporal.indices.tolist()} "
              f"({top5_temporal.values[0].item():.3f})")
        print(f"   Fused:    {top5_fused.indices.tolist()} "
              f"({top5_fused.values[0].item():.3f})")
    
    print("\n" + "="*80)
    print("KEY TAKEAWAYS")
    print("="*80)
    print("""
    1. Two-Stream Architecture:
       - Spatial stream: Processes RGB for appearance
       - Temporal stream: Processes optical flow for motion
       - Complementary information → better performance
    
    2. Optical Flow:
       - Captures motion information between frames
       - (u, v) components encode horizontal and vertical movement
       - Farnebäck: Dense optical flow with polynomial expansion
    
    3. Fusion Strategies:
       - Average: Simple but effective (equal weight)
       - Weighted: Learnable fusion weight α
       - Learned: MLP learns optimal combination
    
    4. Performance Insights:
       - Two-stream >> single RGB stream (historical results)
       - ~10-15% accuracy improvement on action recognition
       - Temporal stream captures what CNN can't from single frames
    
    5. Practical Considerations:
       - Optical flow is expensive to compute
       - Need to store/compute flow offline
       - Two models → 2x parameters and computation
       - Modern alternatives: 3D CNNs, video transformers
    """)


def main():
    """
    Main demonstration of two-stream networks.
    """
    print(__doc__)
    
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run demonstration
    demonstrate_two_stream()


if __name__ == "__main__":
    import cv2  # Import here to avoid issues if not installed
    main()
