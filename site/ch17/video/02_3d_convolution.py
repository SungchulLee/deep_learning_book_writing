"""
Module 34: Video Understanding - Beginner Level
File 02: 3D Convolution - Spatiotemporal Feature Extraction

This file covers 3D convolutional networks for video:
- Understanding 3D convolutions vs 2D convolutions
- Spatiotemporal feature extraction
- Implementing 3D CNNs in PyTorch
- C3D (Convolutional 3D) architecture
- Comparison with 2D CNN alternatives

Mathematical Foundation:
3D Convolution Operation:
    For input V ∈ ℝ^(T×C×H×W) and kernel K ∈ ℝ^(t×c×h×w):
    
    Output(τ, i, j) = Σ Σ Σ Σ V(τ+t', c', i+h', j+w') · K(t', c', h', w')
                      t' c' h' w'
    
Key difference from 2D:
    - 2D Conv: Processes spatial dimensions (H, W) only
    - 3D Conv: Processes spatiotemporal volume (T, H, W) together
    
This allows the network to learn motion patterns directly!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')


#=============================================================================
# PART 1: 2D VS 3D CONVOLUTION COMPARISON
#=============================================================================

class Conv2DExample(nn.Module):
    """
    2D Convolution applied frame-by-frame.
    
    Process:
        Each frame processed independently → No temporal information!
        
    Mathematical operation:
        For each frame I_t ∈ ℝ^(C×H×W):
        Output_t = Conv2D(I_t)
        
    Limitation: Cannot capture motion or temporal dynamics
    """
    
    def __init__(self, in_channels: int = 3, out_channels: int = 64):
        super().__init__()
        
        # Standard 2D convolution (spatial only)
        # Kernel size: (h, w) = (3, 3)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass applying 2D conv to each frame.
        
        Args:
            x: Input video (B, T, C, H, W) or (T, C, H, W)
            
        Returns:
            Output features (B, T, out_C, H, W) or (T, out_C, H, W)
            
        Note: T dimension preserved but frames processed independently
        """
        # Reshape to process all frames together
        # (B, T, C, H, W) → (B*T, C, H, W)
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
            batch_mode = True
        else:
            T, C, H, W = x.shape
            x = x.view(T, C, H, W)
            batch_mode = False
        
        # Apply 2D convolution
        out = self.conv(x)  # (B*T, out_C, H, W)
        
        # Reshape back
        if batch_mode:
            out = out.view(B, T, -1, H, W)
        else:
            out = out.view(T, -1, H, W)
        
        return out


class Conv3DExample(nn.Module):
    """
    3D Convolution for spatiotemporal processing.
    
    Process:
        Processes temporal volume together → Learns motion patterns!
        
    Mathematical operation:
        For video V ∈ ℝ^(T×C×H×W):
        Output = Conv3D(V)
        
        The 3D kernel slides in space AND time:
        K ∈ ℝ^(t_k × C × h_k × w_k)
        
    Advantage: Can learn spatiotemporal features (e.g., walking, jumping)
    """
    
    def __init__(self, in_channels: int = 3, out_channels: int = 64):
        super().__init__()
        
        # 3D convolution (spatiotemporal)
        # Kernel size: (t, h, w) = (3, 3, 3)
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(3, 3, 3),  # temporal_size, height, width
            padding=(1, 1, 1)       # maintain dimensions
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with 3D convolution.
        
        Args:
            x: Input video (B, C, T, H, W) - Note: PyTorch 3D conv format!
            
        Returns:
            Output features (B, out_C, T, H, W)
            
        Important: PyTorch Conv3d expects (B, C, T, H, W) format
        """
        # Input format check
        if x.dim() == 4:
            # Add batch dimension if needed
            x = x.unsqueeze(0)  # (C, T, H, W) → (1, C, T, H, W)
        
        # Apply 3D convolution
        out = self.conv(x)
        
        return out


def demonstrate_2d_vs_3d_convolution():
    """
    Demonstrate the difference between 2D and 3D convolutions.
    """
    print("\n" + "="*80)
    print("2D vs 3D CONVOLUTION COMPARISON")
    print("="*80)
    
    # Create sample video
    B, T, C, H, W = 2, 16, 3, 64, 64
    video = torch.randn(B, T, C, H, W)
    
    print(f"\nInput video shape: {video.shape}")
    print(f"  B={B} (batch), T={T} (time), C={C} (channels)")
    print(f"  H={H} (height), W={W} (width)")
    
    # 2D Convolution
    print("\n1. Applying 2D Convolution (frame-by-frame)...")
    conv2d = Conv2DExample(in_channels=C, out_channels=64)
    
    # Count parameters
    params_2d = sum(p.numel() for p in conv2d.parameters())
    print(f"   Parameters: {params_2d}")
    print(f"   Kernel size: (3, 3) - spatial only")
    
    output_2d = conv2d(video)
    print(f"   Output shape: {output_2d.shape}")
    print(f"   ✗ Frames processed independently - no temporal modeling")
    
    # 3D Convolution
    print("\n2. Applying 3D Convolution (spatiotemporal)...")
    conv3d = Conv3DExample(in_channels=C, out_channels=64)
    
    # Rearrange for Conv3d: (B, T, C, H, W) → (B, C, T, H, W)
    video_3d = video.permute(0, 2, 1, 3, 4)
    
    params_3d = sum(p.numel() for p in conv3d.parameters())
    print(f"   Parameters: {params_3d}")
    print(f"   Kernel size: (3, 3, 3) - spatiotemporal")
    
    output_3d = conv3d(video_3d)
    print(f"   Output shape: {output_3d.shape}")
    print(f"   ✓ Temporal dimension processed - learns motion!")
    
    # Parameter comparison
    print(f"\n3. Parameter Comparison:")
    print(f"   2D Conv: {params_2d:,} parameters")
    print(f"   3D Conv: {params_3d:,} parameters")
    print(f"   Ratio: 3D has {params_3d / params_2d:.1f}x more parameters")
    print(f"   Reason: 3D kernel has additional temporal dimension")


#=============================================================================
# PART 2: 3D CONVOLUTIONAL BLOCKS
#=============================================================================

class Conv3DBlock(nn.Module):
    """
    Basic 3D convolutional block with BatchNorm and ReLU.
    
    Architecture:
        Conv3D → BatchNorm3D → ReLU → MaxPool3D
        
    This is the building block for most 3D CNN architectures.
    """
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, int, int] = (3, 3, 3),
                 stride: Tuple[int, int, int] = (1, 1, 1),
                 padding: Tuple[int, int, int] = (1, 1, 1),
                 use_pooling: bool = True):
        """
        Initialize 3D conv block.
        
        Args:
            in_channels: Input channel dimension
            out_channels: Output channel dimension
            kernel_size: (t, h, w) for temporal, height, width
            stride: Stride for each dimension
            padding: Padding for each dimension
            use_pooling: Whether to apply max pooling
        """
        super().__init__()
        
        # 3D Convolution
        # Weight shape: (out_channels, in_channels, t, h, w)
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False  # BatchNorm handles bias
        )
        
        # 3D Batch Normalization
        # Normalizes across spatial and temporal dimensions
        # Maintains separate stats for each channel
        self.bn = nn.BatchNorm3d(out_channels)
        
        # ReLU activation
        self.relu = nn.ReLU(inplace=True)
        
        # Optional 3D Max Pooling
        # Reduces spatiotemporal dimensions
        self.use_pooling = use_pooling
        if use_pooling:
            self.pool = nn.MaxPool3d(
                kernel_size=(2, 2, 2),  # Pool over time and space
                stride=(2, 2, 2)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through conv block.
        
        Args:
            x: Input tensor (B, C, T, H, W)
            
        Returns:
            Output tensor (B, out_C, T', H', W')
            
        Dimension changes:
            - Conv: maintains or reduces based on stride
            - Pool: reduces by factor of 2 in each dimension
        """
        # Convolution: learns spatiotemporal features
        out = self.conv(x)
        
        # Batch normalization: stabilizes training
        out = self.bn(out)
        
        # Activation: introduces non-linearity
        out = self.relu(out)
        
        # Pooling: reduces dimensionality, increases receptive field
        if self.use_pooling:
            out = self.pool(out)
        
        return out


#=============================================================================
# PART 3: C3D ARCHITECTURE (CLASSIC 3D CNN)
#=============================================================================

class C3D(nn.Module):
    """
    C3D: Learning Spatiotemporal Features with 3D Convolutional Networks.
    
    Paper: Tran et al. "Learning Spatiotemporal Features with 3D Convolutional
           Networks" (ICCV 2015)
    
    Architecture:
        8 convolutional layers with 3x3x3 kernels
        5 max-pooling layers
        2 fully connected layers
        
    Key insight: 3x3x3 kernels throughout the network work best for capturing
                 spatiotemporal features
    
    Input: 16 frames of 112x112 RGB video
    Output: Class probabilities
    """
    
    def __init__(self, num_classes: int = 400, dropout: float = 0.5):
        """
        Initialize C3D network.
        
        Args:
            num_classes: Number of action classes
            dropout: Dropout probability for fully connected layers
        """
        super().__init__()
        
        # Layer 1: Conv3d (3→64)
        # Input: (B, 3, 16, 112, 112)
        self.conv1 = Conv3DBlock(3, 64, use_pooling=True)
        # Output: (B, 64, 8, 56, 56) after pooling
        
        # Layer 2: Conv3d (64→128)
        self.conv2 = Conv3DBlock(64, 128, use_pooling=True)
        # Output: (B, 128, 4, 28, 28)
        
        # Layers 3a, 3b: Conv3d (128→256)
        self.conv3a = Conv3DBlock(128, 256, use_pooling=False)
        self.conv3b = Conv3DBlock(256, 256, use_pooling=True)
        # Output: (B, 256, 2, 14, 14)
        
        # Layers 4a, 4b: Conv3d (256→512)
        self.conv4a = Conv3DBlock(256, 512, use_pooling=False)
        self.conv4b = Conv3DBlock(512, 512, use_pooling=True)
        # Output: (B, 512, 1, 7, 7)
        
        # Layers 5a, 5b: Conv3d (512→512)
        self.conv5a = Conv3DBlock(512, 512, use_pooling=False)
        self.conv5b = Conv3DBlock(512, 512, use_pooling=True)
        # Output: (B, 512, 1, 4, 4) - note temporal dim reduced to 1
        
        # Fully connected layers
        # Flatten: 512 * 1 * 4 * 4 = 8192
        self.fc1 = nn.Linear(512 * 1 * 4 * 4, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)
        
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through C3D.
        
        Args:
            x: Input video (B, C, T, H, W)
               Expected: (B, 3, 16, 112, 112)
               
        Returns:
            Class logits (B, num_classes)
        """
        # Convolutional layers - spatiotemporal feature extraction
        x = self.conv1(x)    # (B, 64, 8, 56, 56)
        x = self.conv2(x)    # (B, 128, 4, 28, 28)
        
        x = self.conv3a(x)   # (B, 256, 4, 28, 28)
        x = self.conv3b(x)   # (B, 256, 2, 14, 14)
        
        x = self.conv4a(x)   # (B, 512, 2, 14, 14)
        x = self.conv4b(x)   # (B, 512, 1, 7, 7)
        
        x = self.conv5a(x)   # (B, 512, 1, 7, 7)
        x = self.conv5b(x)   # (B, 512, 1, 4, 4)
        
        # Flatten for fully connected layers
        x = x.flatten(start_dim=1)  # (B, 512*1*4*4)
        
        # Fully connected layers - classification
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        
        x = self.fc3(x)  # Logits
        
        return x


#=============================================================================
# PART 4: RESIDUAL 3D CNN (R3D)
#=============================================================================

class Residual3DBlock(nn.Module):
    """
    Residual block for 3D convolutions.
    
    Architecture:
        x → Conv3D → BN → ReLU → Conv3D → BN → (+) → ReLU
        └──────────────────────────────────────┘
        
    Residual connection: F(x) + x
    
    Benefits:
        1. Easier gradient flow (helps training deep networks)
        2. Better feature learning
        3. Reduces degradation problem
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        # First conv block
        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1),
            bias=False
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        
        # Second conv block
        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1),
            bias=False
        )
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        # ReLU
        self.relu = nn.ReLU(inplace=True)
        
        # Shortcut connection (if dimensions don't match)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    bias=False
                ),
                nn.BatchNorm3d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.
        
        Args:
            x: Input tensor (B, C, T, H, W)
            
        Returns:
            Output tensor (B, out_C, T, H, W)
            
        Mathematical operation:
            out = ReLU(F(x) + x)
            where F(x) is the residual function
        """
        # Save input for residual connection
        identity = x
        
        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Add residual connection
        identity = self.shortcut(identity)
        out += identity
        
        # Final activation
        out = self.relu(out)
        
        return out


#=============================================================================
# PART 5: VISUALIZATION AND ANALYSIS
#=============================================================================

def visualize_3d_kernels(model: nn.Module):
    """
    Visualize learned 3D convolutional kernels.
    
    Args:
        model: 3D CNN model
    """
    # Get first conv layer
    first_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv3d):
            first_conv = module
            break
    
    if first_conv is None:
        print("No Conv3d layer found")
        return
    
    # Get weights: (out_C, in_C, T, H, W)
    weights = first_conv.weight.data.cpu()
    
    print(f"\nFirst Conv3d layer:")
    print(f"  Weight shape: {weights.shape}")
    print(f"  Kernel size: {first_conv.kernel_size}")
    
    # Visualize a subset of kernels
    num_filters = min(8, weights.shape[0])
    num_temporal = weights.shape[2]
    
    fig, axes = plt.subplots(num_filters, num_temporal, figsize=(12, 10))
    
    for i in range(num_filters):
        for t in range(num_temporal):
            ax = axes[i, t] if num_filters > 1 else axes[t]
            
            # Get kernel at time t (average over input channels)
            kernel = weights[i, :, t, :, :].mean(dim=0)  # (H, W)
            
            # Normalize for visualization
            kernel = (kernel - kernel.min()) / (kernel.max() - kernel.min() + 1e-8)
            
            ax.imshow(kernel.numpy(), cmap='viridis')
            ax.axis('off')
            
            if t == 0:
                ax.set_ylabel(f'Filter {i}', rotation=0, labelpad=30)
            if i == 0:
                ax.set_title(f't={t}')
    
    plt.tight_layout()
    plt.savefig('/home/claude/34_video_understanding/02_3d_kernels.png',
                dpi=150, bbox_inches='tight')
    print(f"Kernel visualization saved to 02_3d_kernels.png")
    plt.close()


def analyze_feature_maps(model: nn.Module, video: torch.Tensor):
    """
    Analyze intermediate feature maps from 3D CNN.
    
    Args:
        model: 3D CNN model
        video: Input video tensor
    """
    print("\nAnalyzing feature maps...")
    
    # Register hook to capture intermediate outputs
    activations = {}
    
    def get_activation(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook
    
    # Register hooks
    layer_names = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv3d):
            module.register_forward_hook(get_activation(name))
            layer_names.append(name)
    
    # Forward pass
    with torch.no_grad():
        _ = model(video)
    
    # Print activation shapes
    print(f"\nFeature map shapes:")
    for name in layer_names[:5]:  # Show first 5 layers
        if name in activations:
            shape = activations[name].shape
            print(f"  {name}: {shape}")


def demonstrate_c3d():
    """
    Demonstrate C3D architecture and its properties.
    """
    print("\n" + "="*80)
    print("C3D ARCHITECTURE DEMONSTRATION")
    print("="*80)
    
    # Create C3D model
    print("\n1. Creating C3D model...")
    model = C3D(num_classes=101)  # UCF-101 has 101 classes
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: ~{total_params * 4 / 1e6:.1f} MB (fp32)")
    
    # Create sample input
    print("\n2. Testing with sample video...")
    batch_size = 2
    video = torch.randn(batch_size, 3, 16, 112, 112)  # (B, C, T, H, W)
    print(f"   Input shape: {video.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(video)
    
    print(f"   Output shape: {output.shape}")
    print(f"   Output: class logits for {output.shape[1]} classes")
    
    # Apply softmax for probabilities
    probs = torch.softmax(output, dim=1)
    top5_probs, top5_indices = torch.topk(probs[0], 5)
    
    print(f"\n3. Top 5 predictions for first video:")
    for i, (prob, idx) in enumerate(zip(top5_probs, top5_indices)):
        print(f"   {i+1}. Class {idx.item()}: {prob.item():.4f}")
    
    # Visualize kernels
    print("\n4. Visualizing learned kernels...")
    visualize_3d_kernels(model)
    
    # Analyze feature maps
    analyze_feature_maps(model, video)


#=============================================================================
# PART 6: EXAMPLE USAGE AND DEMONSTRATIONS
#=============================================================================

def main():
    """
    Main execution function demonstrating 3D convolutions.
    """
    print(__doc__)
    
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Demonstration 1: 2D vs 3D comparison
    demonstrate_2d_vs_3d_convolution()
    
    # Demonstration 2: C3D architecture
    demonstrate_c3d()
    
    # Summary
    print("\n" + "="*80)
    print("KEY TAKEAWAYS")
    print("="*80)
    print("""
    1. 3D Convolution:
       - Extends 2D convolution to temporal dimension
       - Kernel: (t, h, w) processes spatiotemporal volume
       - Learns motion patterns directly from raw pixels
    
    2. 2D vs 3D Trade-offs:
       - 2D: Faster, less memory, no temporal modeling
       - 3D: Slower, more parameters, captures motion
       - 3D has ~3x more parameters due to temporal kernel dimension
    
    3. C3D Architecture:
       - 8 conv layers with 3x3x3 kernels (empirically best)
       - Progressively reduces spatiotemporal dimensions
       - Input: 16 frames of 112x112 RGB
       - ~78M parameters for 101 classes
    
    4. Residual 3D Blocks:
       - Enable training very deep 3D CNNs
       - Better gradient flow through residual connections
       - Used in modern architectures (R3D, I3D)
    
    5. Practical Considerations:
       - 3D CNNs are computationally expensive
       - Need powerful GPUs for training
       - Batch size limited by memory (spatiotemporal volume is large)
       - Benefit: End-to-end learning from raw video
    
    Next: We'll build a simple video classifier using 3D CNNs!
    """)


if __name__ == "__main__":
    main()
