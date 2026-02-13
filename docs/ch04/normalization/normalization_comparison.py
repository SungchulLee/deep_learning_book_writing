"""
Comprehensive Comparison of Normalization Layers
=================================================

This file provides side-by-side comparisons and practical examples
of Batch Norm, Layer Norm, Instance Norm, and Group Norm.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


class NormalizationComparison:
    """
    Class to compare different normalization techniques.
    """
    
    def __init__(self):
        self.normalizations = {
            'BatchNorm': nn.BatchNorm2d(3, affine=False),
            'LayerNorm': nn.LayerNorm([3, 4, 4], elementwise_affine=False),
            'InstanceNorm': nn.InstanceNorm2d(3, affine=False),
            'GroupNorm': nn.GroupNorm(3, 3, affine=False),  # 3 groups for 3 channels
        }
        
        # Set all to eval mode
        for norm in self.normalizations.values():
            if hasattr(norm, 'eval'):
                norm.eval()
    
    def visualize_normalization_axes(self):
        """
        Visualize which axes each normalization method operates on.
        """
        print("=" * 70)
        print("Normalization Axes Visualization")
        print("=" * 70)
        
        print("\nInput tensor shape: (N, C, H, W) = (Batch, Channels, Height, Width)")
        print("\nNormalization axes (what dimensions are averaged over):")
        print("-" * 70)
        
        visualizations = {
            'BatchNorm':     "Axes: [0, 2, 3] → (N, H, W) | Per channel across batch",
            'LayerNorm':     "Axes: [1, 2, 3] → (C, H, W) | Per sample across features",
            'InstanceNorm':  "Axes: [2, 3]    → (H, W)   | Per sample per channel",
            'GroupNorm':     "Axes: [2, 3]    → (H, W)   | Per sample per group",
        }
        
        for name, desc in visualizations.items():
            print(f"{name:15s}: {desc}")
        
        print("\n" + "=" * 70)
    
    def compare_on_sample_data(self):
        """
        Compare all normalization methods on the same input.
        """
        print("\n" + "=" * 70)
        print("Comparing Normalizations on Sample Data")
        print("=" * 70)
        
        torch.manual_seed(42)
        
        # Create sample input: 2 images, 3 channels, 4x4 spatial
        x = torch.randn(2, 3, 4, 4)
        
        # Scale different samples and channels differently
        x[0] *= 5   # First image has larger values
        x[1] *= 0.5  # Second image has smaller values
        x[:, 0] *= 2  # First channel has larger values
        
        print(f"\nInput shape: {x.shape}")
        print(f"Input mean: {x.mean():.4f}, std: {x.std():.4f}")
        
        print("\nOriginal data statistics:")
        for n in range(2):
            for c in range(3):
                mean = x[n, c].mean()
                std = x[n, c].std()
                print(f"  Sample {n}, Channel {c}: mean={mean:7.3f}, std={std:7.3f}")
        
        print("\n" + "-" * 70)
        print("After normalization:")
        print("-" * 70)
        
        for name, norm_layer in self.normalizations.items():
            with torch.no_grad():
                x_norm = norm_layer(x)
            
            print(f"\n{name}:")
            print(f"  Overall: mean={x_norm.mean():.4f}, std={x_norm.std():.4f}")
            
            # Show statistics based on what should be normalized
            if name == 'BatchNorm':
                print("  Per channel (averaged over batch, H, W):")
                for c in range(3):
                    mean = x_norm[:, c].mean()
                    std = x_norm[:, c].std()
                    print(f"    Channel {c}: mean={mean:.4f}, std={std:.4f}")
            
            elif name == 'LayerNorm':
                print("  Per sample (averaged over C, H, W):")
                for n in range(2):
                    mean = x_norm[n].mean()
                    std = x_norm[n].std()
                    print(f"    Sample {n}: mean={mean:.4f}, std={std:.4f}")
            
            elif name == 'InstanceNorm':
                print("  Per sample per channel (averaged over H, W):")
                for n in range(2):
                    for c in range(3):
                        mean = x_norm[n, c].mean()
                        std = x_norm[n, c].std()
                        print(f"    Sample {n}, Channel {c}: mean={mean:.4f}, std={std:.4f}")
    
    def test_batch_size_sensitivity(self):
        """
        Test how different normalizations handle varying batch sizes.
        """
        print("\n" + "=" * 70)
        print("Batch Size Sensitivity Test")
        print("=" * 70)
        
        torch.manual_seed(42)
        
        batch_sizes = [1, 2, 8, 32]
        
        print("\nTesting with different batch sizes:")
        print("(Using the same data distribution)")
        
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 3, 8, 8)
            
            print(f"\n--- Batch size: {batch_size} ---")
            
            for name, norm_layer in self.normalizations.items():
                # Reinitialize to avoid running stats issues
                if name == 'BatchNorm':
                    norm_layer = nn.BatchNorm2d(3, affine=False)
                    norm_layer.eval()
                
                with torch.no_grad():
                    x_norm = norm_layer(x)
                
                print(f"{name:15s}: mean={x_norm.mean():7.4f}, std={x_norm.std():7.4f}")
        
        print("\nObservations:")
        print("- BatchNorm is sensitive to batch size (less stable with small batches)")
        print("- LayerNorm, InstanceNorm, GroupNorm are independent of batch size")


def create_comparison_network():
    """
    Create networks with different normalization layers for comparison.
    """
    print("\n" + "=" * 70)
    print("Example Networks with Different Normalizations")
    print("=" * 70)
    
    class ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels, norm_type='batch'):
            super(ConvBlock, self).__init__()
            
            self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            
            # Choose normalization
            if norm_type == 'batch':
                self.norm = nn.BatchNorm2d(out_channels)
            elif norm_type == 'layer':
                # For LayerNorm with 2D data, we need to specify the shape
                # This is a simplified version
                self.norm = nn.GroupNorm(1, out_channels)  # Equivalent to LayerNorm for single sample
            elif norm_type == 'instance':
                self.norm = nn.InstanceNorm2d(out_channels)
            elif norm_type == 'group':
                self.norm = nn.GroupNorm(8, out_channels)  # 8 groups
            else:
                self.norm = nn.Identity()
            
            self.relu = nn.ReLU(inplace=True)
        
        def forward(self, x):
            x = self.conv(x)
            x = self.norm(x)
            x = self.relu(x)
            return x
    
    # Create networks with different normalizations
    networks = {
        'BatchNorm': nn.Sequential(
            ConvBlock(3, 64, 'batch'),
            ConvBlock(64, 128, 'batch'),
        ),
        'InstanceNorm': nn.Sequential(
            ConvBlock(3, 64, 'instance'),
            ConvBlock(64, 128, 'instance'),
        ),
        'GroupNorm': nn.Sequential(
            ConvBlock(3, 64, 'group'),
            ConvBlock(64, 128, 'group'),
        ),
    }
    
    # Test with sample input
    x = torch.randn(4, 3, 32, 32)
    
    print("\nTesting networks with input shape:", x.shape)
    
    for name, net in networks.items():
        net.eval()
        with torch.no_grad():
            out = net(x)
        print(f"{name:15s}: output shape={out.shape}, mean={out.mean():.4f}, std={out.std():.4f}")


def performance_comparison():
    """
    Compare computational performance of different normalizations.
    """
    print("\n" + "=" * 70)
    print("Performance Characteristics")
    print("=" * 70)
    
    characteristics = {
        'BatchNorm': {
            'Speed': 'Fast',
            'Memory': 'Low (stores running stats)',
            'Batch dependency': 'Yes (sensitive to batch size)',
            'Train/Eval difference': 'Yes (uses different stats)',
        },
        'LayerNorm': {
            'Speed': 'Fast',
            'Memory': 'Low',
            'Batch dependency': 'No (batch independent)',
            'Train/Eval difference': 'No (same computation)',
        },
        'InstanceNorm': {
            'Speed': 'Fast',
            'Memory': 'Low',
            'Batch dependency': 'No (batch independent)',
            'Train/Eval difference': 'No (same computation)',
        },
        'GroupNorm': {
            'Speed': 'Fast',
            'Memory': 'Low',
            'Batch dependency': 'No (batch independent)',
            'Train/Eval difference': 'No (same computation)',
        },
    }
    
    for norm_name, chars in characteristics.items():
        print(f"\n{norm_name}:")
        for key, value in chars.items():
            print(f"  {key:25s}: {value}")


def practical_recommendations():
    """
    Provide practical recommendations for choosing normalization methods.
    """
    print("\n" + "=" * 70)
    print("Practical Recommendations")
    print("=" * 70)
    
    recommendations = """
    Task Type                    | Recommended Norm | Why?
    -----------------------------|------------------|-------------------------
    Image Classification (CNN)   | BatchNorm        | Works well with large batches
    Object Detection            | GroupNorm/SyncBN  | Better for small batches
    Semantic Segmentation       | BatchNorm/GroupNorm| Depends on batch size
    Style Transfer              | InstanceNorm      | Removes instance-specific info
    GANs (Image-to-Image)       | InstanceNorm      | Process samples independently
    Transformers (NLP)          | LayerNorm         | Standard choice, batch independent
    RNNs/LSTMs                  | LayerNorm         | Handles variable sequences well
    Online Learning (batch=1)   | LayerNorm/InstanceNorm | Batch independent
    Small Batch Training        | GroupNorm/LayerNorm | Not sensitive to batch size
    Video Processing            | GroupNorm         | Handles temporal dimension well
    
    Special Cases:
    - If you have small batches (< 8): Use GroupNorm or LayerNorm
    - If training is unstable: Try GroupNorm
    - If you need exact same behavior in train/eval: Use LayerNorm or InstanceNorm
    - If doing multi-GPU training: Use SyncBatchNorm (syncs stats across GPUs)
    """
    
    print(recommendations)


def common_mistakes():
    """
    Highlight common mistakes when using normalization layers.
    """
    print("\n" + "=" * 70)
    print("Common Mistakes to Avoid")
    print("=" * 70)
    
    mistakes = """
    1. FORGETTING TO CALL model.eval()
       - BatchNorm behaves differently in train vs eval mode
       - Always call model.eval() before inference!
    
    2. USING BATCHNORM WITH BATCH SIZE = 1
       - BatchNorm needs multiple samples to compute statistics
       - Use LayerNorm or InstanceNorm instead
    
    3. PLACING NORM BEFORE ACTIVATION
       - Standard: Conv → Norm → Activation
       - Some experiments show Norm → Conv → Activation works better
       - Be consistent in your architecture
    
    4. NOT TUNING MOMENTUM FOR BATCHNORM
       - Default momentum (0.1) may not be optimal
       - For small datasets, try smaller momentum (0.01)
    
    5. USING WRONG NORM FOR THE TASK
       - Don't use BatchNorm for style transfer (use InstanceNorm)
       - Don't use InstanceNorm for classification (use BatchNorm)
    
    6. FREEZING BATCHNORM LAYERS INCORRECTLY
       - If fine-tuning, be careful with BatchNorm layers
       - May need to keep them in eval mode or update running stats
    
    7. NOT CONSIDERING MULTI-GPU TRAINING
       - Standard BatchNorm computes stats per GPU
       - Use SyncBatchNorm for better results across GPUs
    
    8. IGNORING AFFINE PARAMETERS
       - affine=True means learnable scale/shift
       - Usually keep it True, but disable if you want pure normalization
    """
    
    print(mistakes)


def quick_reference():
    """
    Quick reference guide for normalization layers.
    """
    print("\n" + "=" * 70)
    print("Quick Reference Guide")
    print("=" * 70)
    
    reference = """
    PyTorch Implementation:
    
    # Batch Normalization
    nn.BatchNorm1d(num_features)      # For 1D/linear layers
    nn.BatchNorm2d(num_channels)      # For 2D/conv layers
    nn.BatchNorm3d(num_channels)      # For 3D data
    
    # Layer Normalization
    nn.LayerNorm(normalized_shape)    # Specify shape to normalize
    nn.LayerNorm([C, H, W])          # For 2D data
    
    # Instance Normalization
    nn.InstanceNorm1d(num_features)   # For 1D data
    nn.InstanceNorm2d(num_channels)   # For 2D/images
    nn.InstanceNorm3d(num_channels)   # For 3D data
    
    # Group Normalization
    nn.GroupNorm(num_groups, num_channels)  # Divide channels into groups
    
    Common Parameters:
    - eps: Small value for numerical stability (default: 1e-5)
    - momentum: For running stats in BatchNorm (default: 0.1)
    - affine: Learnable scale/shift parameters (default: True)
    - track_running_stats: For BatchNorm (default: True)
    
    Remember:
    - Always call model.eval() for inference with BatchNorm
    - LayerNorm/InstanceNorm: same behavior in train and eval
    - Use .train() and .eval() to switch modes
    """
    
    print(reference)


if __name__ == "__main__":
    comp = NormalizationComparison()
    
    # Run all comparisons
    comp.visualize_normalization_axes()
    comp.compare_on_sample_data()
    comp.test_batch_size_sensitivity()
    
    create_comparison_network()
    performance_comparison()
    practical_recommendations()
    common_mistakes()
    quick_reference()
    
    print("\n" + "=" * 70)
    print("For more details, see individual files:")
    print("  - batch_normalization.py")
    print("  - layer_normalization.py")
    print("  - instance_normalization.py")
    print("=" * 70)
