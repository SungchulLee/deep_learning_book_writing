"""
Basic Residual Block Implementation
====================================
Introduction to residual connections (skip connections) in neural networks.

Key Concept:
Instead of learning H(x), we learn F(x) = H(x) - x, so the output is F(x) + x
This allows gradients to flow directly through the network via the skip connection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """
    Basic Residual Block for ResNet
    
    Architecture:
    Input -> Conv -> BN -> ReLU -> Conv -> BN -> (+) -> ReLU
              |__________________________________|
                    (skip connection)
    """
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection (shortcut)
        self.shortcut = nn.Sequential()
        
        # If dimensions change, use 1x1 conv to match dimensions
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        # Main path
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Add skip connection
        out += self.shortcut(x)
        
        # Final activation
        out = F.relu(out)
        
        return out


class PlainBlock(nn.Module):
    """
    Plain Block (without skip connection) for comparison
    """
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(PlainBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


def demonstrate_gradient_flow():
    """
    Demonstrate how gradients flow through residual connections
    """
    print("=" * 60)
    print("Gradient Flow Demonstration")
    print("=" * 60)
    
    # Create a simple input
    x = torch.randn(1, 64, 32, 32, requires_grad=True)
    
    # Residual block
    res_block = BasicBlock(64, 64)
    res_output = res_block(x)
    loss_res = res_output.sum()
    loss_res.backward()
    res_grad_norm = x.grad.norm().item()
    
    # Plain block
    x.grad = None  # Reset gradient
    plain_block = PlainBlock(64, 64)
    plain_output = plain_block(x)
    loss_plain = plain_output.sum()
    loss_plain.backward()
    plain_grad_norm = x.grad.norm().item()
    
    print(f"\nGradient norm for Residual Block: {res_grad_norm:.4f}")
    print(f"Gradient norm for Plain Block: {plain_grad_norm:.4f}")
    print(f"\nResidual connections help maintain gradient magnitude!")
    print("=" * 60)


def test_blocks():
    """
    Test basic residual block functionality
    """
    print("\n" + "=" * 60)
    print("Testing Residual Blocks")
    print("=" * 60)
    
    # Test with same dimensions
    print("\n1. Same dimensions (64 -> 64)")
    block1 = BasicBlock(64, 64)
    x1 = torch.randn(2, 64, 32, 32)
    out1 = block1(x1)
    print(f"   Input shape:  {x1.shape}")
    print(f"   Output shape: {out1.shape}")
    
    # Test with different dimensions
    print("\n2. Different dimensions (64 -> 128, stride=2)")
    block2 = BasicBlock(64, 128, stride=2)
    x2 = torch.randn(2, 64, 32, 32)
    out2 = block2(x2)
    print(f"   Input shape:  {x2.shape}")
    print(f"   Output shape: {out2.shape}")
    print(f"   Notice: Spatial dimensions halved, channels doubled")
    
    # Count parameters
    print("\n3. Parameter count")
    total_params = sum(p.numel() for p in block1.parameters())
    print(f"   Total parameters in BasicBlock(64, 64): {total_params:,}")
    
    print("=" * 60)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("RESIDUAL CONNECTIONS - BASIC CONCEPTS")
    print("=" * 60)
    
    print("\nKey Benefits of Residual Connections:")
    print("1. Easier gradient flow (addresses vanishing gradient)")
    print("2. Enables training of very deep networks (100+ layers)")
    print("3. Learning identity function is easy (F(x) = 0)")
    print("4. Better optimization landscape")
    
    # Run tests
    test_blocks()
    
    # Demonstrate gradient flow
    demonstrate_gradient_flow()
    
    print("\n" + "=" * 60)
    print("Next: See 02_resnet_implementation.py for full ResNet architecture")
    print("=" * 60 + "\n")
