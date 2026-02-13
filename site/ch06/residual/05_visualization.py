"""
Visualizations for Residual Connections
========================================
Visual demonstrations of:
1. Gradient flow comparison
2. Loss landscape
3. Feature map evolution
4. Network architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import seaborn as sns


# ============================================================================
# GRADIENT FLOW VISUALIZATION
# ============================================================================

def visualize_gradient_flow(plain_model, residual_model, num_layers=10):
    """
    Visualize how gradients flow through deep networks
    with and without residual connections
    """
    print("Visualizing gradient flow...")
    
    # Create input
    x = torch.randn(1, 64, 32, 32, requires_grad=True)
    
    # Forward and backward through plain network
    plain_grads = []
    for i, layer in enumerate(plain_model.layers[:num_layers]):
        x_plain = layer(x)
        loss = x_plain.sum()
        loss.backward(retain_graph=True)
        
        if x.grad is not None:
            plain_grads.append(x.grad.abs().mean().item())
            x.grad.zero_()
    
    # Forward and backward through residual network
    x.grad = None
    residual_grads = []
    current = x
    for i, layer in enumerate(residual_model.layers[:num_layers]):
        # Residual block forward
        identity = current
        out = F.relu(layer['bn1'](layer['conv1'](current)))
        out = layer['bn2'](layer['conv2'](out))
        current = F.relu(out + identity)
        
        loss = current.sum()
        loss.backward(retain_graph=True)
        
        if x.grad is not None:
            residual_grads.append(x.grad.abs().mean().item())
            x.grad.zero_()
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    layers = range(1, num_layers + 1)
    ax.plot(layers, plain_grads, 'b-o', linewidth=2, markersize=8, 
            label='Plain Network', alpha=0.7)
    ax.plot(layers, residual_grads, 'r-o', linewidth=2, markersize=8,
            label='Residual Network', alpha=0.7)
    
    ax.set_xlabel('Layer Depth', fontsize=12)
    ax.set_ylabel('Average Gradient Magnitude', fontsize=12)
    ax.set_title('Gradient Flow: Plain vs Residual Networks', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Add annotation
    ax.annotate('Vanishing gradients\nin plain network',
                xy=(num_layers-1, plain_grads[-1]), xytext=(num_layers-3, plain_grads[-1]*10),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2),
                fontsize=10, color='blue')
    
    plt.tight_layout()
    return fig


# ============================================================================
# LOSS LANDSCAPE VISUALIZATION
# ============================================================================

def visualize_loss_landscape():
    """
    Visualize simplified loss landscape comparison
    Shows how residual connections create smoother optimization paths
    """
    print("Visualizing loss landscape...")
    
    # Create 2D parameter space
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    
    # Simulate loss landscapes
    # Plain network: More rugged with local minima
    Z_plain = (X**2 + Y**2) + 0.5 * np.sin(5*X) * np.sin(5*Y) + 0.3 * X * Y
    
    # Residual network: Smoother landscape
    Z_residual = (X**2 + Y**2) + 0.1 * np.sin(3*X) * np.sin(3*Y)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plain network
    contour1 = axes[0].contourf(X, Y, Z_plain, levels=20, cmap='viridis')
    axes[0].set_title('Loss Landscape: Plain Network\n(Rugged, many local minima)', 
                      fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Parameter 1')
    axes[0].set_ylabel('Parameter 2')
    fig.colorbar(contour1, ax=axes[0], label='Loss')
    
    # Add optimization path (struggles with local minima)
    path_x = np.linspace(-2.5, 0.5, 20) + np.random.randn(20) * 0.3
    path_y = np.linspace(-2.5, 0.5, 20) + np.random.randn(20) * 0.3
    axes[0].plot(path_x, path_y, 'r-o', linewidth=2, markersize=6, 
                alpha=0.7, label='Optimization path')
    axes[0].legend()
    
    # Residual network
    contour2 = axes[1].contourf(X, Y, Z_residual, levels=20, cmap='viridis')
    axes[1].set_title('Loss Landscape: Residual Network\n(Smoother, easier optimization)',
                      fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Parameter 1')
    axes[1].set_ylabel('Parameter 2')
    fig.colorbar(contour2, ax=axes[1], label='Loss')
    
    # Add optimization path (smooth convergence)
    path_x = np.linspace(-2.5, 0, 20)
    path_y = np.linspace(-2.5, 0, 20)
    axes[1].plot(path_x, path_y, 'r-o', linewidth=2, markersize=6,
                alpha=0.7, label='Optimization path')
    axes[1].legend()
    
    plt.tight_layout()
    return fig


# ============================================================================
# ARCHITECTURE VISUALIZATION
# ============================================================================

def draw_residual_block_architecture():
    """
    Draw a clear diagram of residual block architecture
    """
    print("Drawing residual block architecture...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plain Block
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 12)
    ax1.axis('off')
    ax1.set_title('Plain Block (No Skip Connection)', fontsize=14, fontweight='bold', pad=20)
    
    # Draw plain block
    blocks = [
        (4, 10, 'Input\n(H×W×C)', 'lightblue'),
        (4, 8, 'Conv 3×3', 'lightcoral'),
        (4, 7, 'BatchNorm', 'lightgreen'),
        (4, 6, 'ReLU', 'lightyellow'),
        (4, 4, 'Conv 3×3', 'lightcoral'),
        (4, 3, 'BatchNorm', 'lightgreen'),
        (4, 2, 'ReLU', 'lightyellow'),
        (4, 0, 'Output', 'lightblue'),
    ]
    
    for x, y, label, color in blocks:
        box = FancyBboxPatch((x-1, y-0.3), 2, 0.6, boxstyle="round,pad=0.1",
                            edgecolor='black', facecolor=color, linewidth=2)
        ax1.add_patch(box)
        ax1.text(x, y, label, ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Add arrows between blocks
        if y > 0:
            arrow = FancyArrowPatch((x, y-0.3), (x, y-0.7),
                                   arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
            ax1.add_patch(arrow)
    
    # Residual Block
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 12)
    ax2.axis('off')
    ax2.set_title('Residual Block (With Skip Connection)', fontsize=14, fontweight='bold', pad=20)
    
    # Draw residual block
    blocks = [
        (4, 10, 'Input\n(H×W×C)', 'lightblue'),
        (4, 8, 'Conv 3×3', 'lightcoral'),
        (4, 7, 'BatchNorm', 'lightgreen'),
        (4, 6, 'ReLU', 'lightyellow'),
        (4, 4, 'Conv 3×3', 'lightcoral'),
        (4, 3, 'BatchNorm', 'lightgreen'),
        (4, 1.5, 'Add (+)', 'orange'),
        (4, 0, 'Output', 'lightblue'),
    ]
    
    for x, y, label, color in blocks:
        if 'Add' in label:
            # Draw circle for addition
            circle = plt.Circle((x, y), 0.4, edgecolor='black', facecolor=color, linewidth=2)
            ax2.add_patch(circle)
            ax2.text(x, y, label, ha='center', va='center', fontsize=10, fontweight='bold')
        else:
            box = FancyBboxPatch((x-1, y-0.3), 2, 0.6, boxstyle="round,pad=0.1",
                                edgecolor='black', facecolor=color, linewidth=2)
            ax2.add_patch(box)
            ax2.text(x, y, label, ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Add arrows
        if y > 1.5:
            arrow = FancyArrowPatch((x, y-0.3), (x, y-0.7),
                                   arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
            ax2.add_patch(arrow)
        elif y == 1.5:
            # Arrow from BN to Add
            arrow = FancyArrowPatch((x, 2.7), (x, 1.9),
                                   arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
            ax2.add_patch(arrow)
            # Arrow from Add to Output
            arrow = FancyArrowPatch((x, 1.1), (x, 0.3),
                                   arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
            ax2.add_patch(arrow)
    
    # Draw skip connection (the key difference!)
    skip_arrow = FancyArrowPatch((5.5, 10), (5.5, 1.5),
                                arrowstyle='->', mutation_scale=25, 
                                linewidth=3, color='red', linestyle='--')
    ax2.add_patch(skip_arrow)
    ax2.text(6.5, 5.5, 'Skip\nConnection', ha='center', va='center',
            fontsize=11, fontweight='bold', color='red',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    return fig


# ============================================================================
# FEATURE MAP EVOLUTION
# ============================================================================

def visualize_feature_evolution():
    """
    Visualize how features evolve through residual blocks
    """
    print("Visualizing feature evolution...")
    
    from torch import nn
    
    # Create simple models
    class SimpleResBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
            self.conv_skip = nn.Conv2d(3, 16, 1)
        
        def forward(self, x):
            identity = self.conv_skip(x)
            out = F.relu(self.conv1(x))
            out = self.conv2(out)
            return F.relu(out + identity), out, identity
    
    # Create synthetic input
    x = torch.randn(1, 3, 32, 32)
    
    # Forward pass
    model = SimpleResBlock()
    with torch.no_grad():
        final, residual_path, skip_path = model(x)
    
    # Visualize
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    def show_feature_map(ax, tensor, title):
        # Take mean across channels and batch
        feature_map = tensor[0].mean(0).numpy()
        im = ax.imshow(feature_map, cmap='viridis')
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.axis('off')
        return im
    
    # Row 1: Individual channels from input
    for i in range(3):
        show_feature_map(axes[0, i], x[:, i:i+1], f'Input Channel {i+1}')
    axes[0, 3].axis('off')
    axes[0, 3].text(0.5, 0.5, 'Input\n(3 channels)', ha='center', va='center',
                   fontsize=12, fontweight='bold', transform=axes[0, 3].transAxes)
    
    # Row 2: Residual components
    titles = ['Residual Path\n(Learned Features)', 'Skip Connection\n(Identity)',
              'Addition\n(Residual + Skip)', 'Final Output\n(After ReLU)']
    tensors = [residual_path, skip_path, residual_path + skip_path, final]
    
    for i, (tensor, title) in enumerate(zip(tensors, titles)):
        show_feature_map(axes[1, i], tensor, title)
    
    plt.suptitle('Feature Evolution Through Residual Block', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def generate_all_visualizations():
    """
    Generate all visualization plots
    """
    print("=" * 80)
    print("GENERATING RESIDUAL CONNECTION VISUALIZATIONS")
    print("=" * 80)
    
    # Simple models for gradient flow
    class SimpleModels:
        class PlainModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([
                    nn.Sequential(
                        nn.Conv2d(64, 64, 3, padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU()
                    ) for _ in range(10)
                ])
        
        class ResidualModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([
                    nn.ModuleDict({
                        'conv1': nn.Conv2d(64, 64, 3, padding=1),
                        'bn1': nn.BatchNorm2d(64),
                        'conv2': nn.Conv2d(64, 64, 3, padding=1),
                        'bn2': nn.BatchNorm2d(64)
                    }) for _ in range(10)
                ])
    
    plain_model = SimpleModels.PlainModel()
    residual_model = SimpleModels.ResidualModel()
    
    # 1. Gradient Flow
    print("\n1. Creating gradient flow visualization...")
    fig1 = visualize_gradient_flow(plain_model, residual_model)
    plt.savefig('/home/claude/residual_connections/gradient_flow.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Loss Landscape
    print("2. Creating loss landscape visualization...")
    fig2 = visualize_loss_landscape()
    plt.savefig('/home/claude/residual_connections/loss_landscape.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Architecture
    print("3. Creating architecture diagram...")
    fig3 = draw_residual_block_architecture()
    plt.savefig('/home/claude/residual_connections/architecture.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Feature Evolution
    print("4. Creating feature evolution visualization...")
    fig4 = visualize_feature_evolution()
    plt.savefig('/home/claude/residual_connections/feature_evolution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n" + "=" * 80)
    print("All visualizations saved!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - gradient_flow.png      : Gradient magnitudes through depth")
    print("  - loss_landscape.png     : Optimization landscape comparison")
    print("  - architecture.png       : Residual block diagram")
    print("  - feature_evolution.png  : Feature map transformations")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    generate_all_visualizations()
