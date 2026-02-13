"""
Module 52.03: Sample Visualization and Quality Assessment
=========================================================

This module covers visualization techniques for evaluating generated samples
and basic quality metrics based on visual inspection.

Learning Objectives:
-------------------
1. Create effective visualizations of generated samples
2. Perform latent space interpolation
3. Compute reconstruction quality metrics
4. Assess sample diversity visually

Key Concepts:
------------
- Grid visualization of samples
- Latent space interpolation
- Reconstruction error metrics (MSE, SSIM)
- Visual quality assessment

Author: Educational AI Team
Date: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import math

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)


class SampleGridVisualizer:
    """
    Creates grid visualizations of generated samples.
    
    Purpose:
    -------
    Visual inspection is crucial for generative models because:
    1. Metrics may miss perceptual quality issues
    2. Humans are excellent at detecting artifacts
    3. Helps identify mode collapse and diversity
    4. Reveals systematic generation failures
    """
    
    @staticmethod
    def create_image_grid(images: torch.Tensor,
                         nrow: int = 8,
                         padding: int = 2,
                         normalize: bool = True) -> np.ndarray:
        """
        Create a grid of images for visualization.
        
        Args:
            images: Batch of images [batch_size, channels, height, width]
            nrow: Number of images per row
            padding: Pixels between images
            normalize: Whether to normalize to [0, 1]
        
        Returns:
            Grid image as numpy array [height, width, channels]
        """
        batch_size = images.shape[0]
        ncol = (batch_size + nrow - 1) // nrow  # Ceiling division
        
        # Normalize if requested
        if normalize:
            images = (images - images.min()) / (images.max() - images.min() + 1e-8)
        
        # Handle grayscale vs RGB
        if images.shape[1] == 1:
            # Grayscale: [B, 1, H, W] -> [B, H, W]
            images = images.squeeze(1)
            is_grayscale = True
        else:
            # RGB: [B, 3, H, W] -> [B, H, W, 3]
            images = images.permute(0, 2, 3, 1)
            is_grayscale = False
        
        H, W = images.shape[1], images.shape[2]
        
        # Create grid canvas
        grid_h = ncol * H + (ncol + 1) * padding
        grid_w = nrow * W + (nrow + 1) * padding
        
        if is_grayscale:
            grid = np.ones((grid_h, grid_w)) * 0.5  # Gray background
        else:
            grid = np.ones((grid_h, grid_w, 3)) * 0.5
        
        # Place images in grid
        for idx in range(batch_size):
            row = idx // nrow
            col = idx % nrow
            
            y = row * (H + padding) + padding
            x = col * (W + padding) + padding
            
            grid[y:y+H, x:x+W] = images[idx].numpy()
        
        return grid
    
    @staticmethod
    def plot_sample_grid(images: torch.Tensor,
                        title: str = "Generated Samples",
                        save_path: Optional[str] = None):
        """
        Plot and optionally save a grid of samples.
        
        Args:
            images: Batch of images [batch_size, C, H, W]
            title: Plot title
            save_path: Optional path to save figure
        """
        grid = SampleGridVisualizer.create_image_grid(images, nrow=8)
        
        plt.figure(figsize=(12, 12))
        if len(grid.shape) == 2:  # Grayscale
            plt.imshow(grid, cmap='gray')
        else:  # RGB
            plt.imshow(grid)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")
        
        plt.tight_layout()


class LatentSpaceInterpolation:
    """
    Performs linear interpolation in latent space.
    
    Mathematical Foundation:
    -----------------------
    For two latent vectors z1, z2, interpolate:
        z(t) = (1-t) * z1 + t * z2,  where t ∈ [0, 1]
    
    Purpose:
    -------
    1. Assess latent space smoothness
    2. Detect discontinuities or jumps
    3. Visualize learned latent structure
    4. Check for meaningful interpolations
    
    A good generative model should produce smooth transitions
    when interpolating between latent codes.
    """
    
    @staticmethod
    def linear_interpolate(z1: torch.Tensor,
                          z2: torch.Tensor,
                          num_steps: int = 10) -> torch.Tensor:
        """
        Perform linear interpolation between two latent vectors.
        
        Args:
            z1: Start latent vector [latent_dim]
            z2: End latent vector [latent_dim]
            num_steps: Number of interpolation steps
        
        Returns:
            Interpolated latent vectors [num_steps, latent_dim]
        
        Mathematical Formula:
        --------------------
        z(t) = (1-t) * z1 + t * z2
        where t = [0, 1/(n-1), 2/(n-1), ..., 1]
        """
        # Create interpolation weights
        # Shape: [num_steps]
        t = torch.linspace(0, 1, num_steps)
        
        # Expand dimensions for broadcasting
        # z1, z2: [latent_dim] -> [1, latent_dim]
        # t: [num_steps] -> [num_steps, 1]
        t = t.unsqueeze(1)
        z1 = z1.unsqueeze(0)
        z2 = z2.unsqueeze(0)
        
        # Interpolate: z(t) = (1-t) * z1 + t * z2
        # Shape: [num_steps, latent_dim]
        z_interp = (1 - t) * z1 + t * z2
        
        return z_interp
    
    @staticmethod
    def spherical_interpolate(z1: torch.Tensor,
                             z2: torch.Tensor,
                             num_steps: int = 10) -> torch.Tensor:
        """
        Perform spherical linear interpolation (slerp).
        
        Why Slerp?
        ---------
        For distributions like Gaussian (VAE latent space),
        slerp maintains constant distance from origin,
        producing more natural interpolations.
        
        Mathematical Formula:
        --------------------
        slerp(z1, z2; t) = [sin((1-t)θ)/sin(θ)] * z1 + [sin(tθ)/sin(θ)] * z2
        
        where θ = arccos(z1·z2 / (||z1|| ||z2||))
        
        Args:
            z1: Start latent vector [latent_dim]
            z2: End latent vector [latent_dim]
            num_steps: Number of interpolation steps
        
        Returns:
            Interpolated latent vectors [num_steps, latent_dim]
        """
        # Normalize vectors
        z1_norm = F.normalize(z1, dim=0)
        z2_norm = F.normalize(z2, dim=0)
        
        # Compute angle between vectors
        # θ = arccos(z1·z2)
        dot = torch.dot(z1_norm, z2_norm)
        # Clamp to avoid numerical issues
        dot = torch.clamp(dot, -1.0, 1.0)
        theta = torch.acos(dot)
        
        # Handle case where vectors are nearly parallel
        if theta < 1e-6:
            return LatentSpaceInterpolation.linear_interpolate(z1, z2, num_steps)
        
        # Create interpolation weights
        t = torch.linspace(0, 1, num_steps).unsqueeze(1)
        
        # Compute slerp weights
        sin_theta = torch.sin(theta)
        w1 = torch.sin((1 - t) * theta) / sin_theta
        w2 = torch.sin(t * theta) / sin_theta
        
        # Interpolate
        z_interp = w1 * z1 + w2 * z2
        
        return z_interp
    
    @staticmethod
    def visualize_interpolation(decoder,
                               z1: torch.Tensor,
                               z2: torch.Tensor,
                               num_steps: int = 10,
                               use_slerp: bool = False,
                               save_path: Optional[str] = None):
        """
        Visualize interpolation between two latent codes.
        
        Args:
            decoder: Decoder network
            z1: Start latent code
            z2: End latent code
            num_steps: Number of interpolation steps
            use_slerp: Use spherical interpolation instead of linear
            save_path: Optional save path
        """
        # Perform interpolation
        if use_slerp:
            z_interp = LatentSpaceInterpolation.spherical_interpolate(
                z1, z2, num_steps
            )
        else:
            z_interp = LatentSpaceInterpolation.linear_interpolate(
                z1, z2, num_steps
            )
        
        # Decode interpolated latents
        with torch.no_grad():
            images = decoder(z_interp)
        
        # Create visualization
        grid = SampleGridVisualizer.create_image_grid(
            images, nrow=num_steps, padding=2
        )
        
        method = "Spherical" if use_slerp else "Linear"
        plt.figure(figsize=(15, 3))
        if len(grid.shape) == 2:
            plt.imshow(grid, cmap='gray')
        else:
            plt.imshow(grid)
        plt.title(f'{method} Interpolation in Latent Space',
                 fontsize=14, fontweight='bold')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")


class ReconstructionQuality:
    """
    Metrics for assessing reconstruction quality.
    
    Common Metrics:
    --------------
    1. MSE (Mean Squared Error): Pixel-wise differences
    2. PSNR (Peak Signal-to-Noise Ratio): Signal quality
    3. SSIM (Structural Similarity): Perceptual similarity
    
    Use Cases:
    ---------
    - VAE reconstruction quality
    - Image-to-image translation
    - Compression evaluation
    """
    
    @staticmethod
    def compute_mse(original: torch.Tensor,
                    reconstructed: torch.Tensor) -> float:
        """
        Compute Mean Squared Error.
        
        Mathematical Formula:
        --------------------
        MSE = 1/(N*C*H*W) * Σ(original - reconstructed)²
        
        Args:
            original: Original images [B, C, H, W]
            reconstructed: Reconstructed images [B, C, H, W]
        
        Returns:
            MSE value (lower is better)
        """
        # Compute squared differences
        squared_diff = (original - reconstructed) ** 2
        
        # Average over all dimensions
        mse = torch.mean(squared_diff)
        
        return mse.item()
    
    @staticmethod
    def compute_psnr(original: torch.Tensor,
                     reconstructed: torch.Tensor,
                     max_pixel_value: float = 1.0) -> float:
        """
        Compute Peak Signal-to-Noise Ratio.
        
        Mathematical Formula:
        --------------------
        PSNR = 10 * log10(MAX² / MSE)
        
        where MAX is the maximum possible pixel value.
        
        Interpretation:
        --------------
        - Higher PSNR = Better quality
        - PSNR > 30 dB: Good quality
        - PSNR > 40 dB: Excellent quality
        
        Args:
            original: Original images [B, C, H, W]
            reconstructed: Reconstructed images [B, C, H, W]
            max_pixel_value: Maximum pixel value (1.0 for normalized images)
        
        Returns:
            PSNR in decibels (higher is better)
        """
        # Compute MSE
        mse = ReconstructionQuality.compute_mse(original, reconstructed)
        
        # Avoid division by zero
        if mse < 1e-10:
            return 100.0  # Perfect reconstruction
        
        # Compute PSNR
        psnr = 10 * np.log10(max_pixel_value ** 2 / mse)
        
        return psnr
    
    @staticmethod
    def compute_per_sample_mse(original: torch.Tensor,
                              reconstructed: torch.Tensor) -> torch.Tensor:
        """
        Compute MSE for each sample (useful for analysis).
        
        Args:
            original: Original images [B, C, H, W]
            reconstructed: Reconstructed images [B, C, H, W]
        
        Returns:
            Per-sample MSE [B]
        """
        # Compute squared differences
        squared_diff = (original - reconstructed) ** 2
        
        # Average over C, H, W dimensions
        per_sample_mse = torch.mean(squared_diff, dim=[1, 2, 3])
        
        return per_sample_mse


def demonstrate_sample_visualization():
    """
    Demonstrates sample visualization techniques.
    """
    print("=" * 70)
    print("Sample Visualization Demonstration")
    print("=" * 70)
    
    # Generate synthetic images (simulating MNIST-like data)
    batch_size = 64
    images = torch.randn(batch_size, 1, 28, 28)
    # Add some structure (make it look like digits)
    images = torch.sigmoid(images * 2)
    
    print(f"\nGenerated {batch_size} synthetic images")
    print(f"Image shape: {images.shape}")
    
    # Create grid visualization
    print("\nCreating grid visualization...")
    SampleGridVisualizer.plot_sample_grid(
        images,
        title="Generated Samples (8×8 Grid)",
        save_path="/home/claude/sample_grid.png"
    )
    
    # Analyze diversity
    print("\n" + "-" * 70)
    print("Diversity Analysis:")
    print("-" * 70)
    
    # Compute pairwise differences as a simple diversity measure
    # Flatten images
    flat_images = images.reshape(batch_size, -1)
    
    # Compute pairwise L2 distances
    # ||x_i - x_j||
    dists = torch.cdist(flat_images, flat_images, p=2)
    
    # Get upper triangular (exclude diagonal)
    upper_tri = dists[torch.triu(torch.ones_like(dists), diagonal=1) == 1]
    
    print(f"Average pairwise L2 distance: {upper_tri.mean():.4f}")
    print(f"Min pairwise distance: {upper_tri.min():.4f}")
    print(f"Max pairwise distance: {upper_tri.max():.4f}")
    print(f"\nInterpretation:")
    print("  - Low average distance → Low diversity (mode collapse)")
    print("  - High average distance → High diversity")


def demonstrate_latent_interpolation():
    """
    Demonstrates latent space interpolation.
    """
    print("\n" + "=" * 70)
    print("Latent Space Interpolation Demonstration")
    print("=" * 70)
    
    # Simple decoder network (for demonstration)
    class SimpleDecoder(nn.Module):
        def __init__(self, latent_dim=10):
            super().__init__()
            self.fc1 = nn.Linear(latent_dim, 128)
            self.fc2 = nn.Linear(128, 28*28)
        
        def forward(self, z):
            h = F.relu(self.fc1(z))
            x = torch.sigmoid(self.fc2(h))
            return x.view(-1, 1, 28, 28)
    
    decoder = SimpleDecoder(latent_dim=10)
    decoder.eval()
    
    # Sample two random latent codes
    z1 = torch.randn(10)
    z2 = torch.randn(10)
    
    print(f"\nInterpolating between two random latent codes")
    print(f"Latent dimension: {len(z1)}")
    print(f"z1 norm: {torch.norm(z1):.4f}")
    print(f"z2 norm: {torch.norm(z2):.4f}")
    
    # Linear interpolation
    print("\n" + "-" * 70)
    print("Linear Interpolation:")
    print("-" * 70)
    
    z_linear = LatentSpaceInterpolation.linear_interpolate(z1, z2, num_steps=10)
    print(f"Generated {len(z_linear)} interpolated codes")
    print(f"Norms along path: {torch.norm(z_linear, dim=1)}")
    
    # Spherical interpolation
    print("\n" + "-" * 70)
    print("Spherical Interpolation:")
    print("-" * 70)
    
    z_slerp = LatentSpaceInterpolation.spherical_interpolate(z1, z2, num_steps=10)
    print(f"Generated {len(z_slerp)} interpolated codes")
    print(f"Norms along path: {torch.norm(z_slerp, dim=1)}")
    print("\nNote: Spherical interpolation maintains constant norm")


def demonstrate_reconstruction_quality():
    """
    Demonstrates reconstruction quality metrics.
    """
    print("\n" + "=" * 70)
    print("Reconstruction Quality Metrics")
    print("=" * 70)
    
    # Generate original images
    batch_size = 10
    original = torch.randn(batch_size, 1, 28, 28)
    original = torch.sigmoid(original * 2)  # Normalize to [0, 1]
    
    # Create reconstructions with different quality levels
    # Perfect reconstruction
    perfect_recon = original.clone()
    
    # Good reconstruction (small noise)
    good_recon = original + torch.randn_like(original) * 0.05
    good_recon = torch.clamp(good_recon, 0, 1)
    
    # Poor reconstruction (large noise)
    poor_recon = original + torch.randn_like(original) * 0.2
    poor_recon = torch.clamp(poor_recon, 0, 1)
    
    # Compute metrics
    print("\n" + "-" * 70)
    print("Reconstruction Quality Comparison:")
    print("-" * 70)
    
    reconstructions = {
        "Perfect": perfect_recon,
        "Good": good_recon,
        "Poor": poor_recon
    }
    
    print(f"\n{'Reconstruction':<15} {'MSE':<12} {'PSNR (dB)'}")
    print("-" * 70)
    
    for name, recon in reconstructions.items():
        mse = ReconstructionQuality.compute_mse(original, recon)
        psnr = ReconstructionQuality.compute_psnr(original, recon)
        print(f"{name:<15} {mse:<12.6f} {psnr:<10.2f}")
    
    print("\n" + "-" * 70)
    print("Interpretation:")
    print("-" * 70)
    print("MSE: Lower is better (0 = perfect)")
    print("PSNR: Higher is better")
    print("  - >40 dB: Excellent quality")
    print("  - 30-40 dB: Good quality")
    print("  - 20-30 dB: Fair quality")
    print("  - <20 dB: Poor quality")


def main():
    """
    Main function demonstrating sample visualization.
    """
    print("\n" + "=" * 70)
    print("MODULE 52.03: SAMPLE VISUALIZATION AND QUALITY ASSESSMENT")
    print("=" * 70)
    
    # Demonstrate sample visualization
    demonstrate_sample_visualization()
    
    # Demonstrate latent interpolation
    demonstrate_latent_interpolation()
    
    # Demonstrate reconstruction quality
    demonstrate_reconstruction_quality()
    
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
    1. Visual Inspection is Essential:
       - Metrics alone can miss perceptual issues
       - Grid visualization reveals mode collapse
       - Humans excel at detecting artifacts
    
    2. Latent Space Interpolation:
       - Linear: Simple and fast
       - Spherical: Better for Gaussian distributions
       - Smooth transitions indicate good latent structure
    
    3. Reconstruction Metrics:
       - MSE: Simple pixel-wise difference
       - PSNR: Signal quality in decibels
       - Higher PSNR typically means better quality
    
    4. Diversity Assessment:
       - Pairwise sample distances
       - Visual inspection of grid
       - Low diversity suggests mode collapse
    
    5. Best Practices:
       - Always visualize samples
       - Check interpolations for smoothness
       - Use multiple quality metrics
       - Combine quantitative and qualitative evaluation
    """)
    print("=" * 70)


if __name__ == "__main__":
    main()
