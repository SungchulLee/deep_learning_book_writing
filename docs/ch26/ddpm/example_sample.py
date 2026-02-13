"""
Example Sampling Script

Generate samples from a trained diffusion model.
This script loads a checkpoint and generates new MNIST digits.
"""

import torch
import os
from mnist_diffusion import MNISTDiffusion
from diffusion_utils import visualize_samples


def generate_grid(diffusion, n_samples=64, filename='generated_grid.png'):
    """Generate a grid of samples."""
    print(f"Generating {n_samples} samples...")
    samples = diffusion.sample_images(n_samples=n_samples, use_ema=True)
    visualize_samples(samples, nrow=8, filename=filename)
    print(f"Saved to {filename}")


def generate_interpolation(diffusion, n_steps=10):
    """Generate interpolation between random starting points."""
    print(f"Generating interpolation with {n_steps} steps...")
    
    # Generate two random noise vectors
    noise1 = torch.randn(1, 1, 28, 28, device=diffusion.device)
    noise2 = torch.randn(1, 1, 28, 28, device=diffusion.device)
    
    # Interpolate
    alphas = torch.linspace(0, 1, n_steps)
    interpolated_samples = []
    
    for alpha in alphas:
        # Linear interpolation in noise space
        noise = (1 - alpha) * noise1 + alpha * noise2
        
        # Denoise from this starting point
        x_t = noise
        for t in reversed(range(diffusion.timesteps)):
            t_tensor = torch.full((1,), t, device=diffusion.device, dtype=torch.long)
            
            predicted_noise = diffusion.ema_model(x_t, t_tensor)
            
            beta_t = diffusion.diffusion_params['betas'][t]
            sqrt_recip_alpha_t = diffusion.diffusion_params['sqrt_recip_alphas'][t]
            sqrt_one_minus_alpha_cumprod_t = diffusion.diffusion_params['sqrt_one_minus_alphas_cumprod'][t]
            
            mean = sqrt_recip_alpha_t * (x_t - beta_t * predicted_noise / sqrt_one_minus_alpha_cumprod_t)
            
            if t > 0:
                posterior_variance_t = diffusion.diffusion_params['posterior_variance'][t]
                noise_sample = torch.randn_like(x_t)
                x_t = mean + torch.sqrt(posterior_variance_t) * noise_sample
            else:
                x_t = mean
        
        interpolated_samples.append(x_t)
    
    # Visualize
    samples = torch.cat(interpolated_samples, dim=0)
    visualize_samples(samples, nrow=n_steps, filename='interpolation.png')
    print("Saved to interpolation.png")


def main():
    # Check for checkpoint
    checkpoint_files = [
        'mnist_diffusion_model.pt',
        'mnist_diffusion_final.pt',
        'checkpoint_epoch_100.pt'
    ]
    
    checkpoint = None
    for ckpt_file in checkpoint_files:
        if os.path.exists(ckpt_file):
            checkpoint = ckpt_file
            break
    
    if checkpoint is None:
        print("Error: No checkpoint found!")
        print("Please train a model first using example_train.py")
        print("Looking for one of:", checkpoint_files)
        return
    
    print(f"Loading checkpoint: {checkpoint}")
    print("-" * 50)
    
    # Initialize model
    diffusion = MNISTDiffusion(
        timesteps=1000,
        batch_size=64,
        learning_rate=2e-4
    )
    
    # Load checkpoint
    diffusion.load_checkpoint(checkpoint)
    
    print("\nGenerating samples...")
    print("-" * 50)
    
    # Generate various samples
    generate_grid(diffusion, n_samples=64, filename='samples_8x8.png')
    generate_grid(diffusion, n_samples=100, filename='samples_10x10.png')
    
    # Generate interpolation
    generate_interpolation(diffusion, n_steps=10)
    
    # Generate large batch for diversity analysis
    print("\nGenerating large batch for visualization...")
    samples = diffusion.sample_images(n_samples=256)
    visualize_samples(samples, nrow=16, filename='samples_large.png')
    
    print("\n" + "=" * 50)
    print("Sampling complete!")
    print("Generated files:")
    print("  - samples_8x8.png: 64 samples in 8x8 grid")
    print("  - samples_10x10.png: 100 samples in 10x10 grid")
    print("  - samples_large.png: 256 samples")
    print("  - interpolation.png: Smooth interpolation between digits")
    print("=" * 50)


if __name__ == "__main__":
    main()
