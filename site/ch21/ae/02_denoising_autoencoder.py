"""
Module 40.2: Denoising Autoencoder

This script introduces denoising autoencoders (DAE), which learn robust
representations by reconstructing clean data from corrupted inputs.

Key Concepts:
- Adding structured noise to inputs during training
- Learning to denoise and reconstruct
- Comparison with standard autoencoders
- Robustness to noise and corruption
- Applications in image denoising

Mathematical Foundation:
-----------------------
Standard Autoencoder: minimize ||x - f(x)||²
Denoising Autoencoder: minimize ||x - f(x̃)||²

Where:
- x is the clean input
- x̃ = corrupt(x) is the noisy input
- f(x̃) is the reconstruction from noisy input

The model learns to map corrupted inputs back to clean outputs,
forcing it to learn more robust and meaningful features rather
than just copying the input.

Common Corruption Strategies:
1. Gaussian noise: x̃ = x + ε, ε ~ N(0, σ²)
2. Salt-and-pepper noise: randomly set pixels to 0 or 1
3. Masking noise: randomly set pixels to 0
4. Dropout: randomly drop input features

Time: 40 minutes
Level: Beginner-Intermediate
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Callable


# =============================================================================
# PART 1: NOISE CORRUPTION FUNCTIONS
# =============================================================================

def add_gaussian_noise(images: torch.Tensor, noise_factor: float = 0.3) -> torch.Tensor:
    """
    Add Gaussian noise to images.
    
    Corrupted image: x̃ = x + ε where ε ~ N(0, σ²)
    
    Parameters:
    -----------
    images : torch.Tensor, shape (batch_size, ...)
        Clean images
    noise_factor : float
        Standard deviation of Gaussian noise (controls noise level)
        
    Returns:
    --------
    noisy_images : torch.Tensor
        Images with added Gaussian noise, clipped to [0, 1]
    """
    # Sample Gaussian noise with same shape as images
    noise = torch.randn_like(images) * noise_factor
    
    # Add noise and clip to valid range [0, 1]
    noisy_images = images + noise
    noisy_images = torch.clamp(noisy_images, 0.0, 1.0)
    
    return noisy_images


def add_salt_pepper_noise(images: torch.Tensor, noise_prob: float = 0.2) -> torch.Tensor:
    """
    Add salt-and-pepper noise to images.
    
    Randomly set pixels to either 0 (pepper) or 1 (salt).
    
    Parameters:
    -----------
    images : torch.Tensor, shape (batch_size, ...)
        Clean images
    noise_prob : float
        Probability of corrupting each pixel
        
    Returns:
    --------
    noisy_images : torch.Tensor
        Images with salt-and-pepper noise
    """
    noisy_images = images.clone()
    
    # Generate random mask for corruption
    noise_mask = torch.rand_like(images) < noise_prob
    
    # For corrupted pixels, randomly choose salt (1) or pepper (0)
    salt_mask = torch.rand_like(images) > 0.5
    
    # Apply salt and pepper noise
    noisy_images[noise_mask & salt_mask] = 1.0  # Salt
    noisy_images[noise_mask & ~salt_mask] = 0.0  # Pepper
    
    return noisy_images


def add_masking_noise(images: torch.Tensor, mask_prob: float = 0.3) -> torch.Tensor:
    """
    Add masking noise by randomly setting pixels to zero.
    
    This is similar to dropout but applied to input pixels.
    
    Parameters:
    -----------
    images : torch.Tensor, shape (batch_size, ...)
        Clean images
    mask_prob : float
        Probability of masking each pixel
        
    Returns:
    --------
    noisy_images : torch.Tensor
        Images with random pixels masked to 0
    """
    # Generate random mask: 0 where we mask, 1 where we keep
    mask = (torch.rand_like(images) > mask_prob).float()
    
    # Apply mask
    noisy_images = images * mask
    
    return noisy_images


# =============================================================================
# PART 2: DENOISING AUTOENCODER ARCHITECTURE
# =============================================================================

class DenoisingAutoencoder(nn.Module):
    """
    Denoising Autoencoder with same architecture as basic autoencoder.
    
    The key difference is in training: we corrupt inputs but reconstruct
    clean originals, forcing the model to learn robust representations.
    
    Architecture: 784 → 256 → 128 → 64 → 128 → 256 → 784
    """
    
    def __init__(self, input_dim: int = 784, latent_dim: int = 64):
        super(DenoisingAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder: Maps noisy input to latent representation
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
            nn.ReLU()
        )
        
        # Decoder: Reconstructs clean image from latent representation
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: encode noisy input and decode to clean output.
        
        Parameters:
        -----------
        x : torch.Tensor, shape (batch_size, input_dim)
            Input (can be noisy or clean)
            
        Returns:
        --------
        reconstructed : torch.Tensor, shape (batch_size, input_dim)
            Reconstructed (denoised) output
        latent : torch.Tensor, shape (batch_size, latent_dim)
            Latent representation
        """
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent


# =============================================================================
# PART 3: TRAINING FUNCTION WITH NOISE CORRUPTION
# =============================================================================

def train_denoising_autoencoder(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    noise_fn: Callable,
    noise_param: float
) -> float:
    """
    Train denoising autoencoder for one epoch.
    
    Training Process:
    1. Load clean images
    2. Create corrupted versions
    3. Pass corrupted images through encoder-decoder
    4. Compute loss between reconstruction and CLEAN images
    5. Backpropagate and update weights
    
    Parameters:
    -----------
    model : nn.Module
        Denoising autoencoder model
    train_loader : DataLoader
        Training data loader
    criterion : nn.Module
        Loss function (MSE)
    optimizer : optim.Optimizer
        Optimizer
    device : torch.device
        Device for computation
    epoch : int
        Current epoch number
    noise_fn : Callable
        Function to add noise (e.g., add_gaussian_noise)
    noise_param : float
        Parameter for noise function (e.g., noise_factor)
        
    Returns:
    --------
    avg_loss : float
        Average loss over all batches
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (images, _) in enumerate(train_loader):
        # Flatten clean images
        clean_images = images.view(images.size(0), -1).to(device)
        
        # Add noise to create corrupted inputs
        noisy_images = noise_fn(clean_images, noise_param)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass: Encode noisy input, decode to clean output
        reconstructed, _ = model(noisy_images)
        
        # IMPORTANT: Loss is between reconstruction and CLEAN images
        # This forces the model to learn to denoise
        loss = criterion(reconstructed, clean_images)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, '
                  f'Loss: {loss.item():.6f}')
    
    avg_loss = total_loss / num_batches
    return avg_loss


# =============================================================================
# PART 4: EVALUATION AND VISUALIZATION
# =============================================================================

def visualize_denoising_results(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    noise_fn: Callable,
    noise_param: float,
    num_images: int = 10
):
    """
    Visualize denoising results: clean → noisy → reconstructed.
    
    Shows three rows:
    1. Original clean images
    2. Corrupted noisy images
    3. Reconstructed (denoised) images
    
    Parameters:
    -----------
    model : nn.Module
        Trained denoising autoencoder
    test_loader : DataLoader
        Test data loader
    device : torch.device
        Device for inference
    noise_fn : Callable
        Noise corruption function
    noise_param : float
        Noise parameter
    num_images : int
        Number of images to visualize
    """
    model.eval()
    
    # Get test images
    images, labels = next(iter(test_loader))
    images = images[:num_images]
    labels = labels[:num_images]
    
    # Flatten clean images
    clean_images = images.view(images.size(0), -1).to(device)
    
    # Add noise
    noisy_images = noise_fn(clean_images, noise_param)
    
    # Get reconstructions
    with torch.no_grad():
        reconstructed, _ = model(noisy_images)
    
    # Move to CPU and reshape
    clean_np = clean_images.cpu().numpy().reshape(-1, 28, 28)
    noisy_np = noisy_images.cpu().numpy().reshape(-1, 28, 28)
    reconstructed_np = reconstructed.cpu().numpy().reshape(-1, 28, 28)
    
    # Visualize
    fig, axes = plt.subplots(3, num_images, figsize=(15, 5))
    
    for i in range(num_images):
        # Clean image
        axes[0, i].imshow(clean_np[i], cmap='gray', vmin=0, vmax=1)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Clean', fontsize=12, pad=10)
        axes[0, i].text(0.5, -0.1, f'{labels[i].item()}',
                       transform=axes[0, i].transAxes,
                       ha='center', fontsize=10)
        
        # Noisy image
        axes[1, i].imshow(noisy_np[i], cmap='gray', vmin=0, vmax=1)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Noisy', fontsize=12, pad=10)
        
        # Reconstructed image
        axes[2, i].imshow(reconstructed_np[i], cmap='gray', vmin=0, vmax=1)
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_title('Reconstructed', fontsize=12, pad=10)
    
    plt.tight_layout()
    plt.savefig('denoising_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved visualization to 'denoising_results.png'")


def compare_noise_types(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device
):
    """
    Compare different types of noise and reconstruction quality.
    
    Visualizes the same image with different corruption types:
    - Gaussian noise
    - Salt-and-pepper noise
    - Masking noise
    
    Parameters:
    -----------
    model : nn.Module
        Trained denoising autoencoder
    test_loader : DataLoader
        Test data loader
    device : torch.device
        Device for inference
    """
    model.eval()
    
    # Get one test image
    images, labels = next(iter(test_loader))
    clean_image = images[0:1].view(1, -1).to(device)
    label = labels[0].item()
    
    # Define noise types
    noise_types = [
        ('Gaussian', add_gaussian_noise, 0.3),
        ('Salt-Pepper', add_salt_pepper_noise, 0.2),
        ('Masking', add_masking_noise, 0.3)
    ]
    
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    
    # Original image in first column
    clean_np = clean_image.cpu().numpy().reshape(28, 28)
    for row in range(3):
        axes[row, 0].imshow(clean_np, cmap='gray', vmin=0, vmax=1)
        axes[row, 0].axis('off')
        if row == 0:
            axes[row, 0].set_title(f'Clean\n(digit {label})', fontsize=11)
    
    # For each noise type
    with torch.no_grad():
        for row, (noise_name, noise_fn, noise_param) in enumerate(noise_types):
            # Create noisy version
            noisy_image = noise_fn(clean_image, noise_param)
            
            # Reconstruct
            reconstructed, _ = model(noisy_image)
            
            # Convert to numpy
            noisy_np = noisy_image.cpu().numpy().reshape(28, 28)
            recon_np = reconstructed.cpu().numpy().reshape(28, 28)
            
            # Calculate MSE
            mse_noisy = np.mean((clean_np - noisy_np) ** 2)
            mse_recon = np.mean((clean_np - recon_np) ** 2)
            
            # Plot noisy
            axes[row, 1].imshow(noisy_np, cmap='gray', vmin=0, vmax=1)
            axes[row, 1].axis('off')
            if row == 0:
                axes[row, 1].set_title('Noisy', fontsize=11)
            axes[row, 1].text(0.5, -0.15, f'{noise_name}\nMSE: {mse_noisy:.4f}',
                            transform=axes[row, 1].transAxes,
                            ha='center', fontsize=9)
            
            # Plot reconstructed
            axes[row, 2].imshow(recon_np, cmap='gray', vmin=0, vmax=1)
            axes[row, 2].axis('off')
            if row == 0:
                axes[row, 2].set_title('Reconstructed', fontsize=11)
            axes[row, 2].text(0.5, -0.15, f'MSE: {mse_recon:.4f}',
                            transform=axes[row, 2].transAxes,
                            ha='center', fontsize=9)
            
            # Plot difference (error map)
            diff = np.abs(clean_np - recon_np)
            im = axes[row, 3].imshow(diff, cmap='hot', vmin=0, vmax=1)
            axes[row, 3].axis('off')
            if row == 0:
                axes[row, 3].set_title('Error Map', fontsize=11)
    
    # Add colorbar for error maps
    fig.colorbar(im, ax=axes[:, 3], location='right', shrink=0.8, label='Absolute Error')
    
    plt.suptitle('Comparison of Noise Types and Denoising', fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig('noise_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved visualization to 'noise_comparison.png'")


# =============================================================================
# PART 5: COMPARISON WITH STANDARD AUTOENCODER
# =============================================================================

def compare_with_standard_autoencoder(
    denoising_model: nn.Module,
    standard_model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    noise_fn: Callable,
    noise_param: float,
    num_images: int = 5
):
    """
    Compare denoising autoencoder with standard autoencoder on noisy inputs.
    
    This demonstrates that denoising autoencoders learn more robust
    representations and handle noisy inputs better than standard AEs.
    
    Parameters:
    -----------
    denoising_model : nn.Module
        Denoising autoencoder
    standard_model : nn.Module
        Standard autoencoder
    test_loader : DataLoader
        Test data loader
    device : torch.device
        Device for inference
    noise_fn : Callable
        Noise corruption function
    noise_param : float
        Noise parameter
    num_images : int
        Number of images to compare
    """
    denoising_model.eval()
    standard_model.eval()
    
    # Get test images
    images, labels = next(iter(test_loader))
    images = images[:num_images]
    
    # Flatten
    clean_images = images.view(images.size(0), -1).to(device)
    
    # Add noise
    noisy_images = noise_fn(clean_images, noise_param)
    
    # Get reconstructions from both models
    with torch.no_grad():
        denoising_recon, _ = denoising_model(noisy_images)
        standard_recon, _ = standard_model(noisy_images)
    
    # Move to CPU
    clean_np = clean_images.cpu().numpy().reshape(-1, 28, 28)
    noisy_np = noisy_images.cpu().numpy().reshape(-1, 28, 28)
    denoising_np = denoising_recon.cpu().numpy().reshape(-1, 28, 28)
    standard_np = standard_recon.cpu().numpy().reshape(-1, 28, 28)
    
    # Visualize
    fig, axes = plt.subplots(4, num_images, figsize=(12, 10))
    
    for i in range(num_images):
        # Clean
        axes[0, i].imshow(clean_np[i], cmap='gray', vmin=0, vmax=1)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Clean', fontsize=11)
        
        # Noisy
        axes[1, i].imshow(noisy_np[i], cmap='gray', vmin=0, vmax=1)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Noisy Input', fontsize=11)
        
        # Denoising AE reconstruction
        mse_denoising = np.mean((clean_np[i] - denoising_np[i]) ** 2)
        axes[2, i].imshow(denoising_np[i], cmap='gray', vmin=0, vmax=1)
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_title('Denoising AE', fontsize=11)
        axes[2, i].text(0.5, -0.1, f'MSE: {mse_denoising:.4f}',
                       transform=axes[2, i].transAxes,
                       ha='center', fontsize=9)
        
        # Standard AE reconstruction
        mse_standard = np.mean((clean_np[i] - standard_np[i]) ** 2)
        axes[3, i].imshow(standard_np[i], cmap='gray', vmin=0, vmax=1)
        axes[3, i].axis('off')
        if i == 0:
            axes[3, i].set_title('Standard AE', fontsize=11)
        axes[3, i].text(0.5, -0.1, f'MSE: {mse_standard:.4f}',
                       transform=axes[3, i].transAxes,
                       ha='center', fontsize=9)
    
    plt.suptitle('Denoising AE vs Standard AE on Noisy Inputs', fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved visualization to 'model_comparison.png'")


# =============================================================================
# PART 6: DATA LOADING
# =============================================================================

def load_mnist_data(batch_size: int = 128) -> Tuple[DataLoader, DataLoader]:
    """Load MNIST dataset."""
    transform = transforms.Compose([transforms.ToTensor()])
    
    train_dataset = datasets.MNIST(root='./data', train=True,
                                   download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False,
                                  download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                             shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=2)
    
    return train_loader, test_loader


# =============================================================================
# PART 7: MAIN EXECUTION
# =============================================================================

def main():
    """Main function to train denoising autoencoder."""
    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Hyperparameters
    input_dim = 784
    latent_dim = 64
    batch_size = 128
    learning_rate = 0.001
    num_epochs = 10
    
    # Noise configuration
    noise_type = 'gaussian'  # Options: 'gaussian', 'salt_pepper', 'masking'
    noise_param = 0.3  # Adjust based on noise type
    
    print("\n" + "="*60)
    print("DENOISING AUTOENCODER TRAINING")
    print("="*60)
    print(f"Noise type: {noise_type}")
    print(f"Noise parameter: {noise_param}")
    print(f"Latent dimension: {latent_dim}")
    print(f"Learning rate: {learning_rate}")
    print("="*60 + "\n")
    
    # Select noise function
    noise_functions = {
        'gaussian': add_gaussian_noise,
        'salt_pepper': add_salt_pepper_noise,
        'masking': add_masking_noise
    }
    noise_fn = noise_functions[noise_type]
    
    # Load data
    print("Loading MNIST dataset...")
    train_loader, test_loader = load_mnist_data(batch_size)
    
    # Initialize denoising model
    model = DenoisingAutoencoder(input_dim, latent_dim).to(device)
    print(f"\nDenoising Autoencoder architecture:")
    print(model)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 40)
        
        train_loss = train_denoising_autoencoder(
            model, train_loader, criterion, optimizer,
            device, epoch, noise_fn, noise_param
        )
        
        print(f"Epoch {epoch} - Average Loss: {train_loss:.6f}")
    
    # Visualizations
    print("\n" + "="*60)
    print("VISUALIZATIONS")
    print("="*60)
    
    print("\n1. Denoising results...")
    visualize_denoising_results(
        model, test_loader, device, noise_fn, noise_param, num_images=10
    )
    
    print("\n2. Comparing different noise types...")
    compare_noise_types(model, test_loader, device)
    
    # Optionally: Compare with standard autoencoder
    # You would need to train or load a standard autoencoder first
    print("\n3. To compare with standard autoencoder, train a standard model first.")
    
    # Save model
    torch.save(model.state_dict(), 'denoising_autoencoder.pth')
    print("\nModel saved to 'denoising_autoencoder.pth'")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()


# =============================================================================
# EXERCISES
# =============================================================================

"""
EXERCISE 1: Noise Level Analysis
---------------------------------
Train denoising autoencoders with different noise levels:
noise_factors = [0.1, 0.2, 0.3, 0.4, 0.5]

For each:
a) Train model and record final loss
b) Visualize denoising results
c) Plot: noise_factor vs. reconstruction quality

Questions:
- How does noise level affect learned representations?
- Is there an optimal noise level for training?
- Can a model trained on high noise denoise low noise?


EXERCISE 2: Noise Type Robustness
----------------------------------
Train three separate models, each on one noise type:
- Model A: Gaussian noise
- Model B: Salt-and-pepper noise
- Model C: Masking noise

Test each model on ALL noise types.

Questions:
- Does training on one noise type generalize to others?
- Which noise type leads to most robust features?
- Can you train on mixed noise types?


EXERCISE 3: Denoising Real-World Applications
----------------------------------------------
Apply denoising autoencoder to:
a) Fashion-MNIST dataset
b) CIFAR-10 dataset (requires CNN architecture)
c) Add blur instead of noise (use Gaussian blur)

Questions:
- How does performance vary across datasets?
- What architectural changes are needed for color images?
- Can the model learn to deblur images?


EXERCISE 4: Feature Analysis
-----------------------------
Compare representations learned by:
a) Standard autoencoder
b) Denoising autoencoder

Analyze:
- Visualize filters/weights of first layer
- Compute correlation between learned features
- Use latent representations for classification task

Questions:
- Are denoising features more robust?
- Do denoising features transfer better to downstream tasks?
- How do latent spaces differ?


EXERCISE 5: Progressive Denoising
----------------------------------
Implement progressive denoising:
a) Start with heavily corrupted image
b) Pass through decoder multiple times
c) At each step, add small noise and re-denoise

Questions:
- Does iterative denoising improve results?
- How many iterations are optimal?
- Compare with single-pass denoising


ADVANCED CHALLENGE: Blind Denoising
------------------------------------
Train a model that can handle unknown noise types:
1. Create dataset with mixed noise types
2. Train single model on all noise types
3. Test on unseen noise combinations

Can you build a universal denoiser?
"""
