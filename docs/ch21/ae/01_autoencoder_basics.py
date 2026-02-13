"""
Module 40.1: Basic Autoencoder Implementation

This script introduces the fundamental concepts of autoencoders through a simple
implementation on the MNIST dataset. Students will learn:
- Basic autoencoder architecture
- Encoder-decoder structure
- Training process for reconstruction
- Visualization of learned representations
- Latent space exploration

Mathematical Foundation:
-----------------------
An autoencoder learns two functions:
- Encoder: f_θ: X → Z where Z is latent space
- Decoder: g_φ: Z → X̂ where X̂ is reconstruction

Objective: Minimize reconstruction loss
L(θ, φ) = (1/n) Σᵢ ||xᵢ - g_φ(f_θ(xᵢ))||²

For MNIST:
- Input: x ∈ ℝ^784 (28×28 flattened)
- Latent: z ∈ ℝ^64
- Output: x̂ ∈ ℝ^784

Architecture:
------------
Encoder: 784 → 256 → 128 → 64
Decoder: 64 → 128 → 256 → 784

Time: 45 minutes
Level: Beginner
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, List


# =============================================================================
# PART 1: AUTOENCODER ARCHITECTURE
# =============================================================================

class SimpleAutoencoder(nn.Module):
    """
    A simple fully-connected autoencoder for MNIST images.
    
    Architecture:
    - Encoder: Progressively reduces dimensionality
    - Bottleneck: Compressed latent representation
    - Decoder: Mirrors encoder to reconstruct input
    
    Parameters:
    -----------
    input_dim : int
        Dimension of input (784 for flattened 28×28 images)
    latent_dim : int
        Dimension of latent space (compression bottleneck)
    """
    
    def __init__(self, input_dim: int = 784, latent_dim: int = 64):
        super(SimpleAutoencoder, self).__init__()
        
        # Store dimensions for reference
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder: Compresses input to latent representation
        # Each layer reduces dimensionality by approximately 2x
        self.encoder = nn.Sequential(
            # First encoding layer: 784 → 256
            nn.Linear(input_dim, 256),
            nn.ReLU(),  # Non-linearity allows learning complex patterns
            
            # Second encoding layer: 256 → 128
            nn.Linear(256, 128),
            nn.ReLU(),
            
            # Bottleneck layer: 128 → latent_dim (64)
            # This is the compressed representation
            nn.Linear(128, latent_dim),
            nn.ReLU()
        )
        
        # Decoder: Reconstructs input from latent representation
        # Mirrors encoder architecture in reverse
        self.decoder = nn.Sequential(
            # First decoding layer: latent_dim (64) → 128
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            
            # Second decoding layer: 128 → 256
            nn.Linear(128, 256),
            nn.ReLU(),
            
            # Output layer: 256 → 784
            # Sigmoid ensures output is in [0, 1] range (like normalized images)
            nn.Linear(256, input_dim),
            nn.Sigmoid()  # Maps to [0, 1] for pixel values
        )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input into latent representation.
        
        Parameters:
        -----------
        x : torch.Tensor, shape (batch_size, input_dim)
            Input data
            
        Returns:
        --------
        z : torch.Tensor, shape (batch_size, latent_dim)
            Latent representation
        """
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to reconstruct input.
        
        Parameters:
        -----------
        z : torch.Tensor, shape (batch_size, latent_dim)
            Latent representation
            
        Returns:
        --------
        x_reconstructed : torch.Tensor, shape (batch_size, input_dim)
            Reconstructed input
        """
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass through autoencoder.
        
        Parameters:
        -----------
        x : torch.Tensor, shape (batch_size, input_dim)
            Input data
            
        Returns:
        --------
        x_reconstructed : torch.Tensor, shape (batch_size, input_dim)
            Reconstructed input
        z : torch.Tensor, shape (batch_size, latent_dim)
            Latent representation (useful for visualization)
        """
        # Encode input to latent space
        z = self.encode(x)
        
        # Decode latent representation to reconstruct input
        x_reconstructed = self.decode(z)
        
        return x_reconstructed, z


# =============================================================================
# PART 2: DATA LOADING AND PREPROCESSING
# =============================================================================

def load_mnist_data(batch_size: int = 128) -> Tuple[DataLoader, DataLoader]:
    """
    Load and preprocess MNIST dataset.
    
    Preprocessing steps:
    1. Convert images to tensors
    2. Normalize to [0, 1] range (ToTensor does this automatically)
    3. Flatten 28×28 images to 784-dimensional vectors
    
    Parameters:
    -----------
    batch_size : int
        Number of samples per batch
        
    Returns:
    --------
    train_loader : DataLoader
        Training data loader
    test_loader : DataLoader
        Test data loader
    """
    # Define transformation pipeline
    # ToTensor: Converts PIL Image to tensor and scales to [0, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to tensor and normalizes to [0, 1]
    ])
    
    # Download and load training data
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    # Download and load test data
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    # Create data loaders for batching
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data for better generalization
        num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle test data
        num_workers=2
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Batch size: {batch_size}")
    
    return train_loader, test_loader


# =============================================================================
# PART 3: TRAINING FUNCTION
# =============================================================================

def train_autoencoder(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
) -> float:
    """
    Train autoencoder for one epoch.
    
    Training Process:
    1. Forward pass: Encode and decode input
    2. Compute reconstruction loss
    3. Backward pass: Compute gradients
    4. Update weights using optimizer
    
    Parameters:
    -----------
    model : nn.Module
        Autoencoder model
    train_loader : DataLoader
        Training data loader
    criterion : nn.Module
        Loss function (typically MSE)
    optimizer : optim.Optimizer
        Optimizer for updating weights
    device : torch.device
        Device to train on (CPU or CUDA)
    epoch : int
        Current epoch number (for logging)
        
    Returns:
    --------
    avg_loss : float
        Average loss over all batches
    """
    model.train()  # Set model to training mode
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (images, _) in enumerate(train_loader):
        # MNIST images shape: (batch_size, 1, 28, 28)
        # We need to flatten to (batch_size, 784)
        images = images.view(images.size(0), -1).to(device)
        
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        
        # Forward pass: Get reconstruction and latent representation
        reconstructed, latent = model(images)
        
        # Compute reconstruction loss
        # MSE between original and reconstructed images
        loss = criterion(reconstructed, images)
        
        # Backward pass: Compute gradients
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Accumulate loss for averaging
        total_loss += loss.item()
        num_batches += 1
        
        # Print progress every 100 batches
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, '
                  f'Loss: {loss.item():.6f}')
    
    # Return average loss over all batches
    avg_loss = total_loss / num_batches
    return avg_loss


# =============================================================================
# PART 4: EVALUATION FUNCTION
# =============================================================================

def evaluate_autoencoder(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """
    Evaluate autoencoder on test set.
    
    Parameters:
    -----------
    model : nn.Module
        Trained autoencoder model
    test_loader : DataLoader
        Test data loader
    criterion : nn.Module
        Loss function
    device : torch.device
        Device to evaluate on
        
    Returns:
    --------
    avg_loss : float
        Average reconstruction loss on test set
    """
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    num_batches = 0
    
    # Disable gradient computation for evaluation
    with torch.no_grad():
        for images, _ in test_loader:
            # Flatten images
            images = images.view(images.size(0), -1).to(device)
            
            # Forward pass
            reconstructed, _ = model(images)
            
            # Compute loss
            loss = criterion(reconstructed, images)
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    return avg_loss


# =============================================================================
# PART 5: VISUALIZATION FUNCTIONS
# =============================================================================

def visualize_reconstructions(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    num_images: int = 10
):
    """
    Visualize original images and their reconstructions.
    
    This helps assess how well the autoencoder has learned
    to reconstruct the input data.
    
    Parameters:
    -----------
    model : nn.Module
        Trained autoencoder model
    test_loader : DataLoader
        Test data loader
    device : torch.device
        Device to run inference on
    num_images : int
        Number of images to visualize
    """
    model.eval()
    
    # Get one batch of test images
    images, labels = next(iter(test_loader))
    images = images[:num_images]
    labels = labels[:num_images]
    
    # Flatten and move to device
    images_flat = images.view(images.size(0), -1).to(device)
    
    # Get reconstructions
    with torch.no_grad():
        reconstructed, _ = model(images_flat)
    
    # Move back to CPU and reshape
    images_np = images.cpu().numpy()
    reconstructed_np = reconstructed.cpu().numpy().reshape(-1, 28, 28)
    
    # Create visualization
    fig, axes = plt.subplots(2, num_images, figsize=(15, 3))
    
    for i in range(num_images):
        # Plot original image
        axes[0, i].imshow(images_np[i].squeeze(), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontsize=12, pad=10)
        axes[0, i].text(0.5, -0.1, f'{labels[i].item()}', 
                       transform=axes[0, i].transAxes,
                       ha='center', fontsize=10)
        
        # Plot reconstructed image
        axes[1, i].imshow(reconstructed_np[i], cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed', fontsize=12, pad=10)
    
    plt.tight_layout()
    plt.savefig('autoencoder_reconstructions.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved visualization to 'autoencoder_reconstructions.png'")


def visualize_latent_space_2d(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    num_samples: int = 5000
):
    """
    Visualize 2D latent space (only works if latent_dim == 2).
    
    For higher dimensional latent spaces, we would need
    dimensionality reduction (PCA, t-SNE) which is covered
    in the advanced module.
    
    Parameters:
    -----------
    model : nn.Module
        Trained autoencoder model
    test_loader : DataLoader
        Test data loader
    device : torch.device
        Device to run inference on
    num_samples : int
        Number of samples to visualize
    """
    if model.latent_dim != 2:
        print(f"Latent dimension is {model.latent_dim}, not 2. Skipping 2D visualization.")
        print("Train a model with latent_dim=2 to use this visualization.")
        return
    
    model.eval()
    
    latent_vectors = []
    labels_list = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            if len(latent_vectors) * test_loader.batch_size >= num_samples:
                break
            
            # Flatten and encode
            images_flat = images.view(images.size(0), -1).to(device)
            latent = model.encode(images_flat)
            
            latent_vectors.append(latent.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
    
    # Concatenate all batches
    latent_vectors = np.concatenate(latent_vectors, axis=0)[:num_samples]
    labels_array = np.concatenate(labels_list, axis=0)[:num_samples]
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        latent_vectors[:, 0],
        latent_vectors[:, 1],
        c=labels_array,
        cmap='tab10',
        alpha=0.6,
        s=5
    )
    plt.colorbar(scatter, label='Digit Class')
    plt.xlabel('Latent Dimension 1', fontsize=12)
    plt.ylabel('Latent Dimension 2', fontsize=12)
    plt.title('2D Latent Space Visualization', fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('latent_space_2d.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved visualization to 'latent_space_2d.png'")


def interpolate_in_latent_space(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    num_steps: int = 10
):
    """
    Interpolate between two images in latent space.
    
    This demonstrates that the latent space is continuous
    and that interpolation produces meaningful intermediate images.
    
    Process:
    1. Encode two images to get z1 and z2
    2. Linearly interpolate: z_t = (1-t)z1 + t*z2 for t ∈ [0, 1]
    3. Decode each interpolated point
    
    Parameters:
    -----------
    model : nn.Module
        Trained autoencoder model
    test_loader : DataLoader
        Test data loader
    device : torch.device
        Device to run inference on
    num_steps : int
        Number of interpolation steps
    """
    model.eval()
    
    # Get two random images
    images, labels = next(iter(test_loader))
    img1 = images[0:1].view(1, -1).to(device)
    img2 = images[1:2].view(1, -1).to(device)
    label1 = labels[0].item()
    label2 = labels[1].item()
    
    with torch.no_grad():
        # Encode both images
        z1 = model.encode(img1)
        z2 = model.encode(img2)
        
        # Create interpolation steps
        # t ranges from 0 to 1
        interpolated_images = []
        
        for t in np.linspace(0, 1, num_steps):
            # Linear interpolation in latent space
            z_interpolated = (1 - t) * z1 + t * z2
            
            # Decode interpolated latent vector
            img_interpolated = model.decode(z_interpolated)
            interpolated_images.append(img_interpolated.cpu().numpy())
    
    # Visualize interpolation
    fig, axes = plt.subplots(1, num_steps, figsize=(15, 2))
    
    for i in range(num_steps):
        img = interpolated_images[i].reshape(28, 28)
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
        if i == 0:
            axes[i].set_title(f'Start\n(digit {label1})', fontsize=10)
        elif i == num_steps - 1:
            axes[i].set_title(f'End\n(digit {label2})', fontsize=10)
    
    plt.suptitle('Interpolation in Latent Space', fontsize=14, y=1.05)
    plt.tight_layout()
    plt.savefig('latent_interpolation.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved visualization to 'latent_interpolation.png'")


# =============================================================================
# PART 6: MAIN EXECUTION
# =============================================================================

def main():
    """
    Main function to train and evaluate the autoencoder.
    """
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Hyperparameters
    input_dim = 784  # 28 * 28
    latent_dim = 64  # Compression to 64 dimensions
    batch_size = 128
    learning_rate = 0.001
    num_epochs = 10
    
    print("\n" + "="*60)
    print("AUTOENCODER TRAINING CONFIGURATION")
    print("="*60)
    print(f"Input dimension: {input_dim}")
    print(f"Latent dimension: {latent_dim}")
    print(f"Compression ratio: {input_dim / latent_dim:.2f}x")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    print(f"Number of epochs: {num_epochs}")
    print("="*60 + "\n")
    
    # Load data
    print("Loading MNIST dataset...")
    train_loader, test_loader = load_mnist_data(batch_size)
    
    # Initialize model
    model = SimpleAutoencoder(input_dim, latent_dim).to(device)
    print(f"\nModel architecture:")
    print(model)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {num_params:,}")
    
    # Loss function and optimizer
    # MSE Loss: L = (1/n) Σ ||x - x̂||²
    criterion = nn.MSELoss()
    
    # Adam optimizer with default parameters
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    
    train_losses = []
    test_losses = []
    
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 40)
        
        # Train for one epoch
        train_loss = train_autoencoder(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Evaluate on test set
        test_loss = evaluate_autoencoder(
            model, test_loader, criterion, device
        )
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        print(f"Epoch {epoch} - Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss', marker='s')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.title('Training and Test Loss Over Time', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nSaved training curves to 'training_curves.png'")
    
    # Visualizations
    print("\n" + "="*60)
    print("VISUALIZATIONS")
    print("="*60)
    
    print("\n1. Visualizing reconstructions...")
    visualize_reconstructions(model, test_loader, device, num_images=10)
    
    print("\n2. Visualizing latent space interpolation...")
    interpolate_in_latent_space(model, test_loader, device, num_steps=10)
    
    # Note: 2D latent space visualization only works with latent_dim=2
    # Uncomment below and retrain with latent_dim=2 to see this visualization
    # print("\n3. Visualizing 2D latent space...")
    # visualize_latent_space_2d(model, test_loader, device)
    
    # Save model
    torch.save(model.state_dict(), 'simple_autoencoder.pth')
    print("\nModel saved to 'simple_autoencoder.pth'")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()


# =============================================================================
# EXERCISES FOR STUDENTS
# =============================================================================

"""
EXERCISE 1: Architecture Exploration
-------------------------------------
Modify the autoencoder architecture:
a) Change latent_dim to 2, 32, 128, and compare reconstruction quality
b) Add more layers (e.g., 784→512→256→128→64)
c) Try different activation functions (LeakyReLU, ELU)

Questions:
- How does compression ratio affect reconstruction quality?
- What happens with latent_dim = 2? Can you visualize the 2D space?
- Does adding more layers improve performance?


EXERCISE 2: Loss Function Analysis
-----------------------------------
Try different loss functions:
a) nn.L1Loss() instead of MSELoss
b) nn.BCELoss() (binary cross-entropy)
c) Combined loss: MSE + L1

Questions:
- How does the choice of loss function affect reconstructions?
- Which loss function gives the best results for MNIST?


EXERCISE 3: Latent Space Exploration
-------------------------------------
Implement the following:
a) Train a model with latent_dim=2 and visualize the 2D space
b) Sample random points in the latent space and decode them
c) Perform arithmetic in latent space (z_new = z1 + z2 - z3)

Questions:
- Is the latent space continuous and smooth?
- Do nearby points in latent space correspond to similar images?
- Can you find meaningful directions in latent space?


EXERCISE 4: Capacity vs. Compression
-------------------------------------
Train models with different latent dimensions:
latent_dims = [2, 8, 16, 32, 64, 128, 256, 512]

For each:
a) Record final test loss
b) Visualize sample reconstructions
c) Plot: latent_dim vs. reconstruction error

Questions:
- What is the optimal latent dimension?
- Is there a point of diminishing returns?
- How does training time scale with latent dimension?


EXERCISE 5: Anomaly Detection
------------------------------
Use the trained autoencoder for anomaly detection:
a) Calculate reconstruction error for each test image
b) Plot histogram of reconstruction errors
c) Identify images with highest reconstruction errors

Questions:
- Are certain digit classes harder to reconstruct?
- Can you use reconstruction error for outlier detection?
- How would you set a threshold for anomaly detection?


ANSWERS TO COMMON QUESTIONS:
-----------------------------

Q: Why use Sigmoid activation in the output layer?
A: MNIST images are normalized to [0, 1], so Sigmoid ensures outputs
   are in the same range. For unnormalized data, use linear activation.

Q: Why is MSE a good loss for image reconstruction?
A: MSE measures pixel-wise squared difference, which is natural for
   continuous pixel values. It's differentiable and convex.

Q: What's the difference between encoding and embedding?
A: Encoding is the transformation function. The encoded representation
   (the output of encoding) is called an embedding or latent vector.

Q: Can autoencoders generate new images?
A: Basic autoencoders have limited generation capability. Sampling
   random points in latent space often gives poor results. VAEs
   (Module 41) address this with probabilistic latent spaces.

Q: How does this relate to PCA?
A: Linear autoencoders (no activation functions) learn similar
   subspaces to PCA, but autoencoders with non-linearities can
   capture more complex patterns that PCA cannot.
"""
