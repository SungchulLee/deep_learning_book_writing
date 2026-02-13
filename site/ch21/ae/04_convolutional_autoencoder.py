"""
Module 40.4: Convolutional Autoencoder

This script introduces convolutional autoencoders (Conv-AE), which use 
convolutional layers instead of fully connected layers to better preserve
spatial structure in image data.

Key Concepts:
- Convolutional encoder with max pooling
- Transposed convolutions (deconvolutions) for upsampling
- Preserving spatial structure
- Parameter efficiency compared to fully connected
- Application to image reconstruction and compression

Mathematical Foundation:
-----------------------
Standard AE: x ∈ ℝ^d → z ∈ ℝ^k → x̂ ∈ ℝ^d

Convolutional AE: X ∈ ℝ^(H×W×C) → Z ∈ ℝ^(h×w×c) → X̂ ∈ ℝ^(H×W×C)

Where spatial dimensions are progressively reduced/increased
through convolution + pooling / transposed convolution.

Encoder Operations:
- Convolution: Extracts spatial features
  Output size: ⌊(input_size + 2*padding - kernel_size)/stride + 1⌋
  
- Max Pooling: Reduces spatial dimensions
  Output size: ⌊input_size / pool_size⌋

Decoder Operations:
- Transposed Convolution (ConvTranspose2d): Upsamples
  Output size: (input_size - 1)*stride - 2*padding + kernel_size + output_padding

Advantages over Fully Connected:
1. Preserves spatial structure
2. Fewer parameters (weight sharing)
3. Translation invariance
4. Better for image data

Time: 55 minutes
Level: Intermediate
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple


# =============================================================================
# PART 1: CONVOLUTIONAL AUTOENCODER ARCHITECTURE
# =============================================================================

class ConvAutoencoder(nn.Module):
    """
    Convolutional Autoencoder for MNIST (28×28 images).
    
    Architecture:
    Encoder:
        28×28×1 → Conv(16) → 28×28×16 → MaxPool → 14×14×16
        14×14×16 → Conv(32) → 14×14×32 → MaxPool → 7×7×32
        7×7×32 → Conv(64) → 7×7×64 → MaxPool → 3×3×64 (latent: 3×3×64 = 576)
    
    Decoder:
        3×3×64 → ConvTranspose(64) → 7×7×64
        7×7×64 → ConvTranspose(32) → 14×14×32
        14×14×32 → ConvTranspose(16) → 28×28×16
        28×28×16 → Conv(1) → 28×28×1
    
    Key design choices:
    - Progressively increase channels while reducing spatial dimensions
    - Use ReLU activations in hidden layers
    - Use Sigmoid in output for [0, 1] range
    - Batch normalization for stable training
    """
    
    def __init__(self, latent_channels: int = 64):
        super(ConvAutoencoder, self).__init__()
        
        self.latent_channels = latent_channels
        
        # ENCODER
        # Input: 1×28×28
        self.encoder = nn.Sequential(
            # Layer 1: 1×28×28 → 16×28×28
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            # MaxPool: 16×28×28 → 16×14×14
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Layer 2: 16×14×14 → 32×14×14
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            # MaxPool: 32×14×14 → 32×7×7
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Layer 3: 32×7×7 → 64×7×7
            nn.Conv2d(32, latent_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(latent_channels),
            # MaxPool: 64×7×7 → 64×3×3 (latent representation)
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # After encoder: 64×3×3 = 576-dimensional latent space
        
        # DECODER
        # Input: 64×3×3
        self.decoder = nn.Sequential(
            # Layer 1: Upsample 64×3×3 → 64×7×7
            nn.ConvTranspose2d(latent_channels, 64, kernel_size=3, 
                             stride=2, padding=0, output_padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            
            # Layer 2: Upsample 64×7×7 → 32×14×14
            nn.ConvTranspose2d(64, 32, kernel_size=3,
                             stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            
            # Layer 3: Upsample 32×14×14 → 16×28×28
            nn.ConvTranspose2d(32, 16, kernel_size=3,
                             stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            
            # Output layer: 16×28×28 → 1×28×28
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input image to latent representation.
        
        Parameters:
        -----------
        x : torch.Tensor, shape (batch_size, 1, 28, 28)
            Input images
            
        Returns:
        --------
        z : torch.Tensor, shape (batch_size, latent_channels, 3, 3)
            Latent representation
        """
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to reconstructed image.
        
        Parameters:
        -----------
        z : torch.Tensor, shape (batch_size, latent_channels, 3, 3)
            Latent representation
            
        Returns:
        --------
        x_reconstructed : torch.Tensor, shape (batch_size, 1, 28, 28)
            Reconstructed images
        """
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode then decode.
        
        Parameters:
        -----------
        x : torch.Tensor, shape (batch_size, 1, 28, 28)
            Input images
            
        Returns:
        --------
        reconstructed : torch.Tensor, shape (batch_size, 1, 28, 28)
            Reconstructed images
        latent : torch.Tensor, shape (batch_size, latent_channels, 3, 3)
            Latent representation
        """
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed, latent


# =============================================================================
# PART 2: DEEPER CONVOLUTIONAL AUTOENCODER
# =============================================================================

class DeepConvAutoencoder(nn.Module):
    """
    Deeper convolutional autoencoder with more layers and skip connections.
    
    This demonstrates:
    - Deeper architecture for learning hierarchical features
    - More aggressive compression
    - Similar to U-Net style architectures
    """
    
    def __init__(self):
        super(DeepConvAutoencoder, self).__init__()
        
        # Encoder: Progressive downsampling
        # 28×28×1 → 14×14×32 → 7×7×64 → 3×3×128 → 1×1×256
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2)  # → 14×14
        )
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2)  # → 7×7
        )
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2)  # → 3×3
        )
        
        self.enc4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=0),  # → 1×1 (bottleneck)
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )
        
        # Decoder: Progressive upsampling
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, padding=0),  # → 3×3
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=0),  # → 7×7
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # → 14×14
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )
        
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # → 28×28
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through deep conv autoencoder."""
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        latent = self.enc4(e3)
        
        # Decoder
        d4 = self.dec4(latent)
        d3 = self.dec3(d4)
        d2 = self.dec2(d3)
        reconstructed = self.dec1(d2)
        
        return reconstructed, latent


# =============================================================================
# PART 3: TRAINING FUNCTIONS
# =============================================================================

def train_conv_autoencoder(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
) -> float:
    """
    Train convolutional autoencoder for one epoch.
    
    Note: Unlike fully connected AE, we don't flatten the images.
    Images remain in (batch, channels, height, width) format.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (images, _) in enumerate(train_loader):
        # Images are already in correct shape: (batch, 1, 28, 28)
        images = images.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        reconstructed, _ = model(images)
        
        # Compute loss
        loss = criterion(reconstructed, images)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, '
                  f'Loss: {loss.item():.6f}')
    
    return total_loss / num_batches


def evaluate_conv_autoencoder(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """Evaluate convolutional autoencoder on test set."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            reconstructed, _ = model(images)
            loss = criterion(reconstructed, images)
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


# =============================================================================
# PART 4: VISUALIZATION FUNCTIONS
# =============================================================================

def visualize_conv_reconstructions(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    num_images: int = 10
):
    """Visualize original and reconstructed images from conv autoencoder."""
    model.eval()
    
    images, labels = next(iter(test_loader))
    images = images[:num_images].to(device)
    labels = labels[:num_images]
    
    with torch.no_grad():
        reconstructed, _ = model(images)
    
    # Move to CPU
    images_np = images.cpu().numpy()
    reconstructed_np = reconstructed.cpu().numpy()
    
    # Visualize
    fig, axes = plt.subplots(2, num_images, figsize=(15, 3))
    
    for i in range(num_images):
        # Original
        axes[0, i].imshow(images_np[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontsize=12)
        axes[0, i].text(0.5, -0.1, f'{labels[i].item()}',
                       transform=axes[0, i].transAxes, ha='center')
        
        # Reconstructed
        axes[1, i].imshow(reconstructed_np[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('conv_reconstructions.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved to 'conv_reconstructions.png'")


def visualize_feature_maps(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device
):
    """
    Visualize feature maps from encoder layers.
    
    This shows what patterns different conv filters detect.
    """
    model.eval()
    
    # Get one image
    images, _ = next(iter(test_loader))
    image = images[0:1].to(device)
    
    # Extract feature maps from each encoder layer
    feature_maps = []
    x = image
    
    # For ConvAutoencoder, manually extract intermediate features
    if isinstance(model, ConvAutoencoder):
        with torch.no_grad():
            for i, layer in enumerate(model.encoder):
                x = layer(x)
                # Save after each Conv2d layer (before pooling)
                if isinstance(layer, nn.Conv2d):
                    feature_maps.append(x.cpu().numpy())
    
    # Visualize feature maps
    num_layers = len(feature_maps)
    fig, axes = plt.subplots(num_layers, 8, figsize=(16, 2 * num_layers))
    
    for layer_idx, fmap in enumerate(feature_maps):
        # fmap shape: (1, channels, height, width)
        num_channels = min(8, fmap.shape[1])
        
        for channel_idx in range(num_channels):
            ax = axes[layer_idx, channel_idx] if num_layers > 1 else axes[channel_idx]
            ax.imshow(fmap[0, channel_idx], cmap='viridis')
            ax.axis('off')
            
            if channel_idx == 0:
                ax.text(-0.1, 0.5, f'Layer {layer_idx + 1}',
                       transform=ax.transAxes, rotation=90,
                       va='center', fontsize=10)
    
    plt.suptitle('Encoder Feature Maps', fontsize=14)
    plt.tight_layout()
    plt.savefig('feature_maps.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved to 'feature_maps.png'")


def compare_fc_vs_conv(
    fc_model: nn.Module,
    conv_model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    num_images: int = 5
):
    """
    Compare fully connected vs convolutional autoencoder reconstructions.
    
    This demonstrates that conv AE preserves spatial structure better.
    """
    fc_model.eval()
    conv_model.eval()
    
    images, labels = next(iter(test_loader))
    images_orig = images[:num_images]
    labels = labels[:num_images]
    
    # For FC model, need to flatten
    images_flat = images_orig.view(images_orig.size(0), -1).to(device)
    images_conv = images_orig.to(device)
    
    with torch.no_grad():
        # FC reconstruction
        fc_recon, _ = fc_model(images_flat)
        fc_recon = fc_recon.view(-1, 1, 28, 28)
        
        # Conv reconstruction
        conv_recon, _ = conv_model(images_conv)
    
    # Move to CPU
    images_np = images_orig.numpy()
    fc_np = fc_recon.cpu().numpy()
    conv_np = conv_recon.cpu().numpy()
    
    # Visualize
    fig, axes = plt.subplots(3, num_images, figsize=(12, 7))
    
    for i in range(num_images):
        # Original
        axes[0, i].imshow(images_np[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontsize=11)
        
        # FC reconstruction
        fc_mse = np.mean((images_np[i, 0] - fc_np[i, 0]) ** 2)
        axes[1, i].imshow(fc_np[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('FC Autoencoder', fontsize=11)
        axes[1, i].text(0.5, -0.1, f'MSE: {fc_mse:.4f}',
                       transform=axes[1, i].transAxes, ha='center', fontsize=9)
        
        # Conv reconstruction
        conv_mse = np.mean((images_np[i, 0] - conv_np[i, 0]) ** 2)
        axes[2, i].imshow(conv_np[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_title('Conv Autoencoder', fontsize=11)
        axes[2, i].text(0.5, -0.1, f'MSE: {conv_mse:.4f}',
                       transform=axes[2, i].transAxes, ha='center', fontsize=9)
    
    plt.suptitle('FC vs Conv Autoencoder Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig('fc_vs_conv_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved to 'fc_vs_conv_comparison.png'")


# =============================================================================
# PART 5: PARAMETER COMPARISON
# =============================================================================

def count_parameters(model: nn.Module) -> int:
    """Count total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compare_model_sizes():
    """
    Compare parameter counts between FC and Conv autoencoders.
    
    This demonstrates the parameter efficiency of convolutional layers.
    """
    from torch.nn import Linear, Conv2d
    
    # Simple FC autoencoder
    fc_encoder = nn.Sequential(
        Linear(784, 256), nn.ReLU(),
        Linear(256, 128), nn.ReLU(),
        Linear(128, 64), nn.ReLU()
    )
    fc_decoder = nn.Sequential(
        Linear(64, 128), nn.ReLU(),
        Linear(128, 256), nn.ReLU(),
        Linear(256, 784), nn.Sigmoid()
    )
    
    # Conv autoencoder (from our ConvAutoencoder class)
    conv_ae = ConvAutoencoder()
    
    # Count parameters
    fc_params = count_parameters(fc_encoder) + count_parameters(fc_decoder)
    conv_params = count_parameters(conv_ae)
    
    print("\n" + "="*60)
    print("PARAMETER COMPARISON")
    print("="*60)
    print(f"Fully Connected AE:   {fc_params:,} parameters")
    print(f"Convolutional AE:     {conv_params:,} parameters")
    print(f"Reduction factor:     {fc_params / conv_params:.2f}x fewer params")
    print("="*60)


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
    """Main function to train convolutional autoencoder."""
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Hyperparameters
    batch_size = 128
    learning_rate = 0.001
    num_epochs = 10
    
    print("\n" + "="*60)
    print("CONVOLUTIONAL AUTOENCODER TRAINING")
    print("="*60)
    
    # Compare model sizes
    compare_model_sizes()
    
    # Load data
    print("\nLoading MNIST dataset...")
    train_loader, test_loader = load_mnist_data(batch_size)
    
    # Initialize model
    model = ConvAutoencoder(latent_channels=64).to(device)
    print(f"\nModel architecture:")
    print(model)
    print(f"\nTotal parameters: {count_parameters(model):,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    
    train_losses = []
    test_losses = []
    
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 40)
        
        train_loss = train_conv_autoencoder(
            model, train_loader, criterion, optimizer, device, epoch
        )
        test_loss = evaluate_conv_autoencoder(
            model, test_loader, criterion, device
        )
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        print(f"Epoch {epoch} - Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")
    
    # Visualizations
    print("\n" + "="*60)
    print("VISUALIZATIONS")
    print("="*60)
    
    print("\n1. Reconstructions...")
    visualize_conv_reconstructions(model, test_loader, device)
    
    print("\n2. Feature maps...")
    visualize_feature_maps(model, test_loader, device)
    
    # Save model
    torch.save(model.state_dict(), 'conv_autoencoder.pth')
    print("\nModel saved to 'conv_autoencoder.pth'")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()


# =============================================================================
# EXERCISES
# =============================================================================

"""
EXERCISE 1: Architecture Exploration
-------------------------------------
Modify the convolutional autoencoder:
a) Different number of channels: [8, 16, 32, 64, 128]
b) Different numbers of layers (2, 3, 4, 5)
c) Different kernel sizes (3×3, 5×5, 7×7)

Questions:
- How do these affect reconstruction quality?
- What's the trade-off between parameters and performance?
- Can you design a more efficient architecture?


EXERCISE 2: Stride vs Pooling
------------------------------
Compare two downsampling strategies:
a) Max pooling after convolution
b) Strided convolutions (stride=2)

Questions:
- Which gives better reconstructions?
- How do parameter counts compare?
- What about training speed?


EXERCISE 3: Color Images (CIFAR-10)
------------------------------------
Adapt the conv autoencoder for CIFAR-10 (32×32×3):
a) Modify input/output channels
b) Adjust architecture for larger images
c) Train and evaluate

Questions:
- How does performance compare to MNIST?
- What architectural changes improve results?
- How many parameters are needed?


EXERCISE 4: Skip Connections
-----------------------------
Implement skip connections (like U-Net):
a) Concatenate encoder features to decoder
b) Compare with standard autoencoder
c) Measure reconstruction quality

Questions:
- Do skip connections improve reconstruction?
- How do they affect training dynamics?
- What about parameter count?


EXERCISE 5: Compression Ratio Analysis
---------------------------------------
Train models with different compression ratios:
compression_ratios = [2, 4, 8, 16, 32, 64]

For each:
a) Design architecture with target compression
b) Measure reconstruction quality
c) Plot: compression vs. quality

Questions:
- What's the maximum useful compression?
- How does this compare to JPEG compression?
"""
