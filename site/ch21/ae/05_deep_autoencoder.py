"""
Module 40.5: Deep (Stacked) Autoencoder

This script introduces deep autoencoders with multiple encoding/decoding layers,
demonstrating hierarchical feature learning and bottleneck compression.

Key Concepts:
- Stacking multiple layers for hierarchical representations
- Greedy layer-wise pretraining (historical approach)
- Deep bottleneck architectures
- Hierarchical feature extraction
- Comparison with shallow autoencoders

Mathematical Foundation:
-----------------------
Shallow AE: x → h → z → h' → x̂ (single hidden layer each side)

Deep AE: x → h₁ → h₂ → ... → z → ... → h'₂ → h'₁ → x̂

Where each hᵢ represents progressively more abstract features:
- h₁: Low-level features (edges, textures)
- h₂: Mid-level features (parts, patterns)
- z: High-level abstract representation
- Decoder mirrors encoder in reverse

Benefits of Depth:
1. Hierarchical feature learning
2. More compact representations
3. Better expressiv

ity with fewer neurons per layer
4. Can model complex non-linear mappings

Historical Note:
Before modern optimization techniques, deep autoencoders were trained
using greedy layer-wise pretraining. Modern approaches with ReLU,
batch norm, and Adam typically don't require this.

Time: 50 minutes
Level: Intermediate-Advanced
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
# PART 1: DEEP AUTOENCODER ARCHITECTURE
# =============================================================================

class DeepAutoencoder(nn.Module):
    """
    Deep autoencoder with multiple encoding and decoding layers.
    
    Architecture: 784 → 512 → 256 → 128 → 32 (bottleneck)
                  32 → 128 → 256 → 512 → 784
    
    This creates a narrow bottleneck with aggressive compression.
    """
    
    def __init__(self):
        super(DeepAutoencoder, self).__init__()
        
        # Encoder: Progressive dimensionality reduction
        # Each layer roughly halves the dimension
        self.encoder = nn.Sequential(
            # Layer 1: 784 → 512
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            
            # Layer 2: 512 → 256
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            
            # Layer 3: 256 → 128
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            
            # Bottleneck: 128 → 32
            # This is the most compressed representation
            nn.Linear(128, 32),
            nn.ReLU()
        )
        
        # Decoder: Mirror of encoder
        self.decoder = nn.Sequential(
            # Expand from bottleneck: 32 → 128
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            
            # Layer 3: 128 → 256
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            
            # Layer 2: 256 → 512
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            
            # Output layer: 512 → 784
            nn.Linear(512, 784),
            nn.Sigmoid()  # Ensure output in [0, 1]
        )
        
        self.latent_dim = 32
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to bottleneck representation."""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from bottleneck."""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full forward pass."""
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z
    
    def get_layer_outputs(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Get outputs from all encoder layers for visualization.
        
        Returns list of activations: [h₁, h₂, h₃, z]
        """
        activations = []
        h = x
        
        for layer in self.encoder:
            h = layer(h)
            # Save after ReLU activations (skip BatchNorm and Dropout)
            if isinstance(layer, nn.Linear):
                # Apply ReLU if next layer is ReLU
                pass
            elif isinstance(layer, nn.ReLU):
                activations.append(h.detach())
        
        return activations


# =============================================================================
# PART 2: VERY DEEP AUTOENCODER
# =============================================================================

class VeryDeepAutoencoder(nn.Module):
    """
    Very deep autoencoder with 6+ encoding/decoding layers.
    
    Architecture: 784 → 512 → 384 → 256 → 128 → 64 → 16 (bottleneck)
    
    Demonstrates that with proper regularization (batch norm, dropout),
    very deep architectures can be trained end-to-end.
    """
    
    def __init__(self, use_residual: bool = False):
        super(VeryDeepAutoencoder, self).__init__()
        
        self.use_residual = use_residual
        self.latent_dim = 16
        
        # Encoder
        self.enc1 = self._make_layer(784, 512)
        self.enc2 = self._make_layer(512, 384)
        self.enc3 = self._make_layer(384, 256)
        self.enc4 = self._make_layer(256, 128)
        self.enc5 = self._make_layer(128, 64)
        self.enc6 = nn.Sequential(
            nn.Linear(64, 16),
            nn.ReLU()
        )
        
        # Decoder
        self.dec6 = self._make_layer(16, 64)
        self.dec5 = self._make_layer(64, 128)
        self.dec4 = self._make_layer(128, 256)
        self.dec3 = self._make_layer(256, 384)
        self.dec2 = self._make_layer(384, 512)
        self.dec1 = nn.Sequential(
            nn.Linear(512, 784),
            nn.Sigmoid()
        )
    
    def _make_layer(self, in_dim: int, out_dim: int) -> nn.Sequential:
        """Create a layer block with normalization and dropout."""
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.BatchNorm1d(out_dim),
            nn.Dropout(0.3)  # Higher dropout for very deep networks
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through very deep autoencoder."""
        # Encoder
        h1 = self.enc1(x)
        h2 = self.enc2(h1)
        h3 = self.enc3(h2)
        h4 = self.enc4(h3)
        h5 = self.enc5(h4)
        z = self.enc6(h5)
        
        # Decoder
        d6 = self.dec6(z)
        d5 = self.dec5(d6)
        d4 = self.dec4(d5)
        d3 = self.dec3(d4)
        d2 = self.dec2(d3)
        x_recon = self.dec1(d2)
        
        return x_recon, z


# =============================================================================
# PART 3: LAYER-WISE PRETRAINING (HISTORICAL APPROACH)
# =============================================================================

class StackedAutoencoder:
    """
    Greedy layer-wise pretraining for deep autoencoders.
    
    Historical approach (2006-2012) before modern optimization.
    Train each layer as a separate autoencoder, then stack them.
    
    Process:
    1. Train first autoencoder: x → h₁ → x
    2. Fix encoder₁, train second autoencoder: h₁ → h₂ → h₁
    3. Continue for all layers
    4. Stack all encoders, fine-tune end-to-end
    """
    
    def __init__(self, layer_dims: List[int]):
        """
        Initialize stacked autoencoder.
        
        Parameters:
        -----------
        layer_dims : List[int]
            Dimensions for each layer, e.g., [784, 512, 256, 128, 32]
        """
        self.layer_dims = layer_dims
        self.autoencoders = []
        self.encoders = []
        
        # Create autoencoder for each layer pair
        for i in range(len(layer_dims) - 1):
            in_dim = layer_dims[i]
            out_dim = layer_dims[i + 1]
            
            # Simple single-layer autoencoder
            ae = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, in_dim),
                nn.Sigmoid() if i == 0 else nn.ReLU()
            )
            self.autoencoders.append(ae)
    
    def pretrain_layer(
        self,
        layer_idx: int,
        data_loader: DataLoader,
        device: torch.device,
        epochs: int = 5
    ):
        """
        Pretrain a single layer.
        
        Parameters:
        -----------
        layer_idx : int
            Index of layer to pretrain (0-indexed)
        data_loader : DataLoader
            Training data
        device : torch.device
            Device for training
        epochs : int
            Number of pretraining epochs
        """
        ae = self.autoencoders[layer_idx].to(device)
        optimizer = optim.Adam(ae.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        print(f"\nPretraining layer {layer_idx + 1}...")
        
        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0
            
            for images, _ in data_loader:
                images = images.view(images.size(0), -1).to(device)
                
                # If not first layer, encode through previous layers
                if layer_idx > 0:
                    with torch.no_grad():
                        for prev_ae in self.autoencoders[:layer_idx]:
                            # Extract encoder part (first 2 layers)
                            encoder = nn.Sequential(*list(prev_ae.children())[:2])
                            images = encoder(images)
                
                # Train current autoencoder
                optimizer.zero_grad()
                reconstructed = ae(images)
                loss = criterion(reconstructed, images)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            print(f"  Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")
        
        # Extract and save encoder
        encoder = nn.Sequential(*list(ae.children())[:2])
        self.encoders.append(encoder)


# =============================================================================
# PART 4: VISUALIZATION OF HIERARCHICAL FEATURES
# =============================================================================

def visualize_hierarchical_features(
    model: DeepAutoencoder,
    test_loader: DataLoader,
    device: torch.device
):
    """
    Visualize activations at different layers of deep autoencoder.
    
    Shows how representations become more abstract in deeper layers.
    """
    model.eval()
    
    # Get one test image
    images, labels = next(iter(test_loader))
    image = images[0:1].view(1, -1).to(device)
    label = labels[0].item()
    
    # Get layer outputs
    with torch.no_grad():
        layer_outputs = model.get_layer_outputs(image)
        reconstructed, _ = model(image)
    
    # Visualize
    num_layers = len(layer_outputs) + 2  # +2 for original and reconstructed
    fig, axes = plt.subplots(1, num_layers, figsize=(3 * num_layers, 3))
    
    # Original image
    axes[0].imshow(image.cpu().reshape(28, 28), cmap='gray')
    axes[0].set_title(f'Input\n(digit {label})', fontsize=10)
    axes[0].axis('off')
    
    # Layer activations as 1D visualizations (since they're vectors)
    for i, activation in enumerate(layer_outputs):
        # Reshape activation to approximate square for visualization
        act_np = activation.cpu().numpy().flatten()
        size = int(np.ceil(np.sqrt(len(act_np))))
        padded = np.zeros(size * size)
        padded[:len(act_np)] = act_np
        act_2d = padded.reshape(size, size)
        
        axes[i + 1].imshow(act_2d, cmap='viridis', aspect='auto')
        axes[i + 1].set_title(f'Layer {i + 1}\n({len(act_np)} dim)', fontsize=10)
        axes[i + 1].axis('off')
    
    # Reconstructed image
    axes[-1].imshow(reconstructed.cpu().reshape(28, 28), cmap='gray')
    axes[-1].set_title('Reconstructed', fontsize=10)
    axes[-1].axis('off')
    
    plt.suptitle('Hierarchical Feature Representations', fontsize=14)
    plt.tight_layout()
    plt.savefig('hierarchical_features.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved to 'hierarchical_features.png'")


def compare_depths(
    shallow_model: nn.Module,
    deep_model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    num_images: int = 5
):
    """
    Compare reconstructions from shallow vs deep autoencoders.
    
    Demonstrates that deeper models can achieve better reconstructions
    with similar or fewer total parameters.
    """
    shallow_model.eval()
    deep_model.eval()
    
    images, labels = next(iter(test_loader))
    images = images[:num_images]
    labels = labels[:num_images]
    images_flat = images.view(images.size(0), -1).to(device)
    
    with torch.no_grad():
        shallow_recon, _ = shallow_model(images_flat)
        deep_recon, _ = deep_model(images_flat)
    
    # Convert to numpy
    images_np = images.numpy()
    shallow_np = shallow_recon.cpu().numpy().reshape(-1, 28, 28)
    deep_np = deep_recon.cpu().numpy().reshape(-1, 28, 28)
    
    # Visualize
    fig, axes = plt.subplots(3, num_images, figsize=(12, 7))
    
    for i in range(num_images):
        # Original
        axes[0, i].imshow(images_np[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontsize=11)
        
        # Shallow reconstruction
        shallow_mse = np.mean((images_np[i, 0] - shallow_np[i]) ** 2)
        axes[1, i].imshow(shallow_np[i], cmap='gray', vmin=0, vmax=1)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Shallow AE', fontsize=11)
        axes[1, i].text(0.5, -0.1, f'MSE: {shallow_mse:.4f}',
                       transform=axes[1, i].transAxes, ha='center', fontsize=9)
        
        # Deep reconstruction
        deep_mse = np.mean((images_np[i, 0] - deep_np[i]) ** 2)
        axes[2, i].imshow(deep_np[i], cmap='gray', vmin=0, vmax=1)
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_title('Deep AE', fontsize=11)
        axes[2, i].text(0.5, -0.1, f'MSE: {deep_mse:.4f}',
                       transform=axes[2, i].transAxes, ha='center', fontsize=9)
    
    plt.suptitle('Shallow vs Deep Autoencoder', fontsize=14)
    plt.tight_layout()
    plt.savefig('depth_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved to 'depth_comparison.png'")


# =============================================================================
# PART 5: TRAINING AND UTILITIES
# =============================================================================

def train_deep_autoencoder(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
) -> float:
    """Train deep autoencoder for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (images, _) in enumerate(train_loader):
        images = images.view(images.size(0), -1).to(device)
        
        optimizer.zero_grad()
        reconstructed, _ = model(images)
        loss = criterion(reconstructed, images)
        loss.backward()
        
        # Gradient clipping for deep networks
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}')
    
    return total_loss / num_batches


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
# PART 6: MAIN EXECUTION
# =============================================================================

def main():
    """Main function."""
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Hyperparameters
    batch_size = 128
    learning_rate = 0.001
    num_epochs = 15
    
    print("\n" + "="*60)
    print("DEEP AUTOENCODER TRAINING")
    print("="*60)
    
    # Load data
    print("\nLoading MNIST dataset...")
    train_loader, test_loader = load_mnist_data(batch_size)
    
    # Initialize deep model
    model = DeepAutoencoder().to(device)
    print(f"\nDeep Autoencoder architecture:")
    print(model)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {num_params:,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Optional: Learning rate scheduler for deep networks
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    # Training
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    
    best_loss = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 40)
        
        train_loss = train_deep_autoencoder(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        print(f"Epoch {epoch} - Average Loss: {train_loss:.6f}")
        
        # Update learning rate based on loss
        scheduler.step(train_loss)
        
        # Save best model
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), 'deep_autoencoder_best.pth')
    
    # Visualizations
    print("\n" + "="*60)
    print("VISUALIZATIONS")
    print("="*60)
    
    print("\n1. Hierarchical features...")
    visualize_hierarchical_features(model, test_loader, device)
    
    # Save final model
    torch.save(model.state_dict(), 'deep_autoencoder.pth')
    print("\nModel saved")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()


# =============================================================================
# EXERCISES
# =============================================================================

"""
EXERCISE 1: Depth vs Performance
---------------------------------
Train autoencoders with different depths:
- Shallow: 784 → 128 → 32 → 128 → 784
- Medium: 784 → 512 → 256 → 32 → 256 → 512 → 784
- Deep: 784 → 512 → 384 → 256 → 128 → 32 → ... (mirror)
- Very Deep: 6-8 layers each side

Compare:
a) Reconstruction quality
b) Training time
c) Parameter count
d) Convergence speed

Questions:
- Is deeper always better?
- What's the optimal depth for MNIST?
- How does depth affect feature interpretability?


EXERCISE 2: Width vs Depth Trade-off
-------------------------------------
Compare two architectures with similar parameter counts:
- Wide & Shallow: 784 → 1024 → 32 → 1024 → 784
- Narrow & Deep: 784 → 256 → 128 → 64 → 32 → 64 → 128 → 256 → 784

Questions:
- Which performs better?
- How do learned features differ?
- What about training stability?


EXERCISE 3: Layer-wise Pretraining
-----------------------------------
Implement and compare:
a) End-to-end training (modern approach)
b) Greedy layer-wise pretraining (historical approach)

For layer-wise:
1. Train each layer separately
2. Stack and fine-tune
3. Compare with end-to-end

Questions:
- Is pretraining still beneficial?
- How much does it help convergence?
- When might it be necessary?


EXERCISE 4: Residual Connections
---------------------------------
Add residual (skip) connections to deep autoencoder:
h_{i+1} = f(h_i) + h_i (requires matching dimensions)

Compare with standard deep autoencoder.

Questions:
- Do residuals improve deep network training?
- How deep can you go with residuals?
- Impact on reconstruction quality?


EXERCISE 5: Bottleneck Analysis
--------------------------------
Fix depth, vary bottleneck size:
bottleneck_dims = [8, 16, 32, 64, 128, 256]

For each:
a) Train deep autoencoder
b) Measure reconstruction quality
c) Analyze latent space structure

Questions:
- How does bottleneck size affect deep networks differently than shallow?
- What's the minimum viable bottleneck for deep architectures?
"""
