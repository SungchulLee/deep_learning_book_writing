#!/usr/bin/env python3
# ==========================================================
# 02_ae_cnn_commented.py
# ==========================================================
# COMPREHENSIVE AUTOENCODER TUTORIAL: CONVOLUTIONAL ARCHITECTURE
#
# This script demonstrates a Convolutional Neural Network (CNN) based
# autoencoder. CNNs are MUCH better for images than fully connected layers!
#
# WHY CNN FOR IMAGES?
# 1. Spatial structure: CNNs preserve 2D spatial relationships
# 2. Translation invariance: Detects features regardless of position
# 3. Parameter efficiency: Far fewer parameters than fully connected
# 4. Hierarchical features: Learns edges → textures → parts → objects
#
# ARCHITECTURE:
# Input (1×28×28) → Conv Encoder → Bottleneck (64×7×7) → Conv Decoder → Output (1×28×28)
#
# ENCODER: Convolutional layers with downsampling
#   28×28 → Conv → 14×14 → Conv → 7×7 (spatial compression)
#
# DECODER: Transposed convolutions with upsampling
#   7×7 → ConvTranspose → 14×14 → ConvTranspose → 28×28 (spatial expansion)
#
# ADVANTAGES OVER FULLY CONNECTED:
# - Better reconstructions (preserves spatial structure)
# - Fewer parameters (more efficient)
# - Learns meaningful visual features
# - State-of-the-art for image compression
#
# Run: python 02_ae_cnn_commented.py
# ==========================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import time

# ==========================================================
# CONFIGURATION
# ==========================================================

# Model architecture
# Convolutional channel progression: 1 → 32 → 64 → 128
input_channels = 1       # Grayscale images (1 channel)
hidden_channels = [32, 64, 128]  # Channel dimensions through network
latent_channels = 64     # Channels in bottleneck (7×7×64 = 3,136D latent)

# Training parameters
batch_size = 128
learning_rate = 1e-3
num_epochs = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("="*60)
print("CONVOLUTIONAL AUTOENCODER FOR MNIST")
print("="*60)
print(f"Configuration:")
print(f"  • Input shape: 1×28×28")
print(f"  • Channel progression: 1 → 32 → 64 → 128")
print(f"  • Bottleneck: 64×7×7 = 3,136 dimensions")
print(f"  • Compression ratio: 784/3136 = 0.25x (more params but better quality)")
print(f"  • Batch size: {batch_size}")
print(f"  • Learning rate: {learning_rate}")
print(f"  • Epochs: {num_epochs}")
print(f"  • Device: {device}")
print("="*60)

# ==========================================================
# STEP 1: LOAD MNIST DATASET
# ==========================================================

transform = transforms.Compose([
    transforms.ToTensor(),
])

print("\nLoading MNIST dataset...")
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                          num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                         num_workers=2, pin_memory=True)

print(f"✓ Training samples: {len(train_dataset):,}")
print(f"✓ Test samples: {len(test_dataset):,}")

# ==========================================================
# STEP 2: DEFINE CNN AUTOENCODER ARCHITECTURE
# ==========================================================
# CNN ENCODER:
# 1×28×28 → Conv(32) → 32×28×28 → MaxPool → 32×14×14
#         → Conv(64) → 64×14×14 → MaxPool → 64×7×7
#         → Conv(128) → 128×7×7
#
# CNN DECODER:
# 128×7×7 → Conv(64) → 64×7×7
#         → Upsample → 64×14×14 → Conv(32) → 32×14×14
#         → Upsample → 32×28×28 → Conv(1) → 1×28×28
#
# KEY OPERATIONS:
# - Conv2d: Applies learnable filters to extract features
# - MaxPool2d: Downsampling (reduces spatial dimensions)
# - ConvTranspose2d or Upsample: Upsampling (increases spatial dimensions)
# - BatchNorm2d: Normalizes activations (improves training)
# - ReLU: Non-linear activation
# - Sigmoid: Output activation (ensures [0,1] range)

class ConvolutionalAutoencoder(nn.Module):
    """
    Convolutional Autoencoder for MNIST digits.
    
    Uses convolutional layers to preserve spatial structure,
    resulting in much better image reconstructions than fully
    connected autoencoders.
    
    Architecture:
        Encoder: 1×28×28 → 32×14×14 → 64×7×7 → 128×7×7
        Decoder: 128×7×7 → 64×7×7 → 32×14×14 → 1×28×28
    """
    
    def __init__(self):
        super(ConvolutionalAutoencoder, self).__init__()
        
        # ==================== ENCODER ====================
        # Progressively downsample and increase channels
        # This compresses spatial dimensions while extracting features
        
        self.encoder = nn.Sequential(
            # Block 1: 1×28×28 → 32×28×28 → 32×14×14
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            # Conv2d parameters:
            #   - in_channels=1 (grayscale)
            #   - out_channels=32 (learn 32 different features)
            #   - kernel_size=3 (3×3 filter)
            #   - stride=1 (move filter 1 pixel at a time)
            #   - padding=1 (add 1 pixel border to maintain size)
            nn.BatchNorm2d(32),  # Normalize activations for stable training
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 28×28 → 14×14 (downsample by 2)
            
            # Block 2: 32×14×14 → 64×14×14 → 64×7×7
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 14×14 → 7×7 (downsample by 2)
            
            # Block 3: 64×7×7 → 128×7×7 (increase feature depth)
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # No pooling here - keep 7×7 spatial dimensions
        )
        
        # ==================== DECODER ====================
        # Progressively upsample and decrease channels
        # This reconstructs the original spatial dimensions
        
        self.decoder = nn.Sequential(
            # Block 1: 128×7×7 → 64×7×7
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Upsample: 64×7×7 → 64×14×14
            nn.Upsample(scale_factor=2, mode='nearest'),
            # Alternative: nn.ConvTranspose2d (learnable upsampling)
            # Upsample is simpler and often works just as well
            
            # Block 2: 64×14×14 → 32×14×14
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Upsample: 32×14×14 → 32×28×28
            nn.Upsample(scale_factor=2, mode='nearest'),
            
            # Block 3: 32×28×28 → 1×28×28 (reconstruct original)
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()  # Output in [0, 1] range
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialize convolutional weights using Kaiming initialization.
        This is specifically designed for ReLU activations.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def encode(self, x):
        """
        Encode input image to latent representation.
        
        Args:
            x: Input tensor (batch, 1, 28, 28)
        
        Returns:
            Latent code (batch, 128, 7, 7)
        """
        return self.encoder(x)
    
    def decode(self, z):
        """
        Decode latent representation to image.
        
        Args:
            z: Latent code (batch, 128, 7, 7)
        
        Returns:
            Reconstructed image (batch, 1, 28, 28)
        """
        return self.decoder(z)
    
    def forward(self, x):
        """
        Full forward pass: encode then decode.
        
        Args:
            x: Input tensor (batch, 1, 28, 28)
        
        Returns:
            Reconstructed image (batch, 1, 28, 28)
        """
        z = self.encode(x)
        reconstruction = self.decode(z)
        return reconstruction

# Create model
model = ConvolutionalAutoencoder().to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
encoder_params = sum(p.numel() for p in model.encoder.parameters())
decoder_params = sum(p.numel() for p in model.decoder.parameters())

print(f"\nModel Architecture:")
print(model)
print(f"\nParameters:")
print(f"  • Total: {total_params:,}")
print(f"  • Encoder: {encoder_params:,}")
print(f"  • Decoder: {decoder_params:,}")
print(f"\nNote: Fewer parameters than FC autoencoder but better performance!")

# ==========================================================
# STEP 3: LOSS FUNCTION AND OPTIMIZER
# ==========================================================

criterion = nn.BCELoss()  # Binary Cross-Entropy
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

# Optional: Learning rate scheduler
# Reduces learning rate when training plateaus
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, verbose=True
)

print(f"\nTraining Configuration:")
print(f"  • Loss: {criterion.__class__.__name__}")
print(f"  • Optimizer: {optimizer.__class__.__name__}")
print(f"  • Scheduler: ReduceLROnPlateau")

# ==========================================================
# STEP 4: TRAINING LOOP
# ==========================================================

print("\n" + "="*60)
print("TRAINING")
print("="*60)

train_losses = []
test_losses = []
start_time = time.time()

for epoch in range(num_epochs):
    # ========== TRAINING ==========
    model.train()
    train_loss = 0.0
    
    for batch_idx, (images, _) in enumerate(train_loader):
        images = images.to(device)  # (batch, 1, 28, 28)
        
        # Forward pass
        reconstructions = model(images)
        loss = criterion(reconstructions, images)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        if (batch_idx + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                  f"Batch [{batch_idx+1}/{len(train_loader)}], "
                  f"Loss: {loss.item():.6f}")
    
    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # ========== EVALUATION ==========
    model.eval()
    test_loss = 0.0
    
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            reconstructions = model(images)
            loss = criterion(reconstructions, images)
            test_loss += loss.item()
    
    avg_test_loss = test_loss / len(test_loader)
    test_losses.append(avg_test_loss)
    
    # Update learning rate based on validation loss
    scheduler.step(avg_test_loss)
    
    print(f"\n{'='*60}")
    print(f"Epoch [{epoch+1}/{num_epochs}] Summary:")
    print(f"  • Training Loss: {avg_train_loss:.6f}")
    print(f"  • Test Loss: {avg_test_loss:.6f}")
    print(f"  • Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
    print(f"{'='*60}\n")

training_time = time.time() - start_time
print(f"\n✓ Training complete in {training_time:.2f} seconds")

# Save model
torch.save(model.state_dict(), 'autoencoder_cnn.pth')
print(f"✓ Model saved to 'autoencoder_cnn.pth'")

# ==========================================================
# STEP 5: VISUALIZATION - TRAINING HISTORY
# ==========================================================

print("\nCreating visualizations...")

fig, ax = plt.subplots(figsize=(10, 6))
epochs_range = range(1, num_epochs + 1)
ax.plot(epochs_range, train_losses, 'b-o', label='Training Loss', linewidth=2)
ax.plot(epochs_range, test_losses, 'r-s', label='Test Loss', linewidth=2)
ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax.set_ylabel('Loss (BCE)', fontsize=12, fontweight='bold')
ax.set_title('Convolutional Autoencoder Training History', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('02_ae_cnn_training.png', dpi=150, bbox_inches='tight')
print("✓ Saved: 02_ae_cnn_training.png")

# ==========================================================
# STEP 6: VISUALIZATION - RECONSTRUCTIONS
# ==========================================================

model.eval()
with torch.no_grad():
    images, labels = next(iter(test_loader))
    images = images.to(device)
    reconstructions = model(images)
    
    images_np = images.cpu().numpy()
    reconstructions_np = reconstructions.cpu().numpy()
    labels_np = labels.numpy()

n_samples = 10
rng = np.random.default_rng(42)
indices = rng.choice(len(images_np), n_samples, replace=False)

fig, axes = plt.subplots(2, n_samples, figsize=(15, 3))
fig.suptitle('CNN Autoencoder: Original vs Reconstruction\n(Notice sharper details vs FC version)', 
             fontsize=14, fontweight='bold')

for i in range(n_samples):
    idx = indices[i]
    
    # Original
    axes[0, i].imshow(images_np[idx, 0], cmap='gray', vmin=0, vmax=1)
    axes[0, i].axis('off')
    if i == 0:
        axes[0, i].set_title('Original', fontweight='bold')
    axes[0, i].text(0.5, -0.15, f'{labels_np[idx]}', ha='center', va='top',
                    transform=axes[0, i].transAxes, fontsize=11, fontweight='bold')
    
    # Reconstruction
    axes[1, i].imshow(reconstructions_np[idx, 0], cmap='gray', vmin=0, vmax=1)
    axes[1, i].axis('off')
    if i == 0:
        axes[1, i].set_title('Reconstructed', fontweight='bold')
    
    mse = np.mean((images_np[idx] - reconstructions_np[idx])**2)
    axes[1, i].text(0.5, -0.15, f'MSE: {mse:.4f}', ha='center', va='top',
                    transform=axes[1, i].transAxes, fontsize=8, color='red')

plt.tight_layout()
plt.savefig('02_ae_cnn_reconstruction.png', dpi=150, bbox_inches='tight')
print("✓ Saved: 02_ae_cnn_reconstruction.png")

# ==========================================================
# STEP 7: VISUALIZATION - LEARNED FILTERS
# ==========================================================
# Visualize what the first convolutional layer learned
# These are the low-level features (edges, corners, etc.)

print("\nVisualizing learned convolutional filters...")

# Get first conv layer weights: (32, 1, 3, 3)
first_conv_weights = model.encoder[0].weight.data.cpu()

fig, axes = plt.subplots(4, 8, figsize=(12, 6))
fig.suptitle('First Layer Convolutional Filters (32 filters, 3×3 each)', 
             fontsize=14, fontweight='bold')

for i in range(32):
    ax = axes[i // 8, i % 8]
    filter_img = first_conv_weights[i, 0].numpy()
    
    # Normalize for visualization
    filter_img = (filter_img - filter_img.min()) / (filter_img.max() - filter_img.min() + 1e-8)
    
    ax.imshow(filter_img, cmap='viridis', interpolation='nearest')
    ax.set_title(f'F{i+1}', fontsize=8)
    ax.axis('off')

plt.tight_layout()
plt.savefig('02_ae_cnn_filters.png', dpi=150, bbox_inches='tight')
print("✓ Saved: 02_ae_cnn_filters.png")

# ==========================================================
# STEP 8: VISUALIZATION - FEATURE MAPS
# ==========================================================
# Visualize intermediate feature maps to see what the network "sees"

print("\nVisualizing feature maps...")

model.eval()
with torch.no_grad():
    # Get one test image
    test_img = images[0:1].to(device)  # (1, 1, 28, 28)
    
    # Get feature maps from each layer
    x = test_img
    feature_maps = []
    
    for layer in model.encoder:
        x = layer(x)
        if isinstance(layer, nn.Conv2d):
            feature_maps.append(x)

# Visualize feature maps from first conv layer
first_features = feature_maps[0][0].cpu().numpy()  # (32, 28, 28)

fig, axes = plt.subplots(4, 8, figsize=(12, 6))
fig.suptitle('Feature Maps after First Convolution (32 channels)', 
             fontsize=14, fontweight='bold')

for i in range(32):
    ax = axes[i // 8, i % 8]
    ax.imshow(first_features[i], cmap='viridis')
    ax.set_title(f'Ch{i+1}', fontsize=8)
    ax.axis('off')

plt.tight_layout()
plt.savefig('02_ae_cnn_features.png', dpi=150, bbox_inches='tight')
print("✓ Saved: 02_ae_cnn_features.png")

# ==========================================================
# STEP 9: COMPARISON WITH FC AUTOENCODER
# ==========================================================
# Latent space visualization

try:
    from sklearn.decomposition import PCA
    
    # Encode test set with CNN
    model.eval()
    latent_codes_cnn = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            z = model.encode(images)
            # Flatten spatial dimensions: (batch, 128, 7, 7) → (batch, 6272)
            z_flat = z.view(z.size(0), -1)
            latent_codes_cnn.append(z_flat.cpu())
            all_labels.append(labels)
    
    latent_codes_cnn = torch.cat(latent_codes_cnn, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    # Apply PCA for 2D visualization
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent_codes_cnn)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(latent_2d[:, 0], latent_2d[:, 1], c=all_labels, 
                        cmap='tab10', s=3, alpha=0.6)
    cbar = plt.colorbar(scatter, ax=ax, ticks=range(10))
    cbar.set_label('Digit', fontsize=12, fontweight='bold')
    ax.set_xlabel(f'First PC ({pca.explained_variance_ratio_[0]*100:.1f}% var)', 
                  fontsize=12, fontweight='bold')
    ax.set_ylabel(f'Second PC ({pca.explained_variance_ratio_[1]*100:.1f}% var)', 
                  fontsize=12, fontweight='bold')
    ax.set_title('CNN Latent Space (3,136D → 2D via PCA)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('02_ae_cnn_latent_space.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: 02_ae_cnn_latent_space.png")

except Exception as e:
    print(f"Note: Could not create latent space visualization: {e}")

# ==========================================================
# FINAL SUMMARY
# ==========================================================

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print(f"Model: Convolutional Autoencoder")
print(f"  • Architecture: Conv layers with spatial downsampling/upsampling")
print(f"  • Bottleneck: 128×7×7 = 3,136 dimensions")
print(f"  • Parameters: {total_params:,}")
print(f"\nTraining:")
print(f"  • Epochs: {num_epochs}")
print(f"  • Time: {training_time:.2f}s")
print(f"  • Final train loss: {train_losses[-1]:.6f}")
print(f"  • Final test loss: {test_losses[-1]:.6f}")
print(f"\nAdvantages over FC:")
print(f"  • Preserves spatial structure")
print(f"  • Better image quality")
print(f"  • Learns hierarchical features")
print(f"  • More parameter efficient")
print(f"\nOutputs:")
print(f"  • Model: autoencoder_cnn.pth")
print(f"  • Training: 02_ae_cnn_training.png")
print(f"  • Reconstructions: 02_ae_cnn_reconstruction.png")
print(f"  • Filters: 02_ae_cnn_filters.png")
print(f"  • Features: 02_ae_cnn_features.png")
print(f"  • Latent space: 02_ae_cnn_latent_space.png")
print("="*60)

plt.show()
