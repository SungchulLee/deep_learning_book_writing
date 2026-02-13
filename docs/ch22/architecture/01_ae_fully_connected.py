#!/usr/bin/env python3
# ==========================================================
# 01_ae_fully_connected.py
# ==========================================================
# COMPREHENSIVE AUTOENCODER TUTORIAL: FULLY CONNECTED ARCHITECTURE
#
# This script demonstrates the simplest form of autoencoder using
# fully connected (dense) layers. This is the foundation for understanding
# all autoencoder architectures.
#
# WHAT IS AN AUTOENCODER?
# An autoencoder is a neural network that learns to compress data into
# a lower-dimensional representation (encoding) and then reconstruct
# the original data from this compressed form (decoding).
#
# ARCHITECTURE:
# Input (784) → Encoder → Bottleneck (32) → Decoder → Output (784)
#   [28×28 pixels]  [compress]  [code]    [decompress]  [28×28 pixels]
#
# KEY CONCEPTS:
# - Encoder: Compresses high-dimensional input to low-dimensional code
# - Bottleneck: The compressed representation (latent code)
# - Decoder: Reconstructs input from the compressed code
# - Reconstruction loss: How different is output from input?
#
# COMPARISON WITH PCA:
# - PCA: Linear compression (matrix multiplication)
# - Autoencoder: Non-linear compression (neural network)
# - Autoencoder can learn more complex patterns!
#
# Run: python 01_ae_fully_connected.py
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
# Hyperparameters control the training process and model architecture

# Model architecture
input_dim = 784          # 28×28 pixels flattened
hidden_dim = 128         # First hidden layer size
latent_dim = 32          # Bottleneck dimension (compressed representation)
                         # This is analogous to n_components in PCA
                         # 32 → 24.5x compression (784/32)

# Training parameters
batch_size = 128         # Number of samples processed together
learning_rate = 1e-3     # Step size for gradient descent
num_epochs = 20          # How many times to see the full dataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("="*60)
print("FULLY CONNECTED AUTOENCODER FOR MNIST")
print("="*60)
print(f"Configuration:")
print(f"  • Input dimension: {input_dim}")
print(f"  • Hidden dimension: {hidden_dim}")
print(f"  • Latent dimension: {latent_dim}")
print(f"  • Compression ratio: {input_dim/latent_dim:.1f}x")
print(f"  • Batch size: {batch_size}")
print(f"  • Learning rate: {learning_rate}")
print(f"  • Epochs: {num_epochs}")
print(f"  • Device: {device}")
print("="*60)

# ==========================================================
# STEP 1: LOAD AND PREPARE MNIST DATASET
# ==========================================================
# Transform: Convert PIL images to tensors and normalize to [0, 1]

transform = transforms.Compose([
    transforms.ToTensor(),  # Converts to tensor and scales to [0, 1]
])

# Load training and test datasets
print("\nLoading MNIST dataset...")
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# Create data loaders for batching
# DataLoader handles shuffling, batching, and parallel loading
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,           # Shuffle training data each epoch
    num_workers=2,          # Parallel data loading
    pin_memory=True         # Faster data transfer to GPU
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,          # Don't shuffle test data
    num_workers=2,
    pin_memory=True
)

print(f"✓ Training samples: {len(train_dataset):,}")
print(f"✓ Test samples: {len(test_dataset):,}")
print(f"✓ Batches per epoch: {len(train_loader)}")

# ==========================================================
# STEP 2: DEFINE AUTOENCODER ARCHITECTURE
# ==========================================================
# The autoencoder consists of two parts: Encoder and Decoder
#
# ENCODER ARCHITECTURE:
# Input (784) → Linear → ReLU → Linear → ReLU → Linear → Latent (32)
#    784 → 128 → 128 → 64 → 64 → 32
#
# DECODER ARCHITECTURE (mirror of encoder):
# Latent (32) → Linear → ReLU → Linear → ReLU → Linear → Sigmoid → Output (784)
#    32 → 64 → 64 → 128 → 128 → 784
#
# WHY THIS ARCHITECTURE?
# - Symmetric encoder/decoder creates a smooth compression/decompression
# - ReLU activation: Introduces non-linearity (allows complex patterns)
# - Sigmoid output: Ensures output is in [0, 1] like input
# - Gradual dimension reduction: 784 → 128 → 64 → 32 (smooth compression)

class FullyConnectedAutoencoder(nn.Module):
    """
    Fully Connected Autoencoder for MNIST digits.
    
    Architecture:
        Encoder: 784 → 128 → 64 → 32
        Decoder: 32 → 64 → 128 → 784
    
    The bottleneck (32D) forces the network to learn a compressed
    representation that captures the essential features of digits.
    """
    
    def __init__(self, input_dim=784, hidden_dim=128, latent_dim=32):
        """
        Initialize the autoencoder layers.
        
        Args:
            input_dim: Size of input (28×28 = 784 for MNIST)
            hidden_dim: Size of hidden layers
            latent_dim: Size of bottleneck (compressed representation)
        """
        super(FullyConnectedAutoencoder, self).__init__()
        
        # Store dimensions for later use
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # ==================== ENCODER ====================
        # Progressively compress input to latent representation
        
        self.encoder = nn.Sequential(
            # Layer 1: 784 → 128
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),  # inplace=True saves memory
            
            # Layer 2: 128 → 64 (intermediate compression)
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            
            # Layer 3: 64 → 32 (final compression to bottleneck)
            nn.Linear(hidden_dim // 2, latent_dim),
            # No activation here - let latent space be unconstrained
        )
        
        # ==================== DECODER ====================
        # Progressively reconstruct input from latent representation
        # Mirror structure of encoder
        
        self.decoder = nn.Sequential(
            # Layer 1: 32 → 64
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            
            # Layer 2: 64 → 128 (intermediate expansion)
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(inplace=True),
            
            # Layer 3: 128 → 784 (final reconstruction)
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # Sigmoid ensures output is in [0, 1] range
        )
        
        # Initialize weights using Xavier initialization
        # This helps with training stability and convergence
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialize network weights using Xavier/Glorot initialization.
        This helps prevent vanishing/exploding gradients.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def encode(self, x):
        """
        Encode input to latent representation.
        
        Args:
            x: Input tensor (batch_size, 784)
        
        Returns:
            Latent code (batch_size, 32)
        """
        return self.encoder(x)
    
    def decode(self, z):
        """
        Decode latent representation to reconstruction.
        
        Args:
            z: Latent code (batch_size, 32)
        
        Returns:
            Reconstructed input (batch_size, 784)
        """
        return self.decoder(z)
    
    def forward(self, x):
        """
        Full forward pass: encode then decode.
        
        Args:
            x: Input tensor (batch_size, 784)
        
        Returns:
            Reconstructed input (batch_size, 784)
        """
        # Flatten if input is an image (batch, 1, 28, 28)
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)  # Flatten to (batch, 784)
        
        # Encode to latent space
        z = self.encode(x)  # (batch, 32)
        
        # Decode back to input space
        reconstruction = self.decode(z)  # (batch, 784)
        
        return reconstruction

# Create model instance and move to device (GPU/CPU)
model = FullyConnectedAutoencoder(input_dim, hidden_dim, latent_dim)
model = model.to(device)

# Count total parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\nModel Architecture:")
print(model)
print(f"\nParameters:")
print(f"  • Total parameters: {total_params:,}")
print(f"  • Trainable parameters: {trainable_params:,}")

# ==========================================================
# STEP 3: DEFINE LOSS FUNCTION AND OPTIMIZER
# ==========================================================
# LOSS FUNCTION: Binary Cross-Entropy (BCE)
# - Measures difference between input and reconstruction
# - Good for data in [0, 1] range (like normalized images)
# - Formula: -[y*log(ŷ) + (1-y)*log(1-ŷ)]
#
# ALTERNATIVE: Mean Squared Error (MSE)
# - MSE: mean((input - reconstruction)²)
# - Also works well, but BCE is theoretically better for [0,1] data
#
# OPTIMIZER: Adam
# - Adaptive learning rate optimizer
# - Combines benefits of momentum and RMSprop
# - Generally works well out-of-the-box

criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
# criterion = nn.MSELoss()  # Alternative: Mean Squared Error

optimizer = optim.Adam(
    model.parameters(),
    lr=learning_rate,
    weight_decay=1e-5  # L2 regularization to prevent overfitting
)

print(f"\nTraining Configuration:")
print(f"  • Loss function: {criterion.__class__.__name__}")
print(f"  • Optimizer: {optimizer.__class__.__name__}")
print(f"  • Learning rate: {learning_rate}")

# ==========================================================
# STEP 4: TRAINING LOOP
# ==========================================================
# Training process:
# 1. Forward pass: Compute reconstructions
# 2. Compute loss: Compare reconstructions to inputs
# 3. Backward pass: Compute gradients
# 4. Update weights: Adjust parameters to minimize loss
#
# We track loss over time to monitor training progress.

print("\n" + "="*60)
print("TRAINING")
print("="*60)

# Lists to store training history
train_losses = []
test_losses = []

# Training start time
start_time = time.time()

for epoch in range(num_epochs):
    # ========== TRAINING PHASE ==========
    model.train()  # Set model to training mode (enables dropout, etc.)
    train_loss = 0.0
    
    for batch_idx, (images, _) in enumerate(train_loader):
        # images: (batch_size, 1, 28, 28)
        # labels not needed for unsupervised learning
        
        # Move data to device
        images = images.to(device)
        
        # Flatten images: (batch, 1, 28, 28) → (batch, 784)
        images_flat = images.view(images.size(0), -1)
        
        # ===== Forward Pass =====
        # Encode and decode the images
        reconstructions = model(images_flat)
        
        # ===== Compute Loss =====
        # Compare reconstruction to original
        loss = criterion(reconstructions, images_flat)
        
        # ===== Backward Pass =====
        optimizer.zero_grad()  # Clear previous gradients
        loss.backward()        # Compute gradients
        optimizer.step()       # Update weights
        
        # Accumulate loss for this epoch
        train_loss += loss.item()
        
        # Print progress every 100 batches
        if (batch_idx + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                  f"Batch [{batch_idx+1}/{len(train_loader)}], "
                  f"Loss: {loss.item():.6f}")
    
    # Average training loss for this epoch
    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # ========== EVALUATION PHASE ==========
    # Evaluate on test set to check generalization
    model.eval()  # Set model to evaluation mode
    test_loss = 0.0
    
    with torch.no_grad():  # Disable gradient computation (faster, less memory)
        for images, _ in test_loader:
            images = images.to(device)
            images_flat = images.view(images.size(0), -1)
            
            # Forward pass
            reconstructions = model(images_flat)
            
            # Compute loss
            loss = criterion(reconstructions, images_flat)
            test_loss += loss.item()
    
    # Average test loss for this epoch
    avg_test_loss = test_loss / len(test_loader)
    test_losses.append(avg_test_loss)
    
    # Print epoch summary
    print(f"\n{'='*60}")
    print(f"Epoch [{epoch+1}/{num_epochs}] Summary:")
    print(f"  • Average Training Loss: {avg_train_loss:.6f}")
    print(f"  • Average Test Loss: {avg_test_loss:.6f}")
    print(f"{'='*60}\n")

# Training complete
training_time = time.time() - start_time
print(f"\n✓ Training complete in {training_time:.2f} seconds")
print(f"  ({training_time/num_epochs:.2f} seconds per epoch)")

# ==========================================================
# STEP 5: SAVE TRAINED MODEL
# ==========================================================
# Save model weights for later use

torch.save(model.state_dict(), 'autoencoder_fc.pth')
print(f"✓ Model saved to 'autoencoder_fc.pth'")

# ==========================================================
# STEP 6: VISUALIZATION - TRAINING HISTORY
# ==========================================================
# Plot training and test loss over epochs

print("\nCreating visualizations...")

fig, ax = plt.subplots(figsize=(10, 6))

epochs_range = range(1, num_epochs + 1)
ax.plot(epochs_range, train_losses, 'b-o', label='Training Loss', linewidth=2, markersize=5)
ax.plot(epochs_range, test_losses, 'r-s', label='Test Loss', linewidth=2, markersize=5)

ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax.set_ylabel('Loss (BCE)', fontsize=12, fontweight='bold')
ax.set_title('Fully Connected Autoencoder Training History', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim([1, num_epochs])

plt.tight_layout()
plt.savefig('01_ae_fc_training.png', dpi=150, bbox_inches='tight')
print("✓ Saved: 01_ae_fc_training.png")

# ==========================================================
# STEP 7: VISUALIZATION - RECONSTRUCTIONS
# ==========================================================
# Compare original images with their reconstructions

model.eval()
with torch.no_grad():
    # Get a batch of test images
    images, labels = next(iter(test_loader))
    images = images.to(device)
    images_flat = images.view(images.size(0), -1)
    
    # Get reconstructions
    reconstructions = model(images_flat)
    
    # Move to CPU and reshape for visualization
    images_np = images.cpu().view(-1, 28, 28).numpy()
    reconstructions_np = reconstructions.cpu().view(-1, 28, 28).numpy()
    labels_np = labels.numpy()

# Select 10 random samples
n_samples = 10
rng = np.random.default_rng(42)
indices = rng.choice(len(images_np), n_samples, replace=False)

# Create comparison plot
fig, axes = plt.subplots(2, n_samples, figsize=(15, 3))
fig.suptitle('Fully Connected Autoencoder: Original vs Reconstruction', 
             fontsize=14, fontweight='bold')

for i in range(n_samples):
    idx = indices[i]
    
    # Original images (top row)
    axes[0, i].imshow(images_np[idx], cmap='gray', vmin=0, vmax=1)
    axes[0, i].axis('off')
    if i == 0:
        axes[0, i].set_title('Original', fontweight='bold', fontsize=10)
    axes[0, i].text(0.5, -0.15, f'{labels_np[idx]}', 
                    ha='center', va='top', transform=axes[0, i].transAxes,
                    fontsize=11, fontweight='bold')
    
    # Reconstructed images (bottom row)
    axes[1, i].imshow(reconstructions_np[idx], cmap='gray', vmin=0, vmax=1)
    axes[1, i].axis('off')
    if i == 0:
        axes[1, i].set_title('Reconstructed', fontweight='bold', fontsize=10)
    
    # Compute reconstruction error
    mse = np.mean((images_np[idx] - reconstructions_np[idx])**2)
    axes[1, i].text(0.5, -0.15, f'MSE: {mse:.4f}',
                    ha='center', va='top', transform=axes[1, i].transAxes,
                    fontsize=8, color='red')

plt.tight_layout()
plt.savefig('01_ae_fc_reconstruction.png', dpi=150, bbox_inches='tight')
print("✓ Saved: 01_ae_fc_reconstruction.png")

# ==========================================================
# STEP 8: VISUALIZATION - LATENT SPACE (2D PROJECTION)
# ==========================================================
# Visualize the 32D latent codes in 2D using PCA
# This shows how the autoencoder organizes digits in latent space

print("\nComputing latent space visualization...")

# Encode all test images to latent space
model.eval()
latent_codes = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        images_flat = images.view(images.size(0), -1)
        
        # Get latent codes
        z = model.encode(images_flat)
        latent_codes.append(z.cpu())
        all_labels.append(labels)

# Concatenate all batches
latent_codes = torch.cat(latent_codes, dim=0).numpy()  # (10000, 32)
all_labels = torch.cat(all_labels, dim=0).numpy()      # (10000,)

print(f"Latent codes shape: {latent_codes.shape}")

# Apply PCA to reduce 32D to 2D for visualization
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
latent_2d = pca.fit_transform(latent_codes)

print(f"Explained variance by 2 PCs: {pca.explained_variance_ratio_.sum()*100:.1f}%")

# Plot 2D latent space
fig, ax = plt.subplots(figsize=(10, 8))

scatter = ax.scatter(
    latent_2d[:, 0],
    latent_2d[:, 1],
    c=all_labels,
    cmap='tab10',
    s=3,
    alpha=0.6
)

cbar = plt.colorbar(scatter, ax=ax, ticks=range(10))
cbar.set_label('Digit', fontsize=12, fontweight='bold')

ax.set_xlabel(f'First PC ({pca.explained_variance_ratio_[0]*100:.1f}% var)', 
              fontsize=12, fontweight='bold')
ax.set_ylabel(f'Second PC ({pca.explained_variance_ratio_[1]*100:.1f}% var)', 
              fontsize=12, fontweight='bold')
ax.set_title('Latent Space Visualization (32D → 2D via PCA)', 
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('01_ae_fc_latent_space.png', dpi=150, bbox_inches='tight')
print("✓ Saved: 01_ae_fc_latent_space.png")

# ==========================================================
# FINAL SUMMARY
# ==========================================================
print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print(f"Model: Fully Connected Autoencoder")
print(f"  • Architecture: {input_dim} → {hidden_dim} → {latent_dim} → {hidden_dim} → {input_dim}")
print(f"  • Compression ratio: {input_dim/latent_dim:.1f}x")
print(f"  • Parameters: {total_params:,}")
print(f"\nTraining:")
print(f"  • Epochs: {num_epochs}")
print(f"  • Training time: {training_time:.2f}s")
print(f"  • Final train loss: {train_losses[-1]:.6f}")
print(f"  • Final test loss: {test_losses[-1]:.6f}")
print(f"\nOutputs:")
print(f"  • Model weights: autoencoder_fc.pth")
print(f"  • Training plot: 01_ae_fc_training.png")
print(f"  • Reconstructions: 01_ae_fc_reconstruction.png")
print(f"  • Latent space: 01_ae_fc_latent_space.png")
print("="*60)

plt.show()
