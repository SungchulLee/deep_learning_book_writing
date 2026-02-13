#!/usr/bin/env python3
# ==========================================================
# 03_ae_denoising_commented.py
# ==========================================================
# COMPREHENSIVE AUTOENCODER TUTORIAL: DENOISING AUTOENCODER
#
# This script demonstrates a Denoising Autoencoder (DAE), which learns
# to reconstruct clean images from corrupted (noisy) inputs. This is a
# powerful technique for both noise removal and learning robust features.
#
# WHAT IS A DENOISING AUTOENCODER?
# - Training: Add noise to input → Train to reconstruct CLEAN original
# - Effect: Forces network to learn robust, meaningful features
# - Application: Noise removal, feature learning, data preprocessing
#
# KEY INNOVATION:
# Unlike standard autoencoders that learn identity function,
# denoising autoencoders must learn to "denoise" by:
#   1. Identifying meaningful signal vs noise
#   2. Learning robust representations insensitive to corruption
#   3. Reconstructing the clean version
#
# NOISE TYPES DEMONSTRATED:
# - Gaussian noise: Add random normal noise to pixels
# - Salt & pepper: Randomly set pixels to 0 or 1
# - Dropout: Randomly zero out pixels
#
# ARCHITECTURE:
# Based on CNN autoencoder (from 02_ae_cnn.py) but trained with noise
#
# COMPARISON:
# Standard AE: clean → clean (learns compression)
# Denoising AE: noisy → clean (learns denoising + compression)
#
# Run: python 03_ae_denoising_commented.py
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

# Noise parameters
noise_type = 'gaussian'  # Options: 'gaussian', 'salt_pepper', 'dropout'
noise_factor = 0.3       # Amount of noise (0 to 1)
                         # gaussian: std of noise
                         # salt_pepper: fraction of pixels corrupted
                         # dropout: probability of zeroing

# Training parameters
batch_size = 128
learning_rate = 1e-3
num_epochs = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("="*60)
print("DENOISING AUTOENCODER FOR MNIST")
print("="*60)
print(f"Configuration:")
print(f"  • Noise type: {noise_type}")
print(f"  • Noise factor: {noise_factor}")
print(f"  • Architecture: CNN-based (same as 02_ae_cnn.py)")
print(f"  • Batch size: {batch_size}")
print(f"  • Learning rate: {learning_rate}")
print(f"  • Epochs: {num_epochs}")
print(f"  • Device: {device}")
print("="*60)

# ==========================================================
# STEP 1: DEFINE NOISE FUNCTIONS
# ==========================================================
# These functions add various types of corruption to images

def add_gaussian_noise(images, noise_factor=0.3):
    """
    Add Gaussian (normal) noise to images.
    
    This simulates real-world sensor noise or image degradation.
    
    Args:
        images: Clean images tensor (batch, 1, 28, 28)
        noise_factor: Standard deviation of noise
    
    Returns:
        Noisy images, clipped to [0, 1]
    """
    noise = torch.randn_like(images) * noise_factor
    noisy_images = images + noise
    # Clip to valid range [0, 1]
    return torch.clamp(noisy_images, 0., 1.)

def add_salt_pepper_noise(images, noise_factor=0.1):
    """
    Add salt-and-pepper noise (random black/white pixels).
    
    Simulates dead pixels, dust, or transmission errors.
    
    Args:
        images: Clean images tensor (batch, 1, 28, 28)
        noise_factor: Fraction of pixels to corrupt
    
    Returns:
        Noisy images with random pixels set to 0 or 1
    """
    noisy_images = images.clone()
    
    # Generate random mask
    mask = torch.rand_like(images)
    
    # Salt: set fraction/2 pixels to 1 (white)
    salt_mask = mask < (noise_factor / 2)
    noisy_images[salt_mask] = 1.0
    
    # Pepper: set fraction/2 pixels to 0 (black)
    pepper_mask = (mask >= (noise_factor / 2)) & (mask < noise_factor)
    noisy_images[pepper_mask] = 0.0
    
    return noisy_images

def add_dropout_noise(images, noise_factor=0.3):
    """
    Add dropout noise (randomly zero out pixels).
    
    Simulates missing data or occlusions.
    
    Args:
        images: Clean images tensor (batch, 1, 28, 28)
        noise_factor: Probability of zeroing each pixel
    
    Returns:
        Noisy images with random pixels set to 0
    """
    # Generate dropout mask
    mask = torch.rand_like(images) > noise_factor
    noisy_images = images * mask.float()
    return noisy_images

def add_noise(images, noise_type='gaussian', noise_factor=0.3):
    """
    Add noise to images based on specified type.
    
    Args:
        images: Clean images
        noise_type: 'gaussian', 'salt_pepper', or 'dropout'
        noise_factor: Amount of noise
    
    Returns:
        Noisy images
    """
    if noise_type == 'gaussian':
        return add_gaussian_noise(images, noise_factor)
    elif noise_type == 'salt_pepper':
        return add_salt_pepper_noise(images, noise_factor)
    elif noise_type == 'dropout':
        return add_dropout_noise(images, noise_factor)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

# ==========================================================
# STEP 2: LOAD MNIST DATASET
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
# STEP 3: DEFINE DENOISING AUTOENCODER
# ==========================================================
# Using same CNN architecture as 02_ae_cnn.py
# The only difference is the training procedure (add noise to inputs)

class DenoisingAutoencoder(nn.Module):
    """
    Denoising Autoencoder based on CNN architecture.
    
    Same network as CNN autoencoder, but trained to reconstruct
    clean images from noisy inputs.
    
    Architecture:
        Encoder: 1×28×28 → 32×14×14 → 64×7×7 → 128×7×7
        Decoder: 128×7×7 → 64×7×7 → 32×14×14 → 1×28×28
    """
    
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        
        # ==================== ENCODER ====================
        self.encoder = nn.Sequential(
            # Block 1: 1×28×28 → 32×14×14
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2: 32×14×14 → 64×7×7
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3: 64×7×7 → 128×7×7
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        # ==================== DECODER ====================
        self.decoder = nn.Sequential(
            # Block 1: 128×7×7 → 64×7×7
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            
            # Block 2: 64×14×14 → 32×14×14
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            
            # Block 3: 32×28×28 → 1×28×28
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Kaiming initialization for conv layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass: denoise the input.
        
        Args:
            x: Noisy input (batch, 1, 28, 28)
        
        Returns:
            Clean reconstruction (batch, 1, 28, 28)
        """
        z = self.encoder(x)
        clean = self.decoder(z)
        return clean

# Create model
model = DenoisingAutoencoder().to(device)
total_params = sum(p.numel() for p in model.parameters())

print(f"\nModel created: {total_params:,} parameters")

# ==========================================================
# STEP 4: TRAINING WITH NOISE
# ==========================================================
# KEY DIFFERENCE FROM STANDARD AUTOENCODER:
# 1. Add noise to input images
# 2. Train to reconstruct CLEAN original
# 3. Loss compares reconstruction to clean image, not noisy input

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                  factor=0.5, patience=3, verbose=True)

print("\n" + "="*60)
print("TRAINING")
print("="*60)
print(f"Training procedure:")
print(f"  1. Load clean image batch")
print(f"  2. Add {noise_type} noise (factor={noise_factor})")
print(f"  3. Feed noisy image to network")
print(f"  4. Compare output to CLEAN original")
print(f"  5. Backprop and update weights")
print("="*60 + "\n")

train_losses = []
test_losses = []
start_time = time.time()

for epoch in range(num_epochs):
    # ========== TRAINING ==========
    model.train()
    train_loss = 0.0
    
    for batch_idx, (clean_images, _) in enumerate(train_loader):
        clean_images = clean_images.to(device)  # Original clean images
        
        # CRITICAL STEP: Add noise to create corrupted input
        noisy_images = add_noise(clean_images, noise_type, noise_factor)
        
        # Forward pass: Denoise
        # Input: noisy_images, Target: clean_images
        reconstructed = model(noisy_images)
        
        # Loss: How close is reconstruction to CLEAN original?
        loss = criterion(reconstructed, clean_images)
        
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
        for clean_images, _ in test_loader:
            clean_images = clean_images.to(device)
            noisy_images = add_noise(clean_images, noise_type, noise_factor)
            reconstructed = model(noisy_images)
            loss = criterion(reconstructed, clean_images)
            test_loss += loss.item()
    
    avg_test_loss = test_loss / len(test_loader)
    test_losses.append(avg_test_loss)
    scheduler.step(avg_test_loss)
    
    print(f"\n{'='*60}")
    print(f"Epoch [{epoch+1}/{num_epochs}] Summary:")
    print(f"  • Training Loss: {avg_train_loss:.6f}")
    print(f"  • Test Loss: {avg_test_loss:.6f}")
    print(f"  • LR: {optimizer.param_groups[0]['lr']:.2e}")
    print(f"{'='*60}\n")

training_time = time.time() - start_time
print(f"\n✓ Training complete in {training_time:.2f} seconds")

# Save model
torch.save(model.state_dict(), f'autoencoder_denoising_{noise_type}.pth')
print(f"✓ Model saved")

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
ax.set_title(f'Denoising Autoencoder Training ({noise_type} noise)', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'03_ae_denoising_{noise_type}_training.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: 03_ae_denoising_{noise_type}_training.png")

# ==========================================================
# STEP 6: VISUALIZATION - DENOISING RESULTS
# ==========================================================
# Show: Original → Noisy → Denoised (3 rows)

model.eval()
with torch.no_grad():
    clean_images, labels = next(iter(test_loader))
    clean_images = clean_images.to(device)
    
    # Add noise
    noisy_images = add_noise(clean_images, noise_type, noise_factor)
    
    # Denoise
    denoised_images = model(noisy_images)
    
    # To numpy
    clean_np = clean_images.cpu().numpy()
    noisy_np = noisy_images.cpu().numpy()
    denoised_np = denoised_images.cpu().numpy()
    labels_np = labels.numpy()

n_samples = 10
rng = np.random.default_rng(42)
indices = rng.choice(len(clean_np), n_samples, replace=False)

# Create 3-row comparison
fig, axes = plt.subplots(3, n_samples, figsize=(15, 4.5))
fig.suptitle(f'Denoising Autoencoder ({noise_type} noise, factor={noise_factor})\n'
             f'Top: Clean, Middle: Noisy, Bottom: Denoised', 
             fontsize=14, fontweight='bold')

for i in range(n_samples):
    idx = indices[i]
    
    # Clean (row 0)
    axes[0, i].imshow(clean_np[idx, 0], cmap='gray', vmin=0, vmax=1)
    axes[0, i].axis('off')
    if i == 0:
        axes[0, i].set_title('Clean', fontweight='bold', fontsize=10)
    axes[0, i].text(0.5, -0.15, f'{labels_np[idx]}', ha='center', va='top',
                    transform=axes[0, i].transAxes, fontsize=11, fontweight='bold')
    
    # Noisy (row 1)
    axes[1, i].imshow(noisy_np[idx, 0], cmap='gray', vmin=0, vmax=1)
    axes[1, i].axis('off')
    if i == 0:
        axes[1, i].set_title('Noisy', fontweight='bold', fontsize=10)
    
    # Denoised (row 2)
    axes[2, i].imshow(denoised_np[idx, 0], cmap='gray', vmin=0, vmax=1)
    axes[2, i].axis('off')
    if i == 0:
        axes[2, i].set_title('Denoised', fontweight='bold', fontsize=10)
    
    # Compute metrics
    noisy_psnr = -10 * np.log10(np.mean((clean_np[idx] - noisy_np[idx])**2) + 1e-8)
    denoised_psnr = -10 * np.log10(np.mean((clean_np[idx] - denoised_np[idx])**2) + 1e-8)
    improvement = denoised_psnr - noisy_psnr
    
    axes[2, i].text(0.5, -0.15, f'+{improvement:.1f}dB', ha='center', va='top',
                    transform=axes[2, i].transAxes, fontsize=8, 
                    color='green' if improvement > 0 else 'red')

plt.tight_layout()
plt.savefig(f'03_ae_denoising_{noise_type}_results.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: 03_ae_denoising_{noise_type}_results.png")

# ==========================================================
# STEP 7: QUANTITATIVE EVALUATION
# ==========================================================
# Compute average PSNR improvement

print("\nComputing denoising performance metrics...")

model.eval()
total_noisy_psnr = 0.0
total_denoised_psnr = 0.0
n_samples_eval = 0

with torch.no_grad():
    for clean_images, _ in test_loader:
        clean_images = clean_images.to(device)
        noisy_images = add_noise(clean_images, noise_type, noise_factor)
        denoised_images = model(noisy_images)
        
        # Convert to numpy
        clean_np = clean_images.cpu().numpy()
        noisy_np = noisy_images.cpu().numpy()
        denoised_np = denoised_images.cpu().numpy()
        
        # Compute PSNR for each image
        for i in range(len(clean_np)):
            # PSNR = -10 * log10(MSE)
            noisy_mse = np.mean((clean_np[i] - noisy_np[i])**2)
            denoised_mse = np.mean((clean_np[i] - denoised_np[i])**2)
            
            noisy_psnr = -10 * np.log10(noisy_mse + 1e-8)
            denoised_psnr = -10 * np.log10(denoised_mse + 1e-8)
            
            total_noisy_psnr += noisy_psnr
            total_denoised_psnr += denoised_psnr
            n_samples_eval += 1

avg_noisy_psnr = total_noisy_psnr / n_samples_eval
avg_denoised_psnr = total_denoised_psnr / n_samples_eval
avg_improvement = avg_denoised_psnr - avg_noisy_psnr

print(f"\nPerformance on {n_samples_eval} test images:")
print(f"  • Average noisy PSNR: {avg_noisy_psnr:.2f} dB")
print(f"  • Average denoised PSNR: {avg_denoised_psnr:.2f} dB")
print(f"  • Average improvement: {avg_improvement:.2f} dB")

# ==========================================================
# FINAL SUMMARY
# ==========================================================

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print(f"Model: Denoising Autoencoder (CNN-based)")
print(f"  • Noise type: {noise_type}")
print(f"  • Noise factor: {noise_factor}")
print(f"  • Parameters: {total_params:,}")
print(f"\nTraining:")
print(f"  • Epochs: {num_epochs}")
print(f"  • Time: {training_time:.2f}s")
print(f"  • Final test loss: {test_losses[-1]:.6f}")
print(f"\nDenoising Performance:")
print(f"  • PSNR improvement: {avg_improvement:.2f} dB")
print(f"\nKey Insights:")
print(f"  • Trained on noisy→clean pairs")
print(f"  • Learns robust features insensitive to noise")
print(f"  • Can remove noise types it was trained on")
print(f"  • Better than standard filtering for complex noise")
print(f"\nApplications:")
print(f"  • Image denoising and restoration")
print(f"  • Robust feature learning")
print(f"  • Data preprocessing")
print(f"  • Anomaly detection")
print("="*60)

plt.show()
