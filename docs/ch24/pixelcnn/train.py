"""
Training Script for PixelCNN Image Generation

This script demonstrates:
1. Training PixelCNN on MNIST dataset
2. Autoregressive image generation
3. Visualization of generated samples
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from pixelcnn import PixelCNN


def binarize(x: torch.Tensor) -> torch.Tensor:
    """
    Binarize images to black and white.
    
    This simplifies the problem: instead of predicting 256 possible
    values per pixel, we only predict binary (0 or 1).
    
    Args:
        x: Image tensor with values in [0, 1]
        
    Returns:
        Binarized image (0 or 1)
    """
    return (x > 0.5).float()


def train_epoch(model: nn.Module,
                dataloader: torch.utils.data.DataLoader,
                optimizer: optim.Optimizer,
                device: str) -> float:
    """
    Train for one epoch.
    
    Args:
        model: PixelCNN model
        dataloader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        
    Returns:
        Average loss for epoch
    """
    model.train()
    total_loss = 0
    
    # Binary Cross-Entropy Loss
    # For each pixel, we predict probability it's 1 (white)
    criterion = nn.BCEWithLogitsLoss()
    
    for images, _ in dataloader:  # We don't need labels for generation
        # Move to device and binarize
        images = binarize(images.to(device))
        
        # Forward pass
        logits = model(images)
        
        # Compute loss
        # We want to predict the actual image pixel values
        loss = criterion(logits, images)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model: nn.Module,
            dataloader: torch.utils.data.DataLoader,
            device: str) -> float:
    """
    Evaluate model on test set.
    
    Args:
        model: PixelCNN model
        dataloader: Test data loader
        device: Device to evaluate on
        
    Returns:
        Average loss on test set
    """
    model.eval()
    total_loss = 0
    criterion = nn.BCEWithLogitsLoss()
    
    with torch.no_grad():
        for images, _ in dataloader:
            images = binarize(images.to(device))
            logits = model(images)
            loss = criterion(logits, images)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def visualize_samples(model: nn.Module,
                     real_images: torch.Tensor,
                     device: str,
                     n_samples: int = 16):
    """
    Generate and visualize samples.
    
    Args:
        model: Trained PixelCNN
        real_images: Real images for comparison
        device: Device to generate on
        n_samples: Number of samples to generate
    """
    model.eval()
    
    # Generate new images
    print("Generating samples (this may take a minute)...")
    with torch.no_grad():
        generated = model.generate(
            shape=(n_samples, 28, 28),
            device=device
        )
    
    # Create figure
    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    
    # Plot real images in top two rows
    for i in range(2):
        for j in range(8):
            idx = i * 8 + j
            axes[i, j].imshow(real_images[idx, 0].cpu(), cmap='gray')
            axes[i, j].axis('off')
            if j == 0:
                axes[i, j].set_ylabel('Real', rotation=0, labelpad=30, fontsize=12)
    
    # Plot generated images in bottom two rows
    for i in range(2, 4):
        for j in range(8):
            idx = (i - 2) * 8 + j
            axes[i, j].imshow(generated[idx, 0].cpu(), cmap='gray')
            axes[i, j].axis('off')
            if j == 0:
                axes[i, j].set_ylabel('Generated', rotation=0, labelpad=30, fontsize=12)
    
    plt.suptitle('Real Images vs Generated Images', fontsize=16, y=0.98)
    plt.tight_layout()
    
    return fig


def main():
    """
    Main training pipeline
    """
    print("=" * 70)
    print("PixelCNN: Autoregressive Image Generation")
    print("=" * 70)
    
    # ==================== Setup ====================
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    if device == 'cpu':
        print("\nWARNING: Training on CPU will be slow!")
        print("PixelCNN generates pixel-by-pixel, which is computationally intensive.")
        print("Consider using a smaller model or fewer epochs for CPU training.\n")
    
    # Hyperparameters
    BATCH_SIZE = 64
    N_EPOCHS = 20  # Reduce if training on CPU
    LEARNING_RATE = 0.001
    N_CHANNELS = 64
    N_RESIDUAL_BLOCKS = 5
    
    print(f"Hyperparameters:")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Epochs: {N_EPOCHS}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Feature Channels: {N_CHANNELS}")
    print(f"  Residual Blocks: {N_RESIDUAL_BLOCKS}")
    
    # ==================== Load Data ====================
    print(f"\n{'='*70}")
    print("Step 1: Loading MNIST dataset...")
    print(f"{'='*70}")
    
    # Transform: convert to tensor and normalize to [0, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Load MNIST
    train_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )
    
    print(f"\n✓ Loaded MNIST dataset")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  Image size: 28x28")
    
    # ==================== Initialize Model ====================
    print(f"\n{'='*70}")
    print("Step 2: Initializing PixelCNN...")
    print(f"{'='*70}")
    
    model = PixelCNN(
        n_channels=N_CHANNELS,
        n_residual_blocks=N_RESIDUAL_BLOCKS
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel initialized")
    print(f"  Parameters: {n_params:,}")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # ==================== Training ====================
    print(f"\n{'='*70}")
    print("Step 3: Training PixelCNN...")
    print(f"{'='*70}")
    print("\nNote: PixelCNN training is slow because each pixel depends on")
    print("all previous pixels. This is the price of autoregressive modeling!")
    print()
    
    train_losses = []
    test_losses = []
    
    for epoch in tqdm(range(N_EPOCHS), desc="Training"):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        
        # Evaluate
        test_loss = evaluate(model, test_loader, device)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f"\nEpoch {epoch+1}/{N_EPOCHS}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Test Loss: {test_loss:.4f}")
    
    print(f"\n✓ Training complete!")
    print(f"  Final train loss: {train_losses[-1]:.4f}")
    print(f"  Final test loss: {test_losses[-1]:.4f}")
    
    # ==================== Visualization ====================
    print(f"\n{'='*70}")
    print("Step 4: Creating visualizations...")
    print(f"{'='*70}")
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', alpha=0.7)
    plt.plot(test_losses, label='Test Loss', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Binary Cross-Entropy Loss')
    plt.title('PixelCNN: Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('pixelcnn_training.png', dpi=150)
    print("✓ Saved pixelcnn_training.png")
    
    # Get some real images for comparison
    real_images, _ = next(iter(test_loader))
    real_images = binarize(real_images[:16])
    
    # Generate and visualize samples
    fig = visualize_samples(model, real_images, device, n_samples=16)
    plt.savefig('pixelcnn_samples.png', dpi=150)
    print("✓ Saved pixelcnn_samples.png")
    
    # ==================== Generation Demo ====================
    print(f"\n{'='*70}")
    print("Step 5: Demonstrating autoregressive generation...")
    print(f"{'='*70}")
    
    print("\nGenerating a single image step-by-step...")
    print("Watch how the image is filled pixel by pixel!")
    
    # Generate one image and show intermediate steps
    model.eval()
    height, width = 28, 28
    
    # Create frames showing generation process
    frames = []
    sample = torch.zeros(1, 1, height, width).to(device)
    
    # Generate and save frames every 50 pixels
    pixel_count = 0
    with torch.no_grad():
        for i in range(height):
            for j in range(width):
                logits = model(sample)
                probs = torch.sigmoid(logits[:, :, i, j])
                sample[:, :, i, j] = torch.bernoulli(probs)
                
                pixel_count += 1
                if pixel_count % 50 == 0 or pixel_count == height * width:
                    frames.append(sample.clone())
    
    # Visualize generation process
    n_frames = len(frames)
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.flatten()
    
    for idx, frame in enumerate(frames):
        axes[idx].imshow(frame[0, 0].cpu(), cmap='gray')
        axes[idx].set_title(f'Pixel {(idx+1)*50}' if idx < n_frames-1 else 'Complete')
        axes[idx].axis('off')
    
    plt.suptitle('Autoregressive Generation Process', fontsize=14)
    plt.tight_layout()
    plt.savefig('generation_process.png', dpi=150)
    print("✓ Saved generation_process.png")
    
    # ==================== Summary ====================
    print(f"\n{'='*70}")
    print("Training Complete!")
    print(f"{'='*70}")
    print("\nKey Observations:")
    print("1. PixelCNN generates images pixel-by-pixel")
    print("2. Each pixel depends on all previous pixels (autoregressive)")
    print("3. Generation is slow but produces diverse samples")
    print("4. The model learned MNIST digit structure!")
    print("\nCheck the generated PNG files for visualizations.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
