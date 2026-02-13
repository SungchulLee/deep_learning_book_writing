"""
Example 1: Basic U-Net Semantic Segmentation
=============================================

This script implements a complete U-Net architecture from scratch for
binary semantic segmentation. We'll use a synthetic dataset of simple shapes.

Key Concepts:
- U-Net architecture (encoder-decoder with skip connections)
- Binary segmentation (2 classes)
- Pixel-wise cross-entropy loss
- IoU (Intersection over Union) metric
- Data augmentation for segmentation

Author: PyTorch Semantic Segmentation Tutorial
Date: 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import time
import os

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# STEP 1: DEVICE CONFIGURATION
# ============================================================================

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# ============================================================================
# STEP 2: SYNTHETIC DATASET GENERATION
# ============================================================================
"""
For learning purposes, we'll create a synthetic dataset with simple shapes.
Each image will contain circles, rectangles, or triangles that need to be
segmented from the background.

In real applications, you would load actual images and masks.
"""

class SyntheticShapesDataset(Dataset):
    """
    Generate synthetic images with simple shapes for segmentation.
    
    Each sample consists of:
    - Input image: RGB image with a shape drawn on it
    - Mask: Binary mask where 1=shape, 0=background
    """
    
    def __init__(self, num_samples=1000, image_size=256, transform=None):
        """
        Args:
            num_samples: Number of synthetic images to generate
            image_size: Size of square images (image_size × image_size)
            transform: Optional data augmentation
        """
        self.num_samples = num_samples
        self.image_size = image_size
        self.transform = transform
        
        # Pre-generate all samples for consistency
        self.samples = []
        print(f"Generating {num_samples} synthetic images...")
        for i in range(num_samples):
            img, mask = self._generate_sample()
            self.samples.append((img, mask))
            if (i + 1) % 200 == 0:
                print(f"  Generated {i + 1}/{num_samples} images")
        print("Dataset generation complete!\n")
    
    def _generate_sample(self):
        """Generate a single image-mask pair."""
        # Create blank image (random background color)
        bg_color = tuple(np.random.randint(0, 100, 3).tolist())
        img = Image.new('RGB', (self.image_size, self.image_size), bg_color)
        mask = Image.new('L', (self.image_size, self.image_size), 0)
        
        draw_img = ImageDraw.Draw(img)
        draw_mask = ImageDraw.Draw(mask)
        
        # Random shape color (different from background)
        shape_color = tuple(np.random.randint(150, 255, 3).tolist())
        
        # Randomly choose shape type
        shape_type = np.random.choice(['circle', 'rectangle', 'triangle'])
        
        # Random position and size
        center_x = np.random.randint(64, self.image_size - 64)
        center_y = np.random.randint(64, self.image_size - 64)
        size = np.random.randint(30, 70)
        
        if shape_type == 'circle':
            # Draw circle
            bbox = [center_x - size, center_y - size, 
                   center_x + size, center_y + size]
            draw_img.ellipse(bbox, fill=shape_color)
            draw_mask.ellipse(bbox, fill=255)  # 255 = foreground in mask
            
        elif shape_type == 'rectangle':
            # Draw rectangle
            bbox = [center_x - size, center_y - size, 
                   center_x + size, center_y + size]
            draw_img.rectangle(bbox, fill=shape_color)
            draw_mask.rectangle(bbox, fill=255)
            
        else:  # triangle
            # Draw triangle
            points = [
                (center_x, center_y - size),
                (center_x - size, center_y + size),
                (center_x + size, center_y + size)
            ]
            draw_img.polygon(points, fill=shape_color)
            draw_mask.polygon(points, fill=255)
        
        return img, mask
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Returns:
            image: Tensor of shape (3, H, W), normalized to [0, 1]
            mask: Tensor of shape (1, H, W), values in {0, 1}
        """
        img, mask = self.samples[idx]
        
        # Convert PIL Images to numpy arrays
        img_array = np.array(img).astype(np.float32) / 255.0  # Normalize to [0, 1]
        mask_array = np.array(mask).astype(np.float32) / 255.0  # Convert to [0, 1]
        
        # Convert to PyTorch tensors
        # Image: (H, W, C) -> (C, H, W)
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
        # Mask: (H, W) -> (1, H, W)
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)
        
        # Apply augmentation if provided
        if self.transform:
            # For simplicity, we'll apply basic transforms
            # In practice, use albumentations for proper mask augmentation
            img_tensor, mask_tensor = self.transform(img_tensor, mask_tensor)
        
        return img_tensor, mask_tensor


# Simple augmentation function
def simple_augment(img, mask):
    """Apply random horizontal flip to both image and mask."""
    if np.random.random() > 0.5:
        img = torch.flip(img, dims=[2])  # Flip width
        mask = torch.flip(mask, dims=[2])
    return img, mask


# Create datasets
print("Creating datasets...")
train_dataset = SyntheticShapesDataset(num_samples=800, image_size=256)
val_dataset = SyntheticShapesDataset(num_samples=100, image_size=256)
test_dataset = SyntheticShapesDataset(num_samples=100, image_size=256)

# Create data loaders
BATCH_SIZE = 8
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

# ============================================================================
# STEP 3: U-NET ARCHITECTURE IMPLEMENTATION
# ============================================================================
"""
U-Net consists of:
1. Encoder (Contracting Path): Captures context
2. Bottleneck: Deepest features
3. Decoder (Expanding Path): Enables precise localization
4. Skip Connections: Concatenate encoder features to decoder
"""

class DoubleConv(nn.Module):
    """
    Double convolution block: Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU
    This is the basic building block of U-Net.
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net Architecture for Semantic Segmentation.
    
    Architecture:
        Input (3 channels) -> Encoder (with skip connections) -> Bottleneck -> 
        Decoder (with skip connections) -> Output (num_classes channels)
    
    Args:
        in_channels: Number of input channels (3 for RGB)
        num_classes: Number of output classes (2 for binary: background, foreground)
    """
    
    def __init__(self, in_channels=3, num_classes=2):
        super(UNet, self).__init__()
        
        # Encoder (Downsampling path)
        self.enc1 = DoubleConv(in_channels, 64)      # 256x256 -> 256x256
        self.pool1 = nn.MaxPool2d(2)                  # 256x256 -> 128x128
        
        self.enc2 = DoubleConv(64, 128)               # 128x128 -> 128x128
        self.pool2 = nn.MaxPool2d(2)                  # 128x128 -> 64x64
        
        self.enc3 = DoubleConv(128, 256)              # 64x64 -> 64x64
        self.pool3 = nn.MaxPool2d(2)                  # 64x64 -> 32x32
        
        self.enc4 = DoubleConv(256, 512)              # 32x32 -> 32x32
        self.pool4 = nn.MaxPool2d(2)                  # 32x32 -> 16x16
        
        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)       # 16x16 -> 16x16
        
        # Decoder (Upsampling path)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)  # 16x16 -> 32x32
        self.dec4 = DoubleConv(1024, 512)             # 32x32 -> 32x32 (1024 because of concat)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)   # 32x32 -> 64x64
        self.dec3 = DoubleConv(512, 256)              # 64x64 -> 64x64
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)   # 64x64 -> 128x128
        self.dec2 = DoubleConv(256, 128)              # 128x128 -> 128x128
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)    # 128x128 -> 256x256
        self.dec1 = DoubleConv(128, 64)               # 256x256 -> 256x256
        
        # Final output layer
        self.out = nn.Conv2d(64, num_classes, kernel_size=1)  # 1x1 conv to get class scores
    
    def forward(self, x):
        """
        Forward pass through U-Net.
        
        Args:
            x: Input tensor of shape (batch_size, 3, H, W)
        
        Returns:
            Output tensor of shape (batch_size, num_classes, H, W)
        """
        # Encoder with skip connections saved
        enc1 = self.enc1(x)           # 64 channels
        x = self.pool1(enc1)
        
        enc2 = self.enc2(x)           # 128 channels
        x = self.pool2(enc2)
        
        enc3 = self.enc3(x)           # 256 channels
        x = self.pool3(enc3)
        
        enc4 = self.enc4(x)           # 512 channels
        x = self.pool4(enc4)
        
        # Bottleneck
        x = self.bottleneck(x)        # 1024 channels
        
        # Decoder with skip connections
        x = self.upconv4(x)           # Upsample
        x = torch.cat([x, enc4], dim=1)  # Concatenate skip connection
        x = self.dec4(x)
        
        x = self.upconv3(x)
        x = torch.cat([x, enc3], dim=1)
        x = self.dec3(x)
        
        x = self.upconv2(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.dec2(x)
        
        x = self.upconv1(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.dec1(x)
        
        # Final output
        x = self.out(x)
        
        return x


# Create model
print("\nCreating U-Net model...")
model = UNet(in_channels=3, num_classes=2)  # Binary: background (0), foreground (1)
model = model.to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# ============================================================================
# STEP 4: LOSS FUNCTION AND OPTIMIZER
# ============================================================================
"""
For binary segmentation, we use Binary Cross-Entropy Loss with Logits.
This combines sigmoid activation and binary cross-entropy in one function
for numerical stability.

The loss is computed for every pixel:
Loss = -[y*log(p) + (1-y)*log(1-p)]
where y is ground truth (0 or 1) and p is predicted probability
"""

criterion = nn.CrossEntropyLoss()  # For multi-class (even if just 2 classes)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=3, verbose=True
)

print("\nTraining configuration:")
print(f"Loss function: CrossEntropyLoss")
print(f"Optimizer: Adam (lr=0.001)")
print(f"Scheduler: ReduceLROnPlateau (monitor IoU)")

# ============================================================================
# STEP 5: EVALUATION METRICS
# ============================================================================
"""
For segmentation, we use IoU (Intersection over Union), also known as
Jaccard Index. It measures the overlap between predicted and ground truth.

IoU = |A ∩ B| / |A ∪ B|
    = TP / (TP + FP + FN)

Where:
- TP (True Positive): Correctly predicted foreground pixels
- FP (False Positive): Background pixels predicted as foreground
- FN (False Negative): Foreground pixels predicted as background
"""

def calculate_iou(pred, target, threshold=0.5):
    """
    Calculate IoU (Intersection over Union) for binary segmentation.
    
    Args:
        pred: Predicted logits of shape (batch, 2, H, W)
        target: Ground truth masks of shape (batch, 1, H, W) with values in {0, 1}
        threshold: Threshold for converting probabilities to binary predictions
    
    Returns:
        iou: Mean IoU across the batch
    """
    with torch.no_grad():
        # Get predicted class (0 or 1)
        pred_class = torch.argmax(pred, dim=1)  # Shape: (batch, H, W)
        target_class = target.squeeze(1).long()  # Shape: (batch, H, W)
        
        # Calculate intersection and union for foreground class (class 1)
        pred_fg = (pred_class == 1)
        target_fg = (target_class == 1)
        
        intersection = (pred_fg & target_fg).float().sum(dim=(1, 2))
        union = (pred_fg | target_fg).float().sum(dim=(1, 2))
        
        # Avoid division by zero
        iou = (intersection + 1e-6) / (union + 1e-6)
        
        return iou.mean().item()


def calculate_pixel_accuracy(pred, target):
    """Calculate pixel-wise accuracy."""
    with torch.no_grad():
        pred_class = torch.argmax(pred, dim=1)
        target_class = target.squeeze(1).long()
        correct = (pred_class == target_class).float()
        accuracy = correct.mean()
        return accuracy.item()

# ============================================================================
# STEP 6: TRAINING FUNCTION
# ============================================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    running_iou = 0.0
    running_accuracy = 0.0
    
    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        
        # Calculate loss
        # CrossEntropyLoss expects target shape (batch, H, W) with class indices
        masks_class = masks.squeeze(1).long()
        loss = criterion(outputs, masks_class)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        iou = calculate_iou(outputs, masks)
        accuracy = calculate_pixel_accuracy(outputs, masks)
        
        # Statistics
        running_loss += loss.item()
        running_iou += iou
        running_accuracy += accuracy
        
        # Print progress
        if (batch_idx + 1) % 20 == 0:
            print(f'  Batch {batch_idx + 1}/{len(loader)}: '
                  f'Loss: {loss.item():.4f}, '
                  f'IoU: {iou:.4f}, '
                  f'Acc: {accuracy:.4f}')
    
    epoch_loss = running_loss / len(loader)
    epoch_iou = running_iou / len(loader)
    epoch_accuracy = running_accuracy / len(loader)
    
    return epoch_loss, epoch_iou, epoch_accuracy


# ============================================================================
# STEP 7: VALIDATION FUNCTION
# ============================================================================

def validate(model, loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    running_iou = 0.0
    running_accuracy = 0.0
    
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Calculate loss
            masks_class = masks.squeeze(1).long()
            loss = criterion(outputs, masks_class)
            
            # Calculate metrics
            iou = calculate_iou(outputs, masks)
            accuracy = calculate_pixel_accuracy(outputs, masks)
            
            # Statistics
            running_loss += loss.item()
            running_iou += iou
            running_accuracy += accuracy
    
    avg_loss = running_loss / len(loader)
    avg_iou = running_iou / len(loader)
    avg_accuracy = running_accuracy / len(loader)
    
    return avg_loss, avg_iou, avg_accuracy


# ============================================================================
# STEP 8: VISUALIZATION FUNCTION
# ============================================================================

def visualize_predictions(model, dataset, device, num_samples=3):
    """Visualize model predictions."""
    model.eval()
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    
    for i in range(num_samples):
        # Get a sample
        image, mask = dataset[i]
        image_input = image.unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            output = model(image_input)
            pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
        
        # Convert for visualization
        image_np = image.permute(1, 2, 0).cpu().numpy()
        mask_np = mask.squeeze().cpu().numpy()
        
        # Plot
        axes[i, 0].imshow(image_np)
        axes[i, 0].set_title('Input Image')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(mask_np, cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(pred_mask, cmap='gray')
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('segmentation_results.png', dpi=150, bbox_inches='tight')
    print("Visualization saved as 'segmentation_results.png'")
    plt.close()


# ============================================================================
# STEP 9: TRAINING LOOP
# ============================================================================

NUM_EPOCHS = 20

print(f"\n{'='*70}")
print(f"Starting training for {NUM_EPOCHS} epochs...")
print(f"{'='*70}\n")

best_iou = 0.0
start_time = time.time()

for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    print("-" * 70)
    
    # Train
    train_loss, train_iou, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, device
    )
    
    # Validate
    val_loss, val_iou, val_acc = validate(model, val_loader, criterion, device)
    
    # Print results
    print(f"\nEpoch {epoch + 1} Summary:")
    print(f"  Train - Loss: {train_loss:.4f}, IoU: {train_iou:.4f}, Acc: {train_acc:.4f}")
    print(f"  Val   - Loss: {val_loss:.4f}, IoU: {val_iou:.4f}, Acc: {val_acc:.4f}")
    
    # Learning rate scheduling
    scheduler.step(val_iou)
    
    # Save best model
    if val_iou > best_iou:
        best_iou = val_iou
        torch.save(model.state_dict(), 'best_unet_model.pth')
        print(f"  ✓ New best IoU! Model saved.")
    
    print()

total_time = time.time() - start_time
print(f"{'='*70}")
print(f"Training completed in {total_time // 60:.0f}m {total_time % 60:.0f}s")
print(f"Best validation IoU: {best_iou:.4f}")
print(f"{'='*70}\n")

# ============================================================================
# STEP 10: FINAL EVALUATION
# ============================================================================

# Load best model
model.load_state_dict(torch.load('best_unet_model.pth'))

print("Final Test Evaluation:")
print("="*70)
test_loss, test_iou, test_acc = validate(model, test_loader, criterion, device)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test IoU: {test_iou:.4f}")
print(f"Test Pixel Accuracy: {test_acc:.4f}")

# Visualize some predictions
print("\nGenerating visualizations...")
visualize_predictions(model, test_dataset, device, num_samples=5)

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("BASIC U-NET SEGMENTATION COMPLETE!")
print("="*70)
print("\nKey Takeaways:")
print("1. Implemented U-Net architecture from scratch")
print("2. Used skip connections to preserve spatial information")
print("3. Learned binary segmentation (background vs foreground)")
print("4. Used IoU as the primary evaluation metric")
print("5. Applied pixel-wise cross-entropy loss")
print("\nImportant Concepts:")
print("- Encoder-Decoder architecture")
print("- Skip connections preserve fine details")
print("- IoU measures segmentation quality better than accuracy")
print("- Each pixel is classified independently")
print("\nNext Steps:")
print("- Try Example 2 to use pre-trained encoders")
print("- Experiment with different loss functions (Dice loss)")
print("- Try multi-class segmentation (3+ classes)")
print("="*70)
