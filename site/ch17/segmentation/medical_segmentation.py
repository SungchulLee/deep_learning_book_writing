"""
Example 3: Medical Image Segmentation
======================================

This script demonstrates medical image segmentation with domain-specific
techniques, focusing on lesion segmentation in dermoscopic images.

Key Concepts:
- Dice loss for medical imaging
- Handling extreme class imbalance
- Clinical metrics (Dice, Sensitivity, Specificity)
- Medical imaging preprocessing
- Proper validation strategies

Author: PyTorch Semantic Segmentation Tutorial
Date: 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFilter
import time

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# STEP 1: DEVICE CONFIGURATION
# ============================================================================

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# ============================================================================
# STEP 2: SYNTHETIC MEDICAL DATASET
# ============================================================================
"""
We'll create a synthetic dataset simulating dermoscopic images of skin lesions.
In practice, you would use real medical datasets like ISIC, DRIVE, BraTS, etc.

Characteristics of medical images:
- Extreme class imbalance (lesion is small compared to background)
- Irregular shapes
- Varying contrast
- Noise and artifacts
"""

class SyntheticMedicalDataset(Dataset):
    """
    Generate synthetic dermoscopic-like images with lesions.
    
    Simulates skin lesion segmentation task with:
    - Irregular lesion shapes
    - Varying sizes (5-15% of image area)
    - Noise and blur (realistic artifacts)
    - High class imbalance
    """
    
    def __init__(self, num_samples=1000, image_size=256):
        self.num_samples = num_samples
        self.image_size = image_size
        
        print(f"Generating {num_samples} synthetic medical images...")
        self.samples = []
        for i in range(num_samples):
            img, mask = self._generate_sample()
            self.samples.append((img, mask))
            if (i + 1) % 200 == 0:
                print(f"  Generated {i + 1}/{num_samples} images")
        
        # Calculate class distribution
        total_pixels = 0
        lesion_pixels = 0
        for _, mask in self.samples:
            mask_np = np.array(mask)
            total_pixels += mask_np.size
            lesion_pixels += np.sum(mask_np > 0)
        
        self.lesion_ratio = lesion_pixels / total_pixels
        print(f"\nDataset statistics:")
        print(f"  Lesion pixels: {self.lesion_ratio:.2%}")
        print(f"  Background pixels: {1-self.lesion_ratio:.2%}")
        print(f"  Class imbalance ratio: 1:{(1-self.lesion_ratio)/self.lesion_ratio:.1f}")
    
    def _generate_sample(self):
        """Generate a single image-mask pair resembling dermoscopic image."""
        # Create skin-colored background
        skin_color = (
            np.random.randint(200, 240),  # R
            np.random.randint(160, 200),  # G
            np.random.randint(140, 180)   # B
        )
        img = Image.new('RGB', (self.image_size, self.image_size), skin_color)
        mask = Image.new('L', (self.image_size, self.image_size), 0)
        
        draw_img = ImageDraw.Draw(img)
        draw_mask = ImageDraw.Draw(mask)
        
        # Lesion color (darker, brownish)
        lesion_color = (
            np.random.randint(80, 140),
            np.random.randint(60, 100),
            np.random.randint(40, 80)
        )
        
        # Create irregular lesion shape
        center_x = np.random.randint(64, self.image_size - 64)
        center_y = np.random.randint(64, self.image_size - 64)
        
        # Irregular ellipse with multiple distortions
        num_points = 20
        angles = np.linspace(0, 2*np.pi, num_points)
        base_radius = np.random.randint(30, 50)
        
        # Create irregular boundary
        radii = base_radius + np.random.randint(-15, 15, num_points)
        points = [
            (
                center_x + int(radii[i] * np.cos(angles[i])),
                center_y + int(radii[i] * np.sin(angles[i]))
            )
            for i in range(num_points)
        ]
        
        # Draw lesion
        draw_img.polygon(points, fill=lesion_color)
        draw_mask.polygon(points, fill=255)
        
        # Add realistic artifacts
        # 1. Slight blur (simulate camera/skin texture)
        img = img.filter(ImageFilter.GaussianBlur(radius=1))
        
        # 2. Add noise to image
        img_array = np.array(img).astype(np.float32)
        noise = np.random.normal(0, 5, img_array.shape)
        img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_array)
        
        return img, mask
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        img, mask = self.samples[idx]
        
        # Convert to tensors
        img_array = np.array(img).astype(np.float32) / 255.0
        mask_array = np.array(mask).astype(np.float32) / 255.0
        
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # (3, H, W)
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)     # (1, H, W)
        
        return img_tensor, mask_tensor


# Create datasets with patient-level split
# Important: In real medical AI, split by patient, not by image!
print("Creating medical imaging datasets...\n")
train_dataset = SyntheticMedicalDataset(num_samples=800, image_size=256)
val_dataset = SyntheticMedicalDataset(num_samples=100, image_size=256)
test_dataset = SyntheticMedicalDataset(num_samples=100, image_size=256)

# Create data loaders
BATCH_SIZE = 16
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ============================================================================
# STEP 3: U-NET FOR MEDICAL IMAGING
# ============================================================================
"""
U-Net is the gold standard for medical image segmentation.
It was originally designed for biomedical image segmentation.
"""

class DoubleConv(nn.Module):
    """Double convolution block."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
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


class MedicalUNet(nn.Module):
    """
    U-Net optimized for medical imaging.
    Slightly deeper than basic U-Net for better feature extraction.
    """
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        
        # Encoder
        self.enc1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        
        # Final output (no activation, will use BCEWithLogitsLoss)
        self.out = nn.Conv2d(64, num_classes, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        x = self.pool1(enc1)
        
        enc2 = self.enc2(x)
        x = self.pool2(enc2)
        
        enc3 = self.enc3(x)
        x = self.pool3(enc3)
        
        enc4 = self.enc4(x)
        x = self.pool4(enc4)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder with skip connections
        x = self.upconv4(x)
        x = torch.cat([x, enc4], dim=1)
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
        
        # Final output (logits)
        x = self.out(x)
        
        return x


model = MedicalUNet(in_channels=3, num_classes=1)  # Binary: lesion vs background
model = model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"\nModel: Medical U-Net")
print(f"Total parameters: {total_params:,}")

# ============================================================================
# STEP 4: DICE LOSS FOR MEDICAL IMAGING
# ============================================================================
"""
Dice Loss is the standard loss for medical image segmentation.

Advantages over Cross-Entropy:
1. Directly optimizes Dice coefficient (the evaluation metric)
2. Handles class imbalance naturally
3. Focuses on overlap, not individual pixels
4. Better for small objects

Formula:
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    Dice Loss = 1 - Dice
"""

class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation.
    """
    def __init__(self, smooth=1e-6):
        """
        Args:
            smooth: Smoothing constant to avoid division by zero
        """
        super().__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: Logits of shape (batch, 1, H, W)
            targets: Ground truth of shape (batch, 1, H, W) with values in [0, 1]
        
        Returns:
            dice_loss: Scalar loss value
        """
        # Apply sigmoid to convert logits to probabilities
        predictions = torch.sigmoid(predictions)
        
        # Flatten predictions and targets
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        # Calculate intersection and union
        intersection = (predictions * targets).sum()
        union = predictions.sum() + targets.sum()
        
        # Calculate Dice coefficient
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # Dice loss is 1 - Dice
        return 1 - dice


class CombinedLoss(nn.Module):
    """
    Combined Dice + BCE loss.
    
    BCE helps with pixel-wise accuracy, Dice helps with overlap.
    This combination often works best in practice.
    """
    def __init__(self, alpha=0.5, beta=0.5):
        """
        Args:
            alpha: Weight for BCE loss
            beta: Weight for Dice loss
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
    
    def forward(self, predictions, targets):
        bce_loss = self.bce(predictions, targets)
        dice_loss = self.dice(predictions, targets)
        return self.alpha * bce_loss + self.beta * dice_loss


# Use combined loss for best results
criterion = CombinedLoss(alpha=0.5, beta=0.5)

print("\nLoss function: Combined Dice + BCE Loss")
print("  Dice Loss: Optimizes overlap (better for small objects)")
print("  BCE Loss: Pixel-wise accuracy")

# ============================================================================
# STEP 5: MEDICAL METRICS
# ============================================================================
"""
Medical segmentation uses different metrics than general segmentation:
- Dice Coefficient: Primary metric
- Sensitivity (Recall): How many lesion pixels are found?
- Specificity: How many background pixels are correct?
- Precision: Of predicted lesion pixels, how many are correct?
"""

def calculate_dice(pred, target, threshold=0.5, smooth=1e-6):
    """Calculate Dice coefficient."""
    with torch.no_grad():
        pred = torch.sigmoid(pred)
        pred = (pred > threshold).float()
        
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        
        dice = (2. * intersection + smooth) / (union + smooth)
        return dice.item()


def calculate_medical_metrics(pred, target, threshold=0.5):
    """
    Calculate comprehensive medical metrics.
    
    Returns:
        dict with Dice, Sensitivity, Specificity, Precision
    """
    with torch.no_grad():
        pred = torch.sigmoid(pred)
        pred = (pred > threshold).float()
        
        pred = pred.view(-1)
        target = target.view(-1)
        
        # Calculate confusion matrix components
        tp = ((pred == 1) & (target == 1)).float().sum()  # True Positives
        tn = ((pred == 0) & (target == 0)).float().sum()  # True Negatives
        fp = ((pred == 1) & (target == 0)).float().sum()  # False Positives
        fn = ((pred == 0) & (target == 1)).float().sum()  # False Negatives
        
        # Calculate metrics
        smooth = 1e-6
        
        dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
        sensitivity = (tp + smooth) / (tp + fn + smooth)  # Recall, TPR
        specificity = (tn + smooth) / (tn + fp + smooth)  # TNR
        precision = (tp + smooth) / (tp + fp + smooth)    # PPV
        
        return {
            'dice': dice.item(),
            'sensitivity': sensitivity.item(),
            'specificity': specificity.item(),
            'precision': precision.item()
        }


# ============================================================================
# STEP 6: OPTIMIZER
# ============================================================================

optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=5, verbose=True
)

# ============================================================================
# STEP 7: TRAINING FUNCTION
# ============================================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate Dice
        dice = calculate_dice(outputs, masks)
        
        # Statistics
        running_loss += loss.item()
        running_dice += dice
    
    epoch_loss = running_loss / len(loader)
    epoch_dice = running_dice / len(loader)
    
    return epoch_loss, epoch_dice


# ============================================================================
# STEP 8: VALIDATION FUNCTION
# ============================================================================

def validate(model, loader, criterion, device):
    """Validate with comprehensive medical metrics."""
    model.eval()
    running_loss = 0.0
    metrics_sum = {'dice': 0.0, 'sensitivity': 0.0, 'specificity': 0.0, 'precision': 0.0}
    
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Calculate metrics
            metrics = calculate_medical_metrics(outputs, masks)
            
            # Statistics
            running_loss += loss.item()
            for key in metrics_sum:
                metrics_sum[key] += metrics[key]
    
    # Average metrics
    avg_loss = running_loss / len(loader)
    avg_metrics = {key: val / len(loader) for key, val in metrics_sum.items()}
    
    return avg_loss, avg_metrics


# ============================================================================
# STEP 9: VISUALIZATION
# ============================================================================

def visualize_medical_predictions(model, dataset, device, num_samples=3):
    """Visualize predictions with medical overlay style."""
    model.eval()
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    
    for i in range(num_samples):
        image, mask = dataset[np.random.randint(len(dataset))]
        image_input = image.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(image_input)
            pred_prob = torch.sigmoid(output).squeeze().cpu().numpy()
            pred_mask = (pred_prob > 0.5).astype(np.float32)
        
        # Convert for visualization
        image_np = image.permute(1, 2, 0).cpu().numpy()
        mask_np = mask.squeeze().cpu().numpy()
        
        # Create overlay
        overlay = image_np.copy()
        # Ground truth in green
        overlay[mask_np > 0.5] = [0, 1, 0]
        
        overlay_pred = image_np.copy()
        # Prediction in red
        overlay_pred[pred_mask > 0.5] = [1, 0, 0]
        
        # Plot
        axes[i, 0].imshow(image_np)
        axes[i, 0].set_title('Input Image')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(mask_np, cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(pred_prob, cmap='hot')
        axes[i, 2].set_title('Prediction Probability')
        axes[i, 2].axis('off')
        
        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title('Overlay (GT=Green)')
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig('medical_segmentation_results.png', dpi=150, bbox_inches='tight')
    print("Visualization saved as 'medical_segmentation_results.png'")
    plt.close()


# ============================================================================
# STEP 10: TRAINING LOOP
# ============================================================================

NUM_EPOCHS = 30

print(f"\n{'='*70}")
print(f"Starting medical segmentation training for {NUM_EPOCHS} epochs...")
print(f"{'='*70}\n")

best_dice = 0.0
start_time = time.time()

for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    print("-" * 70)
    
    # Train
    train_loss, train_dice = train_one_epoch(
        model, train_loader, criterion, optimizer, device
    )
    
    # Validate
    val_loss, val_metrics = validate(model, val_loader, criterion, device)
    
    # Print results
    print(f"Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}")
    print(f"Val   - Loss: {val_loss:.4f}, Dice: {val_metrics['dice']:.4f}, "
          f"Sens: {val_metrics['sensitivity']:.4f}, Spec: {val_metrics['specificity']:.4f}")
    
    # Learning rate scheduling
    scheduler.step(val_metrics['dice'])
    
    # Save best model
    if val_metrics['dice'] > best_dice:
        best_dice = val_metrics['dice']
        torch.save(model.state_dict(), 'best_medical_segmentation.pth')
        print("✓ New best Dice! Model saved.")
    
    print()

total_time = time.time() - start_time
print(f"{'='*70}")
print(f"Training completed in {total_time // 60:.0f}m {total_time % 60:.0f}s")
print(f"Best validation Dice: {best_dice:.4f}")
print(f"{'='*70}\n")

# ============================================================================
# STEP 11: FINAL TEST EVALUATION
# ============================================================================

model.load_state_dict(torch.load('best_medical_segmentation.pth'))

print("Final Test Evaluation (Clinical Metrics):")
print("="*70)
test_loss, test_metrics = validate(model, test_loader, criterion, device)

print(f"Test Dice Coefficient: {test_metrics['dice']:.4f}")
print(f"Test Sensitivity (Recall): {test_metrics['sensitivity']:.4f}")
print(f"Test Specificity: {test_metrics['specificity']:.4f}")
print(f"Test Precision: {test_metrics['precision']:.4f}")

print("\nClinical Interpretation:")
print(f"- Dice {test_metrics['dice']:.2%}: Overall segmentation quality")
print(f"- Sensitivity {test_metrics['sensitivity']:.2%}: Detected {test_metrics['sensitivity']:.1%} of lesion pixels")
print(f"- Specificity {test_metrics['specificity']:.2%}: Correctly identified {test_metrics['specificity']:.1%} of healthy skin")

# Visualize predictions
print("\nGenerating visualizations...")
visualize_medical_predictions(model, test_dataset, device, num_samples=5)

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("MEDICAL IMAGE SEGMENTATION COMPLETE!")
print("="*70)
print("\nKey Takeaways:")
print("1. Used Dice Loss to handle extreme class imbalance")
print("2. Evaluated with clinical metrics (Dice, Sensitivity, Specificity)")
print("3. Applied U-Net architecture (gold standard for medical imaging)")
print("4. Handled small objects better than cross-entropy")
print("5. Combined Dice + BCE for optimal results")
print("\nMedical AI Best Practices:")
print("✓ Use Dice loss for imbalanced medical data")
print("✓ Report Sensitivity and Specificity, not just accuracy")
print("✓ Split data by patient, not by image")
print("✓ Visualize predictions for clinical validation")
print("✓ Consider false negative cost (missing lesions is dangerous)")
print("\nNext Steps:")
print("- Try Example 4 for advanced techniques")
print("- Apply to real medical datasets (ISIC, DRIVE, BraTS)")
print("- Implement uncertainty quantification")
print("- Add 3D volumetric segmentation (CT/MRI)")
print("="*70)
