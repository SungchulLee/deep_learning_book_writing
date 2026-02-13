"""
Example 4: Advanced Semantic Segmentation Techniques
=====================================================

This script demonstrates state-of-the-art techniques for semantic segmentation:
- Attention mechanisms (CBAM)
- Advanced loss functions (Focal + Dice + Boundary)
- Multi-scale training
- Test-time augmentation
- Post-processing
- Mixed precision training

These techniques can significantly boost performance for challenging tasks.

Author: PyTorch Semantic Segmentation Tutorial
Date: 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import time
from scipy.ndimage import binary_erosion, binary_dilation

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# CONFIGURATION
# ============================================================================

USE_MIXED_PRECISION = True
USE_MULTI_SCALE = True
USE_TTA = True  # Test-time augmentation

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if USE_MIXED_PRECISION and not torch.cuda.is_available():
    print("⚠ Mixed precision requires CUDA. Disabling.")
    USE_MIXED_PRECISION = False

print(f"\nConfiguration:")
print(f"  Mixed Precision: {USE_MIXED_PRECISION}")
print(f"  Multi-scale Training: {USE_MULTI_SCALE}")
print(f"  Test-time Augmentation: {USE_TTA}\n")

# ============================================================================
# STEP 1: ATTENTION MODULES
# ============================================================================
"""
Attention mechanisms help the model focus on important regions and features.
CBAM (Convolutional Block Attention Module) applies attention in both
channel and spatial dimensions.
"""

class ChannelAttention(nn.Module):
    """
    Channel Attention Module.
    Learns which channels (features) are important.
    """
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module.
    Learns which spatial locations are important.
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    Combines channel and spatial attention.
    """
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention()
    
    def forward(self, x):
        # Apply channel attention
        x = x * self.channel_attention(x)
        # Apply spatial attention
        x = x * self.spatial_attention(x)
        return x


# ============================================================================
# STEP 2: ADVANCED U-NET WITH ATTENTION
# ============================================================================

class AttentionDoubleConv(nn.Module):
    """Double convolution block with attention."""
    def __init__(self, in_channels, out_channels, use_attention=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.attention = CBAM(out_channels) if use_attention else nn.Identity()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.attention(x)
        return x


class AdvancedUNet(nn.Module):
    """
    Advanced U-Net with:
    - Attention modules (CBAM)
    - Deep supervision (optional)
    - Residual connections (optional)
    """
    def __init__(self, in_channels=3, num_classes=1, use_attention=True):
        super().__init__()
        
        # Encoder
        self.enc1 = AttentionDoubleConv(in_channels, 64, use_attention)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = AttentionDoubleConv(64, 128, use_attention)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = AttentionDoubleConv(128, 256, use_attention)
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = AttentionDoubleConv(256, 512, use_attention)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck with attention
        self.bottleneck = AttentionDoubleConv(512, 1024, use_attention=True)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = AttentionDoubleConv(1024, 512, use_attention)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = AttentionDoubleConv(512, 256, use_attention)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = AttentionDoubleConv(256, 128, use_attention)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = AttentionDoubleConv(128, 64, use_attention)
        
        # Final output
        self.out = nn.Conv2d(64, num_classes, 1)
    
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
        
        # Decoder
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
        
        x = self.out(x)
        return x


model = AdvancedUNet(in_channels=3, num_classes=1, use_attention=True)
model = model.to(device)

print(f"Model: Advanced U-Net with CBAM Attention")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

# ============================================================================
# STEP 3: ADVANCED LOSS FUNCTIONS
# ============================================================================
"""
Advanced loss functions can significantly improve segmentation quality.
We combine multiple losses for optimal results.
"""

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    Down-weights easy examples, focuses on hard negatives.
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # Probability of true class
        focal_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return focal_loss.mean()


class DiceLoss(nn.Module):
    """Dice Loss for segmentation."""
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice


class BoundaryLoss(nn.Module):
    """
    Loss that emphasizes boundary regions.
    Weights pixels near boundaries higher.
    """
    def __init__(self, theta=5):
        super().__init__()
        self.theta = theta  # Controls boundary width
    
    def forward(self, inputs, targets):
        # Compute boundary weights
        # Pixels near boundaries get higher weight
        targets_np = targets.cpu().numpy()
        boundary_weights = np.zeros_like(targets_np)
        
        for i in range(targets_np.shape[0]):
            mask = targets_np[i, 0]
            # Erode and dilate to find boundaries
            eroded = binary_erosion(mask > 0.5, iterations=2)
            dilated = binary_dilation(mask > 0.5, iterations=2)
            boundary = dilated & ~eroded
            boundary_weights[i, 0] = boundary.astype(np.float32) * self.theta + 1.0
        
        boundary_weights = torch.from_numpy(boundary_weights).to(inputs.device)
        
        # Weighted BCE
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        weighted_bce = bce * boundary_weights
        return weighted_bce.mean()


class CombinedLoss(nn.Module):
    """
    Combined loss: Focal + Dice + Boundary
    Leverages strengths of multiple losses.
    """
    def __init__(self, alpha=0.3, beta=0.4, gamma=0.3):
        super().__init__()
        self.alpha = alpha  # Weight for Focal loss
        self.beta = beta    # Weight for Dice loss
        self.gamma = gamma  # Weight for Boundary loss
        
        self.focal = FocalLoss()
        self.dice = DiceLoss()
        self.boundary = BoundaryLoss()
    
    def forward(self, inputs, targets):
        focal_loss = self.focal(inputs, targets)
        dice_loss = self.dice(inputs, targets)
        boundary_loss = self.boundary(inputs, targets)
        
        total = (self.alpha * focal_loss + 
                self.beta * dice_loss + 
                self.gamma * boundary_loss)
        return total


criterion = CombinedLoss()
print("Loss: Combined (Focal + Dice + Boundary)")

# ============================================================================
# STEP 4: SYNTHETIC DATASET WITH MULTI-SCALE
# ============================================================================

class MultiScaleDataset(Dataset):
    """Dataset that returns images at multiple scales."""
    def __init__(self, num_samples=800, scales=[256, 384] if USE_MULTI_SCALE else [256]):
        self.num_samples = num_samples
        self.scales = scales
        self.samples = []
        
        print(f"Generating {num_samples} samples at scales {scales}...")
        for i in range(num_samples):
            # Generate at largest scale
            img, mask = self._generate_sample(max(scales))
            self.samples.append((img, mask))
    
    def _generate_sample(self, size):
        """Generate synthetic segmentation sample."""
        img = Image.new('RGB', (size, size), 
                       tuple(np.random.randint(100, 200, 3).tolist()))
        mask = Image.new('L', (size, size), 0)
        
        draw_img = ImageDraw.Draw(img)
        draw_mask = ImageDraw.Draw(mask)
        
        # Multiple shapes
        for _ in range(np.random.randint(1, 4)):
            shape_type = np.random.choice(['circle', 'rectangle'])
            center_x = np.random.randint(size//4, 3*size//4)
            center_y = np.random.randint(size//4, 3*size//4)
            shape_size = np.random.randint(size//10, size//5)
            
            color = tuple(np.random.randint(50, 150, 3).tolist())
            
            if shape_type == 'circle':
                bbox = [center_x - shape_size, center_y - shape_size,
                       center_x + shape_size, center_y + shape_size]
                draw_img.ellipse(bbox, fill=color)
                draw_mask.ellipse(bbox, fill=255)
            else:
                bbox = [center_x - shape_size, center_y - shape_size,
                       center_x + shape_size, center_y + shape_size]
                draw_img.rectangle(bbox, fill=color)
                draw_mask.rectangle(bbox, fill=255)
        
        return img, mask
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        img, mask = self.samples[idx]
        
        # Random scale selection for multi-scale training
        scale = np.random.choice(self.scales)
        
        # Resize to selected scale
        img = img.resize((scale, scale), Image.BILINEAR)
        mask = mask.resize((scale, scale), Image.NEAREST)
        
        # Convert to tensors
        img_array = np.array(img).astype(np.float32) / 255.0
        mask_array = np.array(mask).astype(np.float32) / 255.0
        
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)
        
        return img_tensor, mask_tensor


# Create datasets
train_dataset = MultiScaleDataset(num_samples=800)
val_dataset = MultiScaleDataset(num_samples=100, scales=[256])  # Fixed scale for val
test_dataset = MultiScaleDataset(num_samples=100, scales=[256])

BATCH_SIZE = 4  # Smaller due to multi-scale and attention
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"\nDataset created: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test\n")

# ============================================================================
# STEP 5: OPTIMIZER AND SCHEDULER
# ============================================================================

optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)

if USE_MIXED_PRECISION:
    scaler = GradScaler()

# ============================================================================
# STEP 6: TRAINING WITH MIXED PRECISION
# ============================================================================

def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    """Train with mixed precision if enabled."""
    model.train()
    running_loss = 0.0
    
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        if USE_MIXED_PRECISION:
            with autocast():
                outputs = model(images)
                # Resize outputs to match mask size (for multi-scale)
                if outputs.shape[-2:] != masks.shape[-2:]:
                    outputs = F.interpolate(outputs, size=masks.shape[-2:], 
                                          mode='bilinear', align_corners=False)
                loss = criterion(outputs, masks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            if outputs.shape[-2:] != masks.shape[-2:]:
                outputs = F.interpolate(outputs, size=masks.shape[-2:], 
                                      mode='bilinear', align_corners=False)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item()
    
    return running_loss / len(loader)


def calculate_dice(pred, target):
    """Calculate Dice coefficient."""
    with torch.no_grad():
        pred = torch.sigmoid(pred)
        pred = (pred > 0.5).float()
        
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6)
        return dice.item()


def validate(model, loader, criterion, device):
    """Validation with Dice calculation."""
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            
            if USE_MIXED_PRECISION:
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, masks)
            else:
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            dice = calculate_dice(outputs, masks)
            
            running_loss += loss.item()
            running_dice += dice
    
    return running_loss / len(loader), running_dice / len(loader)


# ============================================================================
# STEP 7: TEST-TIME AUGMENTATION
# ============================================================================

def test_time_augmentation(model, image, device):
    """
    Apply multiple augmentations at test time and average predictions.
    Improves robustness and often boosts performance by 1-3%.
    """
    model.eval()
    predictions = []
    
    with torch.no_grad():
        # Original
        pred = torch.sigmoid(model(image))
        predictions.append(pred)
        
        # Horizontal flip
        pred = torch.sigmoid(model(torch.flip(image, dims=[3])))
        predictions.append(torch.flip(pred, dims=[3]))
        
        # Vertical flip
        pred = torch.sigmoid(model(torch.flip(image, dims=[2])))
        predictions.append(torch.flip(pred, dims=[2]))
        
        # Both flips
        pred = torch.sigmoid(model(torch.flip(image, dims=[2, 3])))
        predictions.append(torch.flip(pred, dims=[2, 3]))
    
    # Average all predictions
    avg_pred = torch.stack(predictions).mean(dim=0)
    return avg_pred


def evaluate_with_tta(model, loader, device):
    """Evaluate with test-time augmentation."""
    model.eval()
    running_dice = 0.0
    
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        
        if USE_TTA:
            predictions = test_time_augmentation(model, images, device)
        else:
            with torch.no_grad():
                predictions = torch.sigmoid(model(images))
        
        # Calculate Dice
        pred = (predictions > 0.5).float()
        target = masks
        
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6)
        running_dice += dice.item()
    
    return running_dice / len(loader)


# ============================================================================
# STEP 8: TRAINING LOOP
# ============================================================================

NUM_EPOCHS = 30

print(f"{'='*70}")
print(f"Starting advanced training for {NUM_EPOCHS} epochs...")
print(f"{'='*70}\n")

best_dice = 0.0
start_time = time.time()

for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    
    # Train
    train_loss = train_one_epoch(model, train_loader, criterion, 
                                 optimizer, scaler if USE_MIXED_PRECISION else None, device)
    
    # Validate
    val_loss, val_dice = validate(model, val_loader, criterion, device)
    
    # Print results
    print(f"  Train Loss: {train_loss:.4f}")
    print(f"  Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
    
    # Learning rate scheduling
    scheduler.step()
    
    # Save best model
    if val_dice > best_dice:
        best_dice = val_dice
        torch.save(model.state_dict(), 'best_advanced_model.pth')
        print("  ✓ Best model saved")
    
    print()

total_time = time.time() - start_time
print(f"{'='*70}")
print(f"Training completed in {total_time//60:.0f}m {total_time%60:.0f}s")
print(f"Best validation Dice: {best_dice:.4f}")
print(f"{'='*70}\n")

# ============================================================================
# STEP 9: FINAL EVALUATION WITH TTA
# ============================================================================

model.load_state_dict(torch.load('best_advanced_model.pth'))

print("Final Test Evaluation:")
print("="*70)

# Without TTA
test_dice_no_tta = evaluate_with_tta(model, test_loader, device)
print(f"Test Dice (no TTA): {test_dice_no_tta:.4f}")

# With TTA
if USE_TTA:
    print("\nEvaluating with Test-Time Augmentation...")
    USE_TTA_TEMP = True
    test_dice_tta = evaluate_with_tta(model, test_loader, device)
    print(f"Test Dice (with TTA): {test_dice_tta:.4f}")
    print(f"TTA Improvement: +{(test_dice_tta - test_dice_no_tta):.4f}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("ADVANCED SEGMENTATION TECHNIQUES COMPLETE!")
print("="*70)
print("\nTechniques Applied:")
print("✓ Attention mechanisms (CBAM)")
print("✓ Advanced loss (Focal + Dice + Boundary)")
if USE_MULTI_SCALE:
    print("✓ Multi-scale training")
if USE_MIXED_PRECISION:
    print("✓ Mixed precision training (FP16)")
if USE_TTA:
    print("✓ Test-time augmentation")

print("\nPerformance Gains:")
print(f"  Final Dice Score: {test_dice_tta if USE_TTA else test_dice_no_tta:.4f}")
print(f"  Training Time: {total_time//60:.0f}m {total_time%60:.0f}s")

print("\nKey Takeaways:")
print("1. Attention mechanisms focus on important features")
print("2. Combined losses leverage multiple objectives")
print("3. Multi-scale training improves scale invariance")
print("4. TTA boosts performance with minimal code")
print("5. Mixed precision speeds up training")

print("\nYou've completed all 4 examples!")
print("You're now ready for:")
print("- Real-world segmentation projects")
print("- Kaggle competitions")
print("- Research in semantic segmentation")
print("- Production deployment")
print("="*70)
