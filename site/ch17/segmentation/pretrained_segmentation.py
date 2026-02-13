"""
Example 2: Transfer Learning for Segmentation with Pre-trained Encoders
========================================================================

This script demonstrates how to use pre-trained encoders (ResNet, VGG, etc.)
for semantic segmentation. We'll use the segmentation_models_pytorch library
which provides various architectures with ImageNet pre-trained backbones.

Key Concepts:
- Transfer learning for segmentation
- Pre-trained encoders (ResNet, EfficientNet)
- DeepLabV3+ architecture
- Multi-class segmentation (21 classes)
- Working with real datasets
- Advanced evaluation metrics (mIoU)

Author: PyTorch Semantic Segmentation Tutorial
Date: 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import segmentation_models_pytorch as smp
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# STEP 1: DEVICE CONFIGURATION
# ============================================================================

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

if not torch.cuda.is_available():
    print("⚠ WARNING: CUDA not available. Training will be slow.")
    print("Consider using Google Colab or a GPU instance.\n")

# ============================================================================
# STEP 2: DATASET PREPARATION
# ============================================================================
"""
We'll use PASCAL VOC 2012 dataset - a standard benchmark for segmentation.

PASCAL VOC contains:
- 1,464 training images
- 1,449 validation images  
- 21 classes (20 object classes + background)

Classes: background, aeroplane, bicycle, bird, boat, bottle, bus, car,
         cat, chair, cow, dining table, dog, horse, motorbike, person,
         potted plant, sheep, sofa, train, tv/monitor
"""

# VOC color palette for visualization (RGB values for each class)
VOC_COLORMAP = [
    [0, 0, 0],        # background
    [128, 0, 0],      # aeroplane
    [0, 128, 0],      # bicycle
    [128, 128, 0],    # bird
    [0, 0, 128],      # boat
    [128, 0, 128],    # bottle
    [0, 128, 128],    # bus
    [128, 128, 128],  # car
    [64, 0, 0],       # cat
    [192, 0, 0],      # chair
    [64, 128, 0],     # cow
    [192, 128, 0],    # dining table
    [64, 0, 128],     # dog
    [192, 0, 128],    # horse
    [64, 128, 128],   # motorbike
    [192, 128, 128],  # person
    [0, 64, 0],       # potted plant
    [128, 64, 0],     # sheep
    [0, 192, 0],      # sofa
    [128, 192, 0],    # train
    [0, 64, 128],     # tv/monitor
]

print("Preparing PASCAL VOC 2012 dataset...")
print("(First run will download ~2GB of data)\n")

# Image transformations
# For segmentation, we need to ensure image and mask are transformed together
IMG_SIZE = 512  # DeepLab works well with 512×512

# Define transforms
image_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                        std=[0.229, 0.224, 0.225])
])

mask_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.PILToTensor()
])


class VOCSegmentation(datasets.VOCSegmentation):
    """
    Custom VOC dataset class that applies transforms to both image and mask.
    """
    def __getitem__(self, index):
        img, mask = super().__getitem__(index)
        
        # Apply image transform
        img = image_transform(img)
        
        # Apply mask transform
        mask = mask_transform(mask)
        mask = mask.squeeze(0).long()  # Remove channel dim, convert to long
        
        # VOC uses 255 as ignore index, we keep it
        return img, mask


# Load datasets
train_dataset = VOCSegmentation(
    root='./data',
    year='2012',
    image_set='train',
    download=True
)

val_dataset = VOCSegmentation(
    root='./data',
    year='2012',
    image_set='val',
    download=True
)

# Create data loaders
BATCH_SIZE = 4  # Segmentation is memory-intensive

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=True if torch.cuda.is_available() else False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True if torch.cuda.is_available() else False
)

print(f"Dataset loaded:")
print(f"  Training images: {len(train_dataset)}")
print(f"  Validation images: {len(val_dataset)}")
print(f"  Number of classes: 21")
print(f"  Image size: {IMG_SIZE}×{IMG_SIZE}")
print(f"  Batch size: {BATCH_SIZE}\n")

# ============================================================================
# STEP 3: MODEL SELECTION WITH PRE-TRAINED ENCODER
# ============================================================================
"""
segmentation_models_pytorch (smp) provides many architectures with pre-trained
encoders. We'll use DeepLabV3+ with a ResNet-50 encoder.

Architecture options:
- Unet: Classic U-Net with skip connections
- UnetPlusPlus: Improved U-Net with nested skip connections
- DeepLabV3+: State-of-art with atrous convolutions and ASPP
- FPN: Feature Pyramid Network
- PSPNet: Pyramid Scene Parsing Network

Encoder options:
- resnet18, resnet34, resnet50, resnet101
- efficientnet-b0 to b7
- mobilenet_v2
- vgg16, vgg19
"""

print("Creating DeepLabV3+ model with pre-trained ResNet-50 encoder...")

# Create model with pre-trained encoder
model = smp.DeepLabV3Plus(
    encoder_name="resnet50",           # Choose encoder (resnet50, efficientnet-b0, etc.)
    encoder_weights="imagenet",         # Use ImageNet pre-trained weights
    in_channels=3,                      # Input channels (RGB)
    classes=21,                         # Output classes (VOC has 21)
)

model = model.to(device)

# Model summary
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel: DeepLabV3+ with ResNet-50")
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

print("\nModel Architecture:")
print("  Encoder: ResNet-50 (pre-trained on ImageNet)")
print("  ASPP: Atrous Spatial Pyramid Pooling")
print("  Decoder: Upsampling + refinement")
print("  Output: 21-class segmentation")

# ============================================================================
# STEP 4: LOSS FUNCTION
# ============================================================================
"""
For multi-class segmentation, we use CrossEntropyLoss with:
- ignore_index=255: VOC uses 255 for boundary/uncertain pixels
- Optional class weights for imbalanced classes

Alternative losses:
- Dice Loss: Better for imbalanced classes
- Focal Loss: Focuses on hard examples
- Combined: CE + Dice
"""

# CrossEntropyLoss for multi-class segmentation
criterion = nn.CrossEntropyLoss(ignore_index=255)  # VOC uses 255 for boundaries

# Alternative: Use Dice Loss from smp
# criterion = smp.losses.DiceLoss(mode='multiclass', ignore_index=255)

# Or combined loss
# criterion = smp.losses.DiceLoss(mode='multiclass') + nn.CrossEntropyLoss()

print("\nLoss function: CrossEntropyLoss (ignore boundary pixels)")

# ============================================================================
# STEP 5: OPTIMIZER WITH DISCRIMINATIVE LEARNING RATES
# ============================================================================
"""
For transfer learning, we use different learning rates:
- Encoder (pre-trained): Small LR (0.0001) - fine-tune carefully
- Decoder (random init): Large LR (0.001) - needs more learning

This is called "discriminative learning rates" and is crucial for
effective transfer learning in segmentation.
"""

# Separate encoder and decoder parameters
encoder_params = []
decoder_params = []

for name, param in model.named_parameters():
    if 'encoder' in name:
        encoder_params.append(param)
    else:
        decoder_params.append(param)

# Create optimizer with different learning rates
optimizer = optim.Adam([
    {'params': encoder_params, 'lr': 0.0001},   # Small LR for pre-trained encoder
    {'params': decoder_params, 'lr': 0.001},    # Larger LR for decoder
])

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',      # Maximize mIoU
    factor=0.5,
    patience=3,
    verbose=True
)

print("\nOptimizer: Adam with discriminative learning rates")
print(f"  Encoder LR: 0.0001 (pre-trained, fine-tune)")
print(f"  Decoder LR: 0.001 (random init, learn)")

# ============================================================================
# STEP 6: EVALUATION METRICS
# ============================================================================
"""
For multi-class segmentation, we use:
1. mIoU (mean Intersection over Union): Average IoU across all classes
2. Per-class IoU: IoU for each individual class
3. Pixel Accuracy: Overall pixel classification accuracy

mIoU is the standard metric for segmentation benchmarks.
"""

def calculate_miou(pred, target, num_classes=21, ignore_index=255):
    """
    Calculate mean IoU for multi-class segmentation.
    
    Args:
        pred: Predictions of shape (batch, num_classes, H, W)
        target: Ground truth of shape (batch, H, W)
        num_classes: Number of classes
        ignore_index: Index to ignore (boundaries)
    
    Returns:
        miou: Mean IoU across all classes
        iou_per_class: IoU for each class
    """
    with torch.no_grad():
        # Get predicted class
        pred_class = torch.argmax(pred, dim=1)  # (batch, H, W)
        
        # Initialize IoU storage
        iou_per_class = []
        
        # Calculate IoU for each class
        for cls in range(num_classes):
            # Create binary masks for current class
            pred_mask = (pred_class == cls)
            target_mask = (target == cls)
            
            # Ignore boundary pixels
            valid_mask = (target != ignore_index)
            pred_mask = pred_mask & valid_mask
            target_mask = target_mask & valid_mask
            
            # Calculate intersection and union
            intersection = (pred_mask & target_mask).float().sum()
            union = (pred_mask | target_mask).float().sum()
            
            # Calculate IoU (avoid division by zero)
            if union == 0:
                iou = float('nan')  # Class not present
            else:
                iou = (intersection / union).item()
            
            iou_per_class.append(iou)
        
        # Calculate mean IoU (ignoring NaN values for absent classes)
        valid_ious = [iou for iou in iou_per_class if not np.isnan(iou)]
        miou = np.mean(valid_ious) if valid_ious else 0.0
        
        return miou, iou_per_class


# ============================================================================
# STEP 7: TRAINING FUNCTION
# ============================================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    running_miou = 0.0
    
    pbar = tqdm(loader, desc='Training')
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate mIoU
        miou, _ = calculate_miou(outputs, masks)
        
        # Statistics
        running_loss += loss.item()
        running_miou += miou
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'mIoU': f'{miou:.4f}'
        })
    
    epoch_loss = running_loss / len(loader)
    epoch_miou = running_miou / len(loader)
    
    return epoch_loss, epoch_miou


# ============================================================================
# STEP 8: VALIDATION FUNCTION
# ============================================================================

def validate(model, loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    running_miou = 0.0
    all_ious = [[] for _ in range(21)]  # Store per-class IoUs
    
    with torch.no_grad():
        pbar = tqdm(loader, desc='Validation')
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Calculate metrics
            miou, iou_per_class = calculate_miou(outputs, masks)
            
            # Store per-class IoUs
            for cls, iou in enumerate(iou_per_class):
                if not np.isnan(iou):
                    all_ious[cls].append(iou)
            
            # Statistics
            running_loss += loss.item()
            running_miou += miou
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'mIoU': f'{miou:.4f}'
            })
    
    avg_loss = running_loss / len(loader)
    avg_miou = running_miou / len(loader)
    
    # Calculate per-class average IoU
    avg_iou_per_class = [np.mean(ious) if ious else 0.0 for ious in all_ious]
    
    return avg_loss, avg_miou, avg_iou_per_class


# ============================================================================
# STEP 9: VISUALIZATION FUNCTION
# ============================================================================

def visualize_predictions(model, dataset, device, num_samples=3):
    """Visualize model predictions."""
    model.eval()
    
    # VOC class names
    class_names = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                   'dining table', 'dog', 'horse', 'motorbike', 'person',
                   'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    
    for i in range(num_samples):
        # Get a sample
        image, mask = dataset[np.random.randint(len(dataset))]
        image_input = image.unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            output = model(image_input)
            pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
        
        # Denormalize image for visualization
        image_np = image.cpu().numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_np = std * image_np + mean
        image_np = np.clip(image_np, 0, 1)
        
        mask_np = mask.cpu().numpy()
        mask_np[mask_np == 255] = 0  # Set boundary to background for visualization
        
        # Convert masks to color
        def mask_to_color(mask, colormap):
            h, w = mask.shape
            color_mask = np.zeros((h, w, 3), dtype=np.uint8)
            for i, color in enumerate(colormap):
                color_mask[mask == i] = color
            return color_mask
        
        gt_color = mask_to_color(mask_np, VOC_COLORMAP)
        pred_color = mask_to_color(pred_mask, VOC_COLORMAP)
        
        # Plot
        axes[i, 0].imshow(image_np)
        axes[i, 0].set_title('Input Image')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(gt_color)
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(pred_color)
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('pretrained_segmentation_results.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved as 'pretrained_segmentation_results.png'")
    plt.close()


# ============================================================================
# STEP 10: TRAINING LOOP
# ============================================================================

NUM_EPOCHS = 30

print(f"\n{'='*70}")
print(f"Starting training for {NUM_EPOCHS} epochs...")
print(f"{'='*70}\n")

best_miou = 0.0
start_time = time.time()

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
    print("-" * 70)
    
    # Train
    train_loss, train_miou = train_one_epoch(
        model, train_loader, criterion, optimizer, device
    )
    
    # Validate
    val_loss, val_miou, val_iou_per_class = validate(
        model, val_loader, criterion, device
    )
    
    # Print results
    print(f"\nEpoch {epoch + 1} Summary:")
    print(f"  Train - Loss: {train_loss:.4f}, mIoU: {train_miou:.4f}")
    print(f"  Val   - Loss: {val_loss:.4f}, mIoU: {val_miou:.4f}")
    
    # Learning rate scheduling
    scheduler.step(val_miou)
    
    # Save best model
    if val_miou > best_miou:
        best_miou = val_miou
        torch.save(model.state_dict(), 'best_pretrained_segmentation.pth')
        print(f"  ✓ New best mIoU! Model saved.")
    
    # Print per-class IoU every 5 epochs
    if (epoch + 1) % 5 == 0:
        print("\n  Per-class IoU:")
        class_names = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
                      'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                      'dining table', 'dog', 'horse', 'motorbike', 'person',
                      'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']
        for cls, (name, iou) in enumerate(zip(class_names, val_iou_per_class)):
            if iou > 0:
                print(f"    {name:>15s}: {iou:.4f}")

total_time = time.time() - start_time
print(f"\n{'='*70}")
print(f"Training completed in {total_time // 60:.0f}m {total_time % 60:.0f}s")
print(f"Best validation mIoU: {best_miou:.4f}")
print(f"{'='*70}\n")

# ============================================================================
# STEP 11: FINAL EVALUATION
# ============================================================================

# Load best model
model.load_state_dict(torch.load('best_pretrained_segmentation.pth'))

print("Final Validation Evaluation:")
print("="*70)
val_loss, val_miou, val_iou_per_class = validate(
    model, val_loader, criterion, device
)
print(f"\nFinal Validation mIoU: {val_miou:.4f}")

# Print detailed per-class results
print("\nPer-class IoU:")
print("-" * 70)
class_names = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'dining table', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

for name, iou in zip(class_names, val_iou_per_class):
    if iou > 0:
        print(f"{name:>15s}: {iou:.4f}")

# Visualize predictions
print("\nGenerating visualizations...")
visualize_predictions(model, val_dataset, device, num_samples=5)

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("TRANSFER LEARNING SEGMENTATION COMPLETE!")
print("="*70)
print("\nKey Takeaways:")
print("1. Used pre-trained ResNet-50 encoder from ImageNet")
print("2. Applied DeepLabV3+ architecture with ASPP")
print("3. Used discriminative learning rates (encoder vs decoder)")
print("4. Achieved strong performance on PASCAL VOC (21 classes)")
print("5. Used mIoU as the primary evaluation metric")
print("\nAdvantages of Pre-trained Encoders:")
print("✓ Faster training convergence")
print("✓ Better performance with limited data")
print("✓ Leverages knowledge from millions of ImageNet images")
print("✓ Industry-standard approach")
print("\nNext Steps:")
print("- Try Example 3 for medical image segmentation")
print("- Experiment with different encoders (EfficientNet, VGG)")
print("- Try other architectures (U-Net, FPN)")
print("- Fine-tune hyperparameters")
print("="*70)
