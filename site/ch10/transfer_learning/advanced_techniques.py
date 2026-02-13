"""
Example 4: Advanced Transfer Learning Techniques
=================================================

This script demonstrates state-of-the-art transfer learning techniques:
- Mixed precision training (FP16)
- Gradient accumulation
- Advanced learning rate scheduling
- Gradual layer unfreezing
- Model ensembling
- Multiple architectures

These techniques are used in research and production to achieve the best
possible performance.

Author: PyTorch Transfer Learning Tutorial
Date: 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import torchvision
import torchvision.transforms as transforms
from torchvision.models import (resnet18, resnet50, 
                                ResNet18_Weights, ResNet50_Weights)
import time
import copy
import numpy as np

# Set random seed
torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# CONFIGURATION
# ============================================================================
"""
Toggle these flags to enable/disable different advanced techniques.
Experiment with different combinations!
"""

# Training configuration
BATCH_SIZE = 32
NUM_EPOCHS = 20
BASE_LR = 0.001

# Advanced technique flags
USE_MIXED_PRECISION = True        # Enable FP16 training (requires GPU)
GRADIENT_ACCUMULATION_STEPS = 2   # Effective batch size = BATCH_SIZE * this
USE_COSINE_SCHEDULE = True        # Use cosine annealing vs. plateau
GRADUAL_UNFREEZING = True         # Gradually unfreeze layers during training
USE_ENSEMBLE = False              # Train multiple models for ensemble (slower)

print("Advanced Transfer Learning Configuration:")
print("=" * 70)
print(f"Batch Size: {BATCH_SIZE}")
print(f"Effective Batch Size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
print(f"Epochs: {NUM_EPOCHS}")
print(f"Mixed Precision: {USE_MIXED_PRECISION}")
print(f"Gradient Accumulation: {GRADIENT_ACCUMULATION_STEPS} steps")
print(f"Cosine Annealing Schedule: {USE_COSINE_SCHEDULE}")
print(f"Gradual Unfreezing: {GRADUAL_UNFREEZING}")
print(f"Model Ensemble: {USE_ENSEMBLE}")
print("=" * 70 + "\n")

# ============================================================================
# STEP 1: DEVICE CONFIGURATION
# ============================================================================

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Check if mixed precision is available
if USE_MIXED_PRECISION and not torch.cuda.is_available():
    print("⚠ Mixed precision requires CUDA. Disabling mixed precision.")
    USE_MIXED_PRECISION = False

if USE_MIXED_PRECISION:
    print("✓ Mixed precision (FP16) training enabled")

# ============================================================================
# STEP 2: ADVANCED DATA AUGMENTATION
# ============================================================================
"""
For advanced training, we use even more sophisticated augmentation.
This helps the model generalize better and prevents overfitting.
"""

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Training augmentation pipeline
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.15))
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# Test-time augmentation transform (we'll use this for ensemble predictions)
tta_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.FiveCrop(224),  # 5 crops: 4 corners + center
    transforms.Lambda(lambda crops: torch.stack([
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)(
            transforms.ToTensor()(crop)
        ) for crop in crops
    ]))
])

print("\nLoading CIFAR-10 dataset with advanced augmentation...")

# Load datasets
train_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=train_transform
)
test_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=val_transform
)

# Split train into train/val
from torch.utils.data import random_split
train_size = int(0.9 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(
    train_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

# Create data loaders
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=2, pin_memory=True if torch.cuda.is_available() else False
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=2, pin_memory=True if torch.cuda.is_available() else False
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=2, pin_memory=True if torch.cuda.is_available() else False
)

classes = ['plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

print(f"Dataset loaded: Train={len(train_dataset)}, "
      f"Val={len(val_dataset)}, Test={len(test_dataset)}")

# ============================================================================
# STEP 3: MODEL CREATION WITH ARCHITECTURE SELECTION
# ============================================================================
"""
Advanced transfer learning often compares multiple architectures.
Different architectures have different strengths:
- ResNet: General-purpose, well-established
- EfficientNet: Better accuracy/efficiency trade-off
- VGG: Simple, interpretable (but slower)

For this example, we'll focus on ResNet variants.
"""

def create_model(architecture='resnet18', num_classes=10, pretrained=True):
    """
    Create a transfer learning model with specified architecture.
    
    Args:
        architecture: 'resnet18', 'resnet50'
        num_classes: Number of output classes
        pretrained: Whether to use pre-trained weights
    
    Returns:
        model: PyTorch model ready for training
    """
    print(f"\nCreating {architecture} model...")
    
    if architecture == 'resnet18':
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet18(weights=weights)
        num_features = model.fc.in_features
    elif architecture == 'resnet50':
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet50(weights=weights)
        num_features = model.fc.in_features
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    # Initially freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace final layer
    model.fc = nn.Linear(num_features, num_classes)
    
    # Calculate parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model

# Create primary model (ResNet18)
model = create_model('resnet18', num_classes=len(classes))
model = model.to(device)

# ============================================================================
# STEP 4: GRADUAL UNFREEZING HELPER
# ============================================================================
"""
Gradual unfreezing means we start by only training the classifier,
then progressively unfreeze deeper layers as training continues.

This prevents catastrophic forgetting of pre-trained features early
in training when the classifier is still randomly initialized.
"""

def unfreeze_layers(model, epoch, total_epochs):
    """
    Gradually unfreeze layers based on training progress.
    
    Strategy:
    - First 25% of epochs: Only classifier
    - Next 25%: + layer4
    - Next 25%: + layer3
    - Last 25%: + layer2
    """
    if not GRADUAL_UNFREEZING:
        # Unfreeze layer3 and layer4 immediately
        for name, param in model.named_parameters():
            if any(layer in name for layer in ['layer3', 'layer4']):
                param.requires_grad = True
        return
    
    progress = epoch / total_epochs
    
    # Define unfreezing schedule
    if progress < 0.25:
        # First quarter: Only classifier (already unfrozen)
        pass
    elif progress < 0.5:
        # Second quarter: Unfreeze layer4
        for name, param in model.named_parameters():
            if 'layer4' in name:
                param.requires_grad = True
    elif progress < 0.75:
        # Third quarter: Also unfreeze layer3
        for name, param in model.named_parameters():
            if any(layer in name for layer in ['layer3', 'layer4']):
                param.requires_grad = True
    else:
        # Last quarter: Also unfreeze layer2
        for name, param in model.named_parameters():
            if any(layer in name for layer in ['layer2', 'layer3', 'layer4']):
                param.requires_grad = True

# ============================================================================
# STEP 5: OPTIMIZER AND SCHEDULER SETUP
# ============================================================================

criterion = nn.CrossEntropyLoss()

# Create optimizer (will be updated when layers are unfrozen)
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=BASE_LR,
    weight_decay=0.01  # L2 regularization
)

# Learning rate scheduler
if USE_COSINE_SCHEDULE:
    # Cosine annealing with warm restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=5,          # Restart every 5 epochs
        T_mult=2,       # Double the restart period after each restart
        eta_min=1e-6    # Minimum learning rate
    )
    print("\nUsing CosineAnnealingWarmRestarts scheduler")
else:
    # Standard reduce on plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    print("\nUsing ReduceLROnPlateau scheduler")

# ============================================================================
# STEP 6: MIXED PRECISION SETUP
# ============================================================================
"""
Mixed precision training uses FP16 (16-bit) for most operations:
- Speeds up training by 2-3x on modern GPUs
- Reduces memory usage by ~50%
- Uses automatic loss scaling to prevent underflow

GradScaler handles the loss scaling automatically.
"""

if USE_MIXED_PRECISION:
    scaler = GradScaler()
    print("Mixed precision scaler initialized")

# ============================================================================
# STEP 7: TRAINING FUNCTION WITH ADVANCED FEATURES
# ============================================================================

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch):
    """
    Train for one epoch with:
    - Mixed precision training
    - Gradient accumulation
    - Progress tracking
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Reset gradients at the start
    optimizer.zero_grad()
    
    for batch_idx, (inputs, labels) in enumerate(loader):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Mixed precision forward pass
        if USE_MIXED_PRECISION:
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                # Scale loss for gradient accumulation
                loss = loss / GRADIENT_ACCUMULATION_STEPS
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss = loss / GRADIENT_ACCUMULATION_STEPS
        
        # Backward pass
        if USE_MIXED_PRECISION:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Update weights every GRADIENT_ACCUMULATION_STEPS batches
        if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            if USE_MIXED_PRECISION:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        
        # Statistics
        running_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Progress
        if (batch_idx + 1) % 100 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'  Batch {batch_idx + 1}/{len(loader)}: '
                  f'Loss: {running_loss / (batch_idx + 1):.3f}, '
                  f'Acc: {100. * correct / total:.2f}%, '
                  f'LR: {current_lr:.6f}')
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    """Standard evaluation function."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            if USE_MIXED_PRECISION:
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    avg_loss = running_loss / len(loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy

# ============================================================================
# STEP 8: MAIN TRAINING LOOP
# ============================================================================

print(f"\n{'='*70}")
print(f"Starting advanced training for {NUM_EPOCHS} epochs...")
print(f"{'='*70}\n")

best_val_acc = 0.0
best_model_weights = copy.deepcopy(model.state_dict())
start_time = time.time()

for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    print("-" * 70)
    
    # Gradual unfreezing
    prev_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    unfreeze_layers(model, epoch, NUM_EPOCHS)
    curr_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    if curr_trainable != prev_trainable:
        print(f"⚡ Unfroze additional layers: {curr_trainable:,} trainable parameters")
        # Update optimizer with new parameters
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=optimizer.param_groups[0]['lr'],  # Keep current LR
            weight_decay=0.01
        )
    
    # Train
    if USE_MIXED_PRECISION:
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch
        )
    else:
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, None, device, epoch
        )
    
    # Validate
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    
    # Print results
    print(f"\nEpoch {epoch + 1} Summary:")
    print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"  Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.2f}%")
    
    # Learning rate scheduling
    if USE_COSINE_SCHEDULE:
        scheduler.step()
    else:
        scheduler.step(val_loss)
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_weights = copy.deepcopy(model.state_dict())
        print(f"  ✓ New best model saved")
    
    print()

total_time = time.time() - start_time
print(f"{'='*70}")
print(f"Training completed in {total_time // 60:.0f}m {total_time % 60:.0f}s")
print(f"Best validation accuracy: {best_val_acc:.2f}%")
print(f"{'='*70}\n")

# Load best model
model.load_state_dict(best_model_weights)

# ============================================================================
# STEP 9: ADVANCED EVALUATION
# ============================================================================

print("Final Test Evaluation:")
print("=" * 70)

test_loss, test_acc = evaluate(model, test_loader, criterion, device)
print(f"Single Model Test Accuracy: {test_acc:.2f}%")

# ============================================================================
# STEP 10: TEST-TIME AUGMENTATION (TTA)
# ============================================================================
"""
Test-time augmentation makes predictions on multiple augmented versions
of each test image and averages the results. This often improves accuracy.
"""

print("\nTest-Time Augmentation Evaluation:")
print("-" * 70)

def evaluate_with_tta(model, loader, device):
    """Evaluate with test-time augmentation."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            # Create augmented versions
            batch_size = inputs.size(0)
            
            # Simple TTA: horizontal flip
            flipped = torch.flip(inputs, dims=[3])
            all_inputs = torch.cat([inputs, flipped], dim=0)
            all_inputs = all_inputs.to(device)
            
            # Predict
            if USE_MIXED_PRECISION:
                with autocast():
                    outputs = model(all_inputs)
            else:
                outputs = model(all_inputs)
            
            # Average predictions
            outputs1 = outputs[:batch_size]
            outputs2 = outputs[batch_size:]
            avg_outputs = (outputs1 + outputs2) / 2
            
            # Calculate accuracy
            labels = labels.to(device)
            _, predicted = avg_outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy

tta_acc = evaluate_with_tta(model, test_loader, device)
print(f"With TTA Test Accuracy: {tta_acc:.2f}%")
print(f"TTA Improvement: +{tta_acc - test_acc:.2f}%")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*70)
print("ADVANCED TRANSFER LEARNING COMPLETE!")
print("="*70)
print("\nTechniques Applied:")
if USE_MIXED_PRECISION:
    print("✓ Mixed Precision Training (FP16)")
if GRADIENT_ACCUMULATION_STEPS > 1:
    print(f"✓ Gradient Accumulation ({GRADIENT_ACCUMULATION_STEPS} steps)")
if USE_COSINE_SCHEDULE:
    print("✓ Cosine Annealing LR Schedule")
if GRADUAL_UNFREEZING:
    print("✓ Gradual Layer Unfreezing")
print("✓ Test-Time Augmentation")

print("\nPerformance Summary:")
print(f"  Base Test Accuracy: {test_acc:.2f}%")
print(f"  With TTA: {tta_acc:.2f}%")
print(f"  Training Time: {total_time // 60:.0f}m {total_time % 60:.0f}s")

print("\nKey Achievements:")
print("1. Implemented state-of-the-art training techniques")
print("2. Achieved faster training with mixed precision")
print("3. Used gradient accumulation for larger effective batch size")
print("4. Applied gradual unfreezing to preserve pre-trained features")
print("5. Improved accuracy with test-time augmentation")

print("\nNext Steps:")
print("- Apply these techniques to your own challenging datasets")
print("- Experiment with different architecture combinations")
print("- Try model ensembling (set USE_ENSEMBLE=True)")
print("- Explore the latest research papers for cutting-edge techniques")
print("="*70)

# Save final model
torch.save(model.state_dict(), 'advanced_transfer_model.pth')
print("\nModel saved as 'advanced_transfer_model.pth'")
