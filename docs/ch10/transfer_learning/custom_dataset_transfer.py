"""
Example 3: Transfer Learning with Custom Datasets
==================================================

This script demonstrates how to apply transfer learning to your own custom datasets.
We'll create a synthetic dataset for demonstration, but the code is designed to
work with any properly organized image dataset.

Key Concepts:
- Custom Dataset classes
- Handling imbalanced datasets
- Proper train/val/test splits
- Advanced data augmentation
- Evaluation metrics beyond accuracy

Author: PyTorch Transfer Learning Tutorial
Date: 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.datasets import ImageFolder
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os
from PIL import Image
import time
import copy

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# STEP 1: CREATE A SYNTHETIC DATASET FOR DEMONSTRATION
# ============================================================================
"""
In a real application, you would skip this step and use your own data.
We're creating a synthetic dataset to demonstrate the concepts.

For your own data:
1. Organize images in folders by class
2. Split into train/val/test folders
3. Update the data_dir path
"""

def create_synthetic_dataset():
    """
    Create a synthetic dataset to demonstrate custom dataset handling.
    In practice, you would use your own images organized in folders.
    """
    print("Creating synthetic dataset for demonstration...")
    
    # Create directory structure
    base_dir = './custom_dataset'
    splits = ['train', 'val', 'test']
    classes = ['cat', 'dog', 'bird']  # Example classes
    
    # Number of images per class (imbalanced to show handling techniques)
    class_counts = {
        'train': {'cat': 500, 'dog': 300, 'bird': 200},  # Imbalanced!
        'val': {'cat': 100, 'dog': 60, 'bird': 40},
        'test': {'cat': 100, 'dog': 60, 'bird': 40}
    }
    
    for split in splits:
        for cls in classes:
            path = os.path.join(base_dir, split, cls)
            os.makedirs(path, exist_ok=True)
            
            # Create synthetic images (random colored squares)
            n_images = class_counts[split][cls]
            for i in range(n_images):
                # Create a random image (in practice, these would be your real images)
                img = Image.new('RGB', (64, 64), 
                               color=(np.random.randint(0, 255),
                                     np.random.randint(0, 255),
                                     np.random.randint(0, 255)))
                img.save(os.path.join(path, f'{cls}_{i:04d}.jpg'))
    
    print(f"Synthetic dataset created at: {base_dir}")
    print("\nDataset structure:")
    for split in splits:
        print(f"\n{split}:")
        for cls in classes:
            path = os.path.join(base_dir, split, cls)
            n_files = len(os.listdir(path))
            print(f"  {cls}: {n_files} images")
    
    return base_dir

# Create or use existing dataset
data_dir = create_synthetic_dataset()

# ============================================================================
# STEP 2: DEVICE CONFIGURATION
# ============================================================================

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

# ============================================================================
# STEP 3: ADVANCED DATA AUGMENTATION
# ============================================================================
"""
For custom datasets, especially with limited data, aggressive augmentation
is crucial. We'll use a comprehensive set of transformations.

Augmentation Techniques:
1. Geometric: Rotation, Flip, Crop, Affine
2. Color: ColorJitter, Grayscale
3. Noise/Blur: GaussianBlur (simulated)
4. Normalization: ImageNet statistics
"""

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Training transform with heavy augmentation
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Random zoom
    transforms.RandomHorizontalFlip(p=0.5),               # 50% chance of flip
    transforms.RandomRotation(degrees=15),                # Rotate up to 15 degrees
    transforms.ColorJitter(                               # Random color changes
        brightness=0.3,
        contrast=0.3,
        saturation=0.3,
        hue=0.1
    ),
    transforms.RandomGrayscale(p=0.1),                    # 10% chance of grayscale
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    transforms.RandomErasing(p=0.2)                       # Random erasing (cutout)
])

# Validation/Test transform (no augmentation)
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

print("\nData augmentation configured:")
print("Training: Random crop, flip, rotation, color jitter, erasing")
print("Validation/Test: Center crop only (no augmentation)")

# ============================================================================
# STEP 4: LOAD CUSTOM DATASET USING IMAGEFOLDER
# ============================================================================
"""
PyTorch's ImageFolder is perfect for datasets organized in class folders.
It automatically:
1. Discovers classes from folder names
2. Assigns labels based on alphabetical order
3. Loads images on demand

Alternative: You can create a custom Dataset class for more control.
"""

print("\nLoading custom dataset...")

# Load datasets
train_dataset = ImageFolder(
    root=os.path.join(data_dir, 'train'),
    transform=train_transform
)

val_dataset = ImageFolder(
    root=os.path.join(data_dir, 'val'),
    transform=val_transform
)

test_dataset = ImageFolder(
    root=os.path.join(data_dir, 'test'),
    transform=val_transform
)

# Get class information
classes = train_dataset.classes
class_to_idx = train_dataset.class_to_idx

print(f"\nDataset Information:")
print(f"Number of classes: {len(classes)}")
print(f"Classes: {classes}")
print(f"Class to index mapping: {class_to_idx}")
print(f"\nDataset sizes:")
print(f"Training: {len(train_dataset)} images")
print(f"Validation: {len(val_dataset)} images")
print(f"Test: {len(test_dataset)} images")

# ============================================================================
# STEP 5: ANALYZE CLASS DISTRIBUTION
# ============================================================================
"""
Understanding class distribution is crucial for handling imbalanced datasets.
"""

def get_class_distribution(dataset):
    """Calculate the number of samples per class."""
    class_counts = {}
    for _, label in dataset.imgs:
        class_name = dataset.classes[label]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    return class_counts

train_class_counts = get_class_distribution(train_dataset)

print("\nTraining set class distribution:")
print("-" * 50)
total = sum(train_class_counts.values())
for cls, count in sorted(train_class_counts.items()):
    percentage = 100 * count / total
    print(f"{cls:>10s}: {count:>4d} images ({percentage:>5.1f}%)")

# Calculate class imbalance ratio
max_count = max(train_class_counts.values())
min_count = min(train_class_counts.values())
imbalance_ratio = max_count / min_count
print(f"\nImbalance ratio: {imbalance_ratio:.2f}:1")
if imbalance_ratio > 2:
    print("⚠ Dataset is imbalanced! Will use weighted sampling/loss.")

# ============================================================================
# STEP 6: HANDLE CLASS IMBALANCE
# ============================================================================
"""
Two main strategies for imbalanced datasets:
1. Weighted Random Sampler - Oversample minority classes during training
2. Weighted Loss - Give more importance to minority class errors

We'll use weighted loss here, but show how to use sampler as well.
"""

# Calculate class weights for weighted loss
class_weights = []
for cls in classes:
    count = train_class_counts[cls]
    weight = total / (len(classes) * count)  # Inverse frequency
    class_weights.append(weight)

class_weights = torch.FloatTensor(class_weights).to(device)

print("\nClass weights for balanced loss:")
for cls, weight in zip(classes, class_weights):
    print(f"{cls:>10s}: {weight:.3f}")

# Optional: Create WeightedRandomSampler (alternative approach)
# This oversamples minority classes in each epoch
sample_weights = []
for _, label in train_dataset.imgs:
    class_name = train_dataset.classes[label]
    weight = 1.0 / train_class_counts[class_name]
    sample_weights.append(weight)

# We'll use regular sampler, but here's how to use weighted sampler:
# sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

# ============================================================================
# STEP 7: CREATE DATA LOADERS
# ============================================================================

BATCH_SIZE = 32

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,  # Use shuffle=False if using WeightedRandomSampler
    # sampler=sampler,  # Uncomment to use weighted sampling
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

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True if torch.cuda.is_available() else False
)

# ============================================================================
# STEP 8: LOAD AND CONFIGURE MODEL
# ============================================================================

print("\nLoading pre-trained ResNet18...")
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

# Freeze early layers, unfreeze later layers (fine-tuning)
for param in model.parameters():
    param.requires_grad = False

for name, param in model.named_parameters():
    if any(layer in name for layer in ['layer3', 'layer4']):
        param.requires_grad = True

# Replace final layer
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(classes))

model = model.to(device)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"\nModel configuration:")
print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")

# ============================================================================
# STEP 9: LOSS FUNCTION AND OPTIMIZER
# ============================================================================
"""
Using weighted CrossEntropyLoss to handle class imbalance.
The weights make the loss penalize errors on minority classes more.
"""

# Weighted loss to handle imbalanced classes
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Discriminative learning rates
finetune_params = []
for name, param in model.named_parameters():
    if param.requires_grad and 'fc' not in name:
        finetune_params.append(param)

classifier_params = [model.fc.weight, model.fc.bias]

optimizer = optim.Adam([
    {'params': finetune_params, 'lr': 0.0001},
    {'params': classifier_params, 'lr': 0.001}
])

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, verbose=True
)

print("\nTraining configuration:")
print("Loss: Weighted CrossEntropyLoss (handles class imbalance)")
print("Optimizer: Adam with discriminative learning rates")
print("Scheduler: ReduceLROnPlateau")

# ============================================================================
# STEP 10: TRAINING AND EVALUATION FUNCTIONS
# ============================================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    """Evaluate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = running_loss / len(loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)

# ============================================================================
# STEP 11: TRAINING LOOP
# ============================================================================

NUM_EPOCHS = 20

print(f"\n{'='*70}")
print(f"Starting training for {NUM_EPOCHS} epochs...")
print(f"{'='*70}\n")

best_val_acc = 0.0
best_model_weights = copy.deepcopy(model.state_dict())
start_time = time.time()

for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    print("-" * 70)
    
    # Train
    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, device
    )
    
    # Validate
    val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
    
    # Print results
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.2f}%")
    
    # Learning rate scheduling
    scheduler.step(val_loss)
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_weights = copy.deepcopy(model.state_dict())
        print("✓ New best model saved")
    
    print()

total_time = time.time() - start_time
print(f"{'='*70}")
print(f"Training completed in {total_time // 60:.0f}m {total_time % 60:.0f}s")
print(f"Best validation accuracy: {best_val_acc:.2f}%")
print(f"{'='*70}\n")

# Load best model
model.load_state_dict(best_model_weights)

# ============================================================================
# STEP 12: COMPREHENSIVE TEST EVALUATION
# ============================================================================
"""
For imbalanced datasets, accuracy alone is misleading.
We need to look at per-class metrics:
- Precision: What % of predicted positives are correct?
- Recall: What % of actual positives are found?
- F1-Score: Harmonic mean of precision and recall
"""

print("Final Test Evaluation:")
print("="*70)

test_loss, test_acc, test_preds, test_labels = evaluate(
    model, test_loader, criterion, device
)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.2f}%")

# Detailed classification report
print("\nDetailed Classification Report:")
print("-" * 70)
print(classification_report(test_labels, test_preds, target_names=classes))

# Confusion matrix
print("\nConfusion Matrix:")
print("-" * 70)
cm = confusion_matrix(test_labels, test_preds)
print("Predicted ->")
print(f"{'Actual':>10s} | " + " | ".join(f"{c:>6s}" for c in classes))
print("-" * (12 + 10 * len(classes)))
for i, cls in enumerate(classes):
    print(f"{cls:>10s} | " + " | ".join(f"{cm[i][j]:>6d}" for j in range(len(classes))))

# Save model
print("\nSaving model...")
torch.save(model.state_dict(), 'custom_dataset_model.pth')
print("Model saved as 'custom_dataset_model.pth'")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("CUSTOM DATASET TRANSFER LEARNING COMPLETE!")
print("="*70)
print("\nKey Takeaways:")
print("1. Organized data in standard folder structure (class/images)")
print("2. Used heavy data augmentation for limited data")
print("3. Handled class imbalance with weighted loss")
print("4. Used proper train/val/test splits")
print("5. Evaluated with comprehensive metrics (not just accuracy)")
print("\nApplying to Your Own Data:")
print("1. Organize your images: dataset/train/class_name/*.jpg")
print("2. Update data_dir to your dataset path")
print("3. Adjust number of epochs based on dataset size")
print("4. Run this script!")
print("="*70)
