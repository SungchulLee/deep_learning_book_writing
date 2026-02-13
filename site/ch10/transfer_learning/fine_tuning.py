"""
Example 2: Fine-Tuning a Pre-trained Model
===========================================

This script demonstrates fine-tuning techniques for transfer learning.
Unlike Example 1, we'll train multiple layers with different learning rates.

Key Concepts:
- Discriminative learning rates
- Selective layer unfreezing
- Early stopping
- Validation-based model selection

Author: PyTorch Transfer Learning Tutorial
Date: 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import time
import copy

# Set random seed for reproducibility
torch.manual_seed(42)

# ============================================================================
# STEP 1: DEVICE CONFIGURATION
# ============================================================================

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================================
# STEP 2: DATA PREPARATION WITH AUGMENTATION
# ============================================================================
"""
For fine-tuning, we use more aggressive data augmentation to help the model
generalize better. This is especially important when we're updating more
parameters than in basic transfer learning.

Data Augmentation Techniques:
1. Random horizontal flip - mirrors images
2. Random crop - adds translation invariance
3. Color jitter - makes model robust to lighting changes
"""

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Training transform with augmentation
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),           # Random crop instead of center crop
    transforms.RandomHorizontalFlip(),    # Randomly flip images horizontally
    transforms.ColorJitter(               # Randomly change brightness, contrast, saturation
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# Validation/Test transform without augmentation
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

print("\nLoading CIFAR-10 dataset with augmentation...")

# Load training data
train_dataset_full = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=train_transform
)

# Load test data
test_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=val_transform
)

# ============================================================================
# STEP 3: CREATE TRAIN/VALIDATION SPLIT
# ============================================================================
"""
For fine-tuning, we need a validation set to:
1. Monitor overfitting
2. Implement early stopping
3. Select the best model

We'll use 80% for training and 20% for validation from the training set.
"""

# Split training data into train and validation sets
train_size = int(0.8 * len(train_dataset_full))
val_size = len(train_dataset_full) - train_size

train_dataset, val_dataset = random_split(
    train_dataset_full,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)  # For reproducibility
)

# We need to update the validation dataset's transform
# Create a copy of val_dataset with proper transform
val_dataset.dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=False,
    transform=val_transform  # Use validation transform (no augmentation)
)

BATCH_SIZE = 32

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2
)

classes = ['plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

print(f"\nDataset split:")
print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# ============================================================================
# STEP 4: LOAD PRE-TRAINED MODEL
# ============================================================================

print("\nLoading pre-trained ResNet18 model...")
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

# ============================================================================
# STEP 5: SELECTIVE LAYER UNFREEZING
# ============================================================================
"""
Fine-tuning Strategy:
1. Freeze early layers (conv1, bn1, layer1)
   - These learn generic features that are useful across all vision tasks
2. Unfreeze later layers (layer2, layer3, layer4)
   - These learn more task-specific features that benefit from fine-tuning
3. Replace and train the final classifier

This is called "selective unfreezing" or "partial fine-tuning"
"""

print("\nSetting up fine-tuning strategy...")
print("Freezing early layers, unfreezing later layers")

# First, freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Unfreeze layer2, layer3, and layer4
# These are deeper layers that can benefit from fine-tuning
for name, param in model.named_parameters():
    if any(layer in name for layer in ['layer2', 'layer3', 'layer4']):
        param.requires_grad = True

print("\nLayer-wise freeze status:")
print("-" * 70)
for name, param in model.named_parameters():
    print(f"{name:50s} requires_grad={param.requires_grad}")

# Replace the final layer
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(classes))

# Move model to device
model = model.to(device)

# Count trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTrainable parameters: {trainable_params:,}")
print(f"Total parameters: {total_params:,}")
print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")

# ============================================================================
# STEP 6: DISCRIMINATIVE LEARNING RATES
# ============================================================================
"""
Different parts of the network should use different learning rates:

1. Pre-trained layers (layer2, layer3, layer4): Small LR (0.0001)
   - These already contain useful features
   - We want to fine-tune them slightly, not destroy what they learned

2. New classifier (fc): Large LR (0.001)
   - This is randomly initialized
   - It needs to learn from scratch

This is called "discriminative learning rates" or "differential learning rates"
"""

criterion = nn.CrossEntropyLoss()

# Create parameter groups with different learning rates
print("\nSetting up optimizer with discriminative learning rates:")

# Collect parameters for fine-tuning (layer2, layer3, layer4)
finetune_params = []
for name, param in model.named_parameters():
    if param.requires_grad and name != 'fc.weight' and name != 'fc.bias':
        finetune_params.append(param)

# Parameters for the new classifier (fc layer)
classifier_params = [model.fc.weight, model.fc.bias]

# Define parameter groups with different learning rates
optimizer = optim.Adam([
    {'params': finetune_params, 'lr': 0.0001},    # Small LR for pre-trained layers
    {'params': classifier_params, 'lr': 0.001}     # Large LR for new classifier
])

print(f"- Pre-trained layers learning rate: 0.0001")
print(f"- New classifier learning rate: 0.001")

# ============================================================================
# STEP 7: LEARNING RATE SCHEDULER
# ============================================================================
"""
Learning rate scheduling helps the model converge better.
We'll use ReduceLROnPlateau which reduces learning rate when validation
loss stops improving.

Benefits:
- Helps escape local minima
- Allows for fine-grained optimization later in training
- Automatically adapts to training progress
"""

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',        # Minimize validation loss
    factor=0.5,        # Multiply LR by 0.5 when triggered
    patience=3,        # Wait 3 epochs before reducing LR
    verbose=True       # Print when LR is reduced
)

print(f"\nLearning rate scheduler: ReduceLROnPlateau")
print(f"- Factor: 0.5 (halve LR when triggered)")
print(f"- Patience: 3 epochs")

# ============================================================================
# STEP 8: TRAINING AND VALIDATION FUNCTIONS
# ============================================================================

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, labels) in enumerate(train_loader):
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
        
        if (batch_idx + 1) % 100 == 0:
            print(f'  Batch {batch_idx + 1}/{len(train_loader)}: '
                  f'Loss: {running_loss / (batch_idx + 1):.3f}, '
                  f'Acc: {100. * correct / total:.2f}%')
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    avg_loss = running_loss / len(val_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy

# ============================================================================
# STEP 9: EARLY STOPPING IMPLEMENTATION
# ============================================================================
"""
Early stopping prevents overfitting by stopping training when validation
performance stops improving.

How it works:
1. Track validation loss after each epoch
2. If validation loss doesn't improve for 'patience' epochs, stop
3. Save the best model based on validation accuracy
"""

class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience=5, min_delta=0.0):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        """Check if we should stop training."""
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f'  Early stopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

early_stopping = EarlyStopping(patience=5)

# ============================================================================
# STEP 10: TRAINING LOOP WITH VALIDATION
# ============================================================================

NUM_EPOCHS = 15

print(f"\n{'='*70}")
print(f"Starting fine-tuning for {NUM_EPOCHS} epochs...")
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
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    
    # Print epoch results
    print(f"\nEpoch {epoch + 1} Summary:")
    print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"  Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.2f}%")
    
    # Learning rate scheduling
    scheduler.step(val_loss)
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_weights = copy.deepcopy(model.state_dict())
        print(f"  ✓ New best validation accuracy! Saving model...")
    
    # Early stopping check
    early_stopping(val_loss)
    if early_stopping.early_stop:
        print(f"\nEarly stopping triggered at epoch {epoch + 1}")
        break
    
    print()

total_time = time.time() - start_time
print(f"{'='*70}")
print(f"Training completed in {total_time // 60:.0f}m {total_time % 60:.0f}s")
print(f"Best validation accuracy: {best_val_acc:.2f}%")
print(f"{'='*70}\n")

# Load best model
model.load_state_dict(best_model_weights)

# ============================================================================
# STEP 11: FINAL TEST EVALUATION
# ============================================================================

print("Final Evaluation on Test Set:")
print("-" * 70)

test_loss, test_acc = validate(model, test_loader, criterion, device)
print(f"Final Test Loss: {test_loss:.4f}")
print(f"Final Test Accuracy: {test_acc:.2f}%")

# ============================================================================
# STEP 12: COMPARISON WITH FEATURE EXTRACTION
# ============================================================================

print("\n" + "="*70)
print("COMPARISON: Fine-tuning vs Feature Extraction")
print("="*70)
print("\nFine-tuning advantages:")
print("✓ Better accuracy (typically 3-5% improvement)")
print("✓ Model adapts to specific dataset characteristics")
print("✓ Learns task-specific features in deeper layers")
print("\nFine-tuning trade-offs:")
print("⚠ Longer training time")
print("⚠ Requires more memory")
print("⚠ Risk of overfitting with small datasets")
print("⚠ More hyperparameters to tune")

# Per-class accuracy
print("\nPer-class accuracy:")
print("-" * 70)

class_correct = [0] * len(classes)
class_total = [0] * len(classes)

model.eval()
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += (predicted[i] == label).item()
            class_total[label] += 1

for i in range(len(classes)):
    accuracy = 100 * class_correct[i] / class_total[i]
    print(f"{classes[i]:>10s}: {accuracy:>6.2f}%")

# Save model
print("\nSaving fine-tuned model...")
torch.save(model.state_dict(), 'resnet18_cifar10_finetuned.pth')
print("Model saved as 'resnet18_cifar10_finetuned.pth'")

print("\n" + "="*70)
print("FINE-TUNING COMPLETE!")
print("="*70)
print("\nKey Takeaways:")
print("1. We fine-tuned layers 2, 3, and 4 while keeping early layers frozen")
print("2. We used different learning rates for different parts of the network")
print("3. We implemented early stopping to prevent overfitting")
print("4. We used data augmentation to improve generalization")
print("5. Learning rate scheduling helped optimization")
print("\nNext Steps:")
print("- Try Example 3 to work with custom datasets")
print("- Experiment with unfreezing different layer combinations")
print("- Try different learning rate ratios")
print("="*70)
