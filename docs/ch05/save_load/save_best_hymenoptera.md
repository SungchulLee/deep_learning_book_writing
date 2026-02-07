# Code: Save Best Model (Hymenoptera)

## Overview

This example demonstrates a complete training pipeline with checkpointing using the Hymenoptera (ants vs. bees) dataset—a binary image classification task using transfer learning.

## Complete Example

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os
import copy

# --- Configuration ---
data_dir = 'data/hymenoptera'
batch_size = 32
num_epochs = 25
learning_rate = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Data Transforms ---
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# --- Datasets and DataLoaders ---
image_datasets = {
    phase: datasets.ImageFolder(os.path.join(data_dir, phase),
                                data_transforms[phase])
    for phase in ['train', 'val']
}

dataloaders = {
    phase: DataLoader(image_datasets[phase], batch_size=batch_size,
                      shuffle=(phase == 'train'), num_workers=4)
    for phase in ['train', 'val']
}

dataset_sizes = {phase: len(image_datasets[phase]) for phase in ['train', 'val']}
class_names = image_datasets['train'].classes
print(f"Classes: {class_names}")
print(f"Train: {dataset_sizes['train']}, Val: {dataset_sizes['val']}")

# --- Model: Fine-tune ResNet18 ---
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(class_names))
model = model.to(device)

# --- Optimizer and Scheduler ---
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
criterion = nn.CrossEntropyLoss()

# --- Training with Best Model Saving ---
best_model_state = copy.deepcopy(model.state_dict())
best_val_acc = 0.0

for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in dataloaders['train']:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += (outputs.argmax(1) == labels).sum().item()

    train_loss = running_loss / dataset_sizes['train']
    train_acc = running_corrects / dataset_sizes['train']

    # Validation phase
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    with torch.no_grad():
        for inputs, labels in dataloaders['val']:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += (outputs.argmax(1) == labels).sum().item()

    val_loss = running_loss / dataset_sizes['val']
    val_acc = running_corrects / dataset_sizes['val']

    scheduler.step()

    print(f"Epoch {epoch+1}/{num_epochs} - "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} - "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = copy.deepcopy(model.state_dict())
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_acc': best_val_acc,
        }, 'best_hymenoptera.pt')
        print(f"  → Saved best model (val_acc={best_val_acc:.4f})")

# --- Load Best Model for Inference ---
model.load_state_dict(best_model_state)
model.eval()
print(f"\nBest validation accuracy: {best_val_acc:.4f}")
```

## Key Patterns Demonstrated

This example combines several patterns from earlier sections: separate training and validation transforms, `model.train()`/`model.eval()` switching, `torch.no_grad()` for validation, cosine annealing scheduling, complete checkpoint saving, and best-model tracking based on validation accuracy.
