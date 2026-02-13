#!/usr/bin/env python3
"""
============================================================
Transfer Learning: Saving and Loading Pre-trained Models
============================================================

Learn best practices for working with pre-trained models,
fine-tuning, and saving partial models.

Topics:
- Loading pre-trained models
- Freezing/unfreezing layers
- Saving fine-tuned models
- Partial state dict loading
"""

import torch
import torch.nn as nn
import torchvision.models as models

print("=" * 70)
print("TRANSFER LEARNING SAVE/LOAD TUTORIAL")
print("=" * 70)

# ============================================================
# LOADING PRE-TRAINED MODELS
# ============================================================

print("\n" + "=" * 70)
print("LOADING PRE-TRAINED MODELS")
print("=" * 70)

print("\nLoading ResNet18 with pre-trained weights...")

# Load pre-trained ResNet18
model = models.resnet18(pretrained=False)  # Set to False for demo
print("Model loaded")

# Check number of parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# ============================================================
# MODIFYING FOR TRANSFER LEARNING
# ============================================================

print("\n" + "=" * 70)
print("MODIFYING MODEL FOR TRANSFER LEARNING")
print("=" * 70)

# Get number of input features to final layer
num_features = model.fc.in_features
print(f"\nOriginal classifier input features: {num_features}")
print(f"Original classifier output classes: {model.fc.out_features}")

# Replace final layer for new task (e.g., 10 classes instead of 1000)
num_classes = 10
model.fc = nn.Linear(num_features, num_classes)
print(f"\nNew classifier output classes: {num_classes}")

# ============================================================
# FREEZING LAYERS
# ============================================================

print("\n" + "=" * 70)
print("FREEZING LAYERS")
print("=" * 70)

print("\nFreezing all layers except final classifier...")

# Freeze all parameters
for param in model.parameters():
    param.requires_grad = False

# Unfreeze final layer
for param in model.fc.parameters():
    param.requires_grad = True

# Count trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params:,}")
print(f"Frozen parameters: {total_params - trainable_params:,}")

# ============================================================
# SAVING FINE-TUNED MODEL
# ============================================================

print("\n" + "=" * 70)
print("SAVING FINE-TUNED MODEL")
print("=" * 70)

# Create optimizer only for trainable parameters
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.001
)

# Save checkpoint with transfer learning info
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'num_classes': num_classes,
    'frozen_layers': True,
    'base_model': 'resnet18',
}

filepath = "transfer_learning_checkpoint.pth"
torch.save(checkpoint, filepath)
print(f"\nCheckpoint saved to '{filepath}'")

import os
file_size = os.path.getsize(filepath) / (1024 * 1024)
print(f"File size: {file_size:.2f} MB")

# ============================================================
# LOADING FINE-TUNED MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING FINE-TUNED MODEL")
print("=" * 70)

# Load checkpoint
checkpoint = torch.load(filepath)

# Recreate model with same modifications
model = models.resnet18(pretrained=False)
num_features = model.fc.in_features
num_classes = checkpoint['num_classes']
model.fc = nn.Linear(num_features, num_classes)

# Load state dict
model.load_state_dict(checkpoint['model_state_dict'])
print("\nModel state loaded")

# Recreate optimizer
optimizer = torch.optim.Adam(model.parameters())
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
print("Optimizer state loaded")

# Apply freezing if needed
if checkpoint.get('frozen_layers', False):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
    print("Layer freezing applied")

# ============================================================
# PARTIAL STATE DICT LOADING
# ============================================================

print("\n" + "=" * 70)
print("PARTIAL STATE DICT LOADING")
print("=" * 70)

print("\nLoading only specific layers...")

# Save original state
full_state = model.state_dict()

# Create a new model
new_model = models.resnet18(pretrained=False)
new_model.fc = nn.Linear(num_features, num_classes)

# Load only convolutional layers (not classifier)
pretrained_dict = {k: v for k, v in full_state.items() if 'fc' not in k}

# Get current state
model_dict = new_model.state_dict()

# Update with pretrained weights
model_dict.update(pretrained_dict)

# Load the new state dict
new_model.load_state_dict(model_dict)
print("Partial state dict loaded (excluding fc layer)")

# ============================================================
# HANDLING MISSING/UNEXPECTED KEYS
# ============================================================

print("\n" + "=" * 70)
print("HANDLING MISSING/UNEXPECTED KEYS")
print("=" * 70)

# This demonstrates how to handle state dict mismatches
model = models.resnet18(pretrained=False)
saved_state = model.state_dict()

# Modify model
model.fc = nn.Linear(512, 20)  # Different number of classes

# Load with strict=False to allow mismatches
missing_keys, unexpected_keys = model.load_state_dict(
    saved_state,
    strict=False
)

print(f"\nMissing keys: {len(missing_keys)}")
if missing_keys:
    print(f"  {missing_keys}")

print(f"Unexpected keys: {len(unexpected_keys)}")
if unexpected_keys:
    print(f"  {unexpected_keys}")

print("\nModel loaded with partial matching")

# Cleanup
if os.path.exists(filepath):
    os.remove(filepath)
    print(f"\nCleaned up '{filepath}'")

print("\n" + "=" * 70)
print("TUTORIAL COMPLETE")
print("=" * 70)

print("\nKey Takeaways:")
print("1. Use pre-trained models as feature extractors")
print("2. Freeze early layers, train final layers")
print("3. Save full model state including modifications")
print("4. Use strict=False for partial loading")
print("5. Filter optimizer parameters for frozen layers")
