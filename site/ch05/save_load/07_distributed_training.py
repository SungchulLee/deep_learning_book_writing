#!/usr/bin/env python3
"""
============================================================
Distributed Training: Saving and Loading Checkpoints
============================================================

Learn how to properly save and load models when using
DataParallel or DistributedDataParallel.

Topics:
- DataParallel checkpoints
- DistributedDataParallel checkpoints
- Handling 'module.' prefix
- Multi-GPU best practices
"""

import torch
import torch.nn as nn

print("=" * 70)
print("DISTRIBUTED TRAINING CHECKPOINTS")
print("=" * 70)

# ============================================================
# SIMPLE MODEL FOR DEMONSTRATION
# ============================================================

class SimpleModel(nn.Module):
    """Simple model for demonstration"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 5)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# ============================================================
# DATAPARALLEL SCENARIO
# ============================================================

print("\n" + "=" * 70)
print("SCENARIO 1: DataParallel")
print("=" * 70)

# Create model
model = SimpleModel()

# Check if multiple GPUs are available
if torch.cuda.device_count() > 1:
    print(f"\nUsing {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)
    model_wrapped = True
else:
    print("\nSingle GPU or CPU mode")
    model_wrapped = False

# Simulate training
print("\nSimulating training...")

# SAVING with DataParallel
print("\n--- SAVING ---")

checkpoint_path = "dataparallel_checkpoint.pth"

if model_wrapped:
    # Save the module's state dict (removes 'module.' prefix)
    state_dict = model.module.state_dict()
    print("Saving model.module.state_dict() (without 'module.' prefix)")
else:
    state_dict = model.state_dict()
    print("Saving model.state_dict()")

torch.save({
    'model_state_dict': state_dict,
    'wrapped': model_wrapped,
}, checkpoint_path)

print(f"Checkpoint saved to '{checkpoint_path}'")

# Print first few keys
print("\nFirst few state dict keys:")
for i, key in enumerate(list(state_dict.keys())[:3]):
    print(f"  {key}")

# LOADING with DataParallel
print("\n--- LOADING ---")

checkpoint = torch.load(checkpoint_path)

# Create fresh model
new_model = SimpleModel()

# Load state dict
new_model.load_state_dict(checkpoint['model_state_dict'])
print("State dict loaded into fresh model")

# Wrap with DataParallel if needed
if checkpoint.get('wrapped', False) and torch.cuda.device_count() > 1:
    new_model = nn.DataParallel(new_model)
    print("Model wrapped with DataParallel")

# Cleanup
import os
if os.path.exists(checkpoint_path):
    os.remove(checkpoint_path)


# ============================================================
# HANDLING 'module.' PREFIX
# ============================================================

print("\n" + "=" * 70)
print("SCENARIO 2: Handling 'module.' Prefix")
print("=" * 70)

# Simulate a state dict with 'module.' prefix
model = SimpleModel()
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

state_dict = model.state_dict() if torch.cuda.device_count() > 1 else model.state_dict()

print("\nOriginal state dict keys:")
for i, key in enumerate(list(state_dict.keys())[:3]):
    print(f"  {key}")

# Method 1: Remove 'module.' prefix when saving
from collections import OrderedDict

def remove_module_prefix(state_dict):
    """Remove 'module.' prefix from state dict keys"""
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # Remove 'module.' prefix if present
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    return new_state_dict

clean_state_dict = remove_module_prefix(state_dict)

print("\nCleaned state dict keys:")
for i, key in enumerate(list(clean_state_dict.keys())[:3]):
    print(f"  {key}")

# Method 2: Add 'module.' prefix when loading
def add_module_prefix(state_dict):
    """Add 'module.' prefix to state dict keys"""
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # Add 'module.' prefix if not present
        name = f'module.{k}' if not k.startswith('module.') else k
        new_state_dict[name] = v
    return new_state_dict

# Save clean version
checkpoint_path = "clean_checkpoint.pth"
torch.save({
    'model_state_dict': clean_state_dict,
}, checkpoint_path)

print(f"\nClean checkpoint saved to '{checkpoint_path}'")

# Load into either wrapped or unwrapped model
# Option 1: Load into unwrapped model
new_model = SimpleModel()
checkpoint = torch.load(checkpoint_path)
new_model.load_state_dict(checkpoint['model_state_dict'])
print("Loaded into unwrapped model successfully")

# Option 2: Load into wrapped model (add prefix)
if torch.cuda.device_count() > 1:
    wrapped_model = nn.DataParallel(SimpleModel())
    prefixed_state = add_module_prefix(checkpoint['model_state_dict'])
    wrapped_model.load_state_dict(prefixed_state)
    print("Loaded into wrapped model successfully")

# Cleanup
if os.path.exists(checkpoint_path):
    os.remove(checkpoint_path)


# ============================================================
# BEST PRACTICES
# ============================================================

print("\n" + "=" * 70)
print("BEST PRACTICES FOR DISTRIBUTED TRAINING")
print("=" * 70)

print("""
SAVING:
-------
1. Always save model.module.state_dict() for DataParallel
   - This removes the 'module.' prefix automatically
   - Makes checkpoint portable

2. Save without wrapper-specific keys
   - More flexible for loading

3. Include metadata about wrapping
   - Helps reconstruct model correctly

Example:
    if isinstance(model, nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    
    torch.save({
        'model_state_dict': state_dict,
        'is_parallel': isinstance(model, nn.DataParallel),
    }, path)


LOADING:
--------
1. Load into base model first
   - Then wrap if needed
   - More control over process

2. Use strict=False for flexibility
   - Handles missing/unexpected keys

3. Remove 'module.' prefix if present
   - Use helper function for cleaning

Example:
    model = SimpleModel()
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Wrap if needed
    if use_multi_gpu:
        model = nn.DataParallel(model)


COMMON ISSUES:
--------------
1. RuntimeError: Missing keys / Unexpected keys
   → Check for 'module.' prefix mismatch
   → Use remove_module_prefix() function

2. Model saved with DataParallel, loading without
   → State dict has 'module.' prefix
   → Load with strict=False or clean prefix

3. Model saved without DataParallel, loading with
   → Need to add 'module.' prefix
   → Or load into model.module

4. Checkpoint won't load on different GPU count
   → Save clean state dict without wrapper
   → Reconstruct wrapper after loading
""")

print("\n" + "=" * 70)
print("TUTORIAL COMPLETE")
print("=" * 70)

print("\nKey Takeaways:")
print("1. Save model.module.state_dict() for DataParallel")
print("2. Remove 'module.' prefix when saving")
print("3. Load into base model, then wrap")
print("4. Use helper functions for prefix handling")
print("5. Save metadata about model wrapping")
