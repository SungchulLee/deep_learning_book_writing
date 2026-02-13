#!/usr/bin/env python3
"""
============================================================
PyTorch Model Saving and Loading - Complete Guide
============================================================

This comprehensive tutorial covers all essential methods for saving
and loading PyTorch models, including best practices and common pitfalls.

Key Concepts:
1. torch.save() - Saves Python objects to disk using pickle
2. torch.load() - Loads saved objects from disk
3. state_dict() - Ordered dictionary containing model parameters
4. load_state_dict() - Loads parameters into a model

Author: Based on Patrick Loeber's PyTorch Tutorial
Enhanced with detailed comments and explanations
"""

import torch
import torch.nn as nn
import os

# ============================================================
# CORE METHODS TO REMEMBER
# ============================================================

"""
Three Essential Functions:
--------------------------

1. torch.save(obj, path)
   - Saves any Python object (model, tensor, dictionary, etc.)
   - Uses Python's pickle protocol
   - Path should end with .pt or .pth convention
   
2. torch.load(path)
   - Loads the saved object back into memory
   - Returns the exact object that was saved
   - Can specify device with map_location parameter
   
3. model.load_state_dict(state_dict)
   - Loads model parameters from a state dictionary
   - More flexible and recommended for most use cases
   - Allows model architecture to be defined separately
"""

# ============================================================
# TWO MAIN APPROACHES TO SAVING MODELS
# ============================================================

"""
Approach 1: Save Entire Model (Quick but not recommended)
----------------------------------------------------------
Pros:
  - Simple one-liner
  - Quick for prototyping
  
Cons:
  - Larger file size
  - Less flexible
  - Can break with code refactoring
  - Model class must be available when loading
  - Not recommended for production

Usage:
  torch.save(model, PATH)
  model = torch.load(PATH)
  model.eval()


Approach 2: Save Only State Dict (RECOMMENDED)
-----------------------------------------------
Pros:
  - Smaller file size (only weights, no structure)
  - More flexible and portable
  - Better for version control
  - Can load into different model architectures
  - Industry standard
  
Cons:
  - Requires model class to be defined separately
  - Slightly more code
  
Usage:
  torch.save(model.state_dict(), PATH)
  model = Model(*args, **kwargs)
  model.load_state_dict(torch.load(PATH))
  model.eval()
"""

# ============================================================
# DEFINE A SIMPLE MODEL FOR DEMONSTRATION
# ============================================================

class Model(nn.Module):
    """
    Simple neural network for demonstration purposes.
    
    Architecture:
    - Single linear layer
    - Sigmoid activation
    - Suitable for binary classification
    
    Args:
        n_input_features (int): Number of input features
    """
    
    def __init__(self, n_input_features):
        super(Model, self).__init__()
        # Initialize a single linear layer
        # Maps n_input_features -> 1 output (binary classification)
        self.linear = nn.Linear(n_input_features, 1)
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, n_input_features)
            
        Returns:
            y_pred: Output predictions of shape (batch_size, 1)
                    Values between 0 and 1 due to sigmoid activation
        """
        # Apply linear transformation followed by sigmoid
        # Sigmoid squashes output to range [0, 1] for probability
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred


# ============================================================
# METHOD 1: SAVE AND LOAD ENTIRE MODEL
# ============================================================

print("=" * 60)
print("METHOD 1: Saving and Loading Entire Model")
print("=" * 60)

# Create a model instance
model = Model(n_input_features=6)

# Display initial model parameters
print("\n📊 Initial Model Parameters:")
for name, param in model.named_parameters():
    print(f"{name}: shape {param.shape}")
    print(f"  Values: {param.data.flatten()[:5]}...")  # Show first 5 values

# In practice, you would train your model here
# For demonstration, we'll use the randomly initialized weights
print("\n🔧 (In production: Train your model here)")

# Define file path for saving
FILE = "model_complete.pth"

# SAVE: Entire model (architecture + parameters)
print(f"\n💾 Saving entire model to '{FILE}'...")
torch.save(model, FILE)
print(f"✅ Model saved successfully!")
print(f"   File size: {os.path.getsize(FILE) / 1024:.2f} KB")

# LOAD: Entire model
print(f"\n📂 Loading model from '{FILE}'...")
loaded_model = torch.load(FILE)

# IMPORTANT: Set model to evaluation mode
# This disables dropout and sets batch normalization to eval mode
loaded_model.eval()
print("✅ Model loaded successfully!")

# Verify parameters are identical
print("\n🔍 Verifying Loaded Model Parameters:")
for name, param in loaded_model.named_parameters():
    print(f"{name}: shape {param.shape}")
    print(f"  Values: {param.data.flatten()[:5]}...")

# Verify they match
params_match = all(
    torch.equal(p1, p2) 
    for p1, p2 in zip(model.parameters(), loaded_model.parameters())
)
print(f"\n✓ Parameters match: {params_match}")

# Clean up
if os.path.exists(FILE):
    os.remove(FILE)
    print(f"🗑️  Cleaned up '{FILE}'")


# ============================================================
# METHOD 2: SAVE AND LOAD STATE DICT (RECOMMENDED)
# ============================================================

print("\n" + "=" * 60)
print("METHOD 2: Saving and Loading State Dict (RECOMMENDED)")
print("=" * 60)

# Create a fresh model
model = Model(n_input_features=6)

# Define file path
FILE = "model_state_dict.pth"

# SAVE: Only the state dictionary
print(f"\n💾 Saving model state dict to '{FILE}'...")
torch.save(model.state_dict(), FILE)
print(f"✅ State dict saved successfully!")
print(f"   File size: {os.path.getsize(FILE) / 1024:.2f} KB")

# Display what's in the state dict
print("\n📋 State Dict Contents:")
state_dict = model.state_dict()
for key, value in state_dict.items():
    print(f"  {key}: shape {value.shape}, dtype {value.dtype}")

# LOAD: State dictionary into a new model
print(f"\n📂 Loading state dict from '{FILE}'...")

# STEP 1: Create model architecture (must match saved model)
loaded_model = Model(n_input_features=6)

# STEP 2: Load the state dictionary
# Note: torch.load() returns the dictionary
# load_state_dict() loads it into the model
loaded_model.load_state_dict(torch.load(FILE))

# STEP 3: Set to evaluation mode
loaded_model.eval()
print("✅ State dict loaded successfully!")

# Verify loaded state dict
print("\n🔍 Verifying Loaded State Dict:")
loaded_state_dict = loaded_model.state_dict()
for key, value in loaded_state_dict.items():
    print(f"  {key}: shape {value.shape}")

# Verify they match
state_match = all(
    torch.equal(state_dict[key], loaded_state_dict[key])
    for key in state_dict.keys()
)
print(f"\n✓ State dicts match: {state_match}")

# Clean up
if os.path.exists(FILE):
    os.remove(FILE)
    print(f"🗑️  Cleaned up '{FILE}'")


# ============================================================
# METHOD 3: SAVE AND LOAD TRAINING CHECKPOINT
# ============================================================

print("\n" + "=" * 60)
print("METHOD 3: Saving and Loading Training Checkpoint")
print("=" * 60)

"""
A checkpoint saves the complete training state, allowing you to:
- Resume training from where you left off
- Save best model during training
- Recover from crashes or interruptions

Components typically saved in a checkpoint:
1. Model state dict
2. Optimizer state dict
3. Current epoch number
4. Training loss history
5. Learning rate schedule state
6. Random number generator states (for reproducibility)
"""

# Create model and optimizer
model = Model(n_input_features=6)
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Simulate training progress
print("\n🏋️  Simulating training progress...")
print(f"   Current epoch: 90")
print(f"   Learning rate: {learning_rate}")

# Display optimizer state
print("\n📊 Optimizer State Before Saving:")
print(f"   State dict keys: {list(optimizer.state_dict().keys())}")

# Create a comprehensive checkpoint dictionary
checkpoint = {
    "epoch": 90,                          # Current epoch number
    "model_state": model.state_dict(),     # Model parameters
    "optim_state": optimizer.state_dict(), # Optimizer state (momentum, etc.)
    "loss": 0.123,                         # Optional: current loss
    "accuracy": 0.95,                      # Optional: current accuracy
}

# Save checkpoint
FILE = "checkpoint.pth"
print(f"\n💾 Saving checkpoint to '{FILE}'...")
torch.save(checkpoint, FILE)
print(f"✅ Checkpoint saved successfully!")
print(f"   File size: {os.path.getsize(FILE) / 1024:.2f} KB")

print("\n📦 Checkpoint Contents:")
for key in checkpoint.keys():
    if isinstance(checkpoint[key], dict):
        print(f"  {key}: dict with {len(checkpoint[key])} items")
    else:
        print(f"  {key}: {checkpoint[key]}")

# LOAD: Restore training state
print(f"\n📂 Loading checkpoint from '{FILE}'...")

# STEP 1: Recreate model and optimizer
model = Model(n_input_features=6)
optimizer = torch.optim.SGD(model.parameters(), lr=0)  # lr will be restored

# STEP 2: Load checkpoint
checkpoint = torch.load(FILE)

# STEP 3: Restore model state
model.load_state_dict(checkpoint['model_state'])

# STEP 4: Restore optimizer state
optimizer.load_state_dict(checkpoint['optim_state'])

# STEP 5: Restore other training parameters
epoch = checkpoint['epoch']
loss = checkpoint.get('loss', None)  # Use .get() for optional keys
accuracy = checkpoint.get('accuracy', None)

print("✅ Checkpoint loaded successfully!")
print(f"\n📊 Restored Training State:")
print(f"   Epoch: {epoch}")
print(f"   Loss: {loss}")
print(f"   Accuracy: {accuracy}")
print(f"   Learning rate: {optimizer.param_groups[0]['lr']}")

# IMPORTANT: Set appropriate mode
print("\n⚙️  Setting Model Mode:")
print("   For inference: model.eval()")
print("   For continued training: model.train()")

# For inference
model.eval()
print("   Current mode: EVAL")

# OR for continued training
# model.train()
# print("   Current mode: TRAIN")

# Clean up
if os.path.exists(FILE):
    os.remove(FILE)
    print(f"\n🗑️  Cleaned up '{FILE}'")


# ============================================================
# DEVICE-SPECIFIC SAVING AND LOADING
# ============================================================

print("\n" + "=" * 60)
print("DEVICE-SPECIFIC SAVING AND LOADING")
print("=" * 60)

"""
When working with GPUs, you need to be careful about where models
are saved from and loaded to. PyTorch provides flexibility for
all common scenarios.
"""

print("\n📱 Scenario 1: Save on GPU, Load on CPU")
print("-" * 60)
print("""
# Training on GPU
device = torch.device("cuda")
model.to(device)
torch.save(model.state_dict(), PATH)

# Inference on CPU
device = torch.device('cpu')
model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH, map_location=device))
model.to(device)  # Not strictly necessary for CPU, but good practice
""")

print("\n🖥️  Scenario 2: Save on GPU, Load on GPU")
print("-" * 60)
print("""
# Training on GPU
device = torch.device("cuda")
model.to(device)
torch.save(model.state_dict(), PATH)

# Inference on same GPU
model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH))  # Will load to same device
model.to(device)

# NOTE: Remember to move input tensors to GPU too!
# input_tensor = input_tensor.to(device)
""")

print("\n🔄 Scenario 3: Save on CPU, Load on GPU")
print("-" * 60)
print("""
# Training on CPU
torch.save(model.state_dict(), PATH)

# Inference on GPU
device = torch.device("cuda")
model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH, map_location="cuda:0"))  # Specify GPU
model.to(device)

# This is common when training on a local machine without GPU
# and deploying to a GPU server
""")

print("\n🎯 Scenario 4: Multi-GPU Considerations")
print("-" * 60)
print("""
# If model was trained with DataParallel or DistributedDataParallel:

# Save: Remove 'module.' prefix if present
state_dict = model.state_dict()
if list(state_dict.keys())[0].startswith('module.'):
    # Remove 'module.' prefix
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove 'module.' prefix
        new_state_dict[name] = v
    torch.save(new_state_dict, PATH)
else:
    torch.save(state_dict, PATH)

# Load: Can load on single GPU or multi-GPU
model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.to(device)

# For multi-GPU:
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
""")


# ============================================================
# BEST PRACTICES AND IMPORTANT NOTES
# ============================================================

print("\n" + "=" * 60)
print("BEST PRACTICES AND IMPORTANT NOTES")
print("=" * 60)

print("""
✅ DO's:
--------
1. Use .state_dict() approach for saving models in production
2. Always call model.eval() before inference
3. Save checkpoints during training to prevent data loss
4. Use meaningful file names with version/epoch info
   Example: 'model_epoch50_acc0.95.pth'
5. Save hyperparameters along with checkpoint
6. Use map_location when loading to different devices
7. Verify loaded model works before deleting old checkpoints

❌ DON'Ts:
----------
1. Don't save entire model in production code
2. Don't forget to call model.eval() for inference
3. Don't mix up train() and eval() modes
4. Don't save temporary/cached tensors in checkpoint
5. Don't assume same device when loading
6. Don't ignore warnings about missing or unexpected keys

🔍 Common Issues:
----------------
1. RuntimeError: state dict size mismatch
   → Model architecture doesn't match saved state dict
   
2. CUDA out of memory when loading
   → Use map_location='cpu' to load to CPU first
   
3. Different behavior during inference vs training
   → Forgot to call model.eval()
   
4. Can't resume training properly
   → Forgot to save/load optimizer state
   
5. Module prefix mismatch with DataParallel
   → Need to handle 'module.' prefix in keys

📚 Additional Resources:
-----------------------
- PyTorch Docs: https://pytorch.org/tutorials/beginner/saving_loading_models.html
- Checkpoint Management: See advanced tutorials
- Model Deployment: See ONNX and TorchScript tutorials
""")

print("\n" + "=" * 60)
print("TUTORIAL COMPLETE!")
print("=" * 60)
print("\n💡 Key Takeaways:")
print("   1. Use state_dict() for production models")
print("   2. Save checkpoints for training recovery")
print("   3. Always use model.eval() for inference")
print("   4. Be mindful of device placement (CPU/GPU)")
print("   5. Include training state in checkpoints")
print("\n📖 Next Steps:")
print("   - Check out advanced checkpoint management")
print("   - Learn about model versioning")
print("   - Explore ONNX export for deployment")
print("   - Study distributed training checkpoints")
