"""
Module 64.01: Basic Model Saving and Loading
============================================

This module demonstrates fundamental techniques for saving and loading PyTorch models.
Students will learn the differences between saving complete models versus state dictionaries,
and best practices for model persistence.

Learning Objectives:
-------------------
1. Understand PyTorch model serialization
2. Save and load model state dictionaries
3. Save and load complete models
4. Implement checkpoint management
5. Handle model persistence best practices

Time Estimate: 45 minutes
Difficulty: Beginner
Prerequisites: Module 18 (Save and Load Models), Module 20 (Feedforward Networks)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Optional


# ============================================================================
# PART 1: Simple Model Definition
# ============================================================================

class SimpleClassifier(nn.Module):
    """
    A simple feedforward neural network for demonstration purposes.
    
    Architecture:
    - Input layer: 784 features (28x28 flattened images)
    - Hidden layer 1: 128 neurons with ReLU activation
    - Hidden layer 2: 64 neurons with ReLU activation
    - Output layer: 10 classes (for MNIST-like classification)
    
    This serves as our example model for saving/loading demonstrations.
    """
    
    def __init__(self, input_size: int = 784, hidden1: int = 128, 
                 hidden2: int = 64, num_classes: int = 10):
        """
        Initialize the classifier architecture.
        
        Args:
            input_size: Number of input features (default: 784 for 28x28 images)
            hidden1: Number of neurons in first hidden layer
            hidden2: Number of neurons in second hidden layer
            num_classes: Number of output classes
        """
        super(SimpleClassifier, self).__init__()
        
        # Store architecture parameters for reconstruction
        self.input_size = input_size
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.num_classes = num_classes
        
        # Define the network layers
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # No activation on output (logits)
        return x


# ============================================================================
# PART 2: Method 1 - Saving and Loading State Dictionary (RECOMMENDED)
# ============================================================================

def save_state_dict(model: nn.Module, optimizer: optim.Optimizer, 
                    epoch: int, loss: float, filepath: str) -> None:
    """
    Save model state dictionary along with training metadata.
    
    This is the RECOMMENDED approach for saving PyTorch models because:
    1. More flexible - can load into different model architectures
    2. More portable - doesn't serialize the entire model structure
    3. Better for version control - smaller file sizes
    4. Easier to debug - clear separation between model definition and weights
    
    Args:
        model: The PyTorch model to save
        optimizer: The optimizer (to save its state for resuming training)
        epoch: Current training epoch
        loss: Current training loss
        filepath: Path where the checkpoint will be saved
        
    Structure of saved checkpoint:
        {
            'epoch': int,              # Training epoch number
            'model_state_dict': dict,  # Model weights and biases
            'optimizer_state_dict': dict,  # Optimizer state
            'loss': float,             # Training loss value
            'timestamp': str,          # When checkpoint was created
            'architecture': dict       # Model architecture parameters
        }
    """
    print(f"\n{'='*70}")
    print("SAVING MODEL STATE DICTIONARY")
    print(f"{'='*70}")
    
    # Create checkpoint dictionary with all relevant information
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),  # Only weights, not structure
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        # Save architecture info for model reconstruction
        'architecture': {
            'input_size': model.input_size,
            'hidden1': model.hidden1,
            'hidden2': model.hidden2,
            'num_classes': model.num_classes
        }
    }
    
    # Save to disk
    torch.save(checkpoint, filepath)
    
    # Get file size for reporting
    file_size = Path(filepath).stat().st_size / 1024  # Convert to KB
    
    print(f"✓ Checkpoint saved successfully!")
    print(f"  Location: {filepath}")
    print(f"  File size: {file_size:.2f} KB")
    print(f"  Epoch: {epoch}")
    print(f"  Loss: {loss:.4f}")
    print(f"  Timestamp: {checkpoint['timestamp']}")
    

def load_state_dict(filepath: str, model: nn.Module, 
                   optimizer: Optional[optim.Optimizer] = None) -> Dict:
    """
    Load model state dictionary from a checkpoint file.
    
    This function demonstrates the proper way to load a saved state dictionary:
    1. Load the checkpoint dictionary
    2. Restore model weights
    3. Optionally restore optimizer state (if resuming training)
    4. Return metadata for tracking
    
    Args:
        filepath: Path to the saved checkpoint
        model: The model instance to load weights into
        optimizer: Optional optimizer to restore state (for resuming training)
        
    Returns:
        Dictionary containing checkpoint metadata (epoch, loss, timestamp)
        
    Note:
        The model architecture must match the saved state dictionary!
        You need to create the model instance before loading.
    """
    print(f"\n{'='*70}")
    print("LOADING MODEL STATE DICTIONARY")
    print(f"{'='*70}")
    
    # Check if file exists
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
    
    # Load checkpoint from disk
    checkpoint = torch.load(filepath)
    
    # Restore model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Restore optimizer state if provided (for resuming training)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"✓ Checkpoint loaded successfully!")
    print(f"  Location: {filepath}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Loss: {checkpoint['loss']:.4f}")
    print(f"  Timestamp: {checkpoint.get('timestamp', 'N/A')}")
    
    # Return metadata for reference
    return {
        'epoch': checkpoint['epoch'],
        'loss': checkpoint['loss'],
        'timestamp': checkpoint.get('timestamp', 'N/A'),
        'architecture': checkpoint.get('architecture', {})
    }


# ============================================================================
# PART 3: Method 2 - Saving and Loading Complete Model (NOT RECOMMENDED)
# ============================================================================

def save_complete_model(model: nn.Module, filepath: str) -> None:
    """
    Save the entire model (structure + weights) using torch.save().
    
    WARNING: This method is NOT RECOMMENDED for production because:
    1. Less flexible - hard to modify model architecture later
    2. Can break if PyTorch version changes
    3. Requires the exact same model class definition to load
    4. Larger file sizes
    5. Pickle dependency issues
    
    Use this only for quick prototyping or when you're certain about
    the deployment environment.
    
    Args:
        model: The complete PyTorch model to save
        filepath: Path where the model will be saved
    """
    print(f"\n{'='*70}")
    print("SAVING COMPLETE MODEL (NOT RECOMMENDED FOR PRODUCTION)")
    print(f"{'='*70}")
    
    # Save entire model object
    torch.save(model, filepath)
    
    file_size = Path(filepath).stat().st_size / 1024
    
    print(f"✓ Complete model saved")
    print(f"  Location: {filepath}")
    print(f"  File size: {file_size:.2f} KB")
    print(f"  ⚠️  Warning: This method is less portable and flexible")


def load_complete_model(filepath: str) -> nn.Module:
    """
    Load a complete model that was saved with torch.save(model, ...).
    
    Disadvantages:
    - Requires the model class definition to be available
    - May fail if PyTorch versions differ
    - Less control over loading process
    
    Args:
        filepath: Path to the saved model
        
    Returns:
        The loaded model ready for inference
    """
    print(f"\n{'='*70}")
    print("LOADING COMPLETE MODEL")
    print(f"{'='*70}")
    
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    # Load entire model object
    model = torch.load(filepath)
    
    print(f"✓ Complete model loaded")
    print(f"  Location: {filepath}")
    print(f"  Model type: {type(model).__name__}")
    
    return model


# ============================================================================
# PART 4: Checkpoint Management
# ============================================================================

class CheckpointManager:
    """
    Manages multiple model checkpoints with automatic cleanup.
    
    Features:
    - Automatic checkpoint naming with timestamps
    - Keep only the best N checkpoints (by validation loss)
    - Easy checkpoint loading and model recovery
    - JSON metadata tracking
    
    This is useful for long training runs where you want to:
    1. Save regularly but not keep all checkpoints (storage limits)
    2. Track which checkpoint performed best
    3. Easily resume training from any checkpoint
    """
    
    def __init__(self, checkpoint_dir: str, max_checkpoints: int = 5):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to store checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.metadata_file = self.checkpoint_dir / "checkpoints_metadata.json"
        
        # Load existing metadata or create new
        self.metadata = self._load_metadata()
        
    def _load_metadata(self) -> Dict:
        """Load checkpoint metadata from JSON file."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {'checkpoints': []}
    
    def _save_metadata(self):
        """Save checkpoint metadata to JSON file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def save_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer,
                       epoch: int, loss: float, metrics: Optional[Dict] = None) -> str:
        """
        Save a new checkpoint and manage existing checkpoints.
        
        Args:
            model: Model to save
            optimizer: Optimizer to save
            epoch: Current epoch
            loss: Validation loss (used for ranking checkpoints)
            metrics: Optional dictionary of additional metrics
            
        Returns:
            Path to saved checkpoint
        """
        # Create checkpoint filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"checkpoint_epoch{epoch:03d}_{timestamp}.pt"
        filepath = self.checkpoint_dir / filename
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'metrics': metrics or {},
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        torch.save(checkpoint, filepath)
        
        # Update metadata
        checkpoint_info = {
            'filename': filename,
            'epoch': epoch,
            'loss': loss,
            'timestamp': checkpoint['timestamp'],
            'metrics': metrics or {}
        }
        self.metadata['checkpoints'].append(checkpoint_info)
        
        # Sort by loss (best first) and keep only max_checkpoints
        self.metadata['checkpoints'].sort(key=lambda x: x['loss'])
        if len(self.metadata['checkpoints']) > self.max_checkpoints:
            # Remove worst checkpoints
            removed = self.metadata['checkpoints'][self.max_checkpoints:]
            for ckpt in removed:
                ckpt_path = self.checkpoint_dir / ckpt['filename']
                if ckpt_path.exists():
                    ckpt_path.unlink()
            self.metadata['checkpoints'] = self.metadata['checkpoints'][:self.max_checkpoints]
        
        self._save_metadata()
        
        print(f"\n✓ Checkpoint saved: {filename}")
        print(f"  Epoch: {epoch}, Loss: {loss:.4f}")
        print(f"  Total checkpoints: {len(self.metadata['checkpoints'])}")
        
        return str(filepath)
    
    def get_best_checkpoint(self) -> Optional[str]:
        """Return path to the best checkpoint (lowest loss)."""
        if not self.metadata['checkpoints']:
            return None
        best = self.metadata['checkpoints'][0]  # Already sorted by loss
        return str(self.checkpoint_dir / best['filename'])
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Return path to the most recent checkpoint."""
        if not self.metadata['checkpoints']:
            return None
        latest = max(self.metadata['checkpoints'], key=lambda x: x['epoch'])
        return str(self.checkpoint_dir / latest['filename'])
    
    def list_checkpoints(self):
        """Print all available checkpoints."""
        print(f"\n{'='*70}")
        print("AVAILABLE CHECKPOINTS")
        print(f"{'='*70}")
        if not self.metadata['checkpoints']:
            print("No checkpoints found.")
            return
        
        for i, ckpt in enumerate(self.metadata['checkpoints'], 1):
            print(f"\n{i}. {ckpt['filename']}")
            print(f"   Epoch: {ckpt['epoch']}, Loss: {ckpt['loss']:.4f}")
            print(f"   Timestamp: {ckpt['timestamp']}")
            if ckpt.get('metrics'):
                print(f"   Metrics: {ckpt['metrics']}")


# ============================================================================
# PART 5: Best Practices Demonstrations
# ============================================================================

def demonstrate_device_handling():
    """
    Demonstrate proper device handling when saving/loading models.
    
    Key concept: Models can be trained on GPU but deployed on CPU.
    You need to handle device mapping properly during loading.
    """
    print(f"\n{'='*70}")
    print("BEST PRACTICE: DEVICE HANDLING")
    print(f"{'='*70}")
    
    # Create a model
    model = SimpleClassifier()
    
    # Simulate training on GPU (if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"Model training device: {device}")
    
    # Save model
    filepath = "models/model_device_demo.pt"
    Path("models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), filepath)
    print(f"✓ Model saved from {device}")
    
    # Load model on CPU (common in production)
    model_cpu = SimpleClassifier()
    # Use map_location to load GPU-trained model on CPU
    model_cpu.load_state_dict(torch.load(filepath, map_location='cpu'))
    print(f"✓ Model loaded to CPU (regardless of training device)")
    
    # Load model on GPU (if available)
    if torch.cuda.is_available():
        model_gpu = SimpleClassifier()
        model_gpu.load_state_dict(torch.load(filepath, map_location='cuda:0'))
        print(f"✓ Model loaded to GPU")
    
    print("\nKey takeaway: Use map_location parameter for cross-device loading")


def demonstrate_inference_mode():
    """
    Demonstrate proper model setup for inference.
    
    Critical concepts:
    1. Use model.eval() to disable dropout and batchnorm training behavior
    2. Use torch.no_grad() to disable gradient computation (saves memory)
    3. These are ESSENTIAL for consistent inference results
    """
    print(f"\n{'='*70}")
    print("BEST PRACTICE: INFERENCE MODE SETUP")
    print(f"{'='*70}")
    
    # Create and "train" a model (just initialize with random weights)
    model = SimpleClassifier()
    
    # Create dummy input
    dummy_input = torch.randn(1, 784)
    
    print("\n1. INCORRECT: Model in training mode (inconsistent results)")
    # Training mode (default) - dropout and batchnorm behave differently
    output_train1 = model(dummy_input)
    output_train2 = model(dummy_input)
    print(f"   Output 1: {output_train1[0, :3].detach()}")  # First 3 values
    print(f"   Output 2: {output_train2[0, :3].detach()}")
    print(f"   ⚠️  Outputs may differ slightly due to dropout")
    
    print("\n2. CORRECT: Model in evaluation mode")
    model.eval()  # Switch to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        output_eval1 = model(dummy_input)
        output_eval2 = model(dummy_input)
    print(f"   Output 1: {output_eval1[0, :3]}")
    print(f"   Output 2: {output_eval2[0, :3]}")
    print(f"   ✓ Outputs are identical (deterministic)")
    
    print("\nKey takeaways:")
    print("  1. Always call model.eval() before inference")
    print("  2. Wrap inference code with torch.no_grad()")
    print("  3. This ensures consistent results and reduces memory usage")


# ============================================================================
# PART 6: Main Demonstration
# ============================================================================

def main():
    """
    Main demonstration of all model saving/loading techniques.
    """
    print("\n" + "="*70)
    print("MODULE 64.01: BASIC MODEL SAVING AND LOADING")
    print("="*70)
    
    # Create models directory
    Path("models").mkdir(exist_ok=True)
    
    # Initialize model and optimizer
    model = SimpleClassifier()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Simulate training progress
    epoch = 10
    loss = 0.1234
    
    # -----------------------------------------------------------------------
    # Demonstration 1: State Dictionary (Recommended Method)
    # -----------------------------------------------------------------------
    print("\n" + "="*70)
    print("DEMONSTRATION 1: STATE DICTIONARY APPROACH (RECOMMENDED)")
    print("="*70)
    
    # Save state dictionary
    save_state_dict(
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        loss=loss,
        filepath="models/model_state_dict.pt"
    )
    
    # Create new model instance and load weights
    model_loaded = SimpleClassifier()  # Must create model first
    optimizer_loaded = optim.Adam(model_loaded.parameters(), lr=0.001)
    
    metadata = load_state_dict(
        filepath="models/model_state_dict.pt",
        model=model_loaded,
        optimizer=optimizer_loaded
    )
    
    # Verify loading
    print(f"\n✓ Model loaded successfully with metadata:")
    print(f"  Epoch: {metadata['epoch']}")
    print(f"  Loss: {metadata['loss']:.4f}")
    
    # -----------------------------------------------------------------------
    # Demonstration 2: Complete Model (Not Recommended)
    # -----------------------------------------------------------------------
    print("\n" + "="*70)
    print("DEMONSTRATION 2: COMPLETE MODEL APPROACH (NOT RECOMMENDED)")
    print("="*70)
    
    save_complete_model(model, "models/complete_model.pt")
    model_complete = load_complete_model("models/complete_model.pt")
    
    # -----------------------------------------------------------------------
    # Demonstration 3: Checkpoint Manager
    # -----------------------------------------------------------------------
    print("\n" + "="*70)
    print("DEMONSTRATION 3: CHECKPOINT MANAGER")
    print("="*70)
    
    manager = CheckpointManager("models/checkpoints", max_checkpoints=3)
    
    # Simulate saving multiple checkpoints during training
    for ep in range(1, 6):
        # Simulate decreasing loss
        loss_val = 1.0 / ep
        metrics = {
            'accuracy': 0.5 + ep * 0.08,
            'f1_score': 0.45 + ep * 0.09
        }
        manager.save_checkpoint(model, optimizer, ep, loss_val, metrics)
    
    # List all checkpoints
    manager.list_checkpoints()
    
    # Get best checkpoint
    best_ckpt = manager.get_best_checkpoint()
    print(f"\n✓ Best checkpoint: {best_ckpt}")
    
    # -----------------------------------------------------------------------
    # Demonstration 4: Best Practices
    # -----------------------------------------------------------------------
    demonstrate_device_handling()
    demonstrate_inference_mode()
    
    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "="*70)
    print("SUMMARY: KEY TAKEAWAYS")
    print("="*70)
    print("""
    1. ALWAYS use state_dict approach for production:
       ✓ More flexible and portable
       ✓ Smaller file sizes
       ✓ Better version control
       
    2. Save training metadata along with model weights:
       ✓ Epoch number
       ✓ Loss/metrics
       ✓ Optimizer state (for resuming training)
       ✓ Timestamp and architecture info
       
    3. Use CheckpointManager for long training runs:
       ✓ Automatic checkpoint management
       ✓ Keep only best N checkpoints
       ✓ Easy recovery from best checkpoint
       
    4. Handle device mapping correctly:
       ✓ Use map_location parameter when loading
       ✓ Support CPU inference for GPU-trained models
       
    5. Set proper inference mode:
       ✓ Call model.eval() before inference
       ✓ Use torch.no_grad() context manager
       ✓ Ensures consistent and efficient inference
    """)
    
    print("\n✅ Module 64.01 completed successfully!")
    print("Next: Module 64.02 - TorchScript Basics")


if __name__ == "__main__":
    main()
