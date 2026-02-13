#!/usr/bin/env python3
"""
PyTorch Learning Rate Scheduler Demonstration
==============================================

This script demonstrates various learning rate schedulers available in PyTorch.
Learning rate scheduling is a crucial technique for training deep learning models
effectively. It adjusts the learning rate during training to help the model
converge better and potentially achieve better performance.

Why Learning Rate Scheduling?
------------------------------
1. Start with higher learning rate for faster initial convergence
2. Reduce learning rate as training progresses for fine-tuning
3. Escape local minima by adjusting learning rate dynamically
4. Improve model generalization and final performance

Available Schedulers:
---------------------
1. StepLR: Decay LR by gamma every step_size epochs
2. MultiStepLR: Decay LR at specific milestone epochs
3. ExponentialLR: Decay LR exponentially by gamma each epoch
4. CosineAnnealingLR: Cosine annealing schedule
5. OneCycleLR: One cycle learning rate policy
6. CyclicLR: Cyclical learning rate with triangular pattern
7. ReduceLROnPlateau: Reduce LR when metric plateaus

Example Usage:
--------------
Basic examples for running different schedulers:

1. Step Scheduler (reduce LR every 20 epochs by 0.5):
   python scheduler.py --scheduler step --epochs 50 --lr 0.1 --step_size 20 --gamma 0.5

2. MultiStep Scheduler (reduce at epochs 10, 20, 30):
   python scheduler.py --scheduler multistep --epochs 50 --milestones 10,20,30 --gamma 0.3

3. Cosine Annealing (smooth cosine decay):
   python scheduler.py --scheduler cosine --epochs 50 --t_max 50

4. OneCycle (modern approach, fast training):
   python scheduler.py --scheduler onecycle --epochs 5 --max_lr 0.2

5. Exponential Decay (constant exponential decay):
   python scheduler.py --scheduler exponential --epochs 50 --gamma 0.95

6. Cyclical LR (triangular pattern, updates per batch):
   python scheduler.py --scheduler cyclical --base_lr 1e-3 --max_lr 1e-1 --mode triangular2 --scheduler_step_on batch

7. Reduce on Plateau (adaptive based on validation loss):
   python scheduler.py --scheduler plateau --epochs 50 --patience 5 --factor 0.5 --scheduler_step_on val_loss

Author: Learning Rate Scheduler Tutorial
License: MIT
"""

# ============================================================================
# IMPORTS
# ============================================================================
from scheduler.config import get_config
from scheduler.data_loader import build_dataloaders
from scheduler.model import TinyMLP
from scheduler.train import Trainer
from scheduler.utils import get_device, set_seed


# ============================================================================
# MAIN FUNCTION
# ============================================================================
def main():
    """
    Main function to run the learning rate scheduler demonstration.
    
    This function orchestrates the entire training process:
    1. Load and parse configuration from command-line arguments
    2. Set random seed for reproducibility
    3. Determine compute device (CPU or GPU)
    4. Build data loaders for training and validation
    5. Initialize the model
    6. Create trainer with specified scheduler
    7. Execute training loop
    
    The function is designed to be flexible and allow experimentation with
    different schedulers and hyperparameters through command-line arguments.
    """
    
    # ========================================================================
    # STEP 1: GET CONFIGURATION
    # ========================================================================
    # Parse command-line arguments to get all configuration settings
    # This includes:
    # - Scheduler type (step, multistep, cosine, etc.)
    # - Training hyperparameters (epochs, batch size, learning rate)
    # - Scheduler-specific parameters (step_size, gamma, milestones, etc.)
    # - Model architecture parameters
    # - Data generation parameters
    cfg = get_config()
    
    # Print configuration summary for reference
    print("\n" + "="*70)
    print("LEARNING RATE SCHEDULER DEMONSTRATION")
    print("="*70)
    print(f"\nScheduler: {cfg.scheduler.upper()}")
    print(f"Epochs: {cfg.epochs}")
    print(f"Initial Learning Rate: {cfg.lr}")
    print(f"Batch Size: {cfg.batch_size}")
    print("="*70 + "\n")
    
    # ========================================================================
    # STEP 2: SET RANDOM SEED
    # ========================================================================
    # Setting a random seed ensures reproducibility of results
    # This affects:
    # - PyTorch random number generation
    # - NumPy random number generation
    # - Python's random module
    # - CUDA operations (if using GPU)
    set_seed(cfg.seed)
    print(f"Random seed set to: {cfg.seed}")
    
    # ========================================================================
    # STEP 3: DETERMINE COMPUTE DEVICE
    # ========================================================================
    # Automatically detect and use GPU if available, otherwise use CPU
    # GPU significantly speeds up training for neural networks
    # The device string can be:
    # - 'cuda' or 'cuda:0' for first GPU
    # - 'cuda:1' for second GPU, etc.
    # - 'cpu' for CPU computation
    device = get_device(cfg.device)
    print(f"Using device: {device}\n")
    
    # ========================================================================
    # STEP 4: BUILD DATA LOADERS
    # ========================================================================
    # Create synthetic dataset and data loaders for training and validation
    # 
    # For this demo, we generate synthetic classification data to focus on
    # scheduler behavior rather than data preprocessing complexities.
    #
    # Returns:
    # - train_loader: DataLoader for training data (shuffled)
    # - val_loader: DataLoader for validation data (not shuffled)
    # - steps_per_epoch: Number of batches per epoch (needed for some schedulers)
    print("Building data loaders...")
    train_loader, val_loader, steps_per_epoch = build_dataloaders(
        n_samples=cfg.n_samples,      # Total number of samples to generate
        batch_size=cfg.batch_size,     # Number of samples per batch
        val_ratio=cfg.val_ratio,       # Fraction of data for validation
        seed=cfg.seed                  # Random seed for data generation
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Steps per epoch: {steps_per_epoch}\n")
    
    # ========================================================================
    # STEP 5: INITIALIZE MODEL
    # ========================================================================
    # Create a small Multi-Layer Perceptron (MLP) for classification
    # 
    # The model architecture is intentionally simple to:
    # - Focus on scheduler effects rather than model complexity
    # - Train quickly for demonstration purposes
    # - Make learning rate effects more visible
    #
    # The model is moved to the appropriate device (CPU or GPU)
    print("Initializing model...")
    model = TinyMLP(
        input_dim=cfg.input_dim,       # Number of input features
        hidden_dim=cfg.hidden_dim,     # Number of hidden layer neurons
        num_classes=cfg.num_classes    # Number of output classes
    ).to(device)
    
    # Print model architecture
    print(f"Model: {model}")
    
    # Count and display total parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}\n")
    
    # ========================================================================
    # STEP 6: CREATE TRAINER
    # ========================================================================
    # The Trainer class handles:
    # - Optimizer creation
    # - Scheduler initialization based on config
    # - Training loop execution
    # - Validation evaluation
    # - Learning rate tracking and visualization
    # - Loss tracking
    #
    # It encapsulates all training logic and makes it easy to switch
    # between different schedulers without changing the main code.
    print("Setting up trainer...")
    trainer = Trainer(
        cfg=cfg,                       # Configuration object
        model=model,                   # Neural network model
        device=device,                 # Compute device
        steps_per_epoch=steps_per_epoch  # Needed for batch-level schedulers
    )
    
    # ========================================================================
    # STEP 7: EXECUTE TRAINING
    # ========================================================================
    # Start the training process
    # This will:
    # 1. Train for specified number of epochs
    # 2. Update learning rate according to scheduler
    # 3. Evaluate on validation set each epoch
    # 4. Track and log metrics
    # 5. Save learning rate history plot
    # 6. Display final results
    print("\nStarting training...")
    print("="*70 + "\n")
    
    trainer.fit(train_loader, val_loader)
    
    # ========================================================================
    # TRAINING COMPLETED
    # ========================================================================
    print("\n" + "="*70)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("="*70)
    print(f"\nScheduler used: {cfg.scheduler}")
    print(f"Final learning rate: {trainer.get_current_lr():.6f}")
    print(f"Learning rate plot saved to: lr_schedule_{cfg.scheduler}.png")
    print("\nCheck the generated plot to visualize how the learning rate")
    print("changed during training!")
    print("="*70 + "\n")


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    """
    Entry point when script is run directly.
    
    This block ensures main() is only called when the script is executed
    directly (not when imported as a module).
    
    This is Python best practice and allows the code to be:
    - Run as a standalone script
    - Imported and used by other modules
    - Tested more easily
    """
    main()


# ============================================================================
# ADDITIONAL NOTES
# ============================================================================
"""
Understanding Learning Rate Schedulers:
----------------------------------------

1. STEPLR:
   - Multiplies LR by gamma every step_size epochs
   - Example: LR=0.1, step_size=30, gamma=0.1
     Epoch 0-29: LR=0.1
     Epoch 30-59: LR=0.01
     Epoch 60-89: LR=0.001
   - Best for: Stable training with periodic LR drops

2. MULTISTEPLR:
   - Like StepLR but with custom milestone epochs
   - Example: LR=0.1, milestones=[30,80], gamma=0.1
     Epoch 0-29: LR=0.1
     Epoch 30-79: LR=0.01
     Epoch 80+: LR=0.001
   - Best for: When you know specific epochs to reduce LR

3. EXPONENTIALLR:
   - Multiplies LR by gamma every epoch
   - Example: LR=0.1, gamma=0.95
     Epoch 0: LR=0.1
     Epoch 1: LR=0.095
     Epoch 2: LR=0.09025
   - Best for: Smooth, gradual LR decay

4. COSINEANNEALING:
   - Decreases LR following cosine function
   - Smooth transition from initial to minimum LR
   - Can be restarted for multiple cycles
   - Best for: Modern training, especially for image models

5. ONECYCLELR:
   - Increases LR to max_lr then decreases
   - Based on the 1cycle policy paper
   - Very effective for fast training
   - Best for: Fast convergence, competitive performance

6. CYCLICLR:
   - Cycles LR between base_lr and max_lr
   - Multiple cycle modes: triangular, triangular2, exp_range
   - Updates every batch (typically)
   - Best for: Avoiding local minima, finding optimal LR range

7. REDUCELRONPLATEAU:
   - Reduces LR when metric stops improving
   - Monitors validation loss or other metrics
   - Adaptive and doesn't require epoch planning
   - Best for: When you're unsure of optimal schedule

Tips for Choosing a Scheduler:
-------------------------------
- Start with ReduceLROnPlateau for baseline
- Try OneCycleLR for fast training
- Use CosineAnnealing for modern architectures
- StepLR/MultiStepLR for classical approaches
- Experiment and compare results!
"""
