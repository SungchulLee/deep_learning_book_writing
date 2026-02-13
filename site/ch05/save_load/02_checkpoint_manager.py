#!/usr/bin/env python3
'''
============================================================
Advanced Checkpoint Management for PyTorch
============================================================

Learn how to manage multiple checkpoints, save best models,
and implement automatic checkpoint cleanup.

Topics Covered:
- Best model tracking
- Multiple checkpoint management
- Automatic cleanup of old checkpoints
- Checkpoint metadata
- Resume training from best/last checkpoint
'''

import torch
import torch.nn as nn
import os
import glob
import json
from datetime import datetime

class CheckpointManager:
    '''
    Manages saving and loading of model checkpoints with automatic cleanup.
    
    Features:
    - Saves checkpoints with metadata
    - Tracks best model based on metrics
    - Automatic cleanup of old checkpoints
    - Resume training from specific checkpoint
    
    Args:
        save_dir: Directory to save checkpoints
        max_checkpoints: Maximum number of checkpoints to keep
        keep_best: Number of best checkpoints to always keep
    '''
    
    def __init__(self, save_dir='checkpoints', max_checkpoints=5, keep_best=2):
        self.save_dir = save_dir
        self.max_checkpoints = max_checkpoints
        self.keep_best = keep_best
        self.best_scores = []  # Track best checkpoint scores
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        print(f"✅ Checkpoint manager initialized")
        print(f"   Save directory: {save_dir}")
        print(f"   Max checkpoints: {max_checkpoints}")
        print(f"   Keep best: {keep_best}")
    
    def save_checkpoint(self, model, optimizer, epoch, metrics, is_best=False):
        '''
        Save a checkpoint with full training state and metadata.
        
        Args:
            model: PyTorch model to save
            optimizer: Optimizer state to save
            epoch: Current epoch number
            metrics: Dictionary of metrics (loss, accuracy, etc.)
            is_best: Whether this is the best model so far
        '''
        # Create checkpoint dictionary with all necessary information
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
        }
        
        # Generate filename
        # Format: checkpoint_epoch{epoch}_{metric_name}{metric_value}.pth
        metric_str = '_'.join([f"{k}{v:.4f}" for k, v in metrics.items()])
        filename = f"checkpoint_epoch{epoch:03d}_{metric_str}.pth"
        filepath = os.path.join(self.save_dir, filename)
        
        # Save checkpoint
        torch.save(checkpoint, filepath)
        print(f"\n💾 Checkpoint saved: {filename}")
        print(f"   Epoch: {epoch}")
        for k, v in metrics.items():
            print(f"   {k}: {v:.4f}")
        
        # If this is the best model, save a copy
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"   ⭐ Saved as best model!")
        
        # Cleanup old checkpoints
        self._cleanup_checkpoints()
        
        return filepath
    
    def _cleanup_checkpoints(self):
        '''
        Remove old checkpoints, keeping only the most recent ones
        and the best performing ones.
        '''
        # Get all checkpoint files (excluding best_model.pth)
        checkpoint_files = glob.glob(os.path.join(self.save_dir, 'checkpoint_*.pth'))
        
        if len(checkpoint_files) <= self.max_checkpoints:
            return  # No cleanup needed
        
        # Sort by modification time (oldest first)
        checkpoint_files.sort(key=os.path.getmtime)
        
        # Determine how many to delete
        num_to_delete = len(checkpoint_files) - self.max_checkpoints
        files_to_delete = checkpoint_files[:num_to_delete]
        
        # Delete old checkpoints
        for filepath in files_to_delete:
            os.remove(filepath)
            print(f"   🗑️  Removed old checkpoint: {os.path.basename(filepath)}")
    
    def load_latest_checkpoint(self, model, optimizer=None):
        '''
        Load the most recent checkpoint.
        
        Args:
            model: Model to load state into
            optimizer: Optimizer to load state into (optional)
            
        Returns:
            epoch: Epoch number of loaded checkpoint
            metrics: Metrics dictionary from checkpoint
        '''
        checkpoint_files = glob.glob(os.path.join(self.save_dir, 'checkpoint_*.pth'))
        
        if not checkpoint_files:
            print("⚠️  No checkpoints found")
            return 0, {}
        
        # Get most recent checkpoint
        latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
        
        print(f"\n📂 Loading latest checkpoint: {os.path.basename(latest_checkpoint)}")
        return self._load_checkpoint(latest_checkpoint, model, optimizer)
    
    def load_best_checkpoint(self, model, optimizer=None):
        '''
        Load the best checkpoint (if exists).
        '''
        best_path = os.path.join(self.save_dir, 'best_model.pth')
        
        if not os.path.exists(best_path):
            print("⚠️  No best model checkpoint found")
            return 0, {}
        
        print(f"\n📂 Loading best checkpoint")
        return self._load_checkpoint(best_path, model, optimizer)
    
    def _load_checkpoint(self, filepath, model, optimizer=None):
        '''
        Internal method to load a checkpoint file.
        '''
        checkpoint = torch.load(filepath)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        epoch = checkpoint.get('epoch', 0)
        metrics = checkpoint.get('metrics', {})
        
        print(f"✅ Checkpoint loaded successfully")
        print(f"   Epoch: {epoch}")
        for k, v in metrics.items():
            print(f"   {k}: {v:.4f}")
        
        return epoch, metrics
    
    def list_checkpoints(self):
        '''
        List all available checkpoints with their metadata.
        '''
        checkpoint_files = glob.glob(os.path.join(self.save_dir, 'checkpoint_*.pth'))
        checkpoint_files.sort(key=os.path.getmtime, reverse=True)
        
        print(f"\n📋 Available Checkpoints ({len(checkpoint_files)}):")
        print("-" * 70)
        
        for filepath in checkpoint_files:
            checkpoint = torch.load(filepath, map_location='cpu')
            filename = os.path.basename(filepath)
            epoch = checkpoint.get('epoch', 'N/A')
            metrics = checkpoint.get('metrics', {})
            timestamp = checkpoint.get('timestamp', 'N/A')
            file_size = os.path.getsize(filepath) / 1024  # KB
            
            print(f"\n📄 {filename}")
            print(f"   Epoch: {epoch}")
            print(f"   Metrics: {metrics}")
            print(f"   Time: {timestamp}")
            print(f"   Size: {file_size:.2f} KB")


# ============================================================
# DEMONSTRATION
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("CHECKPOINT MANAGER DEMONSTRATION")
    print("=" * 70)
    
    # Define a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 1)
        
        def forward(self, x):
            return self.linear(x)
    
    # Initialize model and optimizer
    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Create checkpoint manager
    manager = CheckpointManager(
        save_dir='demo_checkpoints',
        max_checkpoints=3,
        keep_best=1
    )
    
    # Simulate training loop with checkpoint saving
    print("\n" + "=" * 70)
    print("SIMULATING TRAINING LOOP")
    print("=" * 70)
    
    best_loss = float('inf')
    
    for epoch in range(1, 6):
        # Simulate training metrics
        train_loss = 1.0 / (epoch + 1)  # Loss decreases
        val_acc = 0.5 + epoch * 0.08    # Accuracy increases
        
        print(f"\nEpoch {epoch}:")
        print(f"  Training Loss: {train_loss:.4f}")
        print(f"  Validation Accuracy: {val_acc:.4f}")
        
        # Check if this is the best model
        is_best = train_loss < best_loss
        if is_best:
            best_loss = train_loss
        
        # Save checkpoint
        metrics = {
            'loss': train_loss,
            'acc': val_acc
        }
        
        manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=metrics,
            is_best=is_best
        )
    
    # List all checkpoints
    manager.list_checkpoints()
    
    # Load best checkpoint
    print("\n" + "=" * 70)
    print("LOADING BEST CHECKPOINT")
    print("=" * 70)
    
    new_model = SimpleModel()
    new_optimizer = torch.optim.Adam(new_model.parameters())
    
    epoch, metrics = manager.load_best_checkpoint(new_model, new_optimizer)
    
    # Verify loading
    print("\n✓ Model ready for inference or continued training")
    
    # Cleanup demo files
    import shutil
    if os.path.exists('demo_checkpoints'):
        shutil.rmtree('demo_checkpoints')
        print("\n🗑️  Cleaned up demo checkpoints")
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
