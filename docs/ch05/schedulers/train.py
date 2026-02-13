"""
Training Module for Learning Rate Scheduler Demo
=================================================

This module contains the Trainer class which handles:
- Model training loop
- Scheduler initialization and management
- Loss and accuracy tracking
- Learning rate history tracking
- Visualization of results

The Trainer class is designed to make it easy to experiment with
different learning rate schedulers by encapsulating all training logic.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import (
    StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR,
    OneCycleLR, CyclicLR, ReduceLROnPlateau
)
from typing import Optional, List
import time

from scheduler.utils import (
    AverageMeter, calculate_accuracy, plot_learning_rate_schedule,
    plot_training_curves, print_epoch_summary
)


# ============================================================================
# TRAINER CLASS
# ============================================================================
class Trainer:
    """
    Trainer class for managing model training with learning rate schedulers.
    
    This class encapsulates all training logic including:
    - Forward and backward passes
    - Optimizer steps
    - Scheduler updates
    - Metric tracking
    - Visualization
    
    The class is designed to work with any of PyTorch's learning rate
    schedulers, making it easy to compare their effects.
    
    Attributes:
        model (nn.Module): Neural network model
        device (torch.device): Device for computation (CPU or GPU)
        optimizer (optim.Optimizer): Optimizer (Adam by default)
        scheduler: Learning rate scheduler
        criterion (nn.Module): Loss function (CrossEntropyLoss)
        config: Configuration object with all hyperparameters
    """
    
    def __init__(self, cfg, model: nn.Module, device: torch.device, steps_per_epoch: int):
        """
        Initialize the trainer.
        
        Args:
            cfg: Configuration object containing all hyperparameters
            model (nn.Module): Neural network model to train
            device (torch.device): Device for computation
            steps_per_epoch (int): Number of training batches per epoch
                                  (needed for OneCycleLR and CyclicLR)
        """
        self.cfg = cfg
        self.model = model
        self.device = device
        self.steps_per_epoch = steps_per_epoch
        
        # ====================================================================
        # INITIALIZE OPTIMIZER
        # ====================================================================
        # Using Adam optimizer as it generally works well without much tuning
        # You could also experiment with SGD, AdamW, etc.
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=cfg.lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-4  # Small L2 regularization
        )
        
        # ====================================================================
        # INITIALIZE LOSS FUNCTION
        # ====================================================================
        # CrossEntropyLoss combines LogSoftmax and NLLLoss
        # It expects:
        # - Input: raw logits (not probabilities), shape (batch_size, num_classes)
        # - Target: class indices, shape (batch_size,)
        self.criterion = nn.CrossEntropyLoss()
        
        # ====================================================================
        # INITIALIZE SCHEDULER
        # ====================================================================
        # Select and initialize the appropriate scheduler based on config
        self.scheduler = self._create_scheduler()
        
        # ====================================================================
        # TRACKING VARIABLES
        # ====================================================================
        # Lists to store history for plotting
        self.train_losses = []      # Training loss per epoch
        self.val_losses = []        # Validation loss per epoch
        self.train_accs = []        # Training accuracy per epoch
        self.val_accs = []          # Validation accuracy per epoch
        self.lr_history = []        # Learning rate history
        
        # Best model tracking (for saving best checkpoint)
        self.best_val_acc = 0.0
        self.best_epoch = 0
        
        print(f"\nTrainer initialized:")
        print(f"  Optimizer: Adam")
        print(f"  Scheduler: {cfg.scheduler}")
        print(f"  Initial LR: {cfg.lr}")
        print(f"  Steps per epoch: {steps_per_epoch}")
    
    def _create_scheduler(self):
        """
        Create the appropriate learning rate scheduler based on configuration.
        
        Returns:
            Scheduler object from torch.optim.lr_scheduler
        
        This method handles all scheduler types and their specific parameters.
        Each scheduler has different behavior and use cases.
        """
        cfg = self.cfg
        
        print(f"\nCreating {cfg.scheduler} scheduler...")
        
        # ====================================================================
        # STEPLR
        # ====================================================================
        if cfg.scheduler == 'step':
            """
            StepLR: Decays learning rate by gamma every step_size epochs.
            
            Formula: LR = initial_lr * (gamma ^ (epoch // step_size))
            
            Example with lr=0.1, step_size=30, gamma=0.1:
              Epoch 0-29: LR = 0.1
              Epoch 30-59: LR = 0.01
              Epoch 60-89: LR = 0.001
            
            Use case: Classic approach, works well when you know good milestone epochs
            """
            scheduler = StepLR(
                self.optimizer,
                step_size=cfg.step_size,
                gamma=cfg.gamma
            )
            print(f"  Step size: {cfg.step_size}")
            print(f"  Gamma: {cfg.gamma}")
        
        # ====================================================================
        # MULTISTEPLR
        # ====================================================================
        elif cfg.scheduler == 'multistep':
            """
            MultiStepLR: Decays learning rate by gamma at specific milestone epochs.
            
            Formula: LR is multiplied by gamma at each milestone
            
            Example with lr=0.1, milestones=[30,80], gamma=0.1:
              Epoch 0-29: LR = 0.1
              Epoch 30-79: LR = 0.01
              Epoch 80+: LR = 0.001
            
            Use case: When you want precise control over when to reduce LR
            """
            scheduler = MultiStepLR(
                self.optimizer,
                milestones=cfg.milestones,
                gamma=cfg.gamma
            )
            print(f"  Milestones: {cfg.milestones}")
            print(f"  Gamma: {cfg.gamma}")
        
        # ====================================================================
        # EXPONENTIALLR
        # ====================================================================
        elif cfg.scheduler == 'exponential':
            """
            ExponentialLR: Decays learning rate by gamma every epoch.
            
            Formula: LR = initial_lr * (gamma ^ epoch)
            
            Example with lr=0.1, gamma=0.95:
              Epoch 0: LR = 0.1
              Epoch 1: LR = 0.095
              Epoch 2: LR = 0.09025
              Epoch 10: LR = 0.0599
            
            Use case: Smooth exponential decay, no sudden drops
            """
            scheduler = ExponentialLR(
                self.optimizer,
                gamma=cfg.gamma
            )
            print(f"  Gamma: {cfg.gamma}")
        
        # ====================================================================
        # COSINEANNEALING
        # ====================================================================
        elif cfg.scheduler == 'cosine':
            """
            CosineAnnealingLR: Decreases LR following a cosine curve.
            
            Formula: LR = eta_min + (initial_lr - eta_min) * (1 + cos(pi * epoch / T_max)) / 2
            
            Characteristics:
            - Smooth decrease from initial_lr to eta_min
            - Fast decrease at start and end, slower in middle
            - Can be restarted for multiple cycles
            
            Use case: Modern approach, popular for image classification,
                     especially with warm restarts
            """
            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=cfg.t_max,
                eta_min=cfg.eta_min
            )
            print(f"  T_max: {cfg.t_max}")
            print(f"  Eta_min: {cfg.eta_min}")
        
        # ====================================================================
        # ONECYCLELR
        # ====================================================================
        elif cfg.scheduler == 'onecycle':
            """
            OneCycleLR: One cycle learning rate policy.
            
            Based on Leslie Smith's 1cycle policy paper.
            
            Phases:
            1. Warmup: LR increases from initial to max_lr
            2. Annealing: LR decreases from max_lr to initial (or lower)
            
            Benefits:
            - Very fast convergence
            - Often achieves better final performance
            - Regularization effect from high learning rates
            
            Use case: When you want fast training with good results,
                     especially for modern architectures
            """
            scheduler = OneCycleLR(
                self.optimizer,
                max_lr=cfg.max_lr,
                steps_per_epoch=self.steps_per_epoch,
                epochs=cfg.epochs,
                pct_start=cfg.pct_start,
                anneal_strategy=cfg.anneal_strategy
            )
            print(f"  Max LR: {cfg.max_lr}")
            print(f"  Steps per epoch: {self.steps_per_epoch}")
            print(f"  Pct start: {cfg.pct_start}")
            print(f"  Anneal strategy: {cfg.anneal_strategy}")
        
        # ====================================================================
        # CYCLICLR
        # ====================================================================
        elif cfg.scheduler == 'cyclical':
            """
            CyclicLR: Cyclically varies learning rate between bounds.
            
            LR cycles between base_lr and max_lr with different modes:
            - triangular: Constant amplitude
            - triangular2: Amplitude halves each cycle
            - exp_range: Amplitude decays exponentially
            
            Benefits:
            - Helps escape local minima
            - Can find optimal LR range
            - Usually updated per batch, not per epoch
            
            Use case: When you want to explore LR range or avoid local minima
            """
            scheduler = CyclicLR(
                self.optimizer,
                base_lr=cfg.base_lr,
                max_lr=cfg.max_lr,
                step_size_up=cfg.step_size_up,
                mode=cfg.mode,
                cycle_momentum=False  # Disable momentum cycling for simplicity
            )
            print(f"  Base LR: {cfg.base_lr}")
            print(f"  Max LR: {cfg.max_lr}")
            print(f"  Step size up: {cfg.step_size_up}")
            print(f"  Mode: {cfg.mode}")
        
        # ====================================================================
        # REDUCELRONPLATEAU
        # ====================================================================
        elif cfg.scheduler == 'plateau':
            """
            ReduceLROnPlateau: Reduce LR when metric plateaus.
            
            Monitors a metric (usually validation loss) and reduces LR
            when it stops improving for 'patience' epochs.
            
            Formula: If no improvement for 'patience' epochs:
                    new_lr = current_lr * factor
            
            Benefits:
            - Adaptive: responds to training dynamics
            - No need to pre-plan schedule
            - Reduces LR only when needed
            
            Use case: When you're unsure of the right schedule,
                     or for adaptive training
            """
            scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode=cfg.mode_plateau,
                factor=cfg.factor,
                patience=cfg.patience,
                threshold=cfg.threshold,
                verbose=True  # Print when LR is reduced
            )
            print(f"  Mode: {cfg.mode_plateau}")
            print(f"  Factor: {cfg.factor}")
            print(f"  Patience: {cfg.patience}")
            print(f"  Threshold: {cfg.threshold}")
        
        else:
            raise ValueError(f"Unknown scheduler: {cfg.scheduler}")
        
        return scheduler
    
    def get_current_lr(self) -> float:
        """
        Get the current learning rate from the optimizer.
        
        Returns:
            float: Current learning rate
        """
        return self.optimizer.param_groups[0]['lr']
    
    def train_one_epoch(self, train_loader) -> tuple:
        """
        Train the model for one epoch.
        
        Args:
            train_loader (DataLoader): Training data loader
        
        Returns:
            tuple: (average_train_loss, average_train_accuracy)
        """
        # Set model to training mode
        # This enables dropout, batch norm updates, etc.
        self.model.train()
        
        # Metrics tracking
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        
        # Iterate over batches
        for batch_idx, (features, labels) in enumerate(train_loader):
            # Move data to device
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            # ================================================================
            # FORWARD PASS
            # ================================================================
            # Zero gradients from previous iteration
            self.optimizer.zero_grad()
            
            # Forward pass through model
            logits = self.model(features)
            
            # Compute loss
            loss = self.criterion(logits, labels)
            
            # ================================================================
            # BACKWARD PASS
            # ================================================================
            # Compute gradients
            loss.backward()
            
            # Update weights
            self.optimizer.step()
            
            # ================================================================
            # SCHEDULER UPDATE (if batch-level)
            # ================================================================
            # Some schedulers update after each batch
            if self.cfg.scheduler_step_on == 'batch':
                self.scheduler.step()
                self.lr_history.append(self.get_current_lr())
            
            # ================================================================
            # TRACK METRICS
            # ================================================================
            # Calculate accuracy
            with torch.no_grad():
                predictions = logits.argmax(dim=1)
                batch_acc = calculate_accuracy(predictions, labels)
            
            # Update meters
            batch_size = labels.size(0)
            loss_meter.update(loss.item(), batch_size)
            acc_meter.update(batch_acc, batch_size)
        
        return loss_meter.avg, acc_meter.avg
    
    @torch.no_grad()
    def evaluate(self, val_loader) -> tuple:
        """
        Evaluate the model on validation set.
        
        Args:
            val_loader (DataLoader): Validation data loader
        
        Returns:
            tuple: (average_val_loss, average_val_accuracy)
        """
        # Set model to evaluation mode
        # This disables dropout, uses running stats for batch norm, etc.
        self.model.eval()
        
        # Metrics tracking
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        
        # Iterate over validation batches
        # No gradient computation needed for evaluation
        for features, labels in val_loader:
            # Move data to device
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            logits = self.model(features)
            
            # Compute loss
            loss = self.criterion(logits, labels)
            
            # Calculate accuracy
            predictions = logits.argmax(dim=1)
            batch_acc = calculate_accuracy(predictions, labels)
            
            # Update meters
            batch_size = labels.size(0)
            loss_meter.update(loss.item(), batch_size)
            acc_meter.update(batch_acc, batch_size)
        
        return loss_meter.avg, acc_meter.avg
    
    def fit(self, train_loader, val_loader):
        """
        Train the model for multiple epochs.
        
        This is the main training loop that:
        1. Trains for one epoch
        2. Evaluates on validation set
        3. Updates learning rate scheduler
        4. Tracks metrics
        5. Saves best model
        6. Creates visualizations
        
        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
        """
        print("\n" + "="*70)
        print("STARTING TRAINING LOOP")
        print("="*70)
        
        # Record initial learning rate
        if self.cfg.scheduler_step_on != 'batch':
            self.lr_history.append(self.get_current_lr())
        
        # Training loop
        for epoch in range(self.cfg.epochs):
            # Start timer
            epoch_start_time = time.time()
            
            # ================================================================
            # TRAIN FOR ONE EPOCH
            # ================================================================
            train_loss, train_acc = self.train_one_epoch(train_loader)
            
            # ================================================================
            # EVALUATE ON VALIDATION SET
            # ================================================================
            val_loss, val_acc = self.evaluate(val_loader)
            
            # ================================================================
            # UPDATE SCHEDULER (if epoch-level or plateau)
            # ================================================================
            if self.cfg.scheduler_step_on == 'epoch':
                self.scheduler.step()
                self.lr_history.append(self.get_current_lr())
            elif self.cfg.scheduler_step_on == 'val_loss':
                # ReduceLROnPlateau requires the metric to monitor
                self.scheduler.step(val_loss)
                self.lr_history.append(self.get_current_lr())
            
            # ================================================================
            # TRACK METRICS
            # ================================================================
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            
            # Update best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
            
            # ================================================================
            # PRINT PROGRESS
            # ================================================================
            epoch_time = time.time() - epoch_start_time
            print_epoch_summary(
                epoch=epoch,
                total_epochs=self.cfg.epochs,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                current_lr=self.get_current_lr(),
                epoch_time=epoch_time
            )
        
        # ====================================================================
        # TRAINING COMPLETED
        # ====================================================================
        print("\n" + "="*70)
        print("TRAINING COMPLETED")
        print("="*70)
        print(f"\nBest validation accuracy: {self.best_val_acc:.2f}% "
              f"(Epoch {self.best_epoch + 1})")
        print(f"Final validation accuracy: {self.val_accs[-1]:.2f}%")
        print(f"Final learning rate: {self.get_current_lr():.6f}")
        
        # ====================================================================
        # CREATE VISUALIZATIONS
        # ====================================================================
        print("\nCreating visualizations...")
        
        # Plot learning rate schedule
        plot_learning_rate_schedule(
            self.lr_history,
            scheduler_name=self.cfg.scheduler,
            save_path=f'lr_schedule_{self.cfg.scheduler}.png'
        )
        
        # Plot comprehensive training curves
        plot_training_curves(
            self.train_losses,
            self.val_losses,
            self.train_accs,
            self.val_accs,
            self.lr_history,
            scheduler_name=self.cfg.scheduler,
            save_path=f'training_curves_{self.cfg.scheduler}.png'
        )
        
        print("\nAll visualizations saved!")


# ============================================================================
# MODULE TEST
# ============================================================================
if __name__ == '__main__':
    """
    Test the trainer module independently.
    """
    print("Trainer module loaded successfully!")
    print("\nThis module should be used through the main script:")
    print("  python scheduler.py --scheduler step")
