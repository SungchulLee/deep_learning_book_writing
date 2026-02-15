"""
Real-Time Training Progress Visualization Utility

This module provides utilities for visualizing training progress in real-time:
1. Accumulator: Accumulates and tracks statistics (loss, accuracy) across batches
2. Animator: Plots training curves with multiple subplots that update each epoch

These tools help monitor training health and catch issues early (e.g., diverging
loss, overfitting, learning rate too high).

Educational purpose: Chapter 5 - Training Utilities & Monitoring
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from typing import List, Dict, Optional, Tuple


# ============================================================================
# Accumulator Class
# ============================================================================

class Accumulator:
    """
    Accumulates numerical values across training batches/epochs.

    Useful for computing average loss, accuracy, and other metrics without
    storing individual batch values in memory.

    Example:
        acc = Accumulator(2)  # Track 2 metrics
        for X_batch, y_batch in dataloader:
            loss, acc_val = train_step(X_batch, y_batch)
            acc.add(loss * len(X_batch), acc_val * len(X_batch))
        avg_loss, avg_acc = acc.values()
    """

    def __init__(self, n: int):
        """
        Initialize accumulator for n metrics.

        Args:
            n: Number of metrics to track
        """
        self.data = [0.0] * n

    def add(self, *args):
        """
        Add values to the accumulator.

        Args:
            *args: Variable number of values to add (must match n)
        """
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        """Reset all accumulated values to zero."""
        self.data = [0.0] * len(self.data)

    def values(self) -> Tuple[float, ...]:
        """Return accumulated values as tuple."""
        return tuple(self.data)

    def __str__(self) -> str:
        """String representation of accumulated values."""
        return f"Accumulator({self.data})"


# ============================================================================
# Animator Class
# ============================================================================

class Animator:
    """
    Plots training metrics with multiple subplots that update in real-time.

    The animator maintains one or more subplots that are updated at the end of
    each training epoch. This allows monitoring of loss, accuracy, and other
    metrics as training progresses.

    Example:
        animator = Animator(xlabel='epoch', ylabel='loss', xlim=[1, 10],
                           legend=['train loss', 'val loss'])
        for epoch in range(10):
            # ... train for one epoch ...
            animator.add(epoch+1, train_loss, val_loss)
    """

    def __init__(
        self,
        xlabel: str = 'epoch',
        ylabel: str = 'loss',
        xlim: Optional[List[float]] = None,
        ylim: Optional[List[float]] = None,
        legend: Optional[List[str]] = None,
        nrows: int = 1,
        ncols: int = 1,
        figsize: Tuple[int, int] = (7, 4)
    ):
        """
        Initialize the animator.

        Args:
            xlabel: Label for x-axis
            ylabel: Label for y-axis
            xlim: X-axis limits [min, max]
            ylim: Y-axis limits [min, max]
            legend: List of legend labels for each metric
            nrows: Number of subplot rows
            ncols: Number of subplot columns
            figsize: Figure size (width, height)
        """
        # Build a dictionary of axes; key is unique for each subplot
        if legend is None:
            legend = []

        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)

        # Handle case of single subplot (axes is not a list)
        if nrows * ncols == 1:
            self.axes = [self.axes]
        else:
            self.axes = self.axes.flatten()

        # Set up axes with labels and limits
        for ax in self.axes:
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            if xlim is not None:
                ax.set_xlim(*xlim)
            if ylim is not None:
                ax.set_ylim(*ylim)
            if legend:
                ax.legend(legend)
            ax.grid(True, alpha=0.3)

        # Data storage
        self.X, self.Y, self.axes_idx = [], [], 0
        self.legend = legend
        self.xlabel = xlabel
        self.ylabel = ylabel

    def add(self, x: float, *y: float):
        """
        Add a data point to the animator.

        Args:
            x: X-axis value (e.g., epoch number)
            *y: Y-axis values (can add multiple metrics at once)
        """
        if not hasattr(self, 'X'):
            self.X, self.Y = [], []

        # x is a scalar, y can be multiple values
        self.X.append(x)

        # Ensure Y has enough space for all y values
        if len(self.Y) == 0:
            self.Y = [[y_val] for y_val in y]
        else:
            for i, y_val in enumerate(y):
                self.Y[i].append(y_val)

    def draw(self):
        """Update the plot with current data."""
        # Choose which axes to use (for multi-plot support)
        ax = self.axes[self.axes_idx % len(self.axes)]

        ax.cla()  # Clear current axes

        # Plot each metric
        for i, y_vals in enumerate(self.Y):
            ax.plot(self.X, y_vals, linewidth=1.5)

        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)

        # Add legend if available
        if self.legend and len(self.legend) == len(self.Y):
            ax.legend(self.legend)

        ax.grid(True, alpha=0.3)

        # Pause to allow plot update
        if not hasattr(plt, 'is_interactive'):
            plt.pause(0.001)

    def next_subplot(self):
        """Move to next subplot (for multi-plot support)."""
        self.axes_idx += 1

    def show(self):
        """Display the plot."""
        plt.tight_layout()
        plt.show()


# ============================================================================
# Simple MLP for Demo
# ============================================================================

class SimpleMLP(nn.Module):
    """Simple multi-layer perceptron for MNIST classification."""

    def __init__(self, input_dim: int = 784, hidden_dim: int = 128, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # Flatten input
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.net(x)


# ============================================================================
# Training Utilities
# ============================================================================

def load_mnist_data(
    train_size: int = 1000,
    val_size: int = 200,
    batch_size: int = 32
) -> Tuple[DataLoader, DataLoader]:
    """
    Load and prepare MNIST-like synthetic data.

    For demonstration purposes, this creates synthetic data similar to MNIST
    (28x28 = 784 features, 10 classes).

    Args:
        train_size: Number of training samples
        val_size: Number of validation samples
        batch_size: Batch size for DataLoader

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Generate synthetic data (in practice, use torchvision.datasets.MNIST)
    X_train = torch.randn(train_size, 784)
    y_train = torch.randint(0, 10, (train_size,))

    X_val = torch.randn(val_size, 784)
    y_val = torch.randint(0, 10, (val_size,))

    # Create DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def evaluate_accuracy(
    model: nn.Module,
    data_loader: DataLoader,
    device: str = 'cpu'
) -> float:
    """
    Evaluate model accuracy on a dataset.

    Args:
        model: Neural network model
        data_loader: DataLoader for evaluation
        device: Device to run on

    Returns:
        Accuracy as float (0 to 1)
    """
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total if total > 0 else 0.0


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: str = 'cpu'
) -> Tuple[float, float]:
    """
    Train model for one epoch.

    Args:
        model: Neural network model
        train_loader: DataLoader for training
        optimizer: Optimizer (SGD, Adam, etc.)
        criterion: Loss function
        device: Device to train on

    Returns:
        Tuple of (average_loss, average_accuracy)
    """
    model.train()
    acc = Accumulator(2)  # Track loss and correct count

    for X, y in train_loader:
        X, y = X.to(device), y.to(device)

        # Forward pass
        logits = model(X)
        loss = criterion(logits, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate metrics
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            correct = (preds == y).sum().item()

        acc.add(loss.item() * len(y), correct)

    # Return average loss and accuracy
    total_loss, total_correct = acc.values()
    total_samples = len(train_loader.dataset)

    return total_loss / total_samples, total_correct / total_samples


# ============================================================================
# Main Training Demo
# ============================================================================

def main():
    """
    Demo: Train a simple MLP on MNIST-like data with live visualization.
    Shows how Animator and Accumulator monitor training progress.
    """
    print("Training Progress Visualization with Animator")
    print("=" * 70)

    # Configuration
    num_epochs = 20
    learning_rate = 0.001
    batch_size = 32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}, Learning rate: {learning_rate}, Batch size: {batch_size}")

    # Create data
    print("\nLoading synthetic MNIST-like data...")
    train_loader, val_loader = load_mnist_data(
        train_size=2000, val_size=500, batch_size=batch_size
    )
    print(f"Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")

    # Create model, optimizer, loss
    model = SimpleMLP(input_dim=784, hidden_dim=256, num_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create animator for tracking metrics
    animator = Animator(
        xlabel='epoch',
        ylabel='metric',
        xlim=[1, num_epochs],
        ylim=[0, 1],
        legend=['train loss', 'train acc', 'val acc'],
        figsize=(8, 5)
    )

    print("\nTraining...")
    print("-" * 70)

    # Training loop
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        # Train for one epoch
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device=device
        )

        # Evaluate on validation set
        val_acc = evaluate_accuracy(model, val_loader, device=device)

        # Track best validation accuracy
        best_val_acc = max(best_val_acc, val_acc)

        # Add data to animator and draw
        animator.add(epoch + 1, train_loss, train_acc, val_acc)
        animator.draw()

        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:2d}/{num_epochs}: "
                  f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                  f"val_acc={val_acc:.4f}")

    print("-" * 70)
    print(f"\nTraining complete!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    # Display the plot
    animator.show()


if __name__ == "__main__":
    main()
