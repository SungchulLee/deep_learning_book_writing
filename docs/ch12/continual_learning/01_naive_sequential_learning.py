"""
Module 57: Continual Learning - Beginner Level
File 2: Naive Sequential Learning (Baseline)

This script implements the naive sequential learning approach, which serves
as a baseline for continual learning experiments. We'll also implement
proper evaluation metrics to quantify forgetting.

Learning Objectives:
1. Implement proper continual learning evaluation protocol
2. Calculate standard continual learning metrics
3. Understand the baseline against which to compare methods
4. Learn how to structure continual learning experiments

Mathematical Metrics:
1. Average Accuracy (AA): (1/T) Σ Acc_{i,T}
2. Backward Transfer (BWT): (1/(T-1)) Σ (Acc_{i,T} - Acc_{i,i})
3. Forward Transfer (FWT): (1/(T-1)) Σ (Acc_{i,i-1} - Acc_{i,init})
4. Learning Accuracy (LA): (1/T) Σ Acc_{i,i}
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import time
from dataclasses import dataclass


@dataclass
class ContinualLearningMetrics:
    """
    Data class to store all continual learning metrics.
    
    Attributes:
        average_accuracy: Mean accuracy across all tasks at end
        backward_transfer: Measure of forgetting (negative = forgot)
        forward_transfer: Measure of knowledge transfer to new tasks
        learning_accuracy: Average accuracy immediately after learning
        forgetting_per_task: Forgetting amount for each individual task
        accuracy_matrix: Full matrix of all accuracies
    """
    average_accuracy: float
    backward_transfer: float
    forward_transfer: float
    learning_accuracy: float
    forgetting_per_task: List[float]
    accuracy_matrix: np.ndarray


class ContinualLearner:
    """
    Base class for continual learning experiments.
    
    This class provides:
    - Standard evaluation protocol
    - Metric calculation
    - Experiment tracking
    - Visualization utilities
    
    Subclasses can override specific methods to implement different
    continual learning strategies.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 device: torch.device,
                 learning_rate: float = 0.001):
        """
        Initialize continual learner.
        
        Args:
            model: Neural network model
            device: Device to train on (CPU or GPU)
            learning_rate: Learning rate for optimizer
        """
        self.model = model
        self.device = device
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()
        
        # Tracking variables
        self.current_task = 0
        self.task_train_loaders = []
        self.task_test_loaders = []
        self.accuracy_matrix = None
        
    def train_task(self, 
                   train_loader: DataLoader,
                   task_id: int,
                   epochs: int = 5) -> List[float]:
        """
        Train on a single task.
        
        This is the naive approach: just train on current task data
        with standard supervised learning. No special techniques to
        prevent forgetting.
        
        Args:
            train_loader: DataLoader for current task
            task_id: ID of current task
            epochs: Number of training epochs
        
        Returns:
            List of losses per epoch
        """
        # Create fresh optimizer for this task
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        self.model.train()
        losses = []
        
        print(f"\n{'=' * 60}")
        print(f"Training Task {task_id}")
        print('=' * 60)
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                # Move to device
                data, target = data.to(self.device), target.to(self.device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Track statistics
                epoch_loss += loss.item()
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
            
            # Calculate metrics
            avg_loss = epoch_loss / len(train_loader)
            accuracy = 100.0 * correct / total
            losses.append(avg_loss)
            
            print(f"  Epoch {epoch + 1}/{epochs} - "
                  f"Loss: {avg_loss:.4f}, "
                  f"Train Acc: {accuracy:.2f}%")
        
        return losses
    
    def evaluate_task(self, 
                     test_loader: DataLoader,
                     task_id: int) -> Tuple[float, float]:
        """
        Evaluate model on a single task.
        
        Args:
            test_loader: DataLoader for test data
            task_id: ID of task being evaluated
        
        Returns:
            Tuple of (accuracy, loss)
        """
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100.0 * correct / total
        avg_loss = total_loss / len(test_loader)
        
        return accuracy, avg_loss
    
    def train_continual(self,
                       train_loaders: List[DataLoader],
                       test_loaders: List[DataLoader],
                       epochs_per_task: int = 5) -> ContinualLearningMetrics:
        """
        Main continual learning loop.
        
        Process:
        1. For each task sequentially:
           a. Train on task data
           b. Evaluate on ALL tasks (including previous ones)
           c. Record accuracies
        2. Calculate continual learning metrics
        3. Return comprehensive results
        
        Args:
            train_loaders: List of DataLoaders for training
            test_loaders: List of DataLoaders for testing
            epochs_per_task: Epochs to train per task
        
        Returns:
            ContinualLearningMetrics object with all metrics
        """
        num_tasks = len(train_loaders)
        self.task_train_loaders = train_loaders
        self.task_test_loaders = test_loaders
        
        # Initialize accuracy matrix
        # accuracy_matrix[i,j] = accuracy on task i after training on task j
        self.accuracy_matrix = np.zeros((num_tasks, num_tasks))
        
        # Track initial random accuracy (before any training)
        print("\n" + "=" * 70)
        print("INITIAL EVALUATION (Before Any Training)")
        print("=" * 70)
        for task_id in range(num_tasks):
            acc, _ = self.evaluate_task(test_loaders[task_id], task_id)
            print(f"Task {task_id}: {acc:.2f}% (random guess ~ {100/2:.1f}%)")
        
        # Sequential training on each task
        for task_id in range(num_tasks):
            # Train on current task
            self.train_task(
                train_loader=train_loaders[task_id],
                task_id=task_id,
                epochs=epochs_per_task
            )
            
            # Evaluate on ALL tasks seen so far
            print(f"\n{'=' * 60}")
            print(f"Evaluation after Task {task_id}")
            print('=' * 60)
            
            for eval_task_id in range(task_id + 1):
                acc, loss = self.evaluate_task(
                    test_loader=test_loaders[eval_task_id],
                    task_id=eval_task_id
                )
                
                # Store in accuracy matrix
                self.accuracy_matrix[eval_task_id, task_id] = acc
                
                # Print with forgetting information if applicable
                if eval_task_id == task_id:
                    print(f"  Task {eval_task_id}: {acc:.2f}% (just learned)")
                elif eval_task_id < task_id:
                    original_acc = self.accuracy_matrix[eval_task_id, eval_task_id]
                    forgetting = original_acc - acc
                    print(f"  Task {eval_task_id}: {acc:.2f}% "
                          f"(was {original_acc:.2f}%, forgot {forgetting:.2f}%)")
        
        # Calculate and return metrics
        metrics = self.calculate_metrics()
        return metrics
    
    def calculate_metrics(self) -> ContinualLearningMetrics:
        """
        Calculate all continual learning metrics from accuracy matrix.
        
        Metrics Explained:
        
        1. Average Accuracy (AA): 
           Mean accuracy on all tasks after learning all tasks
           AA = (1/T) Σ_{i=1}^T Acc_{i,T}
        
        2. Backward Transfer (BWT):
           Measures forgetting - how much accuracy changed on old tasks
           BWT = (1/(T-1)) Σ_{i=1}^{T-1} (Acc_{i,T} - Acc_{i,i})
           Negative BWT = forgetting, Positive BWT = improvement
        
        3. Forward Transfer (FWT):
           Measures ability to use past knowledge for new tasks
           Measures accuracy on task i before training on it
           FWT = (1/(T-1)) Σ_{i=2}^T (Acc_{i,i-1} - baseline)
        
        4. Learning Accuracy (LA):
           Average accuracy immediately after learning each task
           LA = (1/T) Σ_{i=1}^T Acc_{i,i}
        
        Returns:
            ContinualLearningMetrics object
        """
        num_tasks = self.accuracy_matrix.shape[0]
        
        # 1. Average Accuracy (final column)
        final_accuracies = self.accuracy_matrix[:, -1]
        average_accuracy = np.mean(final_accuracies)
        
        # 2. Backward Transfer (forgetting)
        # Compare final accuracy to accuracy right after learning
        backward_transfer = 0.0
        forgetting_per_task = []
        
        for i in range(num_tasks - 1):  # Exclude last task (no chance to forget)
            initial_acc = self.accuracy_matrix[i, i]  # Right after learning
            final_acc = self.accuracy_matrix[i, num_tasks - 1]  # After all tasks
            forgetting = final_acc - initial_acc  # Negative = forgot
            backward_transfer += forgetting
            forgetting_per_task.append(forgetting)
        
        if num_tasks > 1:
            backward_transfer /= (num_tasks - 1)
        
        # 3. Forward Transfer (not applicable for naive learning)
        # In naive learning, there's no mechanism for forward transfer
        # We would need to evaluate on task i before training on it
        forward_transfer = 0.0  # Placeholder for naive baseline
        
        # 4. Learning Accuracy (diagonal of matrix)
        learning_accuracy = np.mean(np.diag(self.accuracy_matrix))
        
        return ContinualLearningMetrics(
            average_accuracy=average_accuracy,
            backward_transfer=backward_transfer,
            forward_transfer=forward_transfer,
            learning_accuracy=learning_accuracy,
            forgetting_per_task=forgetting_per_task,
            accuracy_matrix=self.accuracy_matrix
        )


def create_simple_model(input_size: int = 784,
                       hidden_size: int = 256,
                       num_classes: int = 2) -> nn.Module:
    """
    Create a simple feedforward network.
    
    Args:
        input_size: Input dimension
        hidden_size: Hidden layer size
        num_classes: Number of output classes
    
    Returns:
        PyTorch model
    """
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, num_classes)
    )
    return model


def create_split_mnist_tasks(num_tasks: int = 5) -> List[List[int]]:
    """Create Split MNIST task configuration."""
    all_digits = list(range(10))
    classes_per_task = 10 // num_tasks
    
    tasks = []
    for i in range(num_tasks):
        task_classes = all_digits[i * classes_per_task:(i + 1) * classes_per_task]
        tasks.append(task_classes)
    
    return tasks


def create_task_dataset(full_dataset, task_classes: List[int]) -> TensorDataset:
    """Create dataset for a specific task by filtering classes."""
    indices = []
    for idx in range(len(full_dataset)):
        _, label = full_dataset[idx]
        if label in task_classes:
            indices.append(idx)
    
    subset = Subset(full_dataset, indices)
    
    data_list = []
    label_list = []
    
    for idx in range(len(subset)):
        img, label = subset[idx]
        data_list.append(img)
        new_label = task_classes.index(label)
        label_list.append(new_label)
    
    data_tensor = torch.stack(data_list)
    label_tensor = torch.tensor(label_list, dtype=torch.long)
    
    return TensorDataset(data_tensor, label_tensor)


def print_metrics_summary(metrics: ContinualLearningMetrics, num_tasks: int):
    """
    Print comprehensive summary of continual learning metrics.
    
    Args:
        metrics: ContinualLearningMetrics object
        num_tasks: Number of tasks
    """
    print("\n" + "=" * 70)
    print("CONTINUAL LEARNING METRICS SUMMARY")
    print("=" * 70)
    
    print(f"\n📊 Key Metrics:")
    print(f"   Average Accuracy (AA):    {metrics.average_accuracy:.2f}%")
    print(f"   Learning Accuracy (LA):   {metrics.learning_accuracy:.2f}%")
    print(f"   Backward Transfer (BWT):  {metrics.backward_transfer:.2f}%")
    print(f"   Forward Transfer (FWT):   {metrics.forward_transfer:.2f}%")
    
    print(f"\n📉 Forgetting Analysis:")
    for i, forgetting in enumerate(metrics.forgetting_per_task):
        status = "✓" if forgetting >= 0 else "✗"
        print(f"   Task {i}: {forgetting:+.2f}% {status}")
    
    print(f"\n📈 Final Accuracy per Task:")
    final_accs = metrics.accuracy_matrix[:, -1]
    for i, acc in enumerate(final_accs):
        print(f"   Task {i}: {acc:.2f}%")
    
    print(f"\n📋 Accuracy Matrix:")
    print(f"   (Rows = Tasks, Columns = After Training Stage)")
    print("   " + "-" * 60)
    print("   Task |", end="")
    for j in range(num_tasks):
        print(f"  T{j}  |", end="")
    print()
    print("   " + "-" * 60)
    
    for i in range(num_tasks):
        print(f"    {i}   |", end="")
        for j in range(num_tasks):
            if j >= i:
                print(f" {metrics.accuracy_matrix[i, j]:4.1f} |", end="")
            else:
                print(f"  --  |", end="")
        print()


def visualize_metrics(metrics: ContinualLearningMetrics, num_tasks: int):
    """
    Create comprehensive visualizations of continual learning results.
    
    Args:
        metrics: ContinualLearningMetrics object
        num_tasks: Number of tasks
    """
    fig = plt.figure(figsize=(18, 5))
    
    # Plot 1: Accuracy Matrix Heatmap
    ax1 = plt.subplot(1, 3, 1)
    im = ax1.imshow(metrics.accuracy_matrix, cmap='RdYlGn', 
                    aspect='auto', vmin=0, vmax=100)
    plt.colorbar(im, ax=ax1, label='Accuracy (%)')
    ax1.set_xlabel('After Training Task', fontsize=11)
    ax1.set_ylabel('Evaluated Task', fontsize=11)
    ax1.set_title('Accuracy Matrix', fontsize=12, fontweight='bold')
    ax1.set_xticks(range(num_tasks))
    ax1.set_yticks(range(num_tasks))
    
    # Add text annotations
    for i in range(num_tasks):
        for j in range(num_tasks):
            if j >= i:
                ax1.text(j, i, f'{metrics.accuracy_matrix[i, j]:.1f}',
                        ha="center", va="center", color="black", fontsize=9)
    
    # Plot 2: Forgetting per Task
    ax2 = plt.subplot(1, 3, 2)
    tasks = list(range(len(metrics.forgetting_per_task)))
    colors = ['red' if f < 0 else 'green' for f in metrics.forgetting_per_task]
    
    bars = ax2.bar(tasks, metrics.forgetting_per_task, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.set_xlabel('Task', fontsize=11)
    ax2.set_ylabel('Backward Transfer (%)', fontsize=11)
    ax2.set_title('Forgetting per Task', fontsize=12, fontweight='bold')
    ax2.set_xticks(tasks)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, metrics.forgetting_per_task)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%', ha='center', 
                va='bottom' if val >= 0 else 'top', fontsize=9)
    
    # Plot 3: Learning vs Final Accuracy
    ax3 = plt.subplot(1, 3, 3)
    tasks = list(range(num_tasks))
    learning_accs = np.diag(metrics.accuracy_matrix)
    final_accs = metrics.accuracy_matrix[:, -1]
    
    x = np.arange(num_tasks)
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, learning_accs, width, 
                    label='Learning Accuracy', color='skyblue', alpha=0.8)
    bars2 = ax3.bar(x + width/2, final_accs, width,
                    label='Final Accuracy', color='coral', alpha=0.8)
    
    ax3.set_xlabel('Task', fontsize=11)
    ax3.set_ylabel('Accuracy (%)', fontsize=11)
    ax3.set_title('Learning vs Final Accuracy', fontsize=12, fontweight='bold')
    ax3.set_xticks(tasks)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim([0, 105])
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('naive_sequential_learning_results.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved as 'naive_sequential_learning_results.png'")
    plt.show()


def main():
    """
    Main function to run naive sequential learning baseline.
    
    This establishes the baseline performance against which we'll
    compare continual learning methods in subsequent scripts.
    """
    print("=" * 70)
    print("NAIVE SEQUENTIAL LEARNING BASELINE")
    print("=" * 70)
    print("\nThis script implements the naive baseline for continual learning:")
    print("  - Train on tasks sequentially")
    print("  - No special techniques to prevent forgetting")
    print("  - Comprehensive evaluation metrics")
    print("\nThis serves as the baseline to beat with continual learning methods.")
    print("=" * 70)
    
    # Configuration
    num_tasks = 5
    epochs_per_task = 5
    batch_size = 128
    learning_rate = 0.001
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    # Create tasks
    task_classes = create_split_mnist_tasks(num_tasks)
    print(f"\nTask Configuration:")
    for i, classes in enumerate(task_classes):
        print(f"  Task {i}: Classes {classes}")
    
    # Create dataloaders
    train_loaders = []
    test_loaders = []
    
    for classes in task_classes:
        train_task_dataset = create_task_dataset(train_dataset, classes)
        test_task_dataset = create_task_dataset(test_dataset, classes)
        
        train_loader = DataLoader(train_task_dataset, batch_size=batch_size, 
                                 shuffle=True)
        test_loader = DataLoader(test_task_dataset, batch_size=batch_size, 
                                shuffle=False)
        
        train_loaders.append(train_loader)
        test_loaders.append(test_loader)
    
    # Create model
    num_classes_per_task = len(task_classes[0])
    model = create_simple_model(
        input_size=784,
        hidden_size=256,
        num_classes=num_classes_per_task
    ).to(device)
    
    print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create continual learner
    learner = ContinualLearner(
        model=model,
        device=device,
        learning_rate=learning_rate
    )
    
    # Run continual learning
    start_time = time.time()
    metrics = learner.train_continual(
        train_loaders=train_loaders,
        test_loaders=test_loaders,
        epochs_per_task=epochs_per_task
    )
    total_time = time.time() - start_time
    
    # Print results
    print_metrics_summary(metrics, num_tasks)
    
    print(f"\n⏱️  Total Training Time: {total_time:.2f} seconds")
    
    # Visualize
    visualize_metrics(metrics, num_tasks)
    
    print("\n" + "=" * 70)
    print("BASELINE ESTABLISHED")
    print("=" * 70)
    print("\nThis naive approach shows significant catastrophic forgetting.")
    print("In the next scripts, we'll implement continual learning methods")
    print("that preserve knowledge of previous tasks while learning new ones:")
    print("  - Script 03: Experience Replay")
    print("  - Intermediate: EWC, LWF, Synaptic Intelligence, etc.")
    print("=" * 70)


if __name__ == "__main__":
    main()
