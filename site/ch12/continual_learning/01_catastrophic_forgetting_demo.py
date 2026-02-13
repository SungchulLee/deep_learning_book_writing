"""
Module 57: Continual Learning - Beginner Level
File 1: Catastrophic Forgetting Demonstration

This script demonstrates the catastrophic forgetting problem in neural networks
when learning tasks sequentially. We'll train a simple network on multiple tasks
and visualize how performance degrades on earlier tasks.

Learning Objectives:
1. Understand what catastrophic forgetting is
2. Visualize the problem with concrete examples
3. Measure forgetting quantitatively
4. Establish baseline for continual learning methods

Mathematical Background:
- For tasks T₁, T₂, ..., T_n, we measure:
  * Accuracy on task i after learning task j: Acc_{i,j}
  * Forgetting: F_i = Acc_{i,i} - Acc_{i,n}
  * Average Forgetting: (1/n) Σ F_i
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
import copy

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class SimpleNetwork(nn.Module):
    """
    Simple feedforward network for demonstrating catastrophic forgetting.
    
    Architecture:
    - Input: 28x28 flattened = 784 dimensions
    - Hidden layers: Two layers with ReLU activations
    - Output: 10 classes (for MNIST digits)
    
    This simple architecture makes catastrophic forgetting more visible.
    """
    
    def __init__(self, input_size: int = 784, hidden_size: int = 256, 
                 num_classes: int = 10):
        super(SimpleNetwork, self).__init__()
        
        # Define network layers
        # Layer 1: Input to first hidden layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        
        # Layer 2: First hidden to second hidden layer
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        
        # Layer 3: Second hidden to output layer
        self.fc3 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 784)
        
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Flatten input if needed (from [batch, 1, 28, 28] to [batch, 784])
        x = x.view(x.size(0), -1)
        
        # Forward propagation
        x = self.relu1(self.fc1(x))  # First hidden layer with ReLU
        x = self.relu2(self.fc2(x))  # Second hidden layer with ReLU
        x = self.fc3(x)               # Output layer (logits)
        
        return x


def create_split_mnist_tasks(num_tasks: int = 5) -> List[List[int]]:
    """
    Create Split MNIST tasks by dividing 10 digits into groups.
    
    Split MNIST is a classic continual learning benchmark where we split
    the 10 MNIST digits into multiple binary (or multi-class) classification tasks.
    
    Example for 5 tasks:
    - Task 0: Digits 0 vs 1
    - Task 1: Digits 2 vs 3
    - Task 2: Digits 4 vs 5
    - Task 3: Digits 6 vs 7
    - Task 4: Digits 8 vs 9
    
    Args:
        num_tasks: Number of tasks to create (default: 5)
    
    Returns:
        List of lists, where each sublist contains digit classes for that task
    """
    all_digits = list(range(10))
    classes_per_task = 10 // num_tasks
    
    tasks = []
    for i in range(num_tasks):
        # Select classes for this task
        task_classes = all_digits[i * classes_per_task:(i + 1) * classes_per_task]
        tasks.append(task_classes)
    
    return tasks


def create_task_dataset(full_dataset, task_classes: List[int]) -> TensorDataset:
    """
    Create a dataset for a specific task by filtering classes.
    
    This function:
    1. Filters the dataset to include only specified classes
    2. Remaps labels to be contiguous (0, 1, 2, ...)
    3. Returns a new TensorDataset
    
    Args:
        full_dataset: Complete dataset (e.g., MNIST)
        task_classes: List of class labels to include in this task
    
    Returns:
        TensorDataset containing only examples from specified classes
    """
    # Find indices of examples belonging to task classes
    indices = []
    for idx in range(len(full_dataset)):
        _, label = full_dataset[idx]
        if label in task_classes:
            indices.append(idx)
    
    # Create subset of dataset
    subset = Subset(full_dataset, indices)
    
    # Extract all data and labels
    data_list = []
    label_list = []
    
    for idx in range(len(subset)):
        img, label = subset[idx]
        data_list.append(img)
        
        # Remap label to be contiguous (e.g., classes [4,5] become [0,1])
        new_label = task_classes.index(label)
        label_list.append(new_label)
    
    # Stack into tensors
    data_tensor = torch.stack(data_list)
    label_tensor = torch.tensor(label_list, dtype=torch.long)
    
    return TensorDataset(data_tensor, label_tensor)


def train_on_task(model: nn.Module, 
                  train_loader: DataLoader,
                  device: torch.device,
                  epochs: int = 5,
                  learning_rate: float = 0.001) -> List[float]:
    """
    Train model on a single task.
    
    Training process:
    1. Set model to training mode
    2. For each epoch:
        a. Iterate through batches
        b. Compute loss (cross-entropy)
        c. Backpropagate gradients
        d. Update weights with optimizer
    3. Track training loss for monitoring
    
    Args:
        model: Neural network to train
        train_loader: DataLoader for training data
        device: Device to train on (CPU or GPU)
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
    
    Returns:
        List of average losses per epoch
    """
    # Define loss function (cross-entropy for classification)
    criterion = nn.CrossEntropyLoss()
    
    # Define optimizer (Adam is robust and commonly used)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Track training losses
    losses = []
    
    model.train()  # Set to training mode (enables dropout, etc.)
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # Move data to device
            data, target = data.to(device), target.to(device)
            
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            
            # Forward pass: compute predictions
            output = model(data)
            
            # Compute loss
            loss = criterion(output, target)
            
            # Backward pass: compute gradients
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Track statistics
            epoch_loss += loss.item()
            num_batches += 1
        
        # Average loss for this epoch
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        
        print(f"  Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return losses


def evaluate_on_task(model: nn.Module,
                    test_loader: DataLoader,
                    device: torch.device) -> Tuple[float, float]:
    """
    Evaluate model performance on a task.
    
    Evaluation metrics:
    - Accuracy: Percentage of correct predictions
    - Loss: Average cross-entropy loss
    
    Args:
        model: Neural network to evaluate
        test_loader: DataLoader for test data
        device: Device to evaluate on
    
    Returns:
        Tuple of (accuracy, loss)
    """
    criterion = nn.CrossEntropyLoss()
    
    model.eval()  # Set to evaluation mode (disables dropout, etc.)
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    # Disable gradient computation for efficiency
    with torch.no_grad():
        for data, target in test_loader:
            # Move data to device
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            
            # Compute loss
            loss = criterion(output, target)
            total_loss += loss.item()
            
            # Compute accuracy
            _, predicted = torch.max(output, 1)  # Get class with highest probability
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    # Calculate metrics
    accuracy = 100.0 * correct / total
    avg_loss = total_loss / len(test_loader)
    
    return accuracy, avg_loss


def demonstrate_catastrophic_forgetting(num_tasks: int = 5,
                                       epochs_per_task: int = 5,
                                       batch_size: int = 128):
    """
    Main function to demonstrate catastrophic forgetting.
    
    Process:
    1. Create Split MNIST tasks
    2. Initialize a fresh model
    3. Train sequentially on each task
    4. After each task, evaluate on ALL previous tasks
    5. Visualize how performance degrades
    
    Key Observation:
    As we learn new tasks, accuracy on previous tasks drops dramatically.
    This is catastrophic forgetting!
    
    Args:
        num_tasks: Number of tasks in Split MNIST
        epochs_per_task: Training epochs per task
        batch_size: Batch size for training
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    # Create Split MNIST tasks
    task_classes = create_split_mnist_tasks(num_tasks)
    print("Task Configuration:")
    for i, classes in enumerate(task_classes):
        print(f"  Task {i}: Classes {classes}")
    print()
    
    # Create task-specific datasets
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
    
    # Initialize model
    num_classes_per_task = len(task_classes[0])
    model = SimpleNetwork(input_size=784, hidden_size=256, 
                         num_classes=num_classes_per_task).to(device)
    
    print(f"Model Architecture:")
    print(model)
    print(f"\nTotal Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # Matrix to store accuracy: rows = tasks, columns = training stages
    # accuracy_matrix[i][j] = accuracy on task i after training on task j
    accuracy_matrix = np.zeros((num_tasks, num_tasks))
    
    # Train on each task sequentially
    print("=" * 70)
    print("SEQUENTIAL TRAINING (Demonstrating Catastrophic Forgetting)")
    print("=" * 70)
    
    for task_id in range(num_tasks):
        print(f"\n{'=' * 70}")
        print(f"Training on Task {task_id} (Classes: {task_classes[task_id]})")
        print('=' * 70)
        
        # Train on current task
        train_on_task(
            model=model,
            train_loader=train_loaders[task_id],
            device=device,
            epochs=epochs_per_task,
            learning_rate=0.001
        )
        
        # Evaluate on ALL tasks seen so far
        print(f"\nEvaluating on all tasks after training Task {task_id}:")
        print("-" * 50)
        
        for eval_task_id in range(task_id + 1):
            accuracy, loss = evaluate_on_task(
                model=model,
                test_loader=test_loaders[eval_task_id],
                device=device
            )
            
            # Store accuracy in matrix
            accuracy_matrix[eval_task_id][task_id] = accuracy
            
            # Print results
            print(f"  Task {eval_task_id}: Accuracy = {accuracy:.2f}%")
            
            # Highlight catastrophic forgetting
            if eval_task_id < task_id:
                original_acc = accuracy_matrix[eval_task_id][eval_task_id]
                forgetting = original_acc - accuracy
                print(f"    (Original: {original_acc:.2f}%, "
                      f"Forgetting: {forgetting:.2f}%)")
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    # Calculate forgetting metrics
    print("\nAccuracy Matrix (rows = tasks, columns = after training task):")
    print("(Each entry shows accuracy on row's task after training column's task)")
    print()
    
    # Print header
    print("Task |", end="")
    for j in range(num_tasks):
        print(f" After T{j} |", end="")
    print(" Forgetting")
    print("-" * (12 + 12 * num_tasks))
    
    # Print accuracy matrix with forgetting calculation
    total_forgetting = 0.0
    for i in range(num_tasks):
        print(f"  {i}  |", end="")
        for j in range(num_tasks):
            if j >= i:  # Only show values after task is learned
                print(f"  {accuracy_matrix[i][j]:5.1f}%  |", end="")
            else:
                print(f"    -    |", end="")
        
        # Calculate forgetting for this task
        if i < num_tasks - 1:  # Don't calculate for last task
            original = accuracy_matrix[i][i]
            final = accuracy_matrix[i][num_tasks - 1]
            forgetting = original - final
            total_forgetting += forgetting
            print(f"  {forgetting:5.1f}%")
        else:
            print("    N/A")
    
    # Calculate average forgetting
    avg_forgetting = total_forgetting / (num_tasks - 1) if num_tasks > 1 else 0
    print()
    print(f"Average Forgetting: {avg_forgetting:.2f}%")
    print(f"Final Average Accuracy: {np.mean(accuracy_matrix[:, -1]):.2f}%")
    
    # Visualize results
    visualize_catastrophic_forgetting(accuracy_matrix, task_classes)
    
    return accuracy_matrix


def visualize_catastrophic_forgetting(accuracy_matrix: np.ndarray,
                                     task_classes: List[List[int]]):
    """
    Create visualizations of catastrophic forgetting.
    
    Creates two plots:
    1. Heatmap of accuracy matrix showing forgetting over time
    2. Line plot showing accuracy trajectory for each task
    
    Args:
        accuracy_matrix: Matrix of accuracies (tasks × training stages)
        task_classes: List of class assignments for each task
    """
    num_tasks = len(task_classes)
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Heatmap of accuracy matrix
    ax1 = axes[0]
    im = ax1.imshow(accuracy_matrix, cmap='RdYlGn', aspect='auto', 
                    vmin=0, vmax=100)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label('Accuracy (%)', rotation=270, labelpad=20)
    
    # Set labels
    ax1.set_xlabel('After Training Task', fontsize=12)
    ax1.set_ylabel('Evaluated Task', fontsize=12)
    ax1.set_title('Catastrophic Forgetting Heatmap', fontsize=14, fontweight='bold')
    
    # Set ticks
    ax1.set_xticks(range(num_tasks))
    ax1.set_yticks(range(num_tasks))
    ax1.set_xticklabels([f'T{i}' for i in range(num_tasks)])
    ax1.set_yticklabels([f'T{i}' for i in range(num_tasks)])
    
    # Add text annotations
    for i in range(num_tasks):
        for j in range(num_tasks):
            if j >= i:  # Only annotate valid entries
                text = ax1.text(j, i, f'{accuracy_matrix[i, j]:.1f}',
                              ha="center", va="center", color="black", fontsize=9)
    
    # Plot 2: Line plot showing accuracy trajectory
    ax2 = axes[1]
    
    for task_id in range(num_tasks):
        # Extract accuracy trajectory for this task
        accuracies = [accuracy_matrix[task_id][j] for j in range(task_id, num_tasks)]
        stages = list(range(task_id, num_tasks))
        
        # Plot line
        ax2.plot(stages, accuracies, marker='o', linewidth=2, 
                label=f'Task {task_id} (Classes {task_classes[task_id]})',
                markersize=8)
        
        # Highlight initial accuracy
        ax2.scatter([task_id], [accuracies[0]], s=150, 
                   marker='*', zorder=5)
    
    ax2.set_xlabel('Training Stage (After Task)', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Accuracy Degradation Over Time', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 105])
    ax2.set_xticks(range(num_tasks))
    ax2.set_xticklabels([f'T{i}' for i in range(num_tasks)])
    
    plt.tight_layout()
    plt.savefig('catastrophic_forgetting_demonstration.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved as 'catastrophic_forgetting_demonstration.png'")
    plt.show()


if __name__ == "__main__":
    """
    Main execution: Demonstrate catastrophic forgetting on Split MNIST
    
    Expected Observations:
    1. Each task achieves ~95-99% accuracy initially
    2. As new tasks are learned, old task accuracy drops dramatically
    3. Average forgetting is typically 40-60%
    4. This proves neural networks forget catastrophically by default
    
    Key Takeaway:
    Standard neural network training is NOT suitable for continual learning.
    We need special techniques to prevent catastrophic forgetting!
    """
    
    print("=" * 70)
    print("CATASTROPHIC FORGETTING DEMONSTRATION")
    print("=" * 70)
    print("\nThis script demonstrates how neural networks forget previous")
    print("tasks when learning new tasks sequentially.")
    print("\nWe'll train on Split MNIST (5 tasks, 2 classes each) and watch")
    print("how performance on earlier tasks degrades dramatically.")
    print("=" * 70)
    
    # Run demonstration
    accuracy_matrix = demonstrate_catastrophic_forgetting(
        num_tasks=5,
        epochs_per_task=5,
        batch_size=128
    )
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("\nCatastrophic forgetting is a fundamental problem in neural networks!")
    print("As we train on new tasks, the model's weights are updated to minimize")
    print("the new task's loss, which destroys information needed for old tasks.")
    print("\nIn the following scripts, we'll explore methods to prevent this:")
    print("  - Script 02: Naive Sequential Learning (baseline)")
    print("  - Script 03: Simple Experience Replay")
    print("  - Intermediate scripts: EWC, LWF, and more!")
    print("=" * 70)
