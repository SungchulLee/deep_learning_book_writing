"""
Module 57: Continual Learning - Intermediate Level
File 1: Elastic Weight Consolidation (EWC)

This script implements Elastic Weight Consolidation, one of the most influential
continual learning methods. EWC prevents catastrophic forgetting by protecting
important parameters from significant changes.

Learning Objectives:
1. Understand Fisher information and its role in measuring parameter importance
2. Implement EWC loss with quadratic penalty
3. Compute Fisher information efficiently
4. Balance plasticity and stability with λ hyperparameter

Mathematical Foundation:
The EWC loss for task τ is:

L(θ) = L_τ(θ) + (λ/2) Σ_i F_i (θ_i - θ*_{τ-1,i})²

where:
- L_τ(θ): Loss on current task τ
- F_i: Fisher information for parameter i (importance weight)
- θ*_{τ-1}: Optimal parameters after previous task
- λ: Regularization strength (hyperparameter)

Fisher Information Matrix (diagonal):
F_i = E_{x~D}[(∂log p(y|x,θ)/∂θ_i)²]

Approximation using empirical samples:
F_i ≈ (1/N) Σ_{n=1}^N (∂L(x_n, θ)/∂θ_i)²

Reference: Kirkpatrick et al., "Overcoming catastrophic forgetting in neural 
networks," PNAS 2017
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import copy
from collections import defaultdict


class EWCLearner:
    """
    Continual learner using Elastic Weight Consolidation.
    
    EWC Strategy:
    1. After learning each task, compute Fisher information matrix
    2. Store optimal parameters and Fisher values
    3. When learning new task, add regularization penalty
       that prevents important parameters from changing too much
    
    The key insight: Not all parameters are equally important.
    Parameters crucial for previous tasks get strong protection,
    while less important ones can be freely modified.
    """
    
    def __init__(self,
                 model: nn.Module,
                 device: torch.device,
                 ewc_lambda: float = 5000,
                 learning_rate: float = 0.001):
        """
        Initialize EWC learner.
        
        Args:
            model: Neural network model
            device: Device to train on
            ewc_lambda: Regularization strength (λ)
                       Higher values = more protection of old parameters
                       Typical range: [100, 10000]
            learning_rate: Learning rate for optimizer
        """
        self.model = model
        self.device = device
        self.ewc_lambda = ewc_lambda
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()
        
        # Store Fisher information and optimal parameters for each task
        # fisher_dict: Dict[task_id -> Dict[param_name -> Fisher values]]
        # optpar_dict: Dict[task_id -> Dict[param_name -> optimal parameters]]
        self.fisher_dict = {}
        self.optpar_dict = {}
        
        # Tracking
        self.accuracy_matrix = None
        self.current_task = 0
    
    def compute_fisher_information(self,
                                   data_loader: DataLoader,
                                   task_id: int,
                                   num_samples: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Compute Fisher information matrix (diagonal approximation).
        
        The Fisher information quantifies how much the model's predictions
        depend on each parameter. High Fisher value = parameter is important
        for current task.
        
        Mathematical Details:
        For classification with cross-entropy loss:
        F_i = E[(∂log p(y|x,θ)/∂θ_i)²]
        
        We approximate this using empirical samples:
        1. For each example (x, y):
           a. Compute log probability: log p(y|x, θ)
           b. Compute gradient: ∂log p(y|x,θ)/∂θ_i
           c. Square the gradient: (∂log p(y|x,θ)/∂θ_i)²
        2. Average over all examples
        
        This gives us a diagonal approximation of the full Fisher matrix,
        which is computationally tractable.
        
        Args:
            data_loader: DataLoader for computing Fisher
            task_id: Current task ID
            num_samples: Number of samples to use (None = use all)
        
        Returns:
            Dictionary mapping parameter names to Fisher values
        """
        print(f"\n  Computing Fisher information for Task {task_id}...")
        
        self.model.eval()
        
        # Initialize Fisher information dictionary
        fisher = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher[name] = torch.zeros_like(param.data)
        
        # Accumulate Fisher information from samples
        num_batches = 0
        total_samples = 0
        
        for batch_idx, (data, target) in enumerate(data_loader):
            # Check if we've processed enough samples
            if num_samples is not None and total_samples >= num_samples:
                break
            
            data, target = data.to(self.device), target.to(self.device)
            
            # Zero gradients
            self.model.zero_grad()
            
            # Forward pass
            output = self.model(data)
            
            # Compute log probabilities
            # For classification: log p(y|x) = log_softmax(output)[y]
            log_probs = F.log_softmax(output, dim=1)
            
            # For each example, compute gradient of log probability
            # with respect to parameters
            batch_size = data.size(0)
            
            for i in range(batch_size):
                # Zero gradients for this example
                self.model.zero_grad()
                
                # Get log probability for true class
                log_prob = log_probs[i, target[i]]
                
                # Compute gradient: ∂log p(y|x)/∂θ
                log_prob.backward(retain_graph=(i < batch_size - 1))
                
                # Accumulate squared gradients
                for name, param in self.model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        # Square and add to Fisher
                        fisher[name] += param.grad.data ** 2
            
            total_samples += batch_size
            num_batches += 1
        
        # Average over all samples
        # Fisher[i] = (1/N) Σ_n (∂log p(y_n|x_n)/∂θ_i)²
        for name in fisher:
            fisher[name] /= total_samples
        
        print(f"  Fisher computed using {total_samples} samples")
        
        # Print statistics about Fisher values
        all_fisher_values = torch.cat([f.flatten() for f in fisher.values()])
        print(f"  Fisher statistics:")
        print(f"    Mean: {all_fisher_values.mean().item():.6f}")
        print(f"    Std:  {all_fisher_values.std().item():.6f}")
        print(f"    Max:  {all_fisher_values.max().item():.6f}")
        print(f"    Min:  {all_fisher_values.min().item():.6f}")
        
        return fisher
    
    def ewc_penalty(self) -> torch.Tensor:
        """
        Compute EWC regularization penalty.
        
        The penalty prevents important parameters from changing too much:
        
        Penalty = (λ/2) Σ_{tasks} Σ_i F_i(task) (θ_i - θ*_i(task))²
        
        where:
        - Sum over all previous tasks
        - Sum over all parameters
        - F_i(task): Fisher information for parameter i on that task
        - θ_i: Current parameter value
        - θ*_i(task): Optimal value after learning that task
        
        Returns:
            EWC penalty term (scalar)
        """
        penalty = torch.tensor(0.0).to(self.device)
        
        # Iterate over all previous tasks
        for task_id in self.fisher_dict.keys():
            fisher = self.fisher_dict[task_id]
            optpar = self.optpar_dict[task_id]
            
            # For each parameter
            for name, param in self.model.named_parameters():
                if name in fisher and param.requires_grad:
                    # Compute squared difference from optimal value
                    # weighted by Fisher information
                    # (λ/2) * F_i * (θ_i - θ*_i)²
                    penalty += (fisher[name] * (param - optpar[name]) ** 2).sum()
        
        # Apply regularization strength
        penalty = (self.ewc_lambda / 2) * penalty
        
        return penalty
    
    def train_task(self,
                  train_loader: DataLoader,
                  task_id: int,
                  epochs: int = 5) -> List[float]:
        """
        Train on a single task with EWC regularization.
        
        Loss function:
        L_total = L_current + EWC_penalty
        
        where:
        - L_current: Cross-entropy loss on current task
        - EWC_penalty: Quadratic penalty on parameter changes
        
        Args:
            train_loader: DataLoader for current task
            task_id: Current task ID
            epochs: Number of training epochs
        
        Returns:
            List of losses per epoch
        """
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        self.model.train()
        losses = []
        
        print(f"\n{'=' * 60}")
        print(f"Training Task {task_id} with EWC (λ={self.ewc_lambda})")
        print('=' * 60)
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            current_loss_sum = 0.0
            ewc_loss_sum = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                output = self.model(data)
                
                # Current task loss
                current_loss = self.criterion(output, target)
                
                # EWC penalty (only if we have previous tasks)
                ewc_loss = torch.tensor(0.0).to(self.device)
                if len(self.fisher_dict) > 0:
                    ewc_loss = self.ewc_penalty()
                
                # Total loss
                total_loss = current_loss + ewc_loss
                
                # Backward pass
                total_loss.backward()
                optimizer.step()
                
                # Track statistics
                epoch_loss += total_loss.item()
                current_loss_sum += current_loss.item()
                ewc_loss_sum += ewc_loss.item()
                
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
            
            # Calculate metrics
            avg_loss = epoch_loss / len(train_loader)
            avg_current = current_loss_sum / len(train_loader)
            avg_ewc = ewc_loss_sum / len(train_loader)
            accuracy = 100.0 * correct / total
            losses.append(avg_loss)
            
            print(f"  Epoch {epoch + 1}/{epochs} - "
                  f"Total Loss: {avg_loss:.4f}, "
                  f"Current: {avg_current:.4f}, "
                  f"EWC: {avg_ewc:.4f}, "
                  f"Train Acc: {accuracy:.2f}%")
        
        return losses
    
    def consolidate_task(self, 
                        train_loader: DataLoader,
                        task_id: int):
        """
        Consolidate knowledge after learning a task.
        
        This involves:
        1. Computing Fisher information matrix
        2. Storing current optimal parameters
        
        These will be used as constraints when learning future tasks.
        
        Args:
            train_loader: DataLoader for computing Fisher
            task_id: Task ID to consolidate
        """
        print(f"\n{'=' * 60}")
        print(f"Consolidating Task {task_id}")
        print('=' * 60)
        
        # Compute Fisher information
        fisher = self.compute_fisher_information(
            data_loader=train_loader,
            task_id=task_id,
            num_samples=1000  # Use subset for efficiency
        )
        
        # Store Fisher information
        self.fisher_dict[task_id] = fisher
        
        # Store optimal parameters
        optpar = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                optpar[name] = param.data.clone()
        
        self.optpar_dict[task_id] = optpar
        
        print(f"  Stored Fisher and optimal parameters for Task {task_id}")
    
    def evaluate_task(self,
                     test_loader: DataLoader,
                     task_id: int) -> Tuple[float, float]:
        """Evaluate model on a task."""
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
                       epochs_per_task: int = 5):
        """
        Main continual learning loop with EWC.
        
        Process for each task:
        1. Train with EWC regularization
        2. Consolidate: compute Fisher and store parameters
        3. Evaluate on all tasks
        """
        num_tasks = len(train_loaders)
        self.accuracy_matrix = np.zeros((num_tasks, num_tasks))
        
        print("\n" + "=" * 70)
        print("CONTINUAL LEARNING WITH ELASTIC WEIGHT CONSOLIDATION (EWC)")
        print("=" * 70)
        print(f"EWC Lambda (λ): {self.ewc_lambda}")
        
        for task_id in range(num_tasks):
            self.current_task = task_id
            
            # Train on current task
            self.train_task(
                train_loader=train_loaders[task_id],
                task_id=task_id,
                epochs=epochs_per_task
            )
            
            # Consolidate: compute Fisher and store parameters
            self.consolidate_task(
                train_loader=train_loaders[task_id],
                task_id=task_id
            )
            
            # Evaluate on all tasks
            print(f"\n{'=' * 60}")
            print(f"Evaluation after Task {task_id}")
            print('=' * 60)
            
            for eval_task_id in range(task_id + 1):
                acc, loss = self.evaluate_task(
                    test_loader=test_loaders[eval_task_id],
                    task_id=eval_task_id
                )
                
                self.accuracy_matrix[eval_task_id, task_id] = acc
                
                if eval_task_id == task_id:
                    print(f"  Task {eval_task_id}: {acc:.2f}% (just learned)")
                elif eval_task_id < task_id:
                    original_acc = self.accuracy_matrix[eval_task_id, eval_task_id]
                    change = acc - original_acc
                    print(f"  Task {eval_task_id}: {acc:.2f}% "
                          f"(was {original_acc:.2f}%, change {change:+.2f}%)")
        
        return self.calculate_metrics()
    
    def calculate_metrics(self):
        """Calculate continual learning metrics."""
        num_tasks = self.accuracy_matrix.shape[0]
        
        average_accuracy = np.mean(self.accuracy_matrix[:, -1])
        
        backward_transfer = 0.0
        forgetting_per_task = []
        
        for i in range(num_tasks - 1):
            initial_acc = self.accuracy_matrix[i, i]
            final_acc = self.accuracy_matrix[i, num_tasks - 1]
            change = final_acc - initial_acc
            backward_transfer += change
            forgetting_per_task.append(change)
        
        if num_tasks > 1:
            backward_transfer /= (num_tasks - 1)
        
        learning_accuracy = np.mean(np.diag(self.accuracy_matrix))
        
        return {
            'average_accuracy': average_accuracy,
            'backward_transfer': backward_transfer,
            'learning_accuracy': learning_accuracy,
            'forgetting_per_task': forgetting_per_task,
            'accuracy_matrix': self.accuracy_matrix
        }


def create_simple_model(input_size: int = 784,
                       hidden_size: int = 256,
                       num_classes: int = 2) -> nn.Module:
    """Create simple feedforward network."""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, num_classes)
    )


def create_split_mnist_tasks(num_tasks: int = 5) -> List[List[int]]:
    """Create Split MNIST task configuration."""
    all_digits = list(range(10))
    classes_per_task = 10 // num_tasks
    return [all_digits[i * classes_per_task:(i + 1) * classes_per_task] 
            for i in range(num_tasks)]


def create_task_dataset(full_dataset, task_classes: List[int]) -> TensorDataset:
    """Create dataset for a specific task."""
    indices = [idx for idx in range(len(full_dataset)) 
               if full_dataset[idx][1] in task_classes]
    subset = Subset(full_dataset, indices)
    
    data_list, label_list = [], []
    for idx in range(len(subset)):
        img, label = subset[idx]
        data_list.append(img)
        label_list.append(task_classes.index(label))
    
    return TensorDataset(torch.stack(data_list), 
                        torch.tensor(label_list, dtype=torch.long))


def visualize_ewc_results(ewc_metrics: dict, num_tasks: int):
    """Visualize EWC results."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Accuracy Matrix Heatmap
    ax1 = axes[0]
    im = ax1.imshow(ewc_metrics['accuracy_matrix'], cmap='RdYlGn',
                    aspect='auto', vmin=0, vmax=100)
    plt.colorbar(im, ax=ax1, label='Accuracy (%)')
    ax1.set_xlabel('After Training Task')
    ax1.set_ylabel('Evaluated Task')
    ax1.set_title('EWC: Accuracy Matrix', fontweight='bold')
    ax1.set_xticks(range(num_tasks))
    ax1.set_yticks(range(num_tasks))
    
    for i in range(num_tasks):
        for j in range(num_tasks):
            if j >= i:
                ax1.text(j, i, f'{ewc_metrics["accuracy_matrix"][i, j]:.1f}',
                        ha="center", va="center", color="black", fontsize=9)
    
    # Plot 2: Backward Transfer per Task
    ax2 = axes[1]
    tasks = list(range(len(ewc_metrics['forgetting_per_task'])))
    colors = ['green' if f >= 0 else 'red' 
              for f in ewc_metrics['forgetting_per_task']]
    
    bars = ax2.bar(tasks, ewc_metrics['forgetting_per_task'], 
                   color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.set_xlabel('Task')
    ax2.set_ylabel('Backward Transfer (%)')
    ax2.set_title('EWC: Forgetting per Task', fontweight='bold')
    ax2.set_xticks(tasks)
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, ewc_metrics['forgetting_per_task']):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%', ha='center',
                va='bottom' if val >= 0 else 'top', fontsize=9)
    
    # Plot 3: Learning vs Final Accuracy
    ax3 = axes[2]
    tasks_range = list(range(num_tasks))
    learning_accs = np.diag(ewc_metrics['accuracy_matrix'])
    final_accs = ewc_metrics['accuracy_matrix'][:, -1]
    
    x = np.arange(num_tasks)
    width = 0.35
    
    ax3.bar(x - width/2, learning_accs, width,
            label='Learning Acc', color='skyblue', alpha=0.8)
    ax3.bar(x + width/2, final_accs, width,
            label='Final Acc', color='coral', alpha=0.8)
    
    ax3.set_xlabel('Task')
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_title('EWC: Learning vs Final Accuracy', fontweight='bold')
    ax3.set_xticks(tasks_range)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim([0, 105])
    
    plt.tight_layout()
    plt.savefig('ewc_results.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved as 'ewc_results.png'")
    plt.show()


def main():
    """Main function demonstrating EWC."""
    print("=" * 70)
    print("ELASTIC WEIGHT CONSOLIDATION (EWC)")
    print("=" * 70)
    print("\nEWC prevents forgetting by:")
    print("  1. Identifying important parameters using Fisher information")
    print("  2. Adding quadratic penalty to prevent their modification")
    print("  3. Balancing learning new tasks with preserving old knowledge")
    print("=" * 70)
    
    # Configuration
    num_tasks = 5
    epochs_per_task = 5
    batch_size = 128
    learning_rate = 0.001
    ewc_lambda = 5000  # Key hyperparameter - controls forgetting vs learning trade-off
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    torch.manual_seed(42)
    np.random.seed(42)
    
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
    train_loaders = [DataLoader(create_task_dataset(train_dataset, classes),
                                batch_size=batch_size, shuffle=True)
                    for classes in task_classes]
    test_loaders = [DataLoader(create_task_dataset(test_dataset, classes),
                              batch_size=batch_size, shuffle=False)
                   for classes in task_classes]
    
    # Create model
    model = create_simple_model(784, 256, 2).to(device)
    print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create EWC learner
    learner = EWCLearner(
        model=model,
        device=device,
        ewc_lambda=ewc_lambda,
        learning_rate=learning_rate
    )
    
    # Run continual learning
    import time
    start_time = time.time()
    metrics = learner.train_continual(train_loaders, test_loaders, epochs_per_task)
    total_time = time.time() - start_time
    
    # Print results
    print("\n" + "=" * 70)
    print("EWC RESULTS")
    print("=" * 70)
    print(f"\n📊 Key Metrics:")
    print(f"   Average Accuracy:     {metrics['average_accuracy']:.2f}%")
    print(f"   Learning Accuracy:    {metrics['learning_accuracy']:.2f}%")
    print(f"   Backward Transfer:    {metrics['backward_transfer']:.2f}%")
    print(f"\n⏱️  Training Time: {total_time:.2f} seconds")
    
    visualize_ewc_results(metrics, num_tasks)
    
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print("\n✓ EWC reduces forgetting by protecting important parameters")
    print("✓ No need to store previous examples (privacy-preserving)")
    print("✓ Computationally efficient during training")
    print("\n⚠️  Considerations:")
    print("  - λ hyperparameter is task-dependent")
    print("  - Fisher computation adds overhead after each task")
    print("  - Diagonal Fisher approximation may be too restrictive")
    print("=" * 70)


if __name__ == "__main__":
    main()
