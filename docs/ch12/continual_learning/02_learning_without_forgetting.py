"""
Module 57: Continual Learning - Intermediate Level
File 2: Learning Without Forgetting (LWF)

This script implements Learning Without Forgetting, a continual learning method
that uses knowledge distillation to preserve previous task knowledge without
storing any examples.

Learning Objectives:
1. Understand knowledge distillation for continual learning
2. Implement LWF loss with soft targets
3. Balance new task learning with old knowledge preservation
4. Learn temperature scaling for soft targets

Mathematical Foundation:
The LWF loss combines:

L_total = L_new + λ * L_distill

where:
- L_new: Cross-entropy loss on new task
- L_distill: Distillation loss preserving old task predictions

Distillation Loss:
L_distill = KL(softmax(z_old/T) || softmax(z_new/T))

where:
- z_old: Logits from old model (frozen)
- z_new: Logits from new model (being trained)
- T: Temperature (higher T = softer probabilities)
- KL: Kullback-Leibler divergence

Temperature Scaling:
softmax_T(z_i) = exp(z_i/T) / Σ_j exp(z_j/T)

Higher T → softer probabilities → easier for model to match

Reference: Li & Hoiem, "Learning without Forgetting," IEEE TPAMI 2017
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


class LWFLearner:
    """
    Continual learner using Learning Without Forgetting.
    
    LWF Strategy:
    1. Before learning new task, save copy of current model
    2. While training on new task:
       a. Forward pass through new data with current model
       b. Forward pass through new data with old model (frozen)
       c. Compute distillation loss to match old model's predictions
       d. Compute new task loss
       e. Backpropagate combined loss
    
    Key Insight: The old model's soft predictions contain knowledge
    about previous tasks. By matching these predictions on new task data,
    we implicitly preserve the old knowledge.
    
    Advantages:
    - No storage of previous examples (privacy-preserving)
    - Works with any neural network architecture
    - Computationally efficient
    
    Limitations:
    - Requires running two forward passes per batch
    - May struggle if new task data is very different from old tasks
    """
    
    def __init__(self,
                 model: nn.Module,
                 device: torch.device,
                 distill_lambda: float = 1.0,
                 temperature: float = 2.0,
                 learning_rate: float = 0.001):
        """
        Initialize LWF learner.
        
        Args:
            model: Neural network model
            device: Device to train on
            distill_lambda: Weight for distillation loss
                           Higher values = more preservation of old knowledge
                           Typical range: [0.5, 5.0]
            temperature: Temperature for softening probability distributions
                        Higher temperature = softer distributions
                        Typical range: [1.0, 4.0]
            learning_rate: Learning rate for optimizer
        """
        self.model = model
        self.device = device
        self.distill_lambda = distill_lambda
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()
        
        # Store previous model for distillation
        # This will be updated after each task
        self.old_model = None
        
        # Tracking
        self.accuracy_matrix = None
        self.current_task = 0
    
    def distillation_loss(self,
                         new_logits: torch.Tensor,
                         old_logits: torch.Tensor,
                         temperature: float) -> torch.Tensor:
        """
        Compute knowledge distillation loss.
        
        The distillation loss is the KL divergence between:
        - Old model's soft predictions (teacher)
        - New model's soft predictions (student)
        
        Mathematical Details:
        KL(P || Q) = Σ_i P(i) * log(P(i) / Q(i))
        
        For numerical stability, we use:
        KL = Σ_i P(i) * (log P(i) - log Q(i))
        
        Temperature scaling:
        - Dividing logits by T before softmax makes distribution softer
        - Higher T → more uniform distribution → easier to match
        - T² factor compensates for scaling (see paper for derivation)
        
        Args:
            new_logits: Logits from current model (student)
            old_logits: Logits from old model (teacher)
            temperature: Temperature for scaling
        
        Returns:
            Distillation loss (scalar)
        """
        # Apply temperature scaling and compute log probabilities
        # log_softmax_T(z) = log(exp(z/T) / Σ exp(z/T))
        #                  = z/T - log(Σ exp(z/T))
        new_log_probs = F.log_softmax(new_logits / temperature, dim=1)
        
        # Compute soft targets from old model
        # softmax_T(z) = exp(z/T) / Σ exp(z/T)
        old_probs = F.softmax(old_logits / temperature, dim=1)
        
        # KL divergence with temperature compensation
        # Factor T² compensates for temperature scaling
        # (see paper: "Distilling the Knowledge in a Neural Network")
        loss = F.kl_div(
            new_log_probs,
            old_probs,
            reduction='batchmean'
        ) * (temperature ** 2)
        
        return loss
    
    def train_task(self,
                  train_loader: DataLoader,
                  task_id: int,
                  epochs: int = 5) -> List[float]:
        """
        Train on a single task with LWF.
        
        Loss function for task τ > 0:
        L = L_new + λ * L_distill
        
        For task 0 (first task), there's no distillation:
        L = L_new
        
        Args:
            train_loader: DataLoader for current task
            task_id: Current task ID
            epochs: Number of training epochs
        
        Returns:
            List of losses per epoch
        """
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        self.model.train()
        
        # Set old model to eval mode (frozen)
        if self.old_model is not None:
            self.old_model.eval()
        
        losses = []
        
        print(f"\n{'=' * 60}")
        print(f"Training Task {task_id} with LWF")
        if task_id > 0:
            print(f"Distillation: λ={self.distill_lambda}, T={self.temperature}")
        else:
            print("First task: No distillation needed")
        print('=' * 60)
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            new_loss_sum = 0.0
            distill_loss_sum = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                
                # ===== Forward pass with current model =====
                new_logits = self.model(data)
                
                # ===== New task loss =====
                new_loss = self.criterion(new_logits, target)
                
                # ===== Distillation loss =====
                distill_loss = torch.tensor(0.0).to(self.device)
                
                if self.old_model is not None:
                    # Forward pass with old model (frozen)
                    with torch.no_grad():
                        old_logits = self.old_model(data)
                    
                    # Compute distillation loss
                    distill_loss = self.distillation_loss(
                        new_logits=new_logits,
                        old_logits=old_logits,
                        temperature=self.temperature
                    )
                
                # ===== Combined loss =====
                total_loss = new_loss + self.distill_lambda * distill_loss
                
                # Backward pass
                total_loss.backward()
                optimizer.step()
                
                # Track statistics
                epoch_loss += total_loss.item()
                new_loss_sum += new_loss.item()
                distill_loss_sum += distill_loss.item()
                
                _, predicted = torch.max(new_logits, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
            
            # Calculate metrics
            avg_loss = epoch_loss / len(train_loader)
            avg_new = new_loss_sum / len(train_loader)
            avg_distill = distill_loss_sum / len(train_loader)
            accuracy = 100.0 * correct / total
            losses.append(avg_loss)
            
            if task_id > 0:
                print(f"  Epoch {epoch + 1}/{epochs} - "
                      f"Total: {avg_loss:.4f}, "
                      f"New: {avg_new:.4f}, "
                      f"Distill: {avg_distill:.4f}, "
                      f"Acc: {accuracy:.2f}%")
            else:
                print(f"  Epoch {epoch + 1}/{epochs} - "
                      f"Loss: {avg_loss:.4f}, "
                      f"Acc: {accuracy:.2f}%")
        
        return losses
    
    def update_old_model(self):
        """
        Update the old model (teacher) after learning a task.
        
        This creates a deep copy of the current model to use as
        the teacher for the next task.
        """
        print("\n  Updating old model for next task...")
        
        # Create deep copy of current model
        self.old_model = copy.deepcopy(self.model)
        
        # Move to device and set to eval mode
        self.old_model.to(self.device)
        self.old_model.eval()
        
        # Freeze parameters (no gradient computation needed)
        for param in self.old_model.parameters():
            param.requires_grad = False
        
        print("  Old model updated and frozen")
    
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
        Main continual learning loop with LWF.
        
        Process for each task:
        1. Train with distillation loss (if not first task)
        2. Update old model for next task
        3. Evaluate on all tasks
        """
        num_tasks = len(train_loaders)
        self.accuracy_matrix = np.zeros((num_tasks, num_tasks))
        
        print("\n" + "=" * 70)
        print("CONTINUAL LEARNING WITH LEARNING WITHOUT FORGETTING (LWF)")
        print("=" * 70)
        print(f"Distillation Lambda (λ): {self.distill_lambda}")
        print(f"Temperature (T): {self.temperature}")
        
        for task_id in range(num_tasks):
            self.current_task = task_id
            
            # Train on current task
            self.train_task(
                train_loader=train_loaders[task_id],
                task_id=task_id,
                epochs=epochs_per_task
            )
            
            # Update old model for next task
            self.update_old_model()
            
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


def visualize_lwf_results(lwf_metrics: dict, num_tasks: int):
    """Visualize LWF results."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Accuracy Matrix Heatmap
    ax1 = axes[0]
    im = ax1.imshow(lwf_metrics['accuracy_matrix'], cmap='RdYlGn',
                    aspect='auto', vmin=0, vmax=100)
    plt.colorbar(im, ax=ax1, label='Accuracy (%)')
    ax1.set_xlabel('After Training Task')
    ax1.set_ylabel('Evaluated Task')
    ax1.set_title('LWF: Accuracy Matrix', fontweight='bold')
    ax1.set_xticks(range(num_tasks))
    ax1.set_yticks(range(num_tasks))
    
    for i in range(num_tasks):
        for j in range(num_tasks):
            if j >= i:
                ax1.text(j, i, f'{lwf_metrics["accuracy_matrix"][i, j]:.1f}',
                        ha="center", va="center", color="black", fontsize=9)
    
    # Plot 2: Backward Transfer
    ax2 = axes[1]
    tasks = list(range(len(lwf_metrics['forgetting_per_task'])))
    colors = ['green' if f >= 0 else 'red'
              for f in lwf_metrics['forgetting_per_task']]
    
    bars = ax2.bar(tasks, lwf_metrics['forgetting_per_task'],
                   color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.set_xlabel('Task')
    ax2.set_ylabel('Backward Transfer (%)')
    ax2.set_title('LWF: Forgetting per Task', fontweight='bold')
    ax2.set_xticks(tasks)
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, lwf_metrics['forgetting_per_task']):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%', ha='center',
                va='bottom' if val >= 0 else 'top', fontsize=9)
    
    # Plot 3: Learning vs Final
    ax3 = axes[2]
    learning_accs = np.diag(lwf_metrics['accuracy_matrix'])
    final_accs = lwf_metrics['accuracy_matrix'][:, -1]
    
    x = np.arange(num_tasks)
    width = 0.35
    
    ax3.bar(x - width/2, learning_accs, width,
            label='Learning Acc', color='skyblue', alpha=0.8)
    ax3.bar(x + width/2, final_accs, width,
            label='Final Acc', color='coral', alpha=0.8)
    
    ax3.set_xlabel('Task')
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_title('LWF: Learning vs Final Accuracy', fontweight='bold')
    ax3.set_xticks(range(num_tasks))
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim([0, 105])
    
    plt.tight_layout()
    plt.savefig('lwf_results.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved as 'lwf_results.png'")
    plt.show()


def main():
    """Main function demonstrating LWF."""
    print("=" * 70)
    print("LEARNING WITHOUT FORGETTING (LWF)")
    print("=" * 70)
    print("\nLWF prevents forgetting by:")
    print("  1. Saving predictions from the model before learning new task")
    print("  2. Using knowledge distillation to preserve these predictions")
    print("  3. No need to store previous examples (privacy-preserving)")
    print("=" * 70)
    
    # Configuration
    num_tasks = 5
    epochs_per_task = 5
    batch_size = 128
    learning_rate = 0.001
    distill_lambda = 1.0  # Weight for distillation loss
    temperature = 2.0     # Temperature for soft targets
    
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
    
    # Create LWF learner
    learner = LWFLearner(
        model=model,
        device=device,
        distill_lambda=distill_lambda,
        temperature=temperature,
        learning_rate=learning_rate
    )
    
    # Run continual learning
    import time
    start_time = time.time()
    metrics = learner.train_continual(train_loaders, test_loaders, epochs_per_task)
    total_time = time.time() - start_time
    
    # Print results
    print("\n" + "=" * 70)
    print("LWF RESULTS")
    print("=" * 70)
    print(f"\n📊 Key Metrics:")
    print(f"   Average Accuracy:     {metrics['average_accuracy']:.2f}%")
    print(f"   Learning Accuracy:    {metrics['learning_accuracy']:.2f}%")
    print(f"   Backward Transfer:    {metrics['backward_transfer']:.2f}%")
    print(f"\n⏱️  Training Time: {total_time:.2f} seconds")
    
    visualize_lwf_results(metrics, num_tasks)
    
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print("\n✓ LWF preserves knowledge without storing examples")
    print("✓ Knowledge distillation is privacy-preserving")
    print("✓ Simple to implement with any architecture")
    print("\n⚠️  Considerations:")
    print("  - Requires two forward passes per batch")
    print("  - λ and T are important hyperparameters")
    print("  - May struggle if task domains are very different")
    print("\n📝 Hyperparameter Tuning Tips:")
    print("  - λ ∈ [0.5, 5.0]: Higher = more preservation")
    print("  - T ∈ [1.0, 4.0]: Higher = softer targets")
    print("=" * 70)


if __name__ == "__main__":
    main()
