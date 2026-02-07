# Catastrophic Forgetting

## Introduction

**Catastrophic forgetting** (also called catastrophic interference) is the fundamental phenomenon that motivates all continual learning research. When neural networks are trained sequentially on multiple tasks, they tend to dramatically forget previously learned information when learning new tasks.

This section provides a rigorous treatment of the problem, including mathematical formalization, empirical demonstration, and analysis of why standard neural networks exhibit this behavior.

## Historical Context

The catastrophic forgetting problem was first identified by McCloskey and Cohen (1989) in their study of connectionist networks learning arithmetic facts. They observed that training a network on new addition problems caused severe interference with previously learned additions—a stark contrast to human learning where new knowledge typically enhances rather than destroys old knowledge.

!!! quote "Original Observation"
    "When a connectionist network is trained on a new set of patterns, the changes in connection weights that allow it to learn the new patterns typically cause it to 'unlearn' patterns that it learned previously."
    — McCloskey & Cohen, 1989

## Mathematical Framework

### Sequential Learning Setup

Consider training a neural network $f_\theta: \mathcal{X} \rightarrow \mathcal{Y}$ on a sequence of tasks:

$$
\mathcal{T}_1 \rightarrow \mathcal{T}_2 \rightarrow \cdots \rightarrow \mathcal{T}_T
$$

Each task $\tau$ has associated:

- Training distribution: $p_\tau(x, y)$
- Training dataset: $\mathcal{D}_\tau = \{(x_i^\tau, y_i^\tau)\}_{i=1}^{N_\tau}$
- Task-specific loss: $\mathcal{L}_\tau(\theta) = \mathbb{E}_{(x,y) \sim p_\tau}[\ell(f_\theta(x), y)]$

### Standard Training Procedure

In naive sequential learning, after task $\tau-1$, we initialize parameters at $\theta_{\tau-1}^*$ and optimize:

$$
\theta_\tau^* = \arg\min_\theta \mathcal{L}_\tau(\theta)
$$

This optimization moves parameters from $\theta_{\tau-1}^*$ to $\theta_\tau^*$ in a direction that minimizes the current task loss, without any constraint on maintaining performance on previous tasks.

### Why Forgetting Occurs

The gradient update at each step is:

$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}_\tau(\theta_t)
$$

This update considers only $\mathcal{L}_\tau$ and is completely agnostic to how the parameter change affects $\mathcal{L}_1, \ldots, \mathcal{L}_{\tau-1}$.

**Key insight**: The loss landscapes of different tasks are generally not aligned. Moving toward a minimum of $\mathcal{L}_\tau$ typically moves away from minima of previous task losses.

## Geometric Interpretation

### Parameter Space View

Consider the loss surfaces of two sequential tasks in parameter space:

```
           θ₁
            ↑
            |      Task 1 Minimum
            |           ●
            |          ╱ ╲
            |         ╱   ╲
            |        ╱     ╲
     ───────┼───────●───────────→ θ₂
            |   Task 2 Minimum
            |
```

Training on Task 2 moves parameters toward Task 2's minimum, away from Task 1's minimum. The severity of forgetting depends on:

1. **Distance between minima**: Further apart → more forgetting
2. **Curvature of loss surfaces**: Sharper minima → more sensitive to movement
3. **Dimensionality**: Higher dimensions provide more paths but also more interference

### Representational Overlap

Neural networks learn distributed representations where the same parameters encode multiple concepts. When training on a new task:

1. **Shared features** may be modified to better serve the new task
2. **Task-specific features** from old tasks may be overwritten
3. **Capacity allocation** shifts toward recent tasks

## Empirical Demonstration

### Split MNIST Benchmark

A standard demonstration uses Split MNIST, where the 10 MNIST digits are divided into 5 binary classification tasks:

| Task | Classes | Description |
|------|---------|-------------|
| $\mathcal{T}_1$ | {0, 1} | Distinguish 0 from 1 |
| $\mathcal{T}_2$ | {2, 3} | Distinguish 2 from 3 |
| $\mathcal{T}_3$ | {4, 5} | Distinguish 4 from 5 |
| $\mathcal{T}_4$ | {6, 7} | Distinguish 6 from 7 |
| $\mathcal{T}_5$ | {8, 9} | Distinguish 8 from 9 |

### PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class SimpleNetwork(nn.Module):
    """
    Simple feedforward network for demonstrating catastrophic forgetting.
    
    Architecture:
        Input (784) → Hidden (256) → Hidden (256) → Output (2)
    
    This architecture makes catastrophic forgetting clearly visible
    while remaining computationally tractable.
    """
    
    def __init__(self, input_size=784, hidden_size=256, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


def create_split_mnist_tasks(num_tasks=5):
    """
    Create Split MNIST task configuration.
    
    Returns:
        List of lists, where tasks[i] contains the digit classes for task i.
    """
    all_digits = list(range(10))
    classes_per_task = 10 // num_tasks
    return [all_digits[i * classes_per_task:(i + 1) * classes_per_task] 
            for i in range(num_tasks)]


def create_task_dataset(full_dataset, task_classes):
    """
    Filter dataset for specific task classes and remap labels.
    
    Args:
        full_dataset: Complete MNIST dataset
        task_classes: List of digit classes for this task
    
    Returns:
        TensorDataset with filtered examples and remapped labels
    """
    indices = [i for i in range(len(full_dataset)) 
               if full_dataset[i][1] in task_classes]
    
    data_list, label_list = [], []
    for idx in indices:
        img, label = full_dataset[idx]
        data_list.append(img)
        # Remap labels to [0, 1] for binary classification
        label_list.append(task_classes.index(label))
    
    return TensorDataset(
        torch.stack(data_list),
        torch.tensor(label_list, dtype=torch.long)
    )


def train_on_task(model, train_loader, device, epochs=5, lr=0.001):
    """Train model on a single task."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()


def evaluate_on_task(model, test_loader, device):
    """Evaluate model accuracy on a task."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return 100.0 * correct / total


def demonstrate_forgetting(num_tasks=5, epochs_per_task=5):
    """
    Main demonstration of catastrophic forgetting.
    
    Returns:
        accuracy_matrix: Matrix where entry [i,j] is accuracy on 
                        task i after training on task j
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_data = torchvision.datasets.MNIST(
        './data', train=True, download=True, transform=transform
    )
    test_data = torchvision.datasets.MNIST(
        './data', train=False, download=True, transform=transform
    )
    
    # Create task dataloaders
    task_classes = create_split_mnist_tasks(num_tasks)
    train_loaders = [
        DataLoader(create_task_dataset(train_data, classes), 
                   batch_size=128, shuffle=True)
        for classes in task_classes
    ]
    test_loaders = [
        DataLoader(create_task_dataset(test_data, classes),
                   batch_size=128, shuffle=False)
        for classes in task_classes
    ]
    
    # Initialize model
    model = SimpleNetwork(num_classes=2).to(device)
    
    # Track accuracy: rows = tasks, columns = training stages
    accuracy_matrix = np.zeros((num_tasks, num_tasks))
    
    # Sequential training
    for task_id in range(num_tasks):
        print(f"\nTraining on Task {task_id} (classes {task_classes[task_id]})")
        
        # Train on current task
        train_on_task(model, train_loaders[task_id], device, epochs_per_task)
        
        # Evaluate on all tasks seen so far
        for eval_id in range(task_id + 1):
            acc = evaluate_on_task(model, test_loaders[eval_id], device)
            accuracy_matrix[eval_id, task_id] = acc
            
            if eval_id < task_id:
                original = accuracy_matrix[eval_id, eval_id]
                forgetting = original - acc
                print(f"  Task {eval_id}: {acc:.1f}% "
                      f"(was {original:.1f}%, forgot {forgetting:.1f}%)")
            else:
                print(f"  Task {eval_id}: {acc:.1f}%")
    
    return accuracy_matrix, task_classes


# Run demonstration
if __name__ == "__main__":
    acc_matrix, tasks = demonstrate_forgetting()
    
    # Calculate metrics
    avg_forgetting = np.mean([
        acc_matrix[i, i] - acc_matrix[i, -1] 
        for i in range(len(tasks) - 1)
    ])
    avg_accuracy = np.mean(acc_matrix[:, -1])
    
    print(f"\n{'='*50}")
    print(f"Average Forgetting: {avg_forgetting:.1f}%")
    print(f"Final Average Accuracy: {avg_accuracy:.1f}%")
```

### Expected Results

Running this demonstration typically produces an accuracy matrix like:

| Task | After T0 | After T1 | After T2 | After T3 | After T4 | Forgetting |
|------|----------|----------|----------|----------|----------|------------|
| 0    | **98.5%** | 62.3% | 55.1% | 51.8% | 49.2% | 49.3% |
| 1    | - | **97.8%** | 58.6% | 52.4% | 50.1% | 47.7% |
| 2    | - | - | **98.2%** | 61.3% | 53.7% | 44.5% |
| 3    | - | - | - | **97.5%** | 58.9% | 38.6% |
| 4    | - | - | - | - | **98.1%** | N/A |

**Key observations**:

1. Each task achieves ~98% accuracy when first learned (diagonal)
2. Performance drops to ~50% (near random) after subsequent tasks
3. Average forgetting is typically 40-50%
4. Final average accuracy is much lower than joint training would achieve

## Factors Influencing Forgetting Severity

### Network Architecture

| Factor | Effect on Forgetting |
|--------|---------------------|
| **Network depth** | Deeper networks may forget more (more parameters to perturb) |
| **Width** | Wider layers provide more capacity but not necessarily less forgetting |
| **Activation functions** | ReLU networks may be more susceptible than sigmoid |
| **Skip connections** | Can help preserve representations |

### Task Characteristics

| Factor | Effect on Forgetting |
|--------|---------------------|
| **Task similarity** | Similar tasks may cause more interference |
| **Task difficulty** | Harder new tasks require more parameter changes |
| **Task sequence** | Order can significantly impact final performance |
| **Dataset size** | More training data per task can increase overwriting |

### Training Hyperparameters

| Factor | Effect on Forgetting |
|--------|---------------------|
| **Learning rate** | Higher rates → faster learning but more forgetting |
| **Number of epochs** | More epochs → better task performance but more forgetting |
| **Batch size** | Can affect the sharpness of minima reached |
| **Optimizer** | Adaptive methods may converge to different minima |

## Theoretical Analysis

### Linear Networks

For linear networks, catastrophic forgetting can be analyzed exactly. Consider a single-layer linear network:

$$
f_\theta(x) = Wx
$$

Training on task $\tau$ with MSE loss produces the optimal solution:

$$
W_\tau^* = Y_\tau X_\tau^T (X_\tau X_\tau^T)^{-1}
$$

When tasks have non-overlapping input distributions, the solution for task $\tau$ completely overwrites the solution for task $\tau-1$ in the relevant subspace.

### Connection to Gradient Interference

The severity of forgetting relates to gradient alignment between tasks. Define the gradient of task $\tau$ at parameters $\theta$:

$$
g_\tau(\theta) = \nabla_\theta \mathcal{L}_\tau(\theta)
$$

**Gradient interference** occurs when:

$$
g_{\tau_1}(\theta)^T g_{\tau_2}(\theta) < 0
$$

This means improving on one task degrades the other. The magnitude of this negative inner product predicts forgetting severity.

## The Stability-Plasticity Dilemma

Catastrophic forgetting reflects a fundamental tension in learning systems:

**Plasticity**: The ability to acquire new knowledge
: Required to learn new tasks effectively

**Stability**: The ability to retain old knowledge  
: Required to avoid catastrophic forgetting

!!! info "The Dilemma"
    Any system must balance these competing demands. Pure stability means no learning; pure plasticity means complete forgetting. Continual learning methods are different approaches to managing this trade-off.

### Biological Perspective

Interestingly, biological neural networks do not exhibit catastrophic forgetting to the same degree. Proposed mechanisms include:

1. **Sparse coding**: Only small subsets of neurons active for any stimulus
2. **Neuromodulation**: Context-dependent plasticity rates
3. **Memory consolidation**: Separate systems for recent vs. remote memories
4. **Neurogenesis**: Growth of new neurons for new information
5. **Synaptic consolidation**: Selective stabilization of important synapses

These biological insights have inspired many continual learning algorithms.

## Visualization of Forgetting

```python
def visualize_forgetting(accuracy_matrix, task_classes):
    """Create comprehensive visualization of catastrophic forgetting."""
    num_tasks = len(task_classes)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Heatmap
    ax1 = axes[0]
    im = ax1.imshow(accuracy_matrix, cmap='RdYlGn', vmin=0, vmax=100)
    ax1.set_xlabel('After Training Task')
    ax1.set_ylabel('Evaluated Task')
    ax1.set_title('Accuracy Matrix: Catastrophic Forgetting', fontweight='bold')
    ax1.set_xticks(range(num_tasks))
    ax1.set_yticks(range(num_tasks))
    ax1.set_xticklabels([f'T{i}' for i in range(num_tasks)])
    ax1.set_yticklabels([f'T{i}' for i in range(num_tasks)])
    
    for i in range(num_tasks):
        for j in range(num_tasks):
            if j >= i:
                ax1.text(j, i, f'{accuracy_matrix[i, j]:.0f}',
                        ha='center', va='center', fontsize=10)
    
    plt.colorbar(im, ax=ax1, label='Accuracy (%)')
    
    # Line plot showing degradation
    ax2 = axes[1]
    for task_id in range(num_tasks):
        accs = [accuracy_matrix[task_id, j] 
                for j in range(task_id, num_tasks)]
        ax2.plot(range(task_id, num_tasks), accs, 
                marker='o', linewidth=2, markersize=8,
                label=f'Task {task_id} (classes {task_classes[task_id]})')
    
    ax2.set_xlabel('Training Stage')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Performance Degradation Over Time', fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.set_ylim([0, 105])
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(num_tasks))
    ax2.set_xticklabels([f'T{i}' for i in range(num_tasks)])
    
    plt.tight_layout()
    plt.savefig('catastrophic_forgetting_visualization.png', dpi=150)
    plt.show()
```

## Implications for Practice

### When Catastrophic Forgetting Matters

1. **Robotics**: Learning new manipulation skills shouldn't erase basic motor control
2. **Medical AI**: Incorporating new diseases shouldn't degrade detection of known conditions
3. **Recommendation systems**: User preferences evolve but historical patterns matter
4. **Natural language**: Models should adapt to new domains without losing general capabilities
5. **Autonomous vehicles**: New scenarios shouldn't cause forgetting of basic driving rules

### When It May Be Acceptable

1. **Single-task applications**: No sequential learning required
2. **Abundant replay**: If all past data can be stored and replayed
3. **Fresh starts**: When past knowledge is deliberately discarded
4. **Transfer-only**: When only forward transfer matters (fine-tuning)

## Summary

Catastrophic forgetting is the central challenge that continual learning methods must address:

- **Definition**: Severe performance degradation on old tasks when learning new ones
- **Cause**: Unconstrained optimization overwrites important weights
- **Severity**: Typically 40-60% accuracy loss on Split MNIST benchmark
- **Fundamental**: Reflects the stability-plasticity dilemma in learning systems

The following sections present various approaches to mitigate catastrophic forgetting, each offering different trade-offs between:

- Memory efficiency (storing past examples or not)
- Computational cost (additional forward/backward passes)
- Privacy preservation (avoiding data storage)
- Scalability (handling many sequential tasks)

## References

1. McCloskey, M., & Cohen, N. J. (1989). Catastrophic interference in connectionist networks: The sequential learning problem. *Psychology of Learning and Motivation*, 24, 109-165.

2. French, R. M. (1999). Catastrophic forgetting in connectionist networks. *Trends in Cognitive Sciences*, 3(4), 128-135.

3. Goodfellow, I. J., Mirza, M., Xiao, D., Courville, A., & Bengio, Y. (2013). An empirical investigation of catastrophic forgetting in gradient-based neural networks. *arXiv preprint arXiv:1312.6211*.

4. Kemker, R., McClure, M., Abitino, A., Hayes, T., & Kanan, C. (2018). Measuring catastrophic forgetting in neural networks. *AAAI Conference on Artificial Intelligence*.
