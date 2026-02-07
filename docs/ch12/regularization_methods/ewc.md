# Elastic Weight Consolidation (EWC)

## Introduction

**Elastic Weight Consolidation (EWC)** is one of the most influential regularization-based methods for continual learning, introduced by Kirkpatrick et al. (2017). The key insight of EWC is that not all neural network parameters are equally important for previously learned tasks—some parameters are critical while others can be freely modified. EWC identifies and protects important parameters using the **Fisher information matrix**.

!!! success "Key Advantages"
    - **No data storage**: Privacy-preserving (doesn't store previous examples)
    - **Fixed memory**: Memory doesn't grow with number of tasks
    - **Principled foundation**: Grounded in Bayesian inference and information theory

## Theoretical Foundation

### Bayesian Perspective

EWC derives from a Bayesian view of learning. Consider the posterior over parameters after learning task A:

$$
p(\theta | \mathcal{D}_A) \propto p(\mathcal{D}_A | \theta) p(\theta)
$$

When learning task B, we want to find parameters that are both good for B and consistent with what we learned from A. The optimal approach uses the posterior from A as the prior for B:

$$
\log p(\theta | \mathcal{D}_A, \mathcal{D}_B) = \log p(\mathcal{D}_B | \theta) + \log p(\theta | \mathcal{D}_A) + \text{const}
$$

The challenge is that $p(\theta | \mathcal{D}_A)$ is intractable for neural networks.

### Laplace Approximation

EWC approximates the posterior $p(\theta | \mathcal{D}_A)$ with a Gaussian:

$$
p(\theta | \mathcal{D}_A) \approx \mathcal{N}(\theta_A^*, F_A^{-1})
$$

where:
- $\theta_A^*$ are the optimal parameters after task A
- $F_A$ is the Fisher information matrix

The log of this Gaussian gives us a quadratic penalty:

$$
\log p(\theta | \mathcal{D}_A) \approx -\frac{1}{2}(\theta - \theta_A^*)^T F_A (\theta - \theta_A^*) + \text{const}
$$

### Fisher Information Matrix

The Fisher information matrix quantifies how much information the data provides about each parameter:

$$
F = \mathbb{E}_{x \sim p(x)} \left[ \nabla_\theta \log p(y|x, \theta) \nabla_\theta \log p(y|x, \theta)^T \right]
$$

**Key property**: Parameters with high Fisher information are those where small changes significantly affect the model's predictions—these are the important parameters to protect.

#### Diagonal Approximation

The full Fisher matrix is $d \times d$ for $d$ parameters, making it computationally intractable. EWC uses a diagonal approximation:

$$
F_i = \mathbb{E}_{x \sim p(x)} \left[ \left( \frac{\partial \log p(y|x, \theta)}{\partial \theta_i} \right)^2 \right]
$$

This reduces storage from $O(d^2)$ to $O(d)$.

## The EWC Loss Function

The complete EWC loss when learning task $\tau$ is:

$$
\mathcal{L}(\theta) = \mathcal{L}_\tau(\theta) + \frac{\lambda}{2} \sum_{i} F_i (\theta_i - \theta_{\tau-1,i}^*)^2
$$

where:
- $\mathcal{L}_\tau(\theta)$: Loss on current task
- $F_i$: Fisher information for parameter $i$ (from previous task)
- $\theta_{\tau-1}^*$: Optimal parameters after previous task
- $\lambda$: Regularization strength (hyperparameter)

!!! info "Intuition"
    The penalty term acts like a spring connecting each parameter to its previous optimal value. The spring constant (stiffness) is determined by the Fisher information—important parameters have stiff springs, while unimportant ones have loose springs.

## PyTorch Implementation

### Complete EWC Learner Class

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, Optional, List
import copy


class EWCLearner:
    """
    Elastic Weight Consolidation for continual learning.
    
    EWC prevents catastrophic forgetting by:
    1. Computing Fisher information after each task (parameter importance)
    2. Adding quadratic penalty to protect important parameters
    3. Balancing new task learning with old knowledge preservation
    
    Reference: Kirkpatrick et al., "Overcoming catastrophic forgetting 
               in neural networks," PNAS 2017
    """
    
    def __init__(self,
                 model: nn.Module,
                 device: torch.device,
                 ewc_lambda: float = 5000.0,
                 learning_rate: float = 0.001,
                 fisher_sample_size: Optional[int] = None):
        """
        Initialize EWC learner.
        
        Args:
            model: Neural network model
            device: Computation device (CPU or GPU)
            ewc_lambda: Regularization strength (λ)
                       Higher values = more protection of old parameters
                       Typical range: [100, 10000]
            learning_rate: Learning rate for optimizer
            fisher_sample_size: Number of samples for Fisher computation
                               (None = use entire dataset)
        """
        self.model = model
        self.device = device
        self.ewc_lambda = ewc_lambda
        self.learning_rate = learning_rate
        self.fisher_sample_size = fisher_sample_size
        
        self.criterion = nn.CrossEntropyLoss()
        
        # Storage for Fisher information and optimal parameters
        # Key: task_id, Value: Dict[param_name -> tensor]
        self.fisher_dict: Dict[int, Dict[str, torch.Tensor]] = {}
        self.optpar_dict: Dict[int, Dict[str, torch.Tensor]] = {}
        
        self.current_task = 0
    
    def compute_fisher_information(self, 
                                   data_loader,
                                   task_id: int) -> Dict[str, torch.Tensor]:
        """
        Compute diagonal Fisher information matrix.
        
        The Fisher information F_i for parameter θ_i is:
        F_i = E[(∂log p(y|x,θ)/∂θ_i)²]
        
        We approximate this empirically:
        F_i ≈ (1/N) Σ_n (∂L(x_n, θ)/∂θ_i)²
        
        Args:
            data_loader: DataLoader for computing Fisher
            task_id: Current task identifier
        
        Returns:
            Dictionary mapping parameter names to Fisher values
        """
        self.model.eval()
        
        # Initialize Fisher accumulator
        fisher = {name: torch.zeros_like(param.data)
                 for name, param in self.model.named_parameters()
                 if param.requires_grad}
        
        total_samples = 0
        
        for batch_idx, (data, target) in enumerate(data_loader):
            # Check sample limit
            if (self.fisher_sample_size is not None and 
                total_samples >= self.fisher_sample_size):
                break
            
            data, target = data.to(self.device), target.to(self.device)
            batch_size = data.size(0)
            
            # Forward pass
            self.model.zero_grad()
            output = self.model(data)
            
            # Compute log probabilities
            log_probs = F.log_softmax(output, dim=1)
            
            # For each sample, compute squared gradients
            for i in range(batch_size):
                if (self.fisher_sample_size is not None and 
                    total_samples >= self.fisher_sample_size):
                    break
                
                self.model.zero_grad()
                
                # Log probability for true class
                log_prob = log_probs[i, target[i]]
                
                # Backpropagate
                log_prob.backward(retain_graph=(i < batch_size - 1))
                
                # Accumulate squared gradients
                for name, param in self.model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        fisher[name] += param.grad.data ** 2
                
                total_samples += 1
        
        # Average over samples
        for name in fisher:
            fisher[name] /= total_samples
        
        return fisher
    
    def compute_ewc_penalty(self) -> torch.Tensor:
        """
        Compute EWC regularization penalty.
        
        Penalty = (λ/2) Σ_tasks Σ_i F_i(task) (θ_i - θ*_i(task))²
        
        Returns:
            EWC penalty term (scalar tensor)
        """
        penalty = torch.tensor(0.0, device=self.device)
        
        # Sum over all previous tasks
        for task_id in self.fisher_dict.keys():
            for name, param in self.model.named_parameters():
                if name in self.fisher_dict[task_id]:
                    fisher = self.fisher_dict[task_id][name]
                    optpar = self.optpar_dict[task_id][name]
                    
                    # Quadratic penalty weighted by Fisher
                    penalty += (fisher * (param - optpar) ** 2).sum()
        
        return (self.ewc_lambda / 2) * penalty
    
    def train_on_task(self,
                      train_loader,
                      test_loader,
                      epochs: int = 5,
                      verbose: bool = True) -> Dict:
        """
        Train on a single task with EWC regularization.
        
        Args:
            train_loader: Training data loader
            test_loader: Test data loader for evaluation
            epochs: Number of training epochs
            verbose: Print training progress
        
        Returns:
            Dictionary with training statistics
        """
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        train_losses = []
        ewc_losses = []
        task_losses = []
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_ewc = 0.0
            epoch_task = 0.0
            num_batches = 0
            
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                output = self.model(data)
                
                # Task-specific loss
                task_loss = self.criterion(output, target)
                
                # EWC penalty (only if we have previous tasks)
                if len(self.fisher_dict) > 0:
                    ewc_loss = self.compute_ewc_penalty()
                else:
                    ewc_loss = torch.tensor(0.0, device=self.device)
                
                # Total loss
                total_loss = task_loss + ewc_loss
                
                # Backward pass
                total_loss.backward()
                optimizer.step()
                
                # Track statistics
                epoch_loss += total_loss.item()
                epoch_ewc += ewc_loss.item()
                epoch_task += task_loss.item()
                num_batches += 1
            
            # Average losses
            avg_loss = epoch_loss / num_batches
            avg_ewc = epoch_ewc / num_batches
            avg_task = epoch_task / num_batches
            
            train_losses.append(avg_loss)
            ewc_losses.append(avg_ewc)
            task_losses.append(avg_task)
            
            if verbose:
                print(f"  Epoch {epoch+1}/{epochs}: "
                      f"Total={avg_loss:.4f}, Task={avg_task:.4f}, EWC={avg_ewc:.4f}")
        
        return {
            'train_losses': train_losses,
            'ewc_losses': ewc_losses,
            'task_losses': task_losses
        }
    
    def consolidate_task(self, data_loader, task_id: int):
        """
        Consolidate knowledge after learning a task.
        
        This computes and stores:
        1. Fisher information (parameter importance)
        2. Optimal parameters (reference point for penalty)
        
        Args:
            data_loader: Data loader for Fisher computation
            task_id: Task identifier
        """
        print(f"  Consolidating Task {task_id}...")
        
        # Compute Fisher information
        fisher = self.compute_fisher_information(data_loader, task_id)
        self.fisher_dict[task_id] = fisher
        
        # Store optimal parameters
        optpar = {name: param.data.clone()
                 for name, param in self.model.named_parameters()
                 if param.requires_grad}
        self.optpar_dict[task_id] = optpar
        
        # Print Fisher statistics
        all_fisher = torch.cat([f.flatten() for f in fisher.values()])
        print(f"  Fisher stats - Mean: {all_fisher.mean():.6f}, "
              f"Max: {all_fisher.max():.6f}, "
              f"Sparsity: {(all_fisher < 1e-6).float().mean():.2%}")
    
    def evaluate(self, test_loader) -> float:
        """Evaluate model accuracy on a task."""
        self.model.eval()
        correct, total = 0, 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        return 100.0 * correct / total
    
    def train_continual(self,
                        train_loaders: List,
                        test_loaders: List,
                        epochs_per_task: int = 5) -> Dict:
        """
        Complete continual learning pipeline.
        
        Args:
            train_loaders: List of training DataLoaders
            test_loaders: List of test DataLoaders
            epochs_per_task: Training epochs per task
        
        Returns:
            Dictionary with accuracy matrix and metrics
        """
        num_tasks = len(train_loaders)
        accuracy_matrix = np.zeros((num_tasks, num_tasks))
        
        for task_id in range(num_tasks):
            print(f"\n{'='*60}")
            print(f"Task {task_id}")
            print('='*60)
            
            # Train on current task
            self.train_on_task(
                train_loaders[task_id],
                test_loaders[task_id],
                epochs=epochs_per_task
            )
            
            # Consolidate (compute Fisher and store parameters)
            self.consolidate_task(train_loaders[task_id], task_id)
            
            # Evaluate on all tasks seen so far
            print(f"\n  Evaluation:")
            for eval_id in range(task_id + 1):
                acc = self.evaluate(test_loaders[eval_id])
                accuracy_matrix[eval_id, task_id] = acc
                
                if eval_id < task_id:
                    original = accuracy_matrix[eval_id, eval_id]
                    change = acc - original
                    print(f"    Task {eval_id}: {acc:.1f}% "
                          f"(was {original:.1f}%, change: {change:+.1f}%)")
                else:
                    print(f"    Task {eval_id}: {acc:.1f}%")
        
        return {
            'accuracy_matrix': accuracy_matrix,
            'num_tasks': num_tasks
        }
```

### Usage Example

```python
import numpy as np

# Configuration
num_tasks = 5
epochs_per_task = 5
ewc_lambda = 5000.0  # Key hyperparameter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create model
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, 2)  # Binary classification per task
).to(device)

# Initialize EWC learner
ewc_learner = EWCLearner(
    model=model,
    device=device,
    ewc_lambda=ewc_lambda,
    learning_rate=0.001
)

# Load data and create task loaders (as shown in previous sections)
# train_loaders, test_loaders = create_split_mnist_tasks(...)

# Run continual learning
results = ewc_learner.train_continual(
    train_loaders, 
    test_loaders,
    epochs_per_task=epochs_per_task
)

# Compute metrics
from continual_metrics import ContinualLearningMetrics
metrics = ContinualLearningMetrics(results['accuracy_matrix'])
metrics.print_report()
```

## Hyperparameter Analysis

### The λ Parameter

The regularization strength $\lambda$ controls the stability-plasticity trade-off:

| λ Value | Effect |
|---------|--------|
| Too low (e.g., 10) | Weak protection → significant forgetting |
| Optimal (e.g., 1000-5000) | Balanced protection and learning |
| Too high (e.g., 100000) | Excessive protection → inability to learn new tasks |

```python
def analyze_lambda_sensitivity(train_loaders, test_loaders, 
                               lambda_values, device):
    """
    Analyze sensitivity to λ hyperparameter.
    
    Args:
        train_loaders: Training data loaders
        test_loaders: Test data loaders
        lambda_values: List of λ values to test
        device: Computation device
    
    Returns:
        Dictionary mapping λ to metrics
    """
    results = {}
    
    for ewc_lambda in lambda_values:
        print(f"\nTesting λ = {ewc_lambda}")
        
        # Fresh model for each λ
        model = create_model().to(device)
        
        learner = EWCLearner(
            model=model,
            device=device,
            ewc_lambda=ewc_lambda
        )
        
        result = learner.train_continual(
            train_loaders, test_loaders, epochs_per_task=5
        )
        
        metrics = ContinualLearningMetrics(result['accuracy_matrix'])
        results[ewc_lambda] = {
            'avg_accuracy': metrics.average_accuracy,
            'backward_transfer': metrics.backward_transfer,
            'learning_accuracy': metrics.learning_accuracy
        }
        
        print(f"  AA: {metrics.average_accuracy:.1f}%, "
              f"BWT: {metrics.backward_transfer:+.1f}%")
    
    return results
```

### Visualization of λ Effect

```python
def plot_lambda_analysis(results):
    """Plot the effect of λ on different metrics."""
    lambdas = list(results.keys())
    aa = [results[l]['avg_accuracy'] for l in lambdas]
    bwt = [results[l]['backward_transfer'] for l in lambdas]
    la = [results[l]['learning_accuracy'] for l in lambdas]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Average Accuracy
    axes[0].semilogx(lambdas, aa, 'o-', linewidth=2, markersize=8)
    axes[0].set_xlabel('λ (log scale)')
    axes[0].set_ylabel('Average Accuracy (%)')
    axes[0].set_title('AA vs λ')
    axes[0].grid(True, alpha=0.3)
    
    # Backward Transfer
    axes[1].semilogx(lambdas, bwt, 'o-', linewidth=2, markersize=8, color='red')
    axes[1].axhline(y=0, color='black', linestyle='--')
    axes[1].set_xlabel('λ (log scale)')
    axes[1].set_ylabel('Backward Transfer (%)')
    axes[1].set_title('BWT vs λ')
    axes[1].grid(True, alpha=0.3)
    
    # Learning Accuracy
    axes[2].semilogx(lambdas, la, 'o-', linewidth=2, markersize=8, color='green')
    axes[2].set_xlabel('λ (log scale)')
    axes[2].set_ylabel('Learning Accuracy (%)')
    axes[2].set_title('LA vs λ')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ewc_lambda_analysis.png', dpi=150)
    plt.show()
```

## EWC Variants

### Online EWC

Instead of storing separate Fisher matrices for each task, Online EWC maintains a running estimate:

$$
\tilde{F} = \gamma \tilde{F}_{\text{old}} + F_{\text{new}}
$$

where $\gamma \in [0, 1]$ is a decay factor.

```python
class OnlineEWC(EWCLearner):
    """
    Online EWC with running Fisher estimate.
    
    Maintains a single Fisher matrix that accumulates importance
    across tasks, rather than storing per-task matrices.
    """
    
    def __init__(self, model, device, ewc_lambda=5000.0, 
                 gamma=0.9, **kwargs):
        super().__init__(model, device, ewc_lambda, **kwargs)
        self.gamma = gamma
        self.running_fisher = None
        self.running_optpar = None
    
    def consolidate_task(self, data_loader, task_id):
        """Update running Fisher estimate."""
        new_fisher = self.compute_fisher_information(data_loader, task_id)
        
        if self.running_fisher is None:
            self.running_fisher = new_fisher
        else:
            # Decay old Fisher and add new
            for name in self.running_fisher:
                self.running_fisher[name] = (
                    self.gamma * self.running_fisher[name] + 
                    new_fisher[name]
                )
        
        # Update reference parameters
        self.running_optpar = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
    
    def compute_ewc_penalty(self):
        """Compute penalty using running Fisher."""
        if self.running_fisher is None:
            return torch.tensor(0.0, device=self.device)
        
        penalty = torch.tensor(0.0, device=self.device)
        
        for name, param in self.model.named_parameters():
            if name in self.running_fisher:
                fisher = self.running_fisher[name]
                optpar = self.running_optpar[name]
                penalty += (fisher * (param - optpar) ** 2).sum()
        
        return (self.ewc_lambda / 2) * penalty
```

### EWC++

EWC++ (Chaudhry et al., 2018) makes two improvements:

1. **Separate regularization per task**: Different λ for each previous task
2. **Normalized Fisher**: Scale Fisher values to prevent explosion

```python
class EWCPlusPlus(EWCLearner):
    """
    EWC++ with normalized Fisher and per-task lambda.
    """
    
    def consolidate_task(self, data_loader, task_id):
        """Compute normalized Fisher."""
        fisher = self.compute_fisher_information(data_loader, task_id)
        
        # Normalize Fisher per layer
        for name in fisher:
            f = fisher[name]
            if f.max() > 0:
                fisher[name] = f / f.max()
        
        self.fisher_dict[task_id] = fisher
        self.optpar_dict[task_id] = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
```

## Computational Considerations

### Time Complexity

| Operation | Complexity | When |
|-----------|------------|------|
| Training forward/backward | $O(N \cdot d)$ | Each batch |
| EWC penalty computation | $O(T \cdot d)$ | Each batch (T = num tasks) |
| Fisher computation | $O(N \cdot d)$ | After each task |

### Memory Complexity

| Storage | Size | Growth |
|---------|------|--------|
| Model parameters | $O(d)$ | Fixed |
| Fisher per task | $O(d)$ | Linear in T |
| Optimal params per task | $O(d)$ | Linear in T |
| **Total EWC overhead** | $O(T \cdot d)$ | Linear in tasks |

!!! warning "Scalability"
    For very long task sequences, consider Online EWC to keep memory constant.

## Advantages and Limitations

### Advantages

1. **No data storage**: Privacy-preserving; doesn't require storing previous examples
2. **Theoretical foundation**: Derived from Bayesian principles
3. **Computational efficiency**: Training overhead is minimal
4. **Interpretability**: Fisher values show which parameters are important

### Limitations

1. **Diagonal approximation**: Ignores parameter correlations
2. **Accumulated constraint**: Constraints compound across tasks
3. **Task-specific λ**: Optimal λ may vary across tasks
4. **Capacity limitation**: Eventually runs out of plastic parameters

## Expected Results

On Split MNIST with default hyperparameters:

| Metric | Naive Baseline | EWC |
|--------|----------------|-----|
| Average Accuracy | ~55% | ~85% |
| Backward Transfer | ~-45% | ~-12% |
| Learning Accuracy | ~98% | ~95% |

EWC significantly reduces forgetting while maintaining good learning capability.

## Summary

Elastic Weight Consolidation provides a principled approach to continual learning by:

1. **Identifying important parameters** through Fisher information
2. **Protecting important parameters** with a quadratic penalty
3. **Allowing flexibility** for unimportant parameters

The method requires no data storage, making it suitable for privacy-sensitive applications, though it may accumulate constraints over many tasks.

## References

1. Kirkpatrick, J., et al. (2017). Overcoming catastrophic forgetting in neural networks. *PNAS*, 114(13), 3521-3526.

2. Huszár, F. (2018). Note on the quadratic penalties in elastic weight consolidation. *PNAS*, 115(11), E2496-E2497.

3. Schwarz, J., et al. (2018). Progress & compress: A scalable framework for continual learning. *ICML*.

4. Chaudhry, A., et al. (2018). Riemannian walk for incremental learning: Understanding forgetting and intransigence. *ECCV*.
