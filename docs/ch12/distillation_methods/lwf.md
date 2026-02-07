# Learning Without Forgetting (LwF)

## Introduction

**Learning Without Forgetting (LwF)** is a knowledge distillation-based approach to continual learning introduced by Li & Hoiem (2017). Unlike experience replay, LwF does not store previous examples. Instead, it uses the current task's data to maintain performance on old tasks by matching the model's outputs to its own previous predictions.

!!! success "Key Advantages"
    - **No data storage**: Privacy-preserving approach
    - **Fixed memory**: Memory doesn't grow with tasks
    - **Leverages current data**: Uses new task data to preserve old knowledge
    - **Simple implementation**: Based on standard knowledge distillation

## Theoretical Foundation

### Knowledge Distillation Review

Knowledge distillation transfers knowledge from a "teacher" model to a "student" model by matching soft probability distributions rather than hard labels:

$$
\mathcal{L}_{\text{distill}} = \text{KL}\left( p_{\text{teacher}}(y|x; T) \| p_{\text{student}}(y|x; T) \right)
$$

where $T$ is the temperature parameter that softens the distributions:

$$
p(y_i|x; T) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
$$

Higher temperature produces softer probability distributions that reveal more information about class relationships.

### LwF Insight

LwF treats the model before training on a new task as the "teacher" and uses it to constrain the model during new task learning. The key insight is:

> **Even though we don't have access to old task data, we can use new task data to approximate what the model would predict on old tasks.**

When we present new task images to the old model, its predictions on old task outputs reflect what it learned previously. By encouraging the updated model to match these predictions, we preserve old knowledge.

## The LwF Loss Function

For task $\tau$, the LwF loss combines:

1. **Cross-entropy** on new task: Standard classification loss
2. **Distillation** on old tasks: Match soft outputs from before training

$$
\mathcal{L}_{\text{LwF}} = \mathcal{L}_{\text{CE}}(y_\tau, \hat{y}_\tau) + \lambda \cdot \mathcal{L}_{\text{distill}}(\hat{y}_{1:\tau-1}^{\text{old}}, \hat{y}_{1:\tau-1}^{\text{new}})
$$

where:
- $\mathcal{L}_{\text{CE}}$: Cross-entropy loss on current task
- $\hat{y}_{1:\tau-1}^{\text{old}}$: Soft predictions from model before training (frozen)
- $\hat{y}_{1:\tau-1}^{\text{new}}$: Current model's predictions on old task heads
- $\lambda$: Balance parameter
- Temperature $T$ is used in distillation

### Distillation Loss Details

The distillation loss uses KL divergence with temperature scaling:

$$
\mathcal{L}_{\text{distill}} = T^2 \cdot \text{KL}\left( \text{softmax}(z^{\text{old}}/T) \| \text{softmax}(z^{\text{new}}/T) \right)
$$

The $T^2$ factor compensates for the reduced gradient magnitude at higher temperatures.

## PyTorch Implementation

### Complete LwF Learner

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import List, Dict, Optional
import copy


class LwFLearner:
    """
    Learning Without Forgetting for continual learning.
    
    LwF prevents forgetting by:
    1. Recording model outputs before training on new task
    2. Using knowledge distillation to match old outputs
    3. Combining distillation loss with new task loss
    
    Reference: Li & Hoiem, "Learning Without Forgetting," TPAMI 2017
    """
    
    def __init__(self,
                 model: nn.Module,
                 device: torch.device,
                 distill_lambda: float = 1.0,
                 temperature: float = 2.0,
                 learning_rate: float = 0.001):
        """
        Initialize LwF learner.
        
        Args:
            model: Neural network model
            device: Computation device
            distill_lambda: Weight for distillation loss
            temperature: Temperature for soft targets
                        Higher = softer distributions
            learning_rate: Learning rate for optimizer
        """
        self.model = model
        self.device = device
        self.distill_lambda = distill_lambda
        self.temperature = temperature
        self.learning_rate = learning_rate
        
        self.criterion = nn.CrossEntropyLoss()
        
        # Store reference model (frozen) for distillation
        self.old_model: Optional[nn.Module] = None
        
        self.current_task = 0
    
    def distillation_loss(self, 
                          new_logits: torch.Tensor,
                          old_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute knowledge distillation loss.
        
        Uses KL divergence between soft probability distributions:
        L_distill = T² × KL(softmax(z_old/T) || softmax(z_new/T))
        
        Args:
            new_logits: Logits from current model
            old_logits: Logits from old (frozen) model
        
        Returns:
            Distillation loss (scalar)
        """
        T = self.temperature
        
        # Soft probabilities
        old_probs = F.softmax(old_logits / T, dim=1)
        new_log_probs = F.log_softmax(new_logits / T, dim=1)
        
        # KL divergence (with T² scaling)
        # Note: F.kl_div expects log probabilities as first argument
        loss = F.kl_div(new_log_probs, old_probs, reduction='batchmean')
        
        # Scale by T² to maintain gradient magnitude
        return loss * (T ** 2)
    
    def save_old_model(self):
        """
        Save current model as reference for distillation.
        
        Called before training on a new task.
        """
        if self.current_task > 0:
            self.old_model = copy.deepcopy(self.model)
            self.old_model.eval()
            
            # Freeze all parameters
            for param in self.old_model.parameters():
                param.requires_grad = False
            
            print("  Old model saved for distillation")
    
    def train_on_task(self,
                      train_loader: DataLoader,
                      epochs: int = 5,
                      verbose: bool = True) -> Dict:
        """
        Train on a task with LwF distillation.
        
        Args:
            train_loader: Training data loader
            epochs: Number of training epochs
            verbose: Print progress
        
        Returns:
            Training statistics
        """
        # Save old model before training
        self.save_old_model()
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        losses = {'total': [], 'task': [], 'distill': []}
        
        for epoch in range(epochs):
            self.model.train()
            epoch_losses = {'total': 0, 'task': 0, 'distill': 0}
            num_batches = 0
            
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass through current model
                output = self.model(data)
                
                # Task-specific loss (cross-entropy)
                task_loss = self.criterion(output, target)
                
                # Distillation loss (if we have an old model)
                distill_loss = torch.tensor(0.0, device=self.device)
                
                if self.old_model is not None:
                    with torch.no_grad():
                        old_output = self.old_model(data)
                    
                    distill_loss = self.distillation_loss(output, old_output)
                
                # Combined loss
                total_loss = task_loss + self.distill_lambda * distill_loss
                
                # Backward pass
                total_loss.backward()
                optimizer.step()
                
                # Track losses
                epoch_losses['total'] += total_loss.item()
                epoch_losses['task'] += task_loss.item()
                epoch_losses['distill'] += distill_loss.item()
                num_batches += 1
            
            # Average losses
            for key in epoch_losses:
                losses[key].append(epoch_losses[key] / num_batches)
            
            if verbose:
                print(f"  Epoch {epoch+1}/{epochs}: "
                      f"Total={losses['total'][-1]:.4f}, "
                      f"Task={losses['task'][-1]:.4f}, "
                      f"Distill={losses['distill'][-1]:.4f}")
        
        self.current_task += 1
        
        return losses
    
    def evaluate(self, test_loader: DataLoader) -> float:
        """Evaluate accuracy on a task."""
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
                        train_loaders: List[DataLoader],
                        test_loaders: List[DataLoader],
                        epochs_per_task: int = 5) -> Dict:
        """
        Complete continual learning with LwF.
        
        Args:
            train_loaders: List of training DataLoaders
            test_loaders: List of test DataLoaders
            epochs_per_task: Training epochs per task
        
        Returns:
            Dictionary with accuracy matrix
        """
        num_tasks = len(train_loaders)
        accuracy_matrix = np.zeros((num_tasks, num_tasks))
        
        for task_id in range(num_tasks):
            print(f"\n{'='*60}")
            print(f"Task {task_id}")
            print('='*60)
            
            # Train on current task with distillation
            self.train_on_task(
                train_loaders[task_id],
                epochs=epochs_per_task
            )
            
            # Evaluate on all tasks
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
        
        return {'accuracy_matrix': accuracy_matrix}
```

## Hyperparameter Analysis

### Temperature Effect

The temperature $T$ controls the softness of probability distributions:

| Temperature | Effect | Use Case |
|-------------|--------|----------|
| T = 1 | Hard targets | Standard training |
| T = 2-4 | Soft targets | Typical distillation |
| T > 4 | Very soft | May lose discriminative info |

```python
def analyze_temperature(train_loaders, test_loaders, device):
    """Analyze effect of temperature on LwF performance."""
    temperatures = [1.0, 2.0, 3.0, 4.0, 5.0]
    results = []
    
    for T in temperatures:
        model = create_model().to(device)
        learner = LwFLearner(
            model=model,
            device=device,
            temperature=T,
            distill_lambda=1.0
        )
        
        result = learner.train_continual(
            train_loaders, test_loaders, epochs_per_task=5
        )
        
        metrics = ContinualLearningMetrics(result['accuracy_matrix'])
        results.append({
            'temperature': T,
            'avg_accuracy': metrics.average_accuracy,
            'backward_transfer': metrics.backward_transfer
        })
    
    return results
```

### Distillation Weight (λ)

The `distill_lambda` parameter balances new learning and knowledge preservation:

| λ Value | Effect |
|---------|--------|
| λ = 0 | No distillation (naive training) |
| λ = 0.5-1.0 | Balanced (typical choice) |
| λ > 2.0 | Strong preservation, may hinder new learning |

## Understanding LwF Behavior

### Why New Task Data Works

LwF uses new task data to compute distillation targets for old tasks. This works because:

1. **Shared representations**: Early layers often learn general features
2. **Output stability**: Model shouldn't dramatically change predictions on any input
3. **Regularization effect**: Distillation acts as a constraint on parameter changes

### Limitations

1. **Task similarity assumption**: Works best when tasks share representations
2. **Domain shift**: Performance degrades if new tasks are very different
3. **No explicit old task training**: Can't directly optimize for old task performance
4. **Catastrophic forgetting still possible**: For dissimilar task sequences

## Advanced LwF Variants

### LwF with Multi-Head Architecture

For task-incremental learning with separate output heads:

```python
class MultiHeadLwFLearner(LwFLearner):
    """
    LwF with task-specific output heads.
    
    Each task gets its own output layer, with shared feature extractor.
    Distillation is applied per-head.
    """
    
    def __init__(self, feature_extractor, num_classes_per_task, device, **kwargs):
        super().__init__(feature_extractor, device, **kwargs)
        
        self.feature_extractor = feature_extractor
        self.num_classes_per_task = num_classes_per_task
        self.task_heads = nn.ModuleList()
        
        # Get feature dimension
        with torch.no_grad():
            dummy = torch.randn(1, 1, 28, 28).to(device)
            feat_dim = feature_extractor(dummy).shape[1]
        
        self.feat_dim = feat_dim
    
    def add_task_head(self):
        """Add a new task-specific output head."""
        head = nn.Linear(self.feat_dim, self.num_classes_per_task)
        head = head.to(self.device)
        self.task_heads.append(head)
    
    def forward(self, x, task_id=None):
        """Forward pass with optional task specification."""
        features = self.feature_extractor(x)
        
        if task_id is not None:
            return self.task_heads[task_id](features)
        else:
            # Return all heads (for distillation)
            return [head(features) for head in self.task_heads]
```

### LwF with Feature Distillation

Distill intermediate features in addition to outputs:

```python
class FeatureLwF(LwFLearner):
    """
    LwF with intermediate feature distillation.
    
    Constrains not just outputs but also intermediate representations
    to be similar to the old model.
    """
    
    def __init__(self, model, device, feature_lambda=0.1, **kwargs):
        super().__init__(model, device, **kwargs)
        self.feature_lambda = feature_lambda
        
        # Register hooks to capture intermediate features
        self.features = {}
        self.old_features = {}
    
    def feature_distillation_loss(self, new_features, old_features):
        """Compute MSE loss between intermediate features."""
        loss = 0.0
        for name in new_features:
            if name in old_features:
                loss += F.mse_loss(new_features[name], old_features[name])
        return loss
```

## Comparison with Other Methods

### LwF vs EWC

| Aspect | LwF | EWC |
|--------|-----|-----|
| Storage | Old model copy | Fisher + optimal params |
| Memory | O(d) | O(T × d) |
| Computation | 2× forward pass | 1× forward + penalty |
| Data needed | None | Fisher from old data |
| Best for | Similar tasks | Any task sequence |

### LwF vs Experience Replay

| Aspect | LwF | Experience Replay |
|--------|-----|-------------------|
| Data storage | None | Examples |
| Privacy | High | Lower |
| Performance | Good | Often better |
| Domain shift | Sensitive | More robust |

## Expected Results

On Split MNIST benchmark:

| Method | Avg Accuracy | BWT |
|--------|--------------|-----|
| Naive | ~55% | -45% |
| LwF | ~80% | -18% |
| EWC | ~85% | -12% |
| ER | ~92% | -6% |

LwF provides meaningful improvement over naive training while being privacy-preserving.

## Practical Considerations

### When to Use LwF

✓ **Good for:**
- Privacy-sensitive applications
- Tasks with shared visual/semantic features
- Moderate number of tasks
- Memory-constrained settings

✗ **Avoid when:**
- Tasks are very dissimilar
- Maximum performance is required
- Long task sequences expected

### Implementation Tips

1. **Temperature tuning**: Start with T=2, increase if outputs too peaked
2. **Lambda selection**: Use validation to tune distill_lambda
3. **Learning rate**: May need lower rate than standard training
4. **Model architecture**: Works better with shared representations

## Summary

Learning Without Forgetting offers a privacy-preserving approach to continual learning:

- **Mechanism**: Knowledge distillation using current task data
- **Advantage**: No storage of previous examples
- **Limitation**: Assumes tasks share representations
- **Best use**: Privacy-sensitive applications with related tasks

## References

1. Li, Z., & Hoiem, D. (2017). Learning without forgetting. *IEEE TPAMI*.

2. Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network. *NeurIPS Workshop*.

3. Dhar, P., et al. (2019). Learning without memorizing. *CVPR*.

4. Jung, H., et al. (2016). Less-forgetting learning in deep neural networks. *AAAI*.
