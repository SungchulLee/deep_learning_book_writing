# Experience Replay

## Introduction

**Experience Replay** is one of the most effective and intuitive approaches to continual learning. The core idea is simple: maintain a memory buffer of examples from previous tasks and replay them during training on new tasks. By mixing old and new examples, the model maintains competence on previous tasks while learning new ones.

!!! success "Key Advantages"
    - **Intuitive mechanism**: Directly addresses forgetting by rehearsing old examples
    - **Strong performance**: Often the most effective method in practice
    - **Flexibility**: Can be combined with other continual learning techniques
    - **Simplicity**: Straightforward to implement and understand

## Theoretical Foundation

### Why Replay Works

Consider the loss function during continual learning. Without replay, we minimize only the current task's loss:

$$
\mathcal{L}(\theta) = \mathcal{L}_\tau(\theta)
$$

With replay, we approximate joint training on all tasks:

$$
\mathcal{L}(\theta) = \mathcal{L}_\tau(\theta) + \sum_{k=1}^{\tau-1} \hat{\mathcal{L}}_k(\theta)
$$

where $\hat{\mathcal{L}}_k(\theta)$ is an estimate of the loss on task $k$ using memory samples.

### Connection to Joint Training

Experience replay can be viewed as an approximation to joint training. If we had unlimited memory and stored all previous examples, replay would be equivalent to training on all data simultaneously—which doesn't exhibit forgetting.

## Memory Buffer Strategies

### Reservoir Sampling

**Reservoir sampling** maintains a fixed-size buffer that uniformly represents all seen examples:

```python
import random
import torch
from typing import List, Tuple, Optional


class ReservoirBuffer:
    """
    Reservoir sampling buffer for experience replay.
    
    Maintains a uniform sample over all seen examples,
    regardless of arrival order.
    """
    
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.buffer: List[Tuple[torch.Tensor, int]] = []
        self.seen_count = 0
    
    def add(self, example: torch.Tensor, label: int):
        """Add example using reservoir sampling."""
        self.seen_count += 1
        
        if len(self.buffer) < self.max_size:
            self.buffer.append((example.clone(), label))
        else:
            prob = self.max_size / self.seen_count
            if random.random() < prob:
                idx = random.randint(0, self.max_size - 1)
                self.buffer[idx] = (example.clone(), label)
    
    def sample(self, batch_size: int) -> Tuple[Optional[torch.Tensor], 
                                                Optional[torch.Tensor]]:
        """Sample a batch from buffer."""
        if len(self.buffer) == 0:
            return None, None
        
        sample_size = min(batch_size, len(self.buffer))
        indices = random.sample(range(len(self.buffer)), sample_size)
        
        examples = torch.stack([self.buffer[i][0] for i in indices])
        labels = torch.tensor([self.buffer[i][1] for i in indices], 
                             dtype=torch.long)
        return examples, labels
```

### Class-Balanced Buffer

Maintains equal representation across classes:

```python
class ClassBalancedBuffer:
    """
    Class-balanced memory buffer.
    
    Ensures equal representation of all classes in memory,
    important for class-incremental learning.
    """
    
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.class_buffers: Dict[int, List[torch.Tensor]] = {}
    
    def add_task(self, examples: torch.Tensor, labels: torch.Tensor):
        """Add examples from a new task."""
        unique_classes = labels.unique().tolist()
        
        for cls in unique_classes:
            mask = labels == cls
            class_examples = examples[mask]
            self.class_buffers[cls] = [ex.clone() for ex in class_examples]
        
        self._rebalance()
    
    def _rebalance(self):
        """Rebalance buffer to maintain class equality."""
        num_classes = len(self.class_buffers)
        if num_classes == 0:
            return
        
        per_class = self.max_size // num_classes
        
        for cls in self.class_buffers:
            if len(self.class_buffers[cls]) > per_class:
                indices = random.sample(
                    range(len(self.class_buffers[cls])), per_class
                )
                self.class_buffers[cls] = [
                    self.class_buffers[cls][i] for i in indices
                ]
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample with class balance."""
        all_examples = []
        all_labels = []
        
        for cls, examples in self.class_buffers.items():
            for ex in examples:
                all_examples.append(ex)
                all_labels.append(cls)
        
        if len(all_examples) == 0:
            return None, None
        
        sample_size = min(batch_size, len(all_examples))
        indices = random.sample(range(len(all_examples)), sample_size)
        
        return (torch.stack([all_examples[i] for i in indices]),
                torch.tensor([all_labels[i] for i in indices], dtype=torch.long))
```

## Complete Experience Replay Implementation

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import List, Dict


class ExperienceReplayLearner:
    """
    Continual learner using experience replay.
    
    Strategy:
    1. Maintain a memory buffer of previous examples
    2. During training, mix current task data with replay samples
    3. Combined loss prevents forgetting while learning new tasks
    """
    
    def __init__(self,
                 model: nn.Module,
                 device: torch.device,
                 memory_size: int = 1000,
                 examples_per_task: int = 200,
                 replay_batch_ratio: float = 0.5,
                 learning_rate: float = 0.001):
        """
        Initialize replay learner.
        
        Args:
            model: Neural network model
            device: Computation device
            memory_size: Total memory buffer capacity
            examples_per_task: Examples to store per task
            replay_batch_ratio: Fraction of batch from replay
            learning_rate: Learning rate for optimizer
        """
        self.model = model
        self.device = device
        self.memory_size = memory_size
        self.examples_per_task = examples_per_task
        self.replay_batch_ratio = replay_batch_ratio
        self.learning_rate = learning_rate
        
        self.criterion = nn.CrossEntropyLoss()
        
        # Memory storage
        self.memory_data: List[torch.Tensor] = []
        self.memory_labels: List[int] = []
    
    def add_to_memory(self, data_loader: DataLoader):
        """
        Add examples from current task to memory.
        
        Uses random selection to choose representative examples.
        
        Args:
            data_loader: DataLoader for current task
        """
        # Collect all task data
        all_data, all_labels = [], []
        for data, labels in data_loader:
            all_data.append(data)
            all_labels.append(labels)
        
        all_data = torch.cat(all_data, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Random selection
        n_available = all_data.size(0)
        n_select = min(self.examples_per_task, n_available)
        indices = torch.randperm(n_available)[:n_select]
        
        # Add to memory
        for idx in indices:
            self.memory_data.append(all_data[idx].clone())
            self.memory_labels.append(all_labels[idx].item())
        
        # Trim if exceeds capacity
        if len(self.memory_data) > self.memory_size:
            self.memory_data = self.memory_data[-self.memory_size:]
            self.memory_labels = self.memory_labels[-self.memory_size:]
        
        print(f"  Memory size: {len(self.memory_data)} examples")
    
    def sample_memory(self, batch_size: int):
        """Sample a batch from memory."""
        if len(self.memory_data) == 0:
            return None, None
        
        sample_size = min(batch_size, len(self.memory_data))
        indices = random.sample(range(len(self.memory_data)), sample_size)
        
        examples = torch.stack([self.memory_data[i] for i in indices])
        labels = torch.tensor([self.memory_labels[i] for i in indices],
                             dtype=torch.long)
        return examples, labels
    
    def train_on_task(self, 
                      train_loader: DataLoader,
                      epochs: int = 5,
                      verbose: bool = True) -> Dict:
        """
        Train on a task with experience replay.
        
        Args:
            train_loader: Training data loader
            epochs: Number of training epochs
            verbose: Print progress
        
        Returns:
            Training statistics
        """
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        losses = {'total': [], 'current': [], 'replay': []}
        
        for epoch in range(epochs):
            self.model.train()
            epoch_losses = {'total': 0, 'current': 0, 'replay': 0}
            num_batches = 0
            
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                batch_size = data.size(0)
                
                optimizer.zero_grad()
                
                # Current task loss
                output = self.model(data)
                current_loss = self.criterion(output, target)
                
                # Replay loss
                replay_loss = torch.tensor(0.0, device=self.device)
                replay_size = int(batch_size * self.replay_batch_ratio)
                
                if replay_size > 0 and len(self.memory_data) > 0:
                    replay_data, replay_labels = self.sample_memory(replay_size)
                    replay_data = replay_data.to(self.device)
                    replay_labels = replay_labels.to(self.device)
                    
                    replay_output = self.model(replay_data)
                    replay_loss = self.criterion(replay_output, replay_labels)
                
                # Combined loss
                total_loss = current_loss + replay_loss
                
                # Backward and optimize
                total_loss.backward()
                optimizer.step()
                
                # Track losses
                epoch_losses['total'] += total_loss.item()
                epoch_losses['current'] += current_loss.item()
                epoch_losses['replay'] += replay_loss.item()
                num_batches += 1
            
            # Average losses
            for key in epoch_losses:
                losses[key].append(epoch_losses[key] / num_batches)
            
            if verbose:
                print(f"  Epoch {epoch+1}/{epochs}: "
                      f"Total={losses['total'][-1]:.4f}, "
                      f"Current={losses['current'][-1]:.4f}, "
                      f"Replay={losses['replay'][-1]:.4f}")
        
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
        Complete continual learning with replay.
        
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
            
            # Train on current task
            self.train_on_task(
                train_loaders[task_id],
                epochs=epochs_per_task
            )
            
            # Add to memory AFTER training
            self.add_to_memory(train_loaders[task_id])
            
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

### Memory Size Effect

| Memory Size | Effect | Trade-off |
|-------------|--------|-----------|
| Small (100) | Limited protection | Memory efficient |
| Medium (500-1000) | Good balance | Recommended |
| Large (5000+) | Near-joint training | Memory intensive |

### Examples Per Task

The `examples_per_task` parameter controls task balance:

```python
def analyze_memory_parameters(train_loaders, test_loaders, device):
    """Analyze effect of memory parameters."""
    
    configs = [
        {'memory_size': 500, 'examples_per_task': 100},
        {'memory_size': 1000, 'examples_per_task': 200},
        {'memory_size': 2000, 'examples_per_task': 400},
    ]
    
    results = []
    for config in configs:
        model = create_model().to(device)
        learner = ExperienceReplayLearner(
            model=model,
            device=device,
            **config
        )
        
        result = learner.train_continual(
            train_loaders, test_loaders, epochs_per_task=5
        )
        
        metrics = ContinualLearningMetrics(result['accuracy_matrix'])
        results.append({
            **config,
            'avg_accuracy': metrics.average_accuracy,
            'backward_transfer': metrics.backward_transfer
        })
    
    return results
```

## Advanced Replay Strategies

### Gradient-Based Selection

Select examples that maximize gradient diversity:

```python
class GradientBasedBuffer:
    """
    Select examples based on gradient information.
    
    Prioritizes examples with large or diverse gradients,
    as these are most informative for training.
    """
    
    def __init__(self, model, max_size, device):
        self.model = model
        self.max_size = max_size
        self.device = device
        self.buffer = []
        self.gradients = []
    
    def compute_gradient_score(self, example, label):
        """Compute gradient magnitude for an example."""
        self.model.zero_grad()
        example = example.unsqueeze(0).to(self.device)
        label = torch.tensor([label], device=self.device)
        
        output = self.model(example)
        loss = F.cross_entropy(output, label)
        loss.backward()
        
        # Sum of squared gradients
        score = sum(p.grad.norm().item() ** 2 
                   for p in self.model.parameters() 
                   if p.grad is not None)
        
        return score
    
    def add_task(self, examples, labels, num_select):
        """Add examples prioritized by gradient score."""
        scores = []
        for i in range(len(examples)):
            score = self.compute_gradient_score(examples[i], labels[i].item())
            scores.append((score, i))
        
        # Select top examples by gradient
        scores.sort(reverse=True)
        selected = [scores[i][1] for i in range(min(num_select, len(scores)))]
        
        for idx in selected:
            if len(self.buffer) < self.max_size:
                self.buffer.append((examples[idx].clone(), labels[idx].item()))
```

### Loss-Based Selection

Prioritize examples with high loss (hard examples):

```python
class LossBasedBuffer:
    """
    Select examples based on loss value.
    
    Hard examples (high loss) may be more valuable for 
    maintaining decision boundaries.
    """
    
    def select_by_loss(self, examples, labels, model, num_select, device):
        """Select examples with highest loss."""
        model.eval()
        
        with torch.no_grad():
            outputs = model(examples.to(device))
            losses = F.cross_entropy(outputs, labels.to(device), 
                                    reduction='none')
        
        # Select highest loss examples
        _, indices = torch.topk(losses, min(num_select, len(losses)))
        
        return examples[indices.cpu()], labels[indices.cpu()]
```

## Comparison with Other Methods

### Replay vs Regularization

| Aspect | Experience Replay | EWC |
|--------|------------------|-----|
| Memory | Stores examples | Stores Fisher + params |
| Privacy | Lower (stores data) | Higher (no data) |
| Performance | Generally better | Good, but limited |
| Scalability | Memory grows or fixed | Linear in tasks |
| Flexibility | Combines easily | Standalone |

### Expected Results

On Split MNIST benchmark:

| Method | Avg Accuracy | BWT | Memory |
|--------|--------------|-----|--------|
| Naive | ~55% | -45% | 0 |
| ER (1000) | ~92% | -6% | 1000 examples |
| EWC | ~85% | -12% | 2×params |

## Practical Considerations

### When to Use Experience Replay

✓ **Good for:**
- When memory storage is acceptable
- Privacy is not a primary concern
- Maximum performance is needed
- Tasks are similar in nature

✗ **Avoid when:**
- Strict privacy requirements
- Very limited memory
- Thousands of sequential tasks

### Implementation Tips

1. **Buffer updates**: Add to memory after training each task
2. **Sampling**: Random sampling usually sufficient
3. **Batch mixing**: 50/50 current/replay is a good default
4. **Data augmentation**: Apply same augmentation to replay samples

## Summary

Experience Replay is a powerful and practical approach to continual learning:

- **Mechanism**: Store and replay previous examples
- **Effectiveness**: Often the best-performing method
- **Trade-off**: Requires memory storage
- **Flexibility**: Easy to combine with other methods

The method's simplicity and strong performance make it a go-to choice when memory constraints are not prohibitive.

## References

1. Robins, A. (1995). Catastrophic forgetting, rehearsal and pseudorehearsal. *Connection Science*.

2. Rolnick, D., et al. (2019). Experience replay for continual learning. *NeurIPS*.

3. Chaudhry, A., et al. (2019). On tiny episodic memories in continual learning. *ICML Workshop*.

4. Buzzega, P., et al. (2020). Dark experience for general continual learning: a strong, simple baseline. *NeurIPS*.
