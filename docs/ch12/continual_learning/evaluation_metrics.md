# Evaluation Metrics for Continual Learning

## Introduction

Proper evaluation is critical in continual learning research. Unlike standard machine learning where a single accuracy metric often suffices, continual learning requires multiple metrics to capture different aspects of performance: the ability to learn new tasks, the tendency to forget old ones, and the transfer of knowledge between tasks.

This section presents the standard evaluation protocol and metrics used in continual learning, along with their mathematical definitions and PyTorch implementations.

## Evaluation Protocol

### The Accuracy Matrix

The foundation of continual learning evaluation is the **accuracy matrix** $A \in \mathbb{R}^{T \times T}$, where:

$$
A_{i,j} = \text{Accuracy on task } i \text{ after training on task } j
$$

Key properties:

- **Lower triangular**: Only entries where $j \geq i$ are defined (can't evaluate before learning)
- **Diagonal**: $A_{i,i}$ is the accuracy immediately after learning task $i$
- **Last column**: $A_{i,T}$ is the final accuracy on task $i$

Example accuracy matrix for 5 tasks:

```
        After Training Task
        T0      T1      T2      T3      T4
T0    [98.5%   62.3%   55.1%   51.8%   49.2%]
T1    [  -     97.8%   58.6%   52.4%   50.1%]
T2    [  -       -     98.2%   61.3%   53.7%]
T3    [  -       -       -     97.5%   58.9%]
T4    [  -       -       -       -     98.1%]
```

### Evaluation Procedure

```python
def evaluate_continual_learning(model, test_loaders, train_loaders, 
                                train_fn, epochs_per_task, device):
    """
    Standard evaluation procedure for continual learning.
    
    Args:
        model: Neural network model
        test_loaders: List of test DataLoaders for each task
        train_loaders: List of train DataLoaders for each task
        train_fn: Function to train on a single task
        epochs_per_task: Training epochs per task
        device: Computation device
    
    Returns:
        accuracy_matrix: T x T matrix of accuracies
    """
    num_tasks = len(train_loaders)
    accuracy_matrix = np.zeros((num_tasks, num_tasks))
    
    for task_id in range(num_tasks):
        # Train on current task
        train_fn(model, train_loaders[task_id], epochs_per_task, device)
        
        # Evaluate on ALL tasks seen so far
        for eval_id in range(task_id + 1):
            accuracy_matrix[eval_id, task_id] = evaluate_single_task(
                model, test_loaders[eval_id], device
            )
    
    return accuracy_matrix


def evaluate_single_task(model, test_loader, device):
    """Compute accuracy on a single task."""
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
```

## Primary Metrics

### Average Accuracy (AA)

The most fundamental metric—mean accuracy across all tasks at the end of training:

$$
\text{AA} = \frac{1}{T} \sum_{i=1}^{T} A_{i,T}
$$

**Interpretation**: Overall performance across all tasks. Higher is better.

**Limitations**: Doesn't distinguish between forgetting and inability to learn.

```python
def average_accuracy(accuracy_matrix):
    """
    Compute average accuracy across all tasks.
    
    Args:
        accuracy_matrix: T x T accuracy matrix
    
    Returns:
        Average final accuracy (percentage)
    """
    return np.mean(accuracy_matrix[:, -1])
```

### Backward Transfer (BWT)

Measures how much learning new tasks affects performance on old tasks:

$$
\text{BWT} = \frac{1}{T-1} \sum_{i=1}^{T-1} (A_{i,T} - A_{i,i})
$$

**Interpretation**:

- **Negative BWT**: Forgetting occurred (common case)
- **Zero BWT**: No forgetting
- **Positive BWT**: Learning new tasks improved old task performance (positive backward transfer)

```python
def backward_transfer(accuracy_matrix):
    """
    Compute backward transfer (negative = forgetting).
    
    Args:
        accuracy_matrix: T x T accuracy matrix
    
    Returns:
        Average backward transfer (percentage)
    """
    T = accuracy_matrix.shape[0]
    if T <= 1:
        return 0.0
    
    bwt = 0.0
    for i in range(T - 1):
        bwt += accuracy_matrix[i, -1] - accuracy_matrix[i, i]
    
    return bwt / (T - 1)
```

### Forward Transfer (FWT)

Measures how much previous learning helps with new tasks:

$$
\text{FWT} = \frac{1}{T-1} \sum_{i=2}^{T} (A_{i,i} - A_i^{\text{rand}})
$$

where $A_i^{\text{rand}}$ is the accuracy of a randomly initialized model on task $i$.

**Interpretation**:

- **Positive FWT**: Previous learning helps new tasks
- **Zero FWT**: No transfer
- **Negative FWT**: Previous learning hurts new task learning

```python
def forward_transfer(accuracy_matrix, random_init_accuracies):
    """
    Compute forward transfer.
    
    Args:
        accuracy_matrix: T x T accuracy matrix
        random_init_accuracies: Accuracies of random model on each task
    
    Returns:
        Average forward transfer (percentage)
    """
    T = accuracy_matrix.shape[0]
    if T <= 1:
        return 0.0
    
    fwt = 0.0
    for i in range(1, T):
        fwt += accuracy_matrix[i, i] - random_init_accuracies[i]
    
    return fwt / (T - 1)
```

### Learning Accuracy (LA)

Measures the model's ability to learn each task when first encountered:

$$
\text{LA} = \frac{1}{T} \sum_{i=1}^{T} A_{i,i}
$$

**Interpretation**: Average accuracy immediately after learning. Should be high for a good learner.

```python
def learning_accuracy(accuracy_matrix):
    """
    Compute average learning accuracy (diagonal mean).
    
    Args:
        accuracy_matrix: T x T accuracy matrix
    
    Returns:
        Average learning accuracy (percentage)
    """
    return np.mean(np.diag(accuracy_matrix))
```

## Secondary Metrics

### Forgetting Measure (FM)

Maximum forgetting observed for each task:

$$
F_i = \max_{j \in \{i, \ldots, T-1\}} (A_{i,j} - A_{i,T})
$$

$$
\text{FM} = \frac{1}{T-1} \sum_{i=1}^{T-1} F_i
$$

**Note**: FM uses the maximum drop rather than the final drop, accounting for non-monotonic forgetting patterns.

```python
def forgetting_measure(accuracy_matrix):
    """
    Compute forgetting measure (max forgetting per task).
    
    Args:
        accuracy_matrix: T x T accuracy matrix
    
    Returns:
        Average maximum forgetting (percentage)
    """
    T = accuracy_matrix.shape[0]
    if T <= 1:
        return 0.0
    
    forgetting = 0.0
    for i in range(T - 1):
        # Max accuracy achieved at any point
        max_acc = np.max(accuracy_matrix[i, i:])
        # Final accuracy
        final_acc = accuracy_matrix[i, -1]
        forgetting += max_acc - final_acc
    
    return forgetting / (T - 1)
```

### Intransigence Measure (IM)

Measures inability to learn new tasks compared to training from scratch:

$$
\text{IM} = \frac{1}{T-1} \sum_{i=2}^{T} (A_i^{\text{joint}} - A_{i,i})
$$

where $A_i^{\text{joint}}$ is the accuracy when training on task $i$ from scratch (or jointly with other tasks).

**Interpretation**: How much does prior learning hinder new task acquisition?

### Memory Stability (MS)

Variance in performance on old tasks over time:

$$
\text{MS}_i = \text{Var}(A_{i,i}, A_{i,i+1}, \ldots, A_{i,T})
$$

$$
\text{MS} = \frac{1}{T-1} \sum_{i=1}^{T-1} \text{MS}_i
$$

Lower variance indicates more stable performance.

```python
def memory_stability(accuracy_matrix):
    """
    Compute memory stability (lower variance = more stable).
    
    Args:
        accuracy_matrix: T x T accuracy matrix
    
    Returns:
        Average variance in task performance
    """
    T = accuracy_matrix.shape[0]
    if T <= 1:
        return 0.0
    
    stability = 0.0
    for i in range(T - 1):
        # Variance of task i's accuracy over time
        task_accs = accuracy_matrix[i, i:]
        stability += np.var(task_accs)
    
    return stability / (T - 1)
```

## Comprehensive Evaluation Class

```python
class ContinualLearningMetrics:
    """
    Comprehensive metrics computation for continual learning.
    
    This class computes and stores all standard metrics from an accuracy matrix.
    """
    
    def __init__(self, accuracy_matrix, random_init_accuracies=None):
        """
        Initialize with accuracy matrix.
        
        Args:
            accuracy_matrix: T x T numpy array of accuracies
            random_init_accuracies: Optional baseline accuracies
        """
        self.accuracy_matrix = accuracy_matrix
        self.T = accuracy_matrix.shape[0]
        self.random_init = random_init_accuracies
        
        # Compute all metrics
        self._compute_metrics()
    
    def _compute_metrics(self):
        """Compute all metrics."""
        # Primary metrics
        self.average_accuracy = np.mean(self.accuracy_matrix[:, -1])
        self.learning_accuracy = np.mean(np.diag(self.accuracy_matrix))
        
        # Backward transfer
        if self.T > 1:
            bwt = sum(self.accuracy_matrix[i, -1] - self.accuracy_matrix[i, i] 
                     for i in range(self.T - 1))
            self.backward_transfer = bwt / (self.T - 1)
        else:
            self.backward_transfer = 0.0
        
        # Forward transfer (if baseline provided)
        if self.random_init is not None and self.T > 1:
            fwt = sum(self.accuracy_matrix[i, i] - self.random_init[i] 
                     for i in range(1, self.T))
            self.forward_transfer = fwt / (self.T - 1)
        else:
            self.forward_transfer = None
        
        # Forgetting measure
        if self.T > 1:
            fm = 0.0
            for i in range(self.T - 1):
                max_acc = np.max(self.accuracy_matrix[i, i:])
                fm += max_acc - self.accuracy_matrix[i, -1]
            self.forgetting_measure = fm / (self.T - 1)
        else:
            self.forgetting_measure = 0.0
        
        # Per-task forgetting
        self.per_task_forgetting = []
        for i in range(self.T - 1):
            self.per_task_forgetting.append(
                self.accuracy_matrix[i, i] - self.accuracy_matrix[i, -1]
            )
        
        # Memory stability
        if self.T > 1:
            stability = sum(np.var(self.accuracy_matrix[i, i:]) 
                           for i in range(self.T - 1))
            self.memory_stability = stability / (self.T - 1)
        else:
            self.memory_stability = 0.0
    
    def summary(self):
        """Return summary dictionary."""
        return {
            'average_accuracy': self.average_accuracy,
            'learning_accuracy': self.learning_accuracy,
            'backward_transfer': self.backward_transfer,
            'forward_transfer': self.forward_transfer,
            'forgetting_measure': self.forgetting_measure,
            'memory_stability': self.memory_stability,
            'per_task_forgetting': self.per_task_forgetting
        }
    
    def print_report(self):
        """Print formatted metrics report."""
        print("=" * 60)
        print("CONTINUAL LEARNING METRICS REPORT")
        print("=" * 60)
        
        print(f"\n📊 Primary Metrics:")
        print(f"   Average Accuracy (AA):      {self.average_accuracy:.2f}%")
        print(f"   Learning Accuracy (LA):     {self.learning_accuracy:.2f}%")
        print(f"   Backward Transfer (BWT):    {self.backward_transfer:+.2f}%")
        if self.forward_transfer is not None:
            print(f"   Forward Transfer (FWT):     {self.forward_transfer:+.2f}%")
        
        print(f"\n📉 Forgetting Analysis:")
        print(f"   Forgetting Measure (FM):    {self.forgetting_measure:.2f}%")
        print(f"   Memory Stability (MS):      {self.memory_stability:.2f}")
        
        print(f"\n📋 Per-Task Forgetting:")
        for i, f in enumerate(self.per_task_forgetting):
            print(f"   Task {i}: {f:+.2f}%")
        
        print("=" * 60)
    
    def plot_metrics(self, save_path=None):
        """Generate visualization of metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Accuracy Matrix Heatmap
        ax1 = axes[0, 0]
        im = ax1.imshow(self.accuracy_matrix, cmap='RdYlGn', 
                        vmin=0, vmax=100, aspect='auto')
        ax1.set_xlabel('After Training Task')
        ax1.set_ylabel('Evaluated Task')
        ax1.set_title('Accuracy Matrix', fontweight='bold')
        ax1.set_xticks(range(self.T))
        ax1.set_yticks(range(self.T))
        plt.colorbar(im, ax=ax1, label='Accuracy (%)')
        
        # Add text annotations
        for i in range(self.T):
            for j in range(self.T):
                if j >= i:
                    ax1.text(j, i, f'{self.accuracy_matrix[i,j]:.0f}',
                            ha='center', va='center', fontsize=9)
        
        # 2. Learning vs Final Accuracy
        ax2 = axes[0, 1]
        x = np.arange(self.T)
        width = 0.35
        learning = np.diag(self.accuracy_matrix)
        final = self.accuracy_matrix[:, -1]
        
        ax2.bar(x - width/2, learning, width, label='Learning Acc', 
                color='skyblue', alpha=0.8)
        ax2.bar(x + width/2, final, width, label='Final Acc',
                color='coral', alpha=0.8)
        ax2.set_xlabel('Task')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Learning vs Final Accuracy', fontweight='bold')
        ax2.legend()
        ax2.set_ylim([0, 105])
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Per-Task Forgetting
        ax3 = axes[1, 0]
        colors = ['red' if f > 0 else 'green' for f in self.per_task_forgetting]
        ax3.bar(range(len(self.per_task_forgetting)), 
                self.per_task_forgetting, color=colors, alpha=0.7)
        ax3.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax3.set_xlabel('Task')
        ax3.set_ylabel('Forgetting (%)')
        ax3.set_title('Per-Task Forgetting', fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Accuracy Trajectories
        ax4 = axes[1, 1]
        for i in range(self.T):
            accs = [self.accuracy_matrix[i, j] if j >= i else np.nan 
                   for j in range(self.T)]
            ax4.plot(range(self.T), accs, marker='o', linewidth=2,
                    label=f'Task {i}')
        ax4.set_xlabel('Training Stage')
        ax4.set_ylabel('Accuracy (%)')
        ax4.set_title('Accuracy Trajectories', fontweight='bold')
        ax4.legend(loc='best', fontsize=9)
        ax4.set_ylim([0, 105])
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
```

## Metric Relationships

The metrics are interconnected:

$$
\text{AA} \approx \text{LA} + \text{BWT}
$$

This approximation holds because:

- LA measures initial learning capability
- BWT measures subsequent change (usually negative)
- AA is the final result after both effects

!!! info "Interpreting Metric Combinations"
    | LA | BWT | AA | Interpretation |
    |----|-----|-----|----------------|
    | High | ~0 | High | Good continual learner |
    | High | Very negative | Low | Severe forgetting |
    | Low | ~0 | Low | Poor learning capability |
    | High | Positive | Very High | Beneficial transfer |

## Practical Considerations

### Multiple Runs

Always report mean ± standard deviation over multiple random seeds:

```python
def evaluate_with_confidence(run_experiment_fn, num_runs=5):
    """
    Run experiment multiple times and compute confidence intervals.
    
    Args:
        run_experiment_fn: Function that returns accuracy matrix
        num_runs: Number of independent runs
    
    Returns:
        Dictionary with mean and std for each metric
    """
    all_metrics = []
    
    for run in range(num_runs):
        torch.manual_seed(run * 42)
        np.random.seed(run * 42)
        
        accuracy_matrix = run_experiment_fn()
        metrics = ContinualLearningMetrics(accuracy_matrix)
        all_metrics.append(metrics.summary())
    
    # Aggregate results
    results = {}
    for key in all_metrics[0].keys():
        if key == 'per_task_forgetting':
            continue
        values = [m[key] for m in all_metrics if m[key] is not None]
        if values:
            results[f'{key}_mean'] = np.mean(values)
            results[f'{key}_std'] = np.std(values)
    
    return results
```

### Reporting Guidelines

Following community standards (Hsu et al., 2018):

1. **Always report**: AA, BWT, LA at minimum
2. **Include baselines**: Naive sequential learning, joint training upper bound
3. **Same architecture**: Compare methods with identical network architectures
4. **Same data splits**: Use consistent task configurations
5. **Computational cost**: Report training time and memory usage

## Comparison with Joint Training

The **joint training upper bound** trains on all tasks simultaneously:

```python
def joint_training_baseline(model, all_loaders, test_loaders, 
                            epochs, device):
    """
    Train jointly on all tasks (upper bound baseline).
    
    This represents the best achievable performance without
    continual learning constraints.
    """
    from torch.utils.data import ConcatDataset
    
    # Combine all training data
    combined_dataset = ConcatDataset([
        loader.dataset for loader in all_loaders
    ])
    combined_loader = DataLoader(
        combined_dataset, batch_size=128, shuffle=True
    )
    
    # Train jointly
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs * len(all_loaders)):  # Scale epochs
        for data, target in combined_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    
    # Evaluate on all tasks
    return [evaluate_single_task(model, loader, device) 
            for loader in test_loaders]
```

## Summary

| Metric | Formula | Measures | Good Value |
|--------|---------|----------|------------|
| AA | $\frac{1}{T}\sum_i A_{i,T}$ | Overall performance | High (>80%) |
| LA | $\frac{1}{T}\sum_i A_{i,i}$ | Learning capability | High (>90%) |
| BWT | $\frac{1}{T-1}\sum_i (A_{i,T} - A_{i,i})$ | Forgetting | ~0 or positive |
| FWT | $\frac{1}{T-1}\sum_i (A_{i,i} - A_i^{\text{rand}})$ | Forward transfer | Positive |
| FM | Max forgetting per task | Worst-case forgetting | Low |
| MS | Variance over time | Stability | Low |

Proper evaluation using these metrics enables fair comparison of continual learning methods and identification of their strengths and weaknesses across different dimensions of performance.

## References

1. Lopez-Paz, D., & Ranzato, M. (2017). Gradient episodic memory for continual learning. *NeurIPS*.

2. Chaudhry, A., et al. (2018). Riemannian walk for incremental learning. *ECCV*.

3. Hsu, Y. C., Liu, Y. C., Ramasamy, A., & Kira, Z. (2018). Re-evaluating continual learning scenarios: A categorization and case for strong baselines. *NeurIPS Workshop*.

4. Díaz-Rodríguez, N., et al. (2018). Don't forget, there is more than forgetting: New metrics for continual learning. *NeurIPS Workshop*.
