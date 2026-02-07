# Benchmark Comparison

## Introduction

This section provides a comprehensive comparison of continual learning methods across standard benchmarks. Understanding how methods perform under different conditions is essential for selecting the appropriate approach for a given application.

## Standard Benchmarks

### Split MNIST

The most common introductory benchmark:

- **Dataset**: MNIST (60K train, 10K test)
- **Tasks**: 5 binary classification tasks
- **Classes per task**: 2 (digits 0-1, 2-3, 4-5, 6-7, 8-9)
- **Scenario**: Task-incremental learning

```python
def create_split_mnist(num_tasks=5):
    """Create Split MNIST benchmark."""
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
    
    # Create task splits
    classes_per_task = 10 // num_tasks
    task_classes = [
        list(range(i * classes_per_task, (i + 1) * classes_per_task))
        for i in range(num_tasks)
    ]
    
    return train_data, test_data, task_classes
```

### Permuted MNIST

Tests adaptation to input distribution shifts:

- **Dataset**: MNIST
- **Tasks**: N tasks with different pixel permutations
- **Scenario**: Domain-incremental learning
- **Challenge**: Same task, different input space

```python
def create_permuted_mnist(num_tasks=10, seed=42):
    """Create Permuted MNIST benchmark."""
    np.random.seed(seed)
    
    permutations = [np.random.permutation(784) for _ in range(num_tasks)]
    
    class PermutedMNIST(torch.utils.data.Dataset):
        def __init__(self, base_dataset, permutation):
            self.base = base_dataset
            self.perm = permutation
        
        def __len__(self):
            return len(self.base)
        
        def __getitem__(self, idx):
            x, y = self.base[idx]
            x = x.view(-1)[self.perm].view(1, 28, 28)
            return x, y
    
    return permutations, PermutedMNIST
```

### Split CIFAR-10/100

More challenging visual recognition:

- **CIFAR-10**: 5 tasks × 2 classes
- **CIFAR-100**: 10 tasks × 10 classes or 20 tasks × 5 classes
- **Challenge**: More complex visual features

### Sequential Mini-ImageNet

Large-scale benchmark:

- **Dataset**: Mini-ImageNet (100 classes, 600 images each)
- **Tasks**: 20 tasks × 5 classes
- **Challenge**: High-resolution images, more classes

## Comprehensive Comparison Framework

```python
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Callable
import time


@dataclass
class ExperimentResult:
    """Store results from a continual learning experiment."""
    method_name: str
    accuracy_matrix: np.ndarray
    average_accuracy: float
    backward_transfer: float
    learning_accuracy: float
    forgetting_measure: float
    training_time: float
    memory_usage_bytes: int
    
    def __repr__(self):
        return (f"{self.method_name}: AA={self.average_accuracy:.1f}%, "
                f"BWT={self.backward_transfer:+.1f}%")


class BenchmarkRunner:
    """
    Framework for running comprehensive continual learning benchmarks.
    
    Ensures fair comparison across methods with:
    - Same random seeds
    - Same model architecture
    - Same data splits
    - Comprehensive metrics
    """
    
    def __init__(self, 
                 train_loaders: List,
                 test_loaders: List,
                 model_factory: Callable,
                 device: torch.device,
                 num_runs: int = 3,
                 epochs_per_task: int = 5):
        """
        Initialize benchmark runner.
        
        Args:
            train_loaders: Training data loaders
            test_loaders: Test data loaders
            model_factory: Function that creates fresh model
            device: Computation device
            num_runs: Number of runs for confidence intervals
            epochs_per_task: Training epochs per task
        """
        self.train_loaders = train_loaders
        self.test_loaders = test_loaders
        self.model_factory = model_factory
        self.device = device
        self.num_runs = num_runs
        self.epochs_per_task = epochs_per_task
        self.num_tasks = len(train_loaders)
    
    def run_method(self, 
                   learner_class,
                   learner_kwargs: dict,
                   method_name: str) -> List[ExperimentResult]:
        """
        Run a single method multiple times.
        
        Args:
            learner_class: Continual learning class
            learner_kwargs: Arguments for learner
            method_name: Name for reporting
        
        Returns:
            List of results from each run
        """
        results = []
        
        for run in range(self.num_runs):
            # Set seeds for reproducibility
            torch.manual_seed(run * 42)
            np.random.seed(run * 42)
            
            # Create fresh model
            model = self.model_factory().to(self.device)
            
            # Create learner
            learner = learner_class(
                model=model,
                device=self.device,
                **learner_kwargs
            )
            
            # Measure time
            start_time = time.time()
            
            # Run continual learning
            result = learner.train_continual(
                self.train_loaders,
                self.test_loaders,
                epochs_per_task=self.epochs_per_task
            )
            
            training_time = time.time() - start_time
            
            # Compute metrics
            acc_matrix = result['accuracy_matrix']
            metrics = self._compute_metrics(acc_matrix)
            
            # Memory usage (approximate)
            memory_usage = self._estimate_memory(learner)
            
            results.append(ExperimentResult(
                method_name=method_name,
                accuracy_matrix=acc_matrix,
                average_accuracy=metrics['average_accuracy'],
                backward_transfer=metrics['backward_transfer'],
                learning_accuracy=metrics['learning_accuracy'],
                forgetting_measure=metrics['forgetting_measure'],
                training_time=training_time,
                memory_usage_bytes=memory_usage
            ))
        
        return results
    
    def _compute_metrics(self, accuracy_matrix: np.ndarray) -> dict:
        """Compute all standard metrics."""
        T = accuracy_matrix.shape[0]
        
        # Average accuracy
        avg_acc = np.mean(accuracy_matrix[:, -1])
        
        # Learning accuracy
        learn_acc = np.mean(np.diag(accuracy_matrix))
        
        # Backward transfer
        if T > 1:
            bwt = sum(accuracy_matrix[i, -1] - accuracy_matrix[i, i]
                     for i in range(T - 1)) / (T - 1)
        else:
            bwt = 0.0
        
        # Forgetting measure
        if T > 1:
            fm = sum(np.max(accuracy_matrix[i, i:]) - accuracy_matrix[i, -1]
                    for i in range(T - 1)) / (T - 1)
        else:
            fm = 0.0
        
        return {
            'average_accuracy': avg_acc,
            'learning_accuracy': learn_acc,
            'backward_transfer': bwt,
            'forgetting_measure': fm
        }
    
    def _estimate_memory(self, learner) -> int:
        """Estimate additional memory used by learner."""
        import sys
        
        memory = 0
        
        # Check for stored examples (replay methods)
        if hasattr(learner, 'memory_data'):
            memory += sum(sys.getsizeof(x.storage()) for x in learner.memory_data)
        
        # Check for Fisher matrices (EWC)
        if hasattr(learner, 'fisher_dict'):
            for task_fisher in learner.fisher_dict.values():
                memory += sum(f.numel() * 4 for f in task_fisher.values())
        
        # Check for old model (LwF)
        if hasattr(learner, 'old_model') and learner.old_model is not None:
            memory += sum(p.numel() * 4 for p in learner.old_model.parameters())
        
        return memory
    
    def run_all(self, methods: Dict[str, tuple]) -> Dict[str, List[ExperimentResult]]:
        """
        Run all specified methods.
        
        Args:
            methods: Dict mapping method name to (learner_class, kwargs)
        
        Returns:
            Dict mapping method name to results
        """
        all_results = {}
        
        for name, (learner_class, kwargs) in methods.items():
            print(f"\n{'='*60}")
            print(f"Running: {name}")
            print('='*60)
            
            results = self.run_method(learner_class, kwargs, name)
            all_results[name] = results
            
            # Print summary
            avg_aa = np.mean([r.average_accuracy for r in results])
            std_aa = np.std([r.average_accuracy for r in results])
            avg_bwt = np.mean([r.backward_transfer for r in results])
            
            print(f"\n{name} Summary ({self.num_runs} runs):")
            print(f"  AA: {avg_aa:.1f}% ± {std_aa:.1f}%")
            print(f"  BWT: {avg_bwt:+.1f}%")
        
        return all_results
    
    def generate_report(self, 
                        all_results: Dict[str, List[ExperimentResult]],
                        save_path: str = None):
        """Generate comprehensive comparison report."""
        
        print("\n" + "="*80)
        print("COMPREHENSIVE BENCHMARK REPORT")
        print("="*80)
        
        # Aggregate results
        summary = []
        for name, results in all_results.items():
            summary.append({
                'Method': name,
                'AA (%)': f"{np.mean([r.average_accuracy for r in results]):.1f} ± {np.std([r.average_accuracy for r in results]):.1f}",
                'BWT (%)': f"{np.mean([r.backward_transfer for r in results]):+.1f}",
                'LA (%)': f"{np.mean([r.learning_accuracy for r in results]):.1f}",
                'FM (%)': f"{np.mean([r.forgetting_measure for r in results]):.1f}",
                'Time (s)': f"{np.mean([r.training_time for r in results]):.1f}",
                'Memory (MB)': f"{np.mean([r.memory_usage_bytes for r in results]) / 1e6:.2f}"
            })
        
        # Print table
        print(f"\n{'Method':<20} {'AA (%)':<15} {'BWT (%)':<12} {'LA (%)':<12} {'FM (%)':<12} {'Time (s)':<12} {'Memory':<12}")
        print("-"*95)
        
        for row in summary:
            print(f"{row['Method']:<20} {row['AA (%)']:<15} {row['BWT (%)']:<12} "
                  f"{row['LA (%)']:<12} {row['FM (%)']:<12} {row['Time (s)']:<12} {row['Memory (MB)']:<12}")
        
        # Visualizations
        self._plot_comparison(all_results, save_path)
        
        return summary
    
    def _plot_comparison(self, all_results, save_path):
        """Generate comparison plots."""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        methods = list(all_results.keys())
        colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
        
        # 1. Average Accuracy comparison
        ax1 = axes[0, 0]
        means = [np.mean([r.average_accuracy for r in all_results[m]]) 
                for m in methods]
        stds = [np.std([r.average_accuracy for r in all_results[m]]) 
               for m in methods]
        
        bars = ax1.bar(range(len(methods)), means, yerr=stds, 
                       color=colors, alpha=0.8, capsize=5)
        ax1.set_ylabel('Average Accuracy (%)')
        ax1.set_title('Average Accuracy Comparison', fontweight='bold')
        ax1.set_xticks(range(len(methods)))
        ax1.set_xticklabels(methods, rotation=15, ha='right')
        ax1.set_ylim([0, 105])
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 2. Backward Transfer comparison
        ax2 = axes[0, 1]
        bwts = [np.mean([r.backward_transfer for r in all_results[m]]) 
               for m in methods]
        
        colors_bwt = ['green' if b >= 0 else 'red' for b in bwts]
        bars = ax2.bar(range(len(methods)), bwts, color=colors_bwt, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax2.set_ylabel('Backward Transfer (%)')
        ax2.set_title('Backward Transfer (Forgetting)', fontweight='bold')
        ax2.set_xticks(range(len(methods)))
        ax2.set_xticklabels(methods, rotation=15, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Training Time
        ax3 = axes[1, 0]
        times = [np.mean([r.training_time for r in all_results[m]]) 
                for m in methods]
        
        ax3.bar(range(len(methods)), times, color=colors, alpha=0.8)
        ax3.set_ylabel('Training Time (seconds)')
        ax3.set_title('Computational Efficiency', fontweight='bold')
        ax3.set_xticks(range(len(methods)))
        ax3.set_xticklabels(methods, rotation=15, ha='right')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Accuracy trajectories (first run of each method)
        ax4 = axes[1, 1]
        
        for i, (method, results) in enumerate(all_results.items()):
            # Use first run's accuracy matrix
            acc_matrix = results[0].accuracy_matrix
            
            # Plot Task 0's trajectory
            trajectory = [acc_matrix[0, j] if j >= 0 else np.nan 
                         for j in range(self.num_tasks)]
            ax4.plot(range(self.num_tasks), trajectory, 
                    marker='o', color=colors[i], linewidth=2,
                    label=method, markersize=6)
        
        ax4.set_xlabel('Training Stage')
        ax4.set_ylabel('Task 0 Accuracy (%)')
        ax4.set_title('Task 0 Accuracy Over Time', fontweight='bold')
        ax4.legend(loc='best', fontsize=9)
        ax4.set_ylim([0, 105])
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nPlot saved to {save_path}")
        
        plt.show()
```

## Running the Comparison

```python
def run_comprehensive_comparison():
    """Run full benchmark comparison."""
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create data
    train_data, test_data, task_classes = create_split_mnist(num_tasks=5)
    
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
    
    # Model factory
    def create_model():
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )
    
    # Create benchmark runner
    runner = BenchmarkRunner(
        train_loaders=train_loaders,
        test_loaders=test_loaders,
        model_factory=create_model,
        device=device,
        num_runs=3,
        epochs_per_task=5
    )
    
    # Define methods to compare
    methods = {
        'Naive': (NaiveLearner, {'learning_rate': 0.001}),
        'EWC (λ=5000)': (EWCLearner, {'ewc_lambda': 5000, 'learning_rate': 0.001}),
        'ER (1000)': (ExperienceReplayLearner, {'memory_size': 1000, 'examples_per_task': 200}),
        'LwF (T=2)': (LwFLearner, {'temperature': 2.0, 'distill_lambda': 1.0}),
    }
    
    # Run all methods
    all_results = runner.run_all(methods)
    
    # Generate report
    summary = runner.generate_report(
        all_results, 
        save_path='continual_learning_comparison.png'
    )
    
    return summary, all_results


if __name__ == "__main__":
    summary, results = run_comprehensive_comparison()
```

## Expected Results Summary

### Split MNIST (5 Tasks)

| Method | AA (%) | BWT (%) | LA (%) | Time (s) |
|--------|--------|---------|--------|----------|
| Naive | 54.2 ± 1.3 | -44.8 | 98.1 | 45 |
| EWC (λ=5000) | 84.7 ± 2.1 | -12.3 | 96.2 | 82 |
| ER (1000) | 91.8 ± 1.5 | -5.8 | 97.4 | 58 |
| LwF (T=2) | 79.3 ± 2.8 | -18.1 | 96.8 | 72 |
| ER + EWC | 93.2 ± 1.2 | -3.9 | 97.1 | 95 |

### Key Findings

1. **Experience Replay** consistently achieves highest accuracy
2. **EWC** provides good protection without storing examples
3. **LwF** offers privacy preservation but weaker protection
4. **Hybrid methods** combine benefits at increased cost

## Method Selection Guide

| Requirement | Recommended Method |
|-------------|-------------------|
| Maximum accuracy | Experience Replay + EWC |
| No data storage | EWC or LwF |
| Privacy-critical | LwF |
| Limited computation | Naive with early stopping |
| Many tasks | Online EWC |
| Similar task domains | LwF |
| Dissimilar tasks | EWC or ER |

## Summary

This comprehensive comparison framework enables:

- **Fair evaluation** across methods
- **Statistical confidence** through multiple runs
- **Multi-dimensional analysis** of performance, memory, and computation
- **Informed method selection** based on application requirements

The benchmarking code can be extended to additional datasets and methods for thorough evaluation of continual learning approaches.

## References

1. Hsu, Y. C., et al. (2018). Re-evaluating continual learning scenarios: A categorization and case for strong baselines.

2. van de Ven, G. M., & Tolias, A. S. (2019). Three scenarios for continual learning.

3. Delange, M., et al. (2021). A continual learning survey: Defying forgetting in classification tasks.

4. Mai, Z., et al. (2022). Online continual learning in image classification: An empirical survey.
