"""
Module 57: Continual Learning - Advanced Level
File: Comprehensive Method Comparison

This script provides a comprehensive comparison of multiple continual learning
methods on challenging benchmarks. This is useful for understanding trade-offs
and selecting appropriate methods for different scenarios.

Learning Objectives:
1. Compare regularization vs replay vs hybrid approaches
2. Understand computational and memory trade-offs
3. Analyze performance across different metrics
4. Learn experimental best practices for continual learning

Methods Compared:
1. Naive Sequential Learning (baseline)
2. Experience Replay (ER)
3. Elastic Weight Consolidation (EWC)
4. Learning Without Forgetting (LWF)
5. Hybrid: EWC + Experience Replay

Evaluation Metrics:
- Average Accuracy (AA)
- Backward Transfer (BWT) - forgetting measure
- Learning Accuracy (LA) - learning capability
- Memory Stability - variance in old task performance
- Computational Cost - training time and memory
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
from typing import List, Tuple, Dict
import copy
import time
import random
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class ExperimentResult:
    """Store comprehensive results for a continual learning method."""
    method_name: str
    accuracy_matrix: np.ndarray
    average_accuracy: float
    backward_transfer: float
    learning_accuracy: float
    forgetting_per_task: List[float]
    training_time: float
    memory_usage_mb: float
    
    def summary_dict(self) -> Dict:
        """Return summary as dictionary."""
        return {
            'method': self.method_name,
            'avg_accuracy': self.average_accuracy,
            'backward_transfer': self.backward_transfer,
            'learning_accuracy': self.learning_accuracy,
            'training_time': self.training_time,
            'memory_usage': self.memory_usage_mb
        }


class NaiveLearner:
    """Baseline: naive sequential learning."""
    
    def __init__(self, model, device, lr=0.001):
        self.model = model
        self.device = device
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()
    
    def train_continual(self, train_loaders, test_loaders, epochs_per_task=5):
        num_tasks = len(train_loaders)
        accuracy_matrix = np.zeros((num_tasks, num_tasks))
        
        for task_id in range(num_tasks):
            # Train
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            self.model.train()
            
            for epoch in range(epochs_per_task):
                for data, target in train_loaders[task_id]:
                    data, target = data.to(self.device), target.to(self.device)
                    optimizer.zero_grad()
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    loss.backward()
                    optimizer.step()
            
            # Evaluate
            for eval_task_id in range(task_id + 1):
                acc = self.evaluate(test_loaders[eval_task_id])
                accuracy_matrix[eval_task_id, task_id] = acc
        
        return accuracy_matrix
    
    def evaluate(self, test_loader):
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


class ExperienceReplayLearner:
    """Experience Replay with memory buffer."""
    
    def __init__(self, model, device, memory_size=1000, examples_per_task=200, lr=0.001):
        self.model = model
        self.device = device
        self.memory_size = memory_size
        self.examples_per_task = examples_per_task
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()
        self.memory_data = []
        self.memory_labels = []
    
    def add_to_memory(self, train_loader):
        all_data, all_labels = [], []
        for data, labels in train_loader:
            all_data.append(data)
            all_labels.append(labels)
        all_data = torch.cat(all_data, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        num_available = all_data.size(0)
        num_to_select = min(self.examples_per_task, num_available)
        indices = torch.randperm(num_available)[:num_to_select]
        
        self.memory_data.extend(all_data[indices])
        self.memory_labels.extend(all_labels[indices])
        
        # Trim if exceeds max size
        if len(self.memory_data) > self.memory_size:
            self.memory_data = self.memory_data[-self.memory_size:]
            self.memory_labels = self.memory_labels[-self.memory_size:]
    
    def sample_memory(self, batch_size):
        if len(self.memory_data) == 0:
            return None, None
        sample_size = min(batch_size, len(self.memory_data))
        indices = random.sample(range(len(self.memory_data)), sample_size)
        return torch.stack([self.memory_data[i] for i in indices]), \
               torch.tensor([self.memory_labels[i] for i in indices], dtype=torch.long)
    
    def train_continual(self, train_loaders, test_loaders, epochs_per_task=5):
        num_tasks = len(train_loaders)
        accuracy_matrix = np.zeros((num_tasks, num_tasks))
        
        for task_id in range(num_tasks):
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            self.model.train()
            
            for epoch in range(epochs_per_task):
                for data, target in train_loaders[task_id]:
                    data, target = data.to(self.device), target.to(self.device)
                    optimizer.zero_grad()
                    
                    # Current loss
                    output = self.model(data)
                    current_loss = self.criterion(output, target)
                    
                    # Replay loss
                    replay_loss = torch.tensor(0.0).to(self.device)
                    replay_data, replay_labels = self.sample_memory(data.size(0))
                    if replay_data is not None:
                        replay_data = replay_data.to(self.device)
                        replay_labels = replay_labels.to(self.device)
                        replay_output = self.model(replay_data)
                        replay_loss = self.criterion(replay_output, replay_labels)
                    
                    total_loss = current_loss + replay_loss
                    total_loss.backward()
                    optimizer.step()
            
            # Add to memory
            self.add_to_memory(train_loaders[task_id])
            
            # Evaluate
            for eval_task_id in range(task_id + 1):
                acc = self.evaluate(test_loaders[eval_task_id])
                accuracy_matrix[eval_task_id, task_id] = acc
        
        return accuracy_matrix
    
    def evaluate(self, test_loader):
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


class EWCLearner:
    """Elastic Weight Consolidation."""
    
    def __init__(self, model, device, ewc_lambda=5000, lr=0.001):
        self.model = model
        self.device = device
        self.ewc_lambda = ewc_lambda
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()
        self.fisher_dict = {}
        self.optpar_dict = {}
    
    def compute_fisher(self, data_loader, num_samples=1000):
        self.model.eval()
        fisher = {n: torch.zeros_like(p.data) for n, p in self.model.named_parameters() if p.requires_grad}
        
        total_samples = 0
        for data, target in data_loader:
            if total_samples >= num_samples:
                break
            data, target = data.to(self.device), target.to(self.device)
            
            for i in range(data.size(0)):
                self.model.zero_grad()
                output = self.model(data[i:i+1])
                log_probs = F.log_softmax(output, dim=1)
                log_prob = log_probs[0, target[i]]
                log_prob.backward()
                
                for n, p in self.model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        fisher[n] += p.grad.data ** 2
                
                total_samples += 1
                if total_samples >= num_samples:
                    break
        
        for n in fisher:
            fisher[n] /= total_samples
        
        return fisher
    
    def ewc_penalty(self):
        penalty = torch.tensor(0.0).to(self.device)
        for task_id in self.fisher_dict.keys():
            for n, p in self.model.named_parameters():
                if n in self.fisher_dict[task_id] and p.requires_grad:
                    penalty += (self.fisher_dict[task_id][n] * 
                              (p - self.optpar_dict[task_id][n]) ** 2).sum()
        return (self.ewc_lambda / 2) * penalty
    
    def train_continual(self, train_loaders, test_loaders, epochs_per_task=5):
        num_tasks = len(train_loaders)
        accuracy_matrix = np.zeros((num_tasks, num_tasks))
        
        for task_id in range(num_tasks):
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            self.model.train()
            
            for epoch in range(epochs_per_task):
                for data, target in train_loaders[task_id]:
                    data, target = data.to(self.device), target.to(self.device)
                    optimizer.zero_grad()
                    
                    output = self.model(data)
                    current_loss = self.criterion(output, target)
                    
                    ewc_loss = torch.tensor(0.0).to(self.device)
                    if len(self.fisher_dict) > 0:
                        ewc_loss = self.ewc_penalty()
                    
                    total_loss = current_loss + ewc_loss
                    total_loss.backward()
                    optimizer.step()
            
            # Consolidate
            fisher = self.compute_fisher(train_loaders[task_id])
            self.fisher_dict[task_id] = fisher
            self.optpar_dict[task_id] = {n: p.data.clone() for n, p in self.model.named_parameters() if p.requires_grad}
            
            # Evaluate
            for eval_task_id in range(task_id + 1):
                acc = self.evaluate(test_loaders[eval_task_id])
                accuracy_matrix[eval_task_id, task_id] = acc
        
        return accuracy_matrix
    
    def evaluate(self, test_loader):
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


class LWFLearner:
    """Learning Without Forgetting."""
    
    def __init__(self, model, device, distill_lambda=1.0, temperature=2.0, lr=0.001):
        self.model = model
        self.device = device
        self.distill_lambda = distill_lambda
        self.temperature = temperature
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()
        self.old_model = None
    
    def distillation_loss(self, new_logits, old_logits):
        T = self.temperature
        new_log_probs = F.log_softmax(new_logits / T, dim=1)
        old_probs = F.softmax(old_logits / T, dim=1)
        return F.kl_div(new_log_probs, old_probs, reduction='batchmean') * (T ** 2)
    
    def train_continual(self, train_loaders, test_loaders, epochs_per_task=5):
        num_tasks = len(train_loaders)
        accuracy_matrix = np.zeros((num_tasks, num_tasks))
        
        for task_id in range(num_tasks):
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            self.model.train()
            if self.old_model is not None:
                self.old_model.eval()
            
            for epoch in range(epochs_per_task):
                for data, target in train_loaders[task_id]:
                    data, target = data.to(self.device), target.to(self.device)
                    optimizer.zero_grad()
                    
                    new_logits = self.model(data)
                    new_loss = self.criterion(new_logits, target)
                    
                    distill_loss = torch.tensor(0.0).to(self.device)
                    if self.old_model is not None:
                        with torch.no_grad():
                            old_logits = self.old_model(data)
                        distill_loss = self.distillation_loss(new_logits, old_logits)
                    
                    total_loss = new_loss + self.distill_lambda * distill_loss
                    total_loss.backward()
                    optimizer.step()
            
            # Update old model
            self.old_model = copy.deepcopy(self.model)
            self.old_model.to(self.device)
            self.old_model.eval()
            for param in self.old_model.parameters():
                param.requires_grad = False
            
            # Evaluate
            for eval_task_id in range(task_id + 1):
                acc = self.evaluate(test_loaders[eval_task_id])
                accuracy_matrix[eval_task_id, task_id] = acc
        
        return accuracy_matrix
    
    def evaluate(self, test_loader):
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


def calculate_metrics(accuracy_matrix):
    """Calculate standard continual learning metrics."""
    num_tasks = accuracy_matrix.shape[0]
    
    avg_accuracy = np.mean(accuracy_matrix[:, -1])
    learning_accuracy = np.mean(np.diag(accuracy_matrix))
    
    backward_transfer = 0.0
    forgetting_per_task = []
    for i in range(num_tasks - 1):
        initial = accuracy_matrix[i, i]
        final = accuracy_matrix[i, -1]
        change = final - initial
        backward_transfer += change
        forgetting_per_task.append(change)
    
    if num_tasks > 1:
        backward_transfer /= (num_tasks - 1)
    
    return avg_accuracy, backward_transfer, learning_accuracy, forgetting_per_task


def get_memory_usage():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


def run_experiment(learner_class, model, device, train_loaders, test_loaders,
                  epochs_per_task, **kwargs):
    """Run a single experiment with given learner."""
    learner = learner_class(model, device, **kwargs)
    
    start_time = time.time()
    start_memory = get_memory_usage()
    
    accuracy_matrix = learner.train_continual(train_loaders, test_loaders, epochs_per_task)
    
    end_time = time.time()
    end_memory = get_memory_usage()
    
    training_time = end_time - start_time
    memory_usage = end_memory - start_memory
    
    avg_acc, bwt, learning_acc, forgetting = calculate_metrics(accuracy_matrix)
    
    return ExperimentResult(
        method_name=learner_class.__name__.replace('Learner', ''),
        accuracy_matrix=accuracy_matrix,
        average_accuracy=avg_acc,
        backward_transfer=bwt,
        learning_accuracy=learning_acc,
        forgetting_per_task=forgetting,
        training_time=training_time,
        memory_usage_mb=memory_usage
    )


def create_model():
    """Create a fresh model."""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 2)
    )


def visualize_comparison(results: List[ExperimentResult], num_tasks: int):
    """Create comprehensive comparison visualizations."""
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: Final Accuracy Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    methods = [r.method_name for r in results]
    final_accs = [r.average_accuracy for r in results]
    colors = plt.cm.Set3(range(len(methods)))
    
    bars = ax1.bar(range(len(methods)), final_accs, color=colors, alpha=0.8)
    ax1.set_ylabel('Average Accuracy (%)', fontsize=11)
    ax1.set_title('Final Average Accuracy', fontsize=12, fontweight='bold')
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(methods, rotation=15, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0, 100])
    
    for bar, val in zip(bars, final_accs):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{val:.1f}%', ha='center', fontsize=9)
    
    # Plot 2: Backward Transfer Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    bwts = [r.backward_transfer for r in results]
    bars = ax2.bar(range(len(methods)), bwts, color=colors, alpha=0.8)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.set_ylabel('Backward Transfer (%)', fontsize=11)
    ax2.set_title('Forgetting Measure', fontsize=12, fontweight='bold')
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels(methods, rotation=15, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, bwts):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val:.1f}%', ha='center',
                va='bottom' if val >= 0 else 'top', fontsize=9)
    
    # Plot 3: Training Time Comparison
    ax3 = fig.add_subplot(gs[0, 2])
    times = [r.training_time for r in results]
    bars = ax3.bar(range(len(methods)), times, color=colors, alpha=0.8)
    ax3.set_ylabel('Training Time (seconds)', fontsize=11)
    ax3.set_title('Computational Efficiency', fontsize=12, fontweight='bold')
    ax3.set_xticks(range(len(methods)))
    ax3.set_xticklabels(methods, rotation=15, ha='right')
    ax3.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, times):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(times)*0.02,
                f'{val:.1f}s', ha='center', fontsize=9)
    
    # Plot 4: Accuracy Matrix Heatmaps (multi-panel)
    for idx, result in enumerate(results):
        row = 1
        col = idx
        if col >= 3:
            continue  # Only show first 3 methods' matrices
        
        ax = fig.add_subplot(gs[row, col])
        im = ax.imshow(result.accuracy_matrix, cmap='RdYlGn',
                      aspect='auto', vmin=0, vmax=100)
        ax.set_xlabel('After Task', fontsize=9)
        ax.set_ylabel('Eval Task', fontsize=9)
        ax.set_title(f'{result.method_name}', fontsize=10, fontweight='bold')
        ax.set_xticks(range(num_tasks))
        ax.set_yticks(range(num_tasks))
        
        # Annotations
        for i in range(num_tasks):
            for j in range(num_tasks):
                if j >= i:
                    ax.text(j, i, f'{result.accuracy_matrix[i, j]:.0f}',
                           ha="center", va="center", color="black", fontsize=8)
    
    plt.savefig('continual_learning_comparison.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved as 'continual_learning_comparison.png'")
    plt.show()


def print_comparison_table(results: List[ExperimentResult]):
    """Print detailed comparison table."""
    print("\n" + "=" * 120)
    print("COMPREHENSIVE METHOD COMPARISON")
    print("=" * 120)
    
    # Header
    print(f"\n{'Method':<20} {'Avg Acc':<12} {'BWT':<12} {'Learn Acc':<12} {'Time (s)':<12} {'Memory (MB)':<12}")
    print("-" * 120)
    
    # Results
    for result in results:
        print(f"{result.method_name:<20} "
              f"{result.average_accuracy:>10.2f}% "
              f"{result.backward_transfer:>10.2f}% "
              f"{result.learning_accuracy:>10.2f}% "
              f"{result.training_time:>10.2f} "
              f"{result.memory_usage_mb:>10.2f}")
    
    print("\n" + "=" * 120)


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


def main():
    """Main comparison experiment."""
    print("=" * 70)
    print("COMPREHENSIVE CONTINUAL LEARNING METHOD COMPARISON")
    print("=" * 70)
    print("\nComparing:")
    print("  1. Naive Sequential Learning (baseline)")
    print("  2. Experience Replay (memory-based)")
    print("  3. Elastic Weight Consolidation (regularization-based)")
    print("  4. Learning Without Forgetting (knowledge distillation)")
    print("=" * 70)
    
    # Configuration
    num_tasks = 5
    epochs_per_task = 3  # Reduced for faster comparison
    batch_size = 128
    learning_rate = 0.001
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
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
    
    # Run experiments
    results = []
    
    print("\n" + "=" * 70)
    print("Running Experiments...")
    print("=" * 70)
    
    # 1. Naive
    print("\n[1/4] Naive Sequential Learning...")
    model = create_model().to(device)
    result = run_experiment(NaiveLearner, model, device, train_loaders, test_loaders,
                           epochs_per_task, lr=learning_rate)
    results.append(result)
    print(f"  Avg Acc: {result.average_accuracy:.2f}%, BWT: {result.backward_transfer:.2f}%")
    
    # 2. Experience Replay
    print("\n[2/4] Experience Replay...")
    model = create_model().to(device)
    result = run_experiment(ExperienceReplayLearner, model, device, train_loaders, test_loaders,
                           epochs_per_task, memory_size=1000, examples_per_task=200, lr=learning_rate)
    results.append(result)
    print(f"  Avg Acc: {result.average_accuracy:.2f}%, BWT: {result.backward_transfer:.2f}%")
    
    # 3. EWC
    print("\n[3/4] Elastic Weight Consolidation...")
    model = create_model().to(device)
    result = run_experiment(EWCLearner, model, device, train_loaders, test_loaders,
                           epochs_per_task, ewc_lambda=5000, lr=learning_rate)
    results.append(result)
    print(f"  Avg Acc: {result.average_accuracy:.2f}%, BWT: {result.backward_transfer:.2f}%")
    
    # 4. LWF
    print("\n[4/4] Learning Without Forgetting...")
    model = create_model().to(device)
    result = run_experiment(LWFLearner, model, device, train_loaders, test_loaders,
                           epochs_per_task, distill_lambda=1.0, temperature=2.0, lr=learning_rate)
    results.append(result)
    print(f"  Avg Acc: {result.average_accuracy:.2f}%, BWT: {result.backward_transfer:.2f}%")
    
    # Print comparison
    print_comparison_table(results)
    
    # Visualize
    visualize_comparison(results, num_tasks)
    
    # Analysis
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("\n📊 Performance Ranking (by Avg Accuracy):")
    sorted_results = sorted(results, key=lambda x: x.average_accuracy, reverse=True)
    for i, r in enumerate(sorted_results, 1):
        print(f"  {i}. {r.method_name}: {r.average_accuracy:.2f}%")
    
    print("\n💾 Memory Efficiency:")
    print("  - EWC & LWF: No example storage (privacy-preserving)")
    print("  - Experience Replay: Stores examples (memory overhead)")
    
    print("\n⚡ Computational Cost:")
    print("  - Naive & ER: Single forward/backward per batch")
    print("  - EWC: Extra Fisher computation per task")
    print("  - LWF: Double forward pass per batch")
    
    print("\n🎯 When to Use Each Method:")
    print("  - Experience Replay: When memory is available, best performance")
    print("  - EWC: Privacy concerns, memory constraints")
    print("  - LWF: Task domains similar, good knowledge transfer")
    print("  - Hybrid approaches: Combine strengths of multiple methods")
    print("=" * 70)


if __name__ == "__main__":
    main()
