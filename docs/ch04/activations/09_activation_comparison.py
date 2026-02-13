#!/usr/bin/env python3
"""
Script 09: Systematic Activation Function Comparison
DIFFICULTY: ⭐⭐⭐⭐ Hard | TIME: 5-8 min | PREREQ: Scripts 01-08
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import time

torch.manual_seed(42)
np.random.seed(42)

class ComparisonNetwork(nn.Module):
    """Network for comparing different activations"""
    def __init__(self, activation_type='relu'):
        super().__init__()
        self.fc1 = nn.Linear(20, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        
        # Select activation
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU(),
            'gelu': nn.GELU(),
            'silu': nn.SiLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid()
        }
        self.activation = activations.get(activation_type, nn.ReLU())
        self.name = activation_type
    
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x

def generate_comparison_data():
    """Generate challenging classification dataset"""
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=15,
        n_redundant=5, n_classes=2, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train).unsqueeze(1)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test).unsqueeze(1)
    
    return X_train, y_train, X_test, y_test

def train_and_evaluate(model, X_train, y_train, X_test, y_test, 
                       epochs=150, lr=0.001, verbose=False):
    """Train model and return metrics"""
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    train_accs, test_accs, losses = [], [], []
    start_time = time.time()
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()
        
        # Metrics
        with torch.no_grad():
            train_acc = ((torch.sigmoid(logits) > 0.5) == y_train).float().mean()
            test_logits = model(X_test)
            test_acc = ((torch.sigmoid(test_logits) > 0.5) == y_test).float().mean()
        
        train_accs.append(train_acc.item())
        test_accs.append(test_acc.item())
        losses.append(loss.item())
        
        if verbose and (epoch + 1) % 30 == 0:
            print(f"  Epoch {epoch+1}: Test Acc = {test_acc.item():.4f}")
    
    training_time = time.time() - start_time
    
    return {
        'train_accs': train_accs,
        'test_accs': test_accs,
        'losses': losses,
        'final_test_acc': test_accs[-1],
        'final_train_acc': train_accs[-1],
        'training_time': training_time,
        'best_test_acc': max(test_accs)
    }

def compare_activations():
    """Compare multiple activation functions"""
    print("\n" + "=" * 70)
    print("COMPARING ACTIVATION FUNCTIONS")
    print("=" * 70)
    
    # Data
    print("\nGenerating dataset...")
    X_train, y_train, X_test, y_test = generate_comparison_data()
    print(f"Features: {X_train.shape[1]}, Samples: {len(X_train)}")
    
    # Activations to compare
    activations = ['relu', 'leaky_relu', 'elu', 'gelu', 'silu', 'tanh', 'sigmoid']
    results = {}
    
    print("\nTraining models with different activations...")
    print("-" * 70)
    
    for activation in activations:
        print(f"\n🔹 Training with {activation.upper()}...")
        model = ComparisonNetwork(activation_type=activation)
        result = train_and_evaluate(
            model, X_train, y_train, X_test, y_test,
            epochs=150, lr=0.001, verbose=False
        )
        results[activation] = result
        print(f"   Final Test Acc: {result['final_test_acc']:.4f}")
        print(f"   Best Test Acc:  {result['best_test_acc']:.4f}")
        print(f"   Training Time:  {result['training_time']:.2f}s")
    
    return results

def plot_comparison_results(results):
    """Visualize comparison results"""
    
    # Extract data
    activations = list(results.keys())
    final_accs = [results[a]['final_test_acc'] for a in activations]
    best_accs = [results[a]['best_test_acc'] for a in activations]
    times = [results[a]['training_time'] for a in activations]
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    
    # Plot 1: Final accuracies
    ax1 = plt.subplot(2, 3, 1)
    colors = plt.cm.viridis(np.linspace(0, 1, len(activations)))
    bars = ax1.bar(range(len(activations)), final_accs, color=colors, alpha=0.8)
    ax1.set_xticks(range(len(activations)))
    ax1.set_xticklabels([a.upper() for a in activations], rotation=45, ha='right')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Final Test Accuracy', fontweight='bold')
    ax1.set_ylim([0.7, 1.0])
    ax1.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(final_accs):
        ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)
    
    # Plot 2: Training time
    ax2 = plt.subplot(2, 3, 2)
    bars2 = ax2.bar(range(len(activations)), times, color=colors, alpha=0.8)
    ax2.set_xticks(range(len(activations)))
    ax2.set_xticklabels([a.upper() for a in activations], rotation=45, ha='right')
    ax2.set_ylabel('Time (seconds)')
    ax2.set_title('Training Time', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Best accuracy
    ax3 = plt.subplot(2, 3, 3)
    bars3 = ax3.bar(range(len(activations)), best_accs, color=colors, alpha=0.8)
    ax3.set_xticks(range(len(activations)))
    ax3.set_xticklabels([a.upper() for a in activations], rotation=45, ha='right')
    ax3.set_ylabel('Accuracy')
    ax3.set_title('Best Test Accuracy', fontweight='bold')
    ax3.set_ylim([0.7, 1.0])
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4-6: Learning curves for top 3
    sorted_acts = sorted(activations, key=lambda a: results[a]['final_test_acc'], reverse=True)[:3]
    
    for idx, activation in enumerate(sorted_acts):
        ax = plt.subplot(2, 3, 4 + idx)
        epochs = range(1, len(results[activation]['test_accs']) + 1)
        ax.plot(epochs, results[activation]['train_accs'], 
                label='Train', linewidth=2, alpha=0.7)
        ax.plot(epochs, results[activation]['test_accs'], 
                label='Test', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{activation.upper()} - Learning Curve', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.5, 1.0])
    
    plt.tight_layout()
    return fig

def print_summary_table(results):
    """Print summary table of results"""
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Activation':<15} {'Final Acc':>10} {'Best Acc':>10} {'Time (s)':>10}")
    print("-" * 70)
    
    # Sort by final accuracy
    sorted_results = sorted(results.items(), 
                          key=lambda x: x[1]['final_test_acc'], 
                          reverse=True)
    
    for activation, result in sorted_results:
        print(f"{activation.upper():<15} "
              f"{result['final_test_acc']:>10.4f} "
              f"{result['best_test_acc']:>10.4f} "
              f"{result['training_time']:>10.2f}")
    
    print("=" * 70)
    
    # Find best
    best = sorted_results[0]
    print(f"\n🏆 Best performing: {best[0].upper()}")
    print(f"   Final accuracy: {best[1]['final_test_acc']:.4f}")

def main():
    print("\n" + "█" * 70)
    print("   Script 09: Activation Function Comparison")
    print("█" * 70)
    
    # Run comparison
    results = compare_activations()
    
    # Visualize
    print("\n" + "=" * 70)
    print("Generating visualization...")
    fig = plot_comparison_results(results)
    
    # Summary
    print_summary_table(results)
    
    print("\n" + "=" * 70)
    print("KEY INSIGHTS:")
    print("=" * 70)
    print("• Modern activations (GELU, SiLU) often outperform ReLU")
    print("• ReLU is still very competitive and faster")
    print("• Sigmoid/Tanh struggle in deep networks (vanishing gradients)")
    print("• Leaky ReLU and ELU are good alternatives to ReLU")
    print("• Performance varies by task and architecture")
    print("• Always compare on YOUR specific problem!")
    
    print("\n✅ Next: Run '10_advanced_techniques.py' for advanced topics!")
    plt.show()

if __name__ == "__main__":
    main()
