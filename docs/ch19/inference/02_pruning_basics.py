"""
BEGINNER LEVEL: Magnitude-Based Pruning

This script demonstrates the fundamentals of neural network pruning - removing
less important weights to create sparse models that are smaller and potentially faster.

Topics Covered:
- What is pruning and why it helps
- Magnitude-based pruning (unstructured)
- Global vs layer-wise pruning
- Sparsity measurement and visualization
- Accuracy recovery with fine-tuning

Mathematical Background:
- Importance score: I(w_i) = |w_i| (magnitude-based)
- Pruning criterion: prune w_i if |w_i| < threshold_τ
- Sparsity: S = (# zero weights) / (# total weights)

The Lottery Ticket Hypothesis:
Dense networks contain sparse subnetworks ("winning tickets") that can train
to comparable accuracy when initialized properly.

Prerequisites:
- Module 20: Feedforward Networks
- Module 15: Optimizers
- Understanding of backpropagation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import copy
import numpy as np
import matplotlib.pyplot as plt

# Import our utility functions
from utils import (
    count_parameters,
    measure_model_sparsity,
    print_sparsity_report,
    compare_model_sizes,
    evaluate_accuracy,
    compare_accuracies,
    plot_layer_sparsity,
    plot_compression_tradeoff,
    seed_everything
)


# ============================================================================
# STEP 1: DEFINE A SIMPLE FEEDFORWARD NETWORK
# ============================================================================

class SimpleFC(nn.Module):
    """
    A simple fully-connected network for MNIST classification.
    
    Architecture:
    - Input: 784 (28x28 flattened)
    - Hidden1: 300 neurons
    - Hidden2: 100 neurons
    - Output: 10 classes
    
    Total parameters: ≈266k
    This is intentionally overparameterized to demonstrate pruning benefits.
    """
    def __init__(self, input_size=784, hidden1=300, hidden2=100, num_classes=10):
        super(SimpleFC, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, num_classes)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        # Flatten input if needed
        x = x.view(x.size(0), -1)  # (B, 1, 28, 28) → (B, 784)
        
        # Hidden layers with ReLU activation
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Output layer (logits)
        x = self.fc3(x)
        
        return x


# ============================================================================
# STEP 2: LOAD DATA
# ============================================================================

def get_mnist_dataloaders(batch_size=128):
    """
    Load MNIST dataset.
    
    Args:
        batch_size: Batch size for data loaders
        
    Returns:
        train_loader, test_loader
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


# ============================================================================
# STEP 3: TRAIN BASELINE MODEL
# ============================================================================

def train_model(model, train_loader, test_loader, epochs=5, lr=0.001, device='cpu'):
    """
    Train the model.
    
    Args:
        model: Neural network
        train_loader: Training data
        test_loader: Test data
        epochs: Number of epochs
        lr: Learning rate
        device: Device to train on
        
    Returns:
        Trained model
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    print(f"\nTraining model for {epochs} epochs...")
    print("-" * 60)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        # Evaluate
        test_acc = evaluate_accuracy(model, test_loader, device)
        train_acc = correct / total
        
        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Train Acc: {train_acc*100:.2f}%, "
              f"Test Acc: {test_acc*100:.2f}%")
    
    print("-" * 60)
    return model


# ============================================================================
# STEP 4: PRUNING FUNCTIONS
# ============================================================================

def get_weight_importance_scores(model):
    """
    Calculate importance scores for all weights in the model.
    
    For magnitude-based pruning, importance = |weight|
    Larger magnitude = more important
    
    Args:
        model: Neural network model
        
    Returns:
        Dictionary mapping parameter names to importance scores
        
    Mathematical justification:
    - Weights with large magnitude have strong influence on outputs
    - Small weights contribute little to final predictions
    - Removing small weights has minimal impact on loss surface
    """
    importance_scores = {}
    
    for name, param in model.named_parameters():
        if 'weight' in name:  # Only prune weights, not biases
            # Importance = absolute value
            importance = torch.abs(param.data)
            importance_scores[name] = importance
    
    return importance_scores


def prune_model_global(model, sparsity_ratio):
    """
    Apply GLOBAL magnitude-based pruning.
    
    Global pruning:
    - Consider ALL weights across ALL layers
    - Find global threshold for given sparsity
    - Prune smallest weights regardless of layer
    
    This can lead to some layers being pruned more than others.
    
    Args:
        model: Model to prune
        sparsity_ratio: Fraction of weights to prune (0.0 to 1.0)
        
    Returns:
        Pruned model, pruning masks
        
    Example:
        If sparsity_ratio = 0.5, we prune the 50% smallest weights globally.
        
    Mathematical process:
        1. Collect all weight magnitudes: W = {|w_1|, |w_2|, ..., |w_n|}
        2. Sort: W_sorted
        3. Find threshold: τ = W_sorted[int(n × sparsity_ratio)]
        4. Mask: m_i = 1 if |w_i| ≥ τ else 0
    """
    print(f"\nApplying global pruning with {sparsity_ratio*100:.1f}% sparsity...")
    
    # Step 1: Collect all weight magnitudes
    all_weights = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            all_weights.append(param.data.abs().view(-1))
    
    # Concatenate into single tensor
    all_weights = torch.cat(all_weights)
    
    # Step 2: Find threshold
    # Sort and get the value at sparsity_ratio percentile
    num_weights = all_weights.numel()
    num_to_prune = int(num_weights * sparsity_ratio)
    
    if num_to_prune == 0:
        print("No weights to prune!")
        return model, {}
    
    # Get threshold by sorting
    sorted_weights, _ = torch.sort(all_weights)
    threshold = sorted_weights[num_to_prune - 1]
    
    print(f"Global threshold for pruning: {threshold:.6f}")
    print(f"Total weights: {num_weights:,}")
    print(f"Weights to prune: {num_to_prune:,}")
    
    # Step 3: Create and apply masks
    masks = {}
    total_pruned = 0
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            # Create mask: 1 for keep, 0 for prune
            mask = (param.data.abs() >= threshold).float()
            masks[name] = mask
            
            # Apply mask (set pruned weights to zero)
            param.data *= mask
            
            # Statistics
            layer_pruned = (mask == 0).sum().item()
            layer_total = mask.numel()
            layer_sparsity = layer_pruned / layer_total
            total_pruned += layer_pruned
            
            print(f"  {name:30s}: {layer_sparsity*100:5.1f}% pruned "
                  f"({layer_pruned:,} / {layer_total:,})")
    
    overall_sparsity = total_pruned / num_weights
    print(f"\nOverall sparsity achieved: {overall_sparsity*100:.2f}%")
    
    return model, masks


def prune_model_layerwise(model, sparsity_ratio):
    """
    Apply LAYER-WISE magnitude-based pruning.
    
    Layer-wise pruning:
    - Prune the same percentage from EACH layer
    - Find per-layer threshold
    - More balanced than global pruning
    
    This prevents any single layer from being pruned too aggressively.
    
    Args:
        model: Model to prune
        sparsity_ratio: Fraction of weights to prune PER LAYER
        
    Returns:
        Pruned model, pruning masks
        
    Comparison with global pruning:
    - Global: Can prune 100% of some layers, 0% of others
    - Layer-wise: Prunes sparsity_ratio from EVERY layer
    """
    print(f"\nApplying layer-wise pruning with {sparsity_ratio*100:.1f}% sparsity per layer...")
    
    masks = {}
    total_weights = 0
    total_pruned = 0
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            # Get layer weights
            layer_weights = param.data.abs().view(-1)
            layer_size = layer_weights.numel()
            
            # Calculate threshold for this layer
            num_to_prune = int(layer_size * sparsity_ratio)
            
            if num_to_prune == 0:
                masks[name] = torch.ones_like(param.data)
                continue
            
            # Sort and get layer-specific threshold
            sorted_weights, _ = torch.sort(layer_weights)
            threshold = sorted_weights[num_to_prune - 1]
            
            # Create mask
            mask = (param.data.abs() >= threshold).float()
            masks[name] = mask
            
            # Apply mask
            param.data *= mask
            
            # Statistics
            layer_pruned = (mask == 0).sum().item()
            layer_sparsity = layer_pruned / layer_size
            
            total_weights += layer_size
            total_pruned += layer_pruned
            
            print(f"  {name:30s}: {layer_sparsity*100:5.1f}% pruned "
                  f"({layer_pruned:,} / {layer_size:,}), threshold={threshold:.6f}")
    
    overall_sparsity = total_pruned / total_weights
    print(f"\nOverall sparsity achieved: {overall_sparsity*100:.2f}%")
    
    return model, masks


def apply_pruning_mask(model, masks):
    """
    Reapply pruning masks to model weights.
    
    This is needed after optimizer updates during fine-tuning
    to ensure pruned weights stay zero.
    
    Args:
        model: Model with potentially non-zero pruned weights
        masks: Dictionary of pruning masks
    """
    for name, param in model.named_parameters():
        if name in masks:
            param.data *= masks[name]


# ============================================================================
# STEP 5: FINE-TUNING PRUNED MODEL
# ============================================================================

def finetune_pruned_model(model, masks, train_loader, test_loader, 
                          epochs=3, lr=0.0001, device='cpu'):
    """
    Fine-tune a pruned model to recover accuracy.
    
    Fine-tuning process:
    1. Use lower learning rate (to preserve good weights)
    2. Train normally
    3. After each optimizer step, reapply masks
    4. This ensures pruned weights stay zero
    
    Why fine-tuning helps:
    - Network learns to compensate for missing weights
    - Remaining weights adjust their values
    - Often recovers most of the accuracy loss
    
    Args:
        model: Pruned model
        masks: Pruning masks
        train_loader: Training data
        test_loader: Test data
        epochs: Fine-tuning epochs
        lr: Learning rate (should be smaller than initial training)
        device: Device to train on
        
    Returns:
        Fine-tuned model
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    print(f"\nFine-tuning pruned model for {epochs} epochs...")
    print(f"Using lower learning rate: {lr}")
    print("-" * 60)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            # CRITICAL: Reapply masks to keep pruned weights at zero
            apply_pruning_mask(model, masks)
            
            train_loss += loss.item()
        
        # Evaluate
        test_acc = evaluate_accuracy(model, test_loader, device)
        
        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Test Acc: {test_acc*100:.2f}%")
    
    print("-" * 60)
    return model


# ============================================================================
# STEP 6: EXPERIMENT WITH DIFFERENT SPARSITY LEVELS
# ============================================================================

def pruning_experiment(base_model, train_loader, test_loader, device='cpu'):
    """
    Run experiments with different sparsity levels to find optimal trade-off.
    
    This demonstrates:
    - How accuracy degrades with increasing sparsity
    - The effectiveness of fine-tuning
    - Optimal sparsity level for deployment
    
    Args:
        base_model: Trained baseline model
        train_loader: Training data
        test_loader: Test data
        device: Device to run on
    """
    # Sparsity levels to test
    sparsity_levels = [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
    
    results = {
        'sparsity': [],
        'accuracy_before_ft': [],
        'accuracy_after_ft': [],
        'compression_ratio': []
    }
    
    # Baseline (no pruning)
    baseline_acc = evaluate_accuracy(base_model, test_loader, device)
    print(f"\nBaseline accuracy (0% sparsity): {baseline_acc*100:.2f}%")
    
    print("\n" + "="*60)
    print("PRUNING EXPERIMENT: Testing Multiple Sparsity Levels")
    print("="*60)
    
    for sparsity in sparsity_levels:
        if sparsity == 0.0:
            # Baseline
            results['sparsity'].append(0.0)
            results['accuracy_before_ft'].append(baseline_acc)
            results['accuracy_after_ft'].append(baseline_acc)
            results['compression_ratio'].append(1.0)
            continue
        
        print(f"\n--- Testing {sparsity*100:.0f}% Sparsity ---")
        
        # Prune model (use global pruning)
        model_copy = copy.deepcopy(base_model)
        pruned_model, masks = prune_model_global(model_copy, sparsity)
        
        # Accuracy before fine-tuning
        acc_before = evaluate_accuracy(pruned_model, test_loader, device)
        print(f"Accuracy before fine-tuning: {acc_before*100:.2f}%")
        
        # Fine-tune
        finetuned_model = finetune_pruned_model(
            pruned_model, masks, train_loader, test_loader,
            epochs=2, lr=0.0001, device=device
        )
        
        # Accuracy after fine-tuning
        acc_after = evaluate_accuracy(finetuned_model, test_loader, device)
        print(f"Accuracy after fine-tuning:  {acc_after*100:.2f}%")
        
        # Compression ratio (approximately)
        compression_ratio = 1.0 / (1.0 - sparsity + 1e-10)
        
        # Store results
        results['sparsity'].append(sparsity)
        results['accuracy_before_ft'].append(acc_before)
        results['accuracy_after_ft'].append(acc_after)
        results['compression_ratio'].append(compression_ratio)
    
    return results


def plot_pruning_results(results):
    """
    Visualize pruning experiment results.
    
    Args:
        results: Dictionary with experiment results
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Accuracy vs Sparsity
    ax1.plot(np.array(results['sparsity']) * 100, 
             np.array(results['accuracy_before_ft']) * 100,
             marker='o', label='Before Fine-tuning', linewidth=2, markersize=8)
    ax1.plot(np.array(results['sparsity']) * 100,
             np.array(results['accuracy_after_ft']) * 100,
             marker='s', label='After Fine-tuning', linewidth=2, markersize=8)
    
    ax1.set_xlabel('Sparsity (%)', fontsize=12)
    ax1.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax1.set_title('Accuracy vs Sparsity Level', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Accuracy vs Compression Ratio
    ax2.plot(results['compression_ratio'],
             np.array(results['accuracy_after_ft']) * 100,
             marker='o', color='green', linewidth=2, markersize=8)
    
    ax2.set_xlabel('Compression Ratio', fontsize=12)
    ax2.set_ylabel('Test Accuracy (%) [After Fine-tuning]', fontsize=12)
    ax2.set_title('Accuracy vs Compression Trade-off', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pruning_experiment_results.png', dpi=150, bbox_inches='tight')
    print("\nResults saved to: pruning_experiment_results.png")
    plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function demonstrating magnitude-based pruning.
    """
    seed_everything(42)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # ========================================================================
    # PART A: TRAIN BASELINE MODEL
    # ========================================================================
    
    print("="*60)
    print("PART A: BASELINE MODEL TRAINING")
    print("="*60)
    
    train_loader, test_loader = get_mnist_dataloaders(batch_size=128)
    
    model = SimpleFC()
    print(f"\nModel parameters: {count_parameters(model):,}")
    
    # Train or load
    try:
        model.load_state_dict(torch.load('simple_fc_mnist.pth', map_location=device))
        print("Loaded pre-trained model")
    except:
        print("Training baseline model...")
        model = train_model(model, train_loader, test_loader, epochs=5, device=device)
        torch.save(model.state_dict(), 'simple_fc_mnist.pth')
    
    baseline_acc = evaluate_accuracy(model, test_loader, device)
    print(f"\nBaseline Test Accuracy: {baseline_acc*100:.2f}%")
    
    # ========================================================================
    # PART B: GLOBAL PRUNING
    # ========================================================================
    
    print("\n" + "="*60)
    print("PART B: GLOBAL MAGNITUDE PRUNING")
    print("="*60)
    
    # Create a copy for pruning
    model_global = copy.deepcopy(model)
    
    # Prune 80% of weights globally
    pruned_model_global, masks_global = prune_model_global(model_global, sparsity_ratio=0.8)
    
    # Check sparsity
    print_sparsity_report(pruned_model_global)
    
    # Visualize layer-wise sparsity
    plot_layer_sparsity(pruned_model_global)
    
    # Accuracy before fine-tuning
    pruned_acc_before = evaluate_accuracy(pruned_model_global, test_loader, device)
    print(f"\nPruned model accuracy (before fine-tuning): {pruned_acc_before*100:.2f}%")
    print(f"Accuracy drop: {(baseline_acc - pruned_acc_before)*100:.2f}%")
    
    # Fine-tune
    finetuned_model_global = finetune_pruned_model(
        pruned_model_global, masks_global, train_loader, test_loader,
        epochs=3, lr=0.0001, device=device
    )
    
    # Accuracy after fine-tuning
    pruned_acc_after = evaluate_accuracy(finetuned_model_global, test_loader, device)
    print(f"\nPruned model accuracy (after fine-tuning): {pruned_acc_after*100:.2f}%")
    print(f"Accuracy recovered: {(pruned_acc_after - pruned_acc_before)*100:.2f}%")
    print(f"Final accuracy drop: {(baseline_acc - pruned_acc_after)*100:.2f}%")
    
    # ========================================================================
    # PART C: LAYER-WISE PRUNING
    # ========================================================================
    
    print("\n" + "="*60)
    print("PART C: LAYER-WISE MAGNITUDE PRUNING")
    print("="*60)
    
    model_layerwise = copy.deepcopy(model)
    
    # Prune 70% per layer
    pruned_model_layerwise, masks_layerwise = prune_model_layerwise(
        model_layerwise, sparsity_ratio=0.7
    )
    
    print_sparsity_report(pruned_model_layerwise)
    
    # Fine-tune
    finetuned_model_layerwise = finetune_pruned_model(
        pruned_model_layerwise, masks_layerwise, train_loader, test_loader,
        epochs=3, lr=0.0001, device=device
    )
    
    layerwise_acc = evaluate_accuracy(finetuned_model_layerwise, test_loader, device)
    print(f"\nLayer-wise pruned accuracy: {layerwise_acc*100:.2f}%")
    
    # ========================================================================
    # PART D: SPARSITY SWEEP
    # ========================================================================
    
    print("\n" + "="*60)
    print("PART D: SPARSITY LEVEL EXPERIMENT")
    print("="*60)
    
    results = pruning_experiment(model, train_loader, test_loader, device)
    
    # Plot results
    plot_pruning_results(results)
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    print("\n" + "="*60)
    print("SUMMARY: KEY TAKEAWAYS")
    print("="*60)
    print("""
    1. Magnitude-Based Pruning:
       ✓ Simple and effective
       ✓ Remove smallest weights first
       ✓ Can achieve 80-90% sparsity with <1% accuracy loss
       ✓ Fine-tuning is crucial for recovery
    
    2. Global vs Layer-wise:
       ✓ Global: More aggressive, some layers highly pruned
       ✓ Layer-wise: More balanced, safer
       ✓ Both benefit from fine-tuning
    
    3. Practical Guidelines:
       - Start with 50% sparsity, then increase
       - Always fine-tune after pruning
       - Monitor layer-wise sparsity distribution
       - Consider structured pruning for actual speedups
    
    4. Limitations:
       - Unstructured sparsity needs special hardware
       - No guaranteed speedup on regular hardware
       - First/last layers are sensitive
       - May hurt out-of-distribution generalization
    
    5. Next Steps:
       - Try structured pruning (Module 05)
       - Implement iterative pruning (Module 06)
       - Combine with quantization (Module 09)
       - Explore lottery ticket hypothesis
    """)
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
