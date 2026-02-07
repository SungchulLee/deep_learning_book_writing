# Temperature Scaling

## Introduction

Temperature scaling is the simplest and most effective post-hoc calibration method. It requires only a single parameter to be learned on a held-out validation set and preserves model accuracy while significantly improving calibration.

## The Method

### Mathematical Formulation

Given pre-softmax logits $\mathbf{z}$, temperature scaling applies:

$$\hat{p}_i = \frac{\exp(z_i / T)}{\sum_{j=1}^K \exp(z_j / T)}$$

where $T > 0$ is the temperature parameter.

**Key insight**: Temperature does not change the argmax, so predictions remain identical. Only confidence levels change.

### Finding Optimal Temperature

The optimal temperature $T^*$ minimizes negative log-likelihood on a validation set:

$$T^* = \arg\min_T \sum_{i=1}^{N_\text{val}} -\log \hat{p}_{y_i}(\mathbf{z}_i, T)$$

This is a simple one-dimensional optimization problem.

## PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Tuple, Optional
import numpy as np


class TemperatureScaling(nn.Module):
    """
    Temperature scaling for post-hoc calibration.
    
    After training a model, find optimal temperature on validation set
    that minimizes NLL, then apply to test predictions.
    
    Advantages:
    - Single parameter (T)
    - Preserves accuracy
    - Simple optimization
    - Very effective in practice
    """
    
    def __init__(self, temperature: float = 1.0):
        """
        Initialize with temperature = 1.0 (no scaling).
        """
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * temperature)
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature scaling.
        
        Args:
            logits: Pre-softmax outputs (batch_size, n_classes)
        
        Returns:
            Calibrated probabilities
        """
        return F.softmax(logits / self.temperature, dim=-1)
    
    def scale_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Return scaled logits (for use with CrossEntropyLoss).
        """
        return logits / self.temperature


def find_optimal_temperature(logits: torch.Tensor, 
                              labels: torch.Tensor,
                              lr: float = 0.01,
                              max_iter: int = 100) -> Tuple[float, float]:
    """
    Find optimal temperature via LBFGS optimization.
    
    Minimizes: NLL(softmax(logits/T), labels)
    
    Args:
        logits: Validation set logits
        labels: True labels
        lr: Learning rate for LBFGS
        max_iter: Maximum iterations
    
    Returns:
        optimal_temperature: Best T found
        final_nll: NLL at optimal T
    """
    temp_scaling = TemperatureScaling()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS([temp_scaling.temperature], lr=lr, max_iter=max_iter)
    
    nll_before = criterion(logits, labels).item()
    
    def closure():
        optimizer.zero_grad()
        scaled_logits = logits / temp_scaling.temperature
        loss = criterion(scaled_logits, labels)
        loss.backward()
        return loss
    
    optimizer.step(closure)
    
    optimal_T = temp_scaling.temperature.item()
    nll_after = criterion(logits / optimal_T, labels).item()
    
    return optimal_T, nll_after


def calibrate_model(model: nn.Module,
                    val_loader: DataLoader,
                    test_loader: DataLoader,
                    device: torch.device = None) -> Tuple[torch.Tensor, float, dict]:
    """
    Complete calibration workflow.
    
    Steps:
    1. Collect validation logits
    2. Find optimal temperature
    3. Apply to test predictions
    4. Return calibrated probabilities and metrics
    
    Args:
        model: Trained model
        val_loader: Validation data for finding T
        test_loader: Test data for evaluation
        device: Computation device
    
    Returns:
        test_probs: Calibrated test probabilities
        optimal_T: Optimal temperature found
        metrics: Calibration metrics before/after
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    # Step 1: Collect validation logits
    val_logits = []
    val_labels = []
    
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            if x.dim() > 2:
                x = x.view(x.size(0), -1)
            logits = model(x)
            val_logits.append(logits.cpu())
            val_labels.append(y)
    
    val_logits = torch.cat(val_logits)
    val_labels = torch.cat(val_labels)
    
    # Metrics before calibration
    probs_before = F.softmax(val_logits, dim=-1)
    ece_before = compute_ece_torch(probs_before, val_labels)
    
    # Step 2: Find optimal temperature
    optimal_T, nll_after = find_optimal_temperature(val_logits, val_labels)
    
    # Metrics after calibration
    probs_after = F.softmax(val_logits / optimal_T, dim=-1)
    ece_after = compute_ece_torch(probs_after, val_labels)
    
    # Step 3: Apply to test set
    test_probs = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            if x.dim() > 2:
                x = x.view(x.size(0), -1)
            logits = model(x)
            probs = F.softmax(logits / optimal_T, dim=-1)
            test_probs.append(probs.cpu())
    
    test_probs = torch.cat(test_probs)
    
    metrics = {
        'ece_before': ece_before,
        'ece_after': ece_after,
        'temperature': optimal_T,
        'improvement': ece_before - ece_after
    }
    
    return test_probs, optimal_T, metrics


def compute_ece_torch(probs: torch.Tensor, 
                      labels: torch.Tensor,
                      n_bins: int = 15) -> float:
    """
    Compute ECE using PyTorch tensors.
    """
    confidences = probs.max(dim=-1)[0]
    predictions = probs.argmax(dim=-1)
    
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_samples = len(confidences)
    
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        prop_in_bin = in_bin.float().sum() / n_samples
        
        if prop_in_bin > 0:
            avg_conf = confidences[in_bin].mean()
            accuracy = (predictions[in_bin] == labels[in_bin]).float().mean()
            ece += torch.abs(avg_conf - accuracy) * prop_in_bin
    
    return ece.item()
```

## Complete Example

```python
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split


def temperature_scaling_example():
    """
    Complete temperature scaling example on MNIST.
    """
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_data = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_data = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    # Split training into train/validation
    train_subset, val_subset = random_split(train_data, [50000, 10000])
    
    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False)
    
    # Create and train model
    print("=" * 60)
    print("Temperature Scaling Example")
    print("=" * 60)
    
    model = nn.Sequential(
        nn.Linear(28*28, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    ).to(device)
    
    print("\n1. Training model...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(5):
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device).view(-1, 28*28), y.to(device)
            
            logits = model(x)
            loss = criterion(logits, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"  Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f}")
    
    # Calibrate
    print("\n2. Finding optimal temperature...")
    test_probs, optimal_T, metrics = calibrate_model(model, val_loader, test_loader, device)
    
    # Results
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Optimal Temperature: {optimal_T:.4f}")
    print(f"ECE Before: {metrics['ece_before']:.4f}")
    print(f"ECE After:  {metrics['ece_after']:.4f}")
    print(f"Improvement: {metrics['improvement']:.4f} ({100*metrics['improvement']/metrics['ece_before']:.1f}% reduction)")
    
    # Check accuracy preserved
    test_labels = torch.cat([y for _, y in test_loader])
    predictions = test_probs.argmax(dim=-1)
    accuracy = (predictions == test_labels).float().mean()
    print(f"\nTest Accuracy: {accuracy:.4f} (preserved)")
    
    return model, optimal_T, metrics


if __name__ == "__main__":
    temperature_scaling_example()
```

## Visualization

```python
import matplotlib.pyplot as plt


def plot_temperature_calibration(val_logits: torch.Tensor,
                                  val_labels: torch.Tensor,
                                  optimal_T: float):
    """
    Visualize effect of temperature scaling.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Before calibration
    probs_before = F.softmax(val_logits, dim=-1)
    conf_before = probs_before.max(dim=-1)[0].numpy()
    preds_before = probs_before.argmax(dim=-1).numpy()
    labels_np = val_labels.numpy()
    
    ece_before, bin_data_before = compute_ece(conf_before, preds_before, labels_np)
    
    ax = axes[0]
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2)
    ax.scatter(bin_data_before['bin_confidences'], bin_data_before['bin_accuracies'],
              s=50, alpha=0.7, edgecolors='black')
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Before (T=1.0)\nECE = {ece_before:.4f}')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    # After calibration
    probs_after = F.softmax(val_logits / optimal_T, dim=-1)
    conf_after = probs_after.max(dim=-1)[0].numpy()
    preds_after = probs_after.argmax(dim=-1).numpy()
    
    ece_after, bin_data_after = compute_ece(conf_after, preds_after, labels_np)
    
    ax = axes[1]
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2)
    ax.scatter(bin_data_after['bin_confidences'], bin_data_after['bin_accuracies'],
              s=50, alpha=0.7, edgecolors='black', c='green')
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'After (T={optimal_T:.2f})\nECE = {ece_after:.4f}')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    # Confidence distribution comparison
    ax = axes[2]
    ax.hist(conf_before, bins=30, alpha=0.5, label=f'Before (T=1)', density=True)
    ax.hist(conf_after, bins=30, alpha=0.5, label=f'After (T={optimal_T:.2f})', density=True)
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Density')
    ax.set_title('Confidence Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def analyze_temperature_range(val_logits: torch.Tensor,
                               val_labels: torch.Tensor,
                               T_range: np.ndarray = np.linspace(0.5, 3.0, 50)):
    """
    Analyze calibration across temperature range.
    """
    eces = []
    nlls = []
    
    criterion = nn.CrossEntropyLoss()
    
    for T in T_range:
        probs = F.softmax(val_logits / T, dim=-1)
        conf = probs.max(dim=-1)[0].numpy()
        preds = probs.argmax(dim=-1).numpy()
        labels_np = val_labels.numpy()
        
        ece, _ = compute_ece(conf, preds, labels_np)
        nll = criterion(val_logits / T, val_labels).item()
        
        eces.append(ece)
        nlls.append(nll)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(T_range, eces, 'b-', linewidth=2)
    axes[0].axvline(x=1.0, color='r', linestyle='--', label='T=1 (no scaling)')
    optimal_idx = np.argmin(nlls)
    axes[0].axvline(x=T_range[optimal_idx], color='g', linestyle='--', label=f'T*={T_range[optimal_idx]:.2f}')
    axes[0].set_xlabel('Temperature')
    axes[0].set_ylabel('ECE')
    axes[0].set_title('ECE vs Temperature')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(T_range, nlls, 'b-', linewidth=2)
    axes[1].axvline(x=1.0, color='r', linestyle='--', label='T=1')
    axes[1].axvline(x=T_range[optimal_idx], color='g', linestyle='--', label=f'T*={T_range[optimal_idx]:.2f}')
    axes[1].set_xlabel('Temperature')
    axes[1].set_ylabel('NLL')
    axes[1].set_title('NLL vs Temperature')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
```

## When Temperature Scaling Works

### Ideal Conditions

- **Overconfident models**: Neural networks trained with cross-entropy
- **Smooth miscalibration**: Systematic overconfidence across all confidence levels
- **Sufficient validation data**: Need enough samples for reliable T estimation

### Limitations

| Limitation | Description |
|------------|-------------|
| **Uniform scaling** | Same T for all inputs (may need different T for different regions) |
| **Cannot fix predictions** | Doesn't improve accuracy, only confidence |
| **Assumes monotonic** | Works best when miscalibration is monotonic |
| **Class-agnostic** | Same T for all classes |

## Comparison with Other Methods

```python
def compare_calibration_methods(model: nn.Module,
                                 val_loader: DataLoader,
                                 test_loader: DataLoader):
    """
    Compare temperature scaling with other post-hoc methods.
    """
    print("=" * 60)
    print("Calibration Method Comparison")
    print("=" * 60)
    
    # Collect validation data
    val_logits, val_labels = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.view(x.size(0), -1)
            val_logits.append(model(x))
            val_labels.append(y)
    
    val_logits = torch.cat(val_logits)
    val_labels = torch.cat(val_labels)
    
    # 1. No calibration
    probs_raw = F.softmax(val_logits, dim=-1)
    ece_raw = compute_ece_torch(probs_raw, val_labels)
    
    # 2. Temperature scaling
    optimal_T, _ = find_optimal_temperature(val_logits, val_labels)
    probs_temp = F.softmax(val_logits / optimal_T, dim=-1)
    ece_temp = compute_ece_torch(probs_temp, val_labels)
    
    # 3. Vector scaling (per-class temperature)
    # Simple implementation: optimize K temperatures
    vector_temps = optimize_vector_scaling(val_logits, val_labels)
    probs_vector = F.softmax(val_logits / vector_temps, dim=-1)
    ece_vector = compute_ece_torch(probs_vector, val_labels)
    
    # Results
    print(f"{'Method':<25} {'ECE':<10} {'Parameters':<15}")
    print("-" * 50)
    print(f"{'No Calibration':<25} {ece_raw:<10.4f} {'0':<15}")
    print(f"{'Temperature Scaling':<25} {ece_temp:<10.4f} {'1':<15}")
    print(f"{'Vector Scaling':<25} {ece_vector:<10.4f} {'K (n_classes)':<15}")


def optimize_vector_scaling(logits: torch.Tensor, 
                             labels: torch.Tensor,
                             n_iter: int = 100) -> torch.Tensor:
    """
    Vector scaling: per-class temperature.
    """
    n_classes = logits.shape[1]
    temperatures = nn.Parameter(torch.ones(n_classes))
    
    optimizer = torch.optim.Adam([temperatures], lr=0.1)
    criterion = nn.CrossEntropyLoss()
    
    for _ in range(n_iter):
        optimizer.zero_grad()
        scaled_logits = logits / temperatures.unsqueeze(0)
        loss = criterion(scaled_logits, labels)
        loss.backward()
        optimizer.step()
        
        # Ensure positive temperatures
        with torch.no_grad():
            temperatures.clamp_(min=0.1)
    
    return temperatures.detach()
```

## Key Takeaways

!!! success "Summary"
    1. **Single parameter $T$** scales all logits uniformly
    2. **Preserves accuracy** — only confidence changes
    3. **Optimize on validation set** by minimizing NLL
    4. **Typical values**: $T \in [1.0, 3.0]$ for overconfident models
    5. **Should be standard practice** for any deployed classifier

## Best Practices

1. **Always use held-out validation** — never calibrate on test set
2. **Check multiple metrics** — ECE, MCE, and NLL
3. **Visualize reliability diagram** — understand calibration pattern
4. **Consider class-specific calibration** for imbalanced problems
5. **Monitor calibration over time** in production

## Exercises

1. **Optimal T Analysis**: Train models of different sizes. Does optimal temperature correlate with model capacity?

2. **Dataset Sensitivity**: Find optimal T on one dataset, apply to another. How transferable is temperature?

3. **Confidence Distribution**: Plot confidence histograms before/after calibration. How does the distribution change?

## References

- Guo, C., et al. (2017). "On Calibration of Modern Neural Networks"
- Platt, J. (1999). "Probabilistic Outputs for Support Vector Machines"
- Kull, M., et al. (2019). "Beyond Temperature Scaling"
