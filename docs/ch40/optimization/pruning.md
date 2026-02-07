# Model Pruning

## Overview

Pruning removes redundant or less important parameters from neural networks to create sparse models with reduced memory footprint and potentially faster inference. The core insight is that neural networks are typically over-parameterized, containing many weights that contribute minimally to the final output.

## Why Pruning?

Modern neural networks are over-parameterized. Key observations:

1. **Redundant weights**: Many weights converge to near-zero after training
2. **Sparse activations**: ReLU networks have sparse intermediate outputs
3. **Lottery ticket hypothesis**: Small subnetworks can achieve similar performance

Benefits of pruning:

| Benefit | Typical Improvement |
|---------|---------------------|
| Model size | 2-10× reduction |
| Inference speed | 1.5-4× faster |
| Memory usage | 2-10× reduction |
| Energy consumption | 2-5× reduction |

## Theoretical Foundation

### Weight Importance

The fundamental question in pruning is: which weights can be removed with minimal impact on model performance? Several criteria have been proposed:

**Magnitude-Based Importance**

The simplest and most widely used criterion assumes that small-magnitude weights contribute less to the output:

$$\text{importance}(w_i) = |w_i|$$

**Gradient-Based Importance**

Weights with small gradients may be less important for the current task:

$$\text{importance}(w_i) = \left| \frac{\partial \mathcal{L}}{\partial w_i} \right|$$

**Taylor Expansion Approximation**

Combining magnitude and gradient information through first-order Taylor expansion:

$$\text{importance}(w_i) = \left| w_i \cdot \frac{\partial \mathcal{L}}{\partial w_i} \right|$$

**Hessian-Based Importance (Optimal Brain Damage)**

Using second-order information to estimate the impact of removing each weight:

$$\text{importance}(w_i) = \frac{1}{2} w_i^2 \cdot H_{ii}$$

where $H_{ii}$ is the diagonal element of the Hessian matrix.

### Pruning Criterion

Given importance scores, we apply a threshold to determine which weights to prune:

$$\text{mask}_i = \mathbf{1}[\text{importance}(w_i) \geq \tau]$$

The sparsity ratio is:

$$s = \frac{\text{number of pruned weights}}{\text{total weights}} = \frac{\sum_i (1 - \text{mask}_i)}{n}$$

## Pruning Taxonomy

### Unstructured vs Structured Pruning

**Unstructured (Weight) Pruning**: Remove individual weights

```
Original:           Pruned (50%):
┌───────────────┐   ┌───────────────┐
│ 0.5 0.1 0.8 │   │ 0.5  0  0.8 │
│ 0.2 0.7 0.3 │ → │  0  0.7  0  │
│ 0.9 0.4 0.6 │   │ 0.9 0.4 0.6 │
└───────────────┘   └───────────────┘
```

- **Pros**: High compression ratios (90%+ possible), fine-grained control
- **Cons**: Requires sparse matrix support for speedup, irregular memory access

**Structured Pruning**: Remove entire filters/channels/layers

```
Original:           Pruned (remove filter 2):
Filter 1  Filter 2  Filter 3     Filter 1  Filter 3
┌─────┐   ┌─────┐   ┌─────┐     ┌─────┐   ┌─────┐
│█████│   │█████│   │█████│  →  │█████│   │█████│
│█████│   │█████│   │█████│     │█████│   │█████│
└─────┘   └─────┘   └─────┘     └─────┘   └─────┘
```

- **Pros**: Works with standard dense operations, immediate speedup
- **Cons**: Lower maximum sparsity (50-70%), coarser granularity

### Mathematical Formulation

For a weight tensor $\mathbf{W} \in \mathbb{R}^{m \times n}$:

$$\mathbf{W}_{\text{pruned}} = \mathbf{W} \odot \mathbf{M}$$

where $\mathbf{M} \in \{0, 1\}^{m \times n}$ is the binary mask and $\odot$ denotes element-wise multiplication.

## PyTorch Implementation

### Basic Magnitude Pruning

```python
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from typing import Dict, List, Tuple, Optional


class SimpleMLP(nn.Module):
    """Example model for demonstrating pruning."""
    
    def __init__(self, input_size: int = 784, hidden_size: int = 256, 
                 num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def apply_magnitude_pruning(model: nn.Module, amount: float = 0.3) -> nn.Module:
    """
    Apply magnitude-based pruning to all linear layers.
    
    Args:
        model: PyTorch model
        amount: Fraction of weights to prune (0.3 = 30%)
        
    Returns:
        Pruned model with masks attached
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
            print(f"Pruned {name}: {amount*100:.0f}% of weights")
    
    return model


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count total and non-zero parameters."""
    total = 0
    nonzero = 0
    
    for param in model.parameters():
        total += param.numel()
        nonzero += torch.count_nonzero(param).item()
    
    return total, nonzero
```

### MagnitudePruner Class

```python
class MagnitudePruner:
    """
    Magnitude-based weight pruning for neural networks.
    
    Implements both global and layer-wise pruning strategies
    with support for iterative pruning schedules.
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.masks = {}
    
    def compute_importance(self, 
                          param: torch.Tensor, 
                          method: str = 'magnitude') -> torch.Tensor:
        """
        Compute importance scores for weights.
        
        Args:
            param: Weight tensor
            method: 'magnitude', 'gradient', or 'taylor'
            
        Returns:
            Importance scores with same shape as param
        """
        if method == 'magnitude':
            return param.data.abs()
        elif method == 'gradient':
            if param.grad is None:
                raise ValueError("Gradients required for gradient-based importance")
            return param.grad.abs()
        elif method == 'taylor':
            if param.grad is None:
                raise ValueError("Gradients required for Taylor importance")
            return (param.data * param.grad).abs()
        else:
            raise ValueError(f"Unknown importance method: {method}")
    
    def prune_global(self, 
                     sparsity: float, 
                     importance_method: str = 'magnitude') -> Dict[str, torch.Tensor]:
        """
        Apply global magnitude pruning across all layers.
        
        Args:
            sparsity: Target sparsity ratio (0.0 to 1.0)
            importance_method: Method to compute importance scores
            
        Returns:
            Dictionary mapping parameter names to binary masks
        """
        # Collect importance scores from all parameters
        all_scores = []
        param_info = []
        
        for name, param in self.model.named_parameters():
            if 'weight' in name and param.dim() > 1:
                importance = self.compute_importance(param, importance_method)
                all_scores.append(importance.flatten())
                param_info.append((name, param, importance.shape))
        
        # Compute global threshold
        all_scores = torch.cat(all_scores)
        threshold = torch.quantile(all_scores, sparsity)
        
        # Create and apply masks
        self.masks = {}
        for name, param, shape in param_info:
            importance = self.compute_importance(param, importance_method)
            mask = (importance >= threshold).float()
            self.masks[name] = mask
            param.data *= mask
        
        return self.masks
    
    def prune_layerwise(self, 
                        sparsity: float,
                        importance_method: str = 'magnitude') -> Dict[str, torch.Tensor]:
        """
        Apply layer-wise pruning with uniform sparsity per layer.
        
        Args:
            sparsity: Target sparsity ratio per layer
            importance_method: Method to compute importance scores
            
        Returns:
            Dictionary mapping parameter names to binary masks
        """
        self.masks = {}
        
        for name, param in self.model.named_parameters():
            if 'weight' in name and param.dim() > 1:
                importance = self.compute_importance(param, importance_method)
                threshold = torch.quantile(importance.flatten(), sparsity)
                mask = (importance >= threshold).float()
                self.masks[name] = mask
                param.data *= mask
        
        return self.masks
    
    def apply_masks(self):
        """Re-apply stored masks (use after optimizer step during fine-tuning)."""
        for name, param in self.model.named_parameters():
            if name in self.masks:
                param.data *= self.masks[name]
    
    def get_sparsity_report(self) -> Dict:
        """
        Compute sparsity statistics for the pruned model.
        
        Returns:
            Dictionary with per-layer and overall sparsity
        """
        report = {}
        total_params = 0
        total_zeros = 0
        
        for name, param in self.model.named_parameters():
            if 'weight' in name and param.dim() > 1:
                numel = param.numel()
                zeros = (param.data.abs() < 1e-8).sum().item()
                sparsity = zeros / numel
                
                report[name] = {
                    'total': numel,
                    'zeros': zeros,
                    'sparsity': sparsity
                }
                
                total_params += numel
                total_zeros += zeros
        
        report['overall'] = {
            'total': total_params,
            'zeros': total_zeros,
            'sparsity': total_zeros / total_params if total_params > 0 else 0
        }
        
        return report
```

### Global Pruning

```python
def apply_global_pruning(model: nn.Module, amount: float = 0.3) -> nn.Module:
    """
    Apply global magnitude pruning across all layers.
    
    Prunes the smallest weights globally rather than per-layer,
    which often yields better accuracy at the same sparsity level.
    """
    parameters_to_prune = []
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            parameters_to_prune.append((module, 'weight'))
    
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount
    )
    
    return model
```

### Making Pruning Permanent

```python
def make_pruning_permanent(model: nn.Module) -> nn.Module:
    """
    Remove pruning reparameterization and make masks permanent.
    
    After this, the model can be saved without pruning hooks.
    """
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            if hasattr(module, 'weight_mask'):
                prune.remove(module, 'weight')
    
    return model
```

### Fine-Tuning Pruned Models

```python
def finetune_pruned_model(model: nn.Module,
                          pruner: MagnitudePruner,
                          train_loader: torch.utils.data.DataLoader,
                          epochs: int = 5,
                          lr: float = 1e-4,
                          device: str = 'cpu') -> nn.Module:
    """
    Fine-tune a pruned model while maintaining sparsity.
    
    Critical: After each optimizer step, re-apply the pruning mask
    to prevent pruned weights from being updated.
    
    Args:
        model: Pruned model
        pruner: MagnitudePruner with stored masks
        train_loader: Training data loader
        epochs: Number of fine-tuning epochs
        lr: Learning rate (typically lower than initial training)
        device: Device to train on
        
    Returns:
        Fine-tuned model
    """
    model = model.to(device)
    model.train()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            # Critical: Re-apply masks to maintain sparsity
            pruner.apply_masks()
            
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}")
    
    return model
```

## Structured Pruning

### Filter Pruning for CNNs

```python
class SimpleCNN(nn.Module):
    """Example CNN for demonstrating structured pruning."""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(64 * 7 * 7, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        return self.fc(x)


def apply_filter_pruning(model: nn.Module, amount: float = 0.3) -> nn.Module:
    """
    Apply structured pruning to remove entire filters.
    
    Filters are ranked by L1 norm and lowest are removed.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.ln_structured(
                module, 
                name='weight', 
                amount=amount, 
                n=1,  # L1 norm
                dim=0  # Prune along output channels (filters)
            )
            print(f"Pruned {name}: removed {amount*100:.0f}% of filters")
    
    return model
```

### StructuredPruner Class

```python
class StructuredPruner:
    """
    Filter and channel pruning for convolutional neural networks.
    
    Removes entire filters to achieve actual speedups without
    requiring sparse tensor operations.
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
    
    def compute_filter_importance(self, 
                                  conv: nn.Conv2d,
                                  method: str = 'l1_norm') -> torch.Tensor:
        """
        Compute importance score for each output filter.
        
        Args:
            conv: Convolutional layer
            method: 'l1_norm', 'l2_norm', or 'geometric_median'
            
        Returns:
            1D tensor of importance scores (length = out_channels)
        """
        weight = conv.weight.data  # Shape: (out_ch, in_ch, kH, kW)
        
        if method == 'l1_norm':
            # Sum of absolute values
            importance = weight.abs().view(weight.size(0), -1).sum(dim=1)
        elif method == 'l2_norm':
            # L2 norm of each filter
            importance = weight.view(weight.size(0), -1).norm(p=2, dim=1)
        elif method == 'geometric_median':
            # Distance from geometric median (prune filters closest to others)
            flat_filters = weight.view(weight.size(0), -1)
            median = flat_filters.mean(dim=0)
            importance = (flat_filters - median).norm(p=2, dim=1)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return importance
    
    def get_pruning_plan(self, 
                         prune_ratio: float,
                         method: str = 'l1_norm') -> Dict:
        """
        Compute which filters to keep in each layer.
        
        Args:
            prune_ratio: Fraction of filters to remove per layer
            method: Importance computation method
            
        Returns:
            Dictionary mapping layer names to indices of filters to keep
        """
        plan = {}
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                importance = self.compute_filter_importance(module, method)
                num_filters = len(importance)
                num_keep = max(1, int(num_filters * (1 - prune_ratio)))
                
                # Get indices of top-k filters by importance
                _, indices = torch.sort(importance, descending=True)
                keep_indices = indices[:num_keep].sort().values
                
                plan[name] = {
                    'keep_indices': keep_indices,
                    'original_filters': num_filters,
                    'remaining_filters': num_keep,
                    'importance_scores': importance
                }
        
        return plan
    
    def prune_conv_layer(self,
                         conv: nn.Conv2d,
                         keep_indices: torch.Tensor,
                         prune_input: bool = False,
                         input_indices: torch.Tensor = None) -> nn.Conv2d:
        """
        Create a new smaller conv layer with only the kept filters.
        
        Args:
            conv: Original convolutional layer
            keep_indices: Indices of output filters to keep
            prune_input: Whether to also prune input channels
            input_indices: If pruning input, which channels to keep
            
        Returns:
            New smaller Conv2d layer
        """
        # Determine new dimensions
        new_out_channels = len(keep_indices)
        new_in_channels = len(input_indices) if prune_input else conv.in_channels
        
        # Create new layer
        new_conv = nn.Conv2d(
            in_channels=new_in_channels,
            out_channels=new_out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=conv.bias is not None
        )
        
        # Copy selected weights
        if prune_input:
            new_conv.weight.data = conv.weight.data[keep_indices][:, input_indices]
        else:
            new_conv.weight.data = conv.weight.data[keep_indices]
        
        if conv.bias is not None:
            new_conv.bias.data = conv.bias.data[keep_indices]
        
        return new_conv
```

### Gradient-Based Filter Importance

```python
def compute_filter_importance(model: nn.Module, 
                              dataloader: torch.utils.data.DataLoader, 
                              criterion: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Compute importance scores for each filter based on gradient magnitude.
    
    Higher scores indicate more important filters.
    """
    importance = {}
    
    # Register hooks to capture gradients
    gradients = {}
    
    def save_grad(name):
        def hook(grad):
            gradients[name] = grad.clone()
        return hook
    
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            handle = module.weight.register_hook(save_grad(name))
            handles.append(handle)
    
    # Accumulate gradient magnitudes
    model.train()
    for batch_idx, (data, target) in enumerate(dataloader):
        model.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        for name in gradients:
            if name not in importance:
                importance[name] = torch.zeros_like(gradients[name])
            importance[name] += gradients[name].abs()
        
        if batch_idx >= 10:  # Use subset for speed
            break
    
    # Remove hooks
    for handle in handles:
        handle.remove()
    
    # Compute per-filter importance (sum over spatial dimensions)
    for name in importance:
        importance[name] = importance[name].sum(dim=[1, 2, 3])
    
    return importance


def prune_by_importance(model: nn.Module, 
                        importance: Dict[str, torch.Tensor], 
                        prune_ratio: float = 0.3) -> None:
    """
    Prune filters with lowest importance scores.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and name in importance:
            scores = importance[name]
            num_filters = len(scores)
            num_to_prune = int(num_filters * prune_ratio)
            
            # Find filters to prune
            _, indices = torch.sort(scores)
            prune_indices = indices[:num_to_prune]
            
            # Create mask
            mask = torch.ones(num_filters, dtype=torch.bool)
            mask[prune_indices] = False
            
            # Apply mask (zero out pruned filters)
            with torch.no_grad():
                module.weight.data[~mask] = 0
            
            print(f"{name}: pruned {num_to_prune}/{num_filters} filters")
```

## Iterative Pruning

### Gradual Sparsity Increase

```python
def iterative_pruning(model: nn.Module, 
                      train_loader: torch.utils.data.DataLoader, 
                      val_loader: torch.utils.data.DataLoader,
                      target_sparsity: float = 0.9, 
                      num_iterations: int = 10,
                      epochs_per_iteration: int = 5,
                      device: str = 'cpu') -> nn.Module:
    """
    Iterative magnitude pruning with fine-tuning.
    
    Gradually increases sparsity to target level, fine-tuning
    between pruning steps for better accuracy retention.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    current_sparsity = 0.0
    sparsity_increment = target_sparsity / num_iterations
    
    for iteration in range(num_iterations):
        current_sparsity += sparsity_increment
        print(f"\nIteration {iteration + 1}/{num_iterations}")
        print(f"Target sparsity: {current_sparsity*100:.1f}%")
        
        # Apply pruning
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # Compute how much more to prune
                current_zeros = (module.weight == 0).float().mean().item()
                additional_prune = (current_sparsity - current_zeros) / (1 - current_zeros)
                additional_prune = max(0, min(1, additional_prune))
                
                if additional_prune > 0:
                    prune.l1_unstructured(module, 'weight', amount=additional_prune)
        
        # Fine-tune
        model.train()
        for epoch in range(epochs_per_iteration):
            total_loss = 0
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            print(f"  Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f}")
        
        # Evaluate
        accuracy = evaluate(model, val_loader, device)
        print(f"  Validation accuracy: {accuracy*100:.2f}%")
    
    return model


def evaluate(model: nn.Module, 
             dataloader: torch.utils.data.DataLoader,
             device: str = 'cpu') -> float:
    """Evaluate model accuracy."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    
    return correct / total
```

### Cubic Sparsity Schedule

```python
def cubic_sparsity_schedule(current_step: int,
                            total_steps: int,
                            initial_sparsity: float = 0.0,
                            final_sparsity: float = 0.9) -> float:
    """
    Cubic sparsity schedule (commonly used in gradual pruning).
    
    Sparsity increases slowly at first, then accelerates.
    
    s(t) = s_f + (s_i - s_f) * (1 - t/T)^3
    """
    progress = current_step / total_steps
    sparsity = final_sparsity + (initial_sparsity - final_sparsity) * (1 - progress) ** 3
    return sparsity
```

## Lottery Ticket Hypothesis

Finding winning tickets (sparse subnetworks):

```python
def find_winning_ticket(model_class, 
                        train_loader: torch.utils.data.DataLoader, 
                        val_loader: torch.utils.data.DataLoader,
                        target_sparsity: float = 0.9, 
                        num_rounds: int = 10,
                        device: str = 'cpu') -> Tuple[nn.Module, Dict]:
    """
    Implementation of Lottery Ticket Hypothesis.
    
    1. Train network to completion
    2. Prune smallest weights
    3. Reset remaining weights to initial values
    4. Repeat
    
    The "winning ticket" is the sparse network with initial weights
    that can be trained to match the original performance.
    """
    # Initialize and save initial weights
    model = model_class().to(device)
    initial_state = {k: v.clone() for k, v in model.state_dict().items()}
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    sparsity_per_round = 1 - (1 - target_sparsity) ** (1 / num_rounds)
    mask = None
    
    for round_idx in range(num_rounds):
        print(f"\nRound {round_idx + 1}/{num_rounds}")
        
        # Train
        model.train()
        for epoch in range(10):
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                # Apply mask if exists
                if mask is not None:
                    with torch.no_grad():
                        for name, param in model.named_parameters():
                            if name in mask:
                                param.data *= mask[name]
        
        # Evaluate
        accuracy = evaluate(model, val_loader, device)
        print(f"Accuracy: {accuracy*100:.2f}%")
        
        # Prune and create mask
        mask = {}
        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() > 1:
                # Compute threshold for this round
                threshold = torch.quantile(
                    param.abs().flatten(),
                    sparsity_per_round
                )
                mask[name] = (param.abs() > threshold).float()
        
        # Reset to initial weights (keeping mask)
        for name, param in model.named_parameters():
            if name in initial_state:
                param.data = initial_state[name].clone().to(device)
                if name in mask:
                    param.data *= mask[name]
        
        # Reset optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Report sparsity
        total_params = sum(m.numel() for m in mask.values())
        nonzero_params = sum(m.sum().item() for m in mask.values())
        sparsity = 1 - nonzero_params / total_params
        print(f"Current sparsity: {sparsity*100:.1f}%")
    
    return model, mask
```

## Hardware-Aware Pruning

### N:M Sparsity Pattern

```python
def apply_n_m_sparsity(weight: torch.Tensor, n: int = 2, m: int = 4) -> torch.Tensor:
    """
    Apply N:M sparsity pattern (e.g., 2:4 for Ampere GPUs).
    
    Keeps N largest values in every M consecutive elements.
    This pattern is accelerated by NVIDIA Ampere tensor cores.
    """
    shape = weight.shape
    weight_flat = weight.view(-1)
    
    # Pad to multiple of m
    pad_size = (m - len(weight_flat) % m) % m
    if pad_size > 0:
        weight_flat = torch.cat([weight_flat, torch.zeros(pad_size)])
    
    # Reshape to groups of m
    weight_groups = weight_flat.view(-1, m)
    
    # Keep top n in each group
    _, indices = torch.topk(weight_groups.abs(), n, dim=1)
    mask = torch.zeros_like(weight_groups)
    mask.scatter_(1, indices, 1)
    
    # Apply mask
    weight_sparse = weight_groups * mask
    
    # Reshape back
    weight_sparse = weight_sparse.view(-1)[:weight.numel()].view(shape)
    
    return weight_sparse
```

## Best Practices

### Sensitivity Analysis

```python
def layer_sensitivity_analysis(model: nn.Module,
                               test_loader: torch.utils.data.DataLoader,
                               sparsity_levels: List[float] = [0.1, 0.3, 0.5, 0.7, 0.9],
                               device: str = 'cpu') -> Dict:
    """
    Analyze how sensitive each layer is to pruning.
    
    Prune one layer at a time to various sparsity levels
    and measure accuracy impact.
    """
    import copy
    
    baseline_acc = evaluate(model, test_loader, device)
    results = {}
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            results[name] = {'sparsity': [], 'accuracy': [], 'acc_drop': []}
            
            for sparsity in sparsity_levels:
                # Create copy and prune only this layer
                model_copy = copy.deepcopy(model).to(device)
                
                for n, m in model_copy.named_modules():
                    if n == name:
                        prune.l1_unstructured(m, name='weight', amount=sparsity)
                        break
                
                acc = evaluate(model_copy, test_loader, device)
                
                results[name]['sparsity'].append(sparsity)
                results[name]['accuracy'].append(acc)
                results[name]['acc_drop'].append(baseline_acc - acc)
    
    return results
```

## Trade-offs and Limitations

### Accuracy vs. Sparsity

| Sparsity | Typical Accuracy Drop | Recovery Difficulty |
|----------|----------------------|---------------------|
| 50% | < 0.5% | Easy (brief fine-tuning) |
| 70% | 0.5-1% | Moderate |
| 90% | 1-3% | Challenging |
| 95% | 3-5%+ | Very challenging |

### Speedup Reality

Unstructured pruning provides theoretical speedups that are difficult to realize without specialized hardware:

| Sparsity | Theoretical Speedup | Actual Speedup (GPU) |
|----------|--------------------|--------------------|
| 50% | 2× | ~1.0× (no speedup) |
| 90% | 10× | 1.5-2× (with sparse ops) |
| 95% | 20× | 2-3× (with sparse ops) |

Structured pruning achieves actual speedups proportional to the reduction in computation.

### When to Use Each Method

| Method | Best For | Typical Speedup | Accuracy Loss |
|--------|----------|-----------------|---------------|
| Unstructured (50%) | Maximum compression | None* | 0-1% |
| Unstructured (90%) | Extreme compression | None* | 1-5% |
| Structured (30%) | Immediate speedup | 1.3-1.5× | 0-2% |
| Structured (50%) | Significant speedup | 1.5-2× | 1-5% |
| N:M Sparsity | GPU acceleration | 1.5-2× | 0-1% |

*Unstructured pruning requires sparse matrix support for speedup

## Summary

Pruning is a powerful technique for model compression:

1. **Unstructured pruning**: Maximum flexibility, requires sparse support
2. **Structured pruning**: Immediate speedups, coarser granularity
3. **Iterative pruning**: Better accuracy at high sparsity
4. **Hardware-aware**: N:M patterns for modern GPUs

Key recommendations:
- Start with magnitude-based pruning
- Use iterative pruning with fine-tuning for high sparsity
- Consider structured pruning for deployment without sparse support
- Validate accuracy at each sparsity level
- Test on target hardware for actual speedup

## References

1. Han, S., et al. "Learning both Weights and Connections for Efficient Neural Networks." NeurIPS 2015.
2. Li, H., et al. "Pruning Filters for Efficient ConvNets." ICLR 2017.
3. Molchanov, P., et al. "Importance Estimation for Neural Network Pruning." CVPR 2019.
4. Frankle, J. & Carlin, M. "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks." ICLR 2019.
5. Blalock, D., et al. "What is the State of Neural Network Pruning?" MLSys 2020.
6. Zhou, A., et al. "Learning N:M Fine-grained Structured Sparse Neural Networks From Scratch." ICLR 2021.


---

# Iterative Magnitude Pruning

## Overview

Iterative Magnitude Pruning (IMP) is a pruning technique that progressively removes weights over multiple training cycles, allowing the network to adapt to increasing sparsity. This approach is closely associated with the Lottery Ticket Hypothesis, which posits that dense networks contain sparse subnetworks ("winning tickets") that can achieve comparable accuracy when trained in isolation.

## The Lottery Ticket Hypothesis

### Core Insight

Frankle and Carlin (2019) demonstrated that:

> A randomly-initialized, dense neural network contains a subnetwork that, when trained in isolation, can match the test accuracy of the original network after training for at most the same number of iterations.

Formally, for a network $f(x; \theta_0)$ with initial parameters $\theta_0$, there exists a mask $m \in \{0, 1\}^{|\theta|}$ such that:

$$\text{accuracy}(f(x; m \odot \theta_0)) \geq \text{accuracy}(f(x; \theta_0)) - \epsilon$$

with $\|m\|_0 \ll |\theta|$ (far fewer non-zero weights).

### Key Finding: Weight Rewinding

The original winning tickets use the **initial weights** $\theta_0$. Simply reinitializing pruned networks randomly does not achieve the same accuracy—the specific initialization matters.

Later work (Frankle et al., 2019) found that "rewinding" to early-training weights $\theta_k$ (after $k$ iterations) often works better than using $\theta_0$, especially for larger networks.

## Iterative Magnitude Pruning Algorithm

### Standard IMP

```
Algorithm: Iterative Magnitude Pruning (IMP)

Input: Network f, initial weights θ₀, training data D, 
       pruning rate p, target sparsity s, training epochs T

1. Train network to convergence: θ_T = Train(f, θ₀, D, T)
2. Save initial weights: θ_init = θ₀

3. While current_sparsity < target_sparsity:
   a. Compute magnitude scores: |θ_T|
   b. Prune p% of smallest-magnitude weights: 
      mask = Prune(|θ_T|, p)
   c. Reset weights to initialization: 
      θ ← mask ⊙ θ_init
   d. Retrain: θ_T = Train(f, θ, D, T)
   e. Update current_sparsity

4. Return: final mask, final weights θ_T
```

### Rewinding Variant

```
Algorithm: IMP with Weight Rewinding

Input: Network f, training data D, rewind iteration k,
       pruning rate p, target sparsity s

1. Train for k iterations: θ_k = Train(f, θ₀, D, k)
2. Save rewind checkpoint: θ_rewind = θ_k

3. Continue training to convergence: θ_T = Train(f, θ_k, D, T-k)

4. While current_sparsity < target_sparsity:
   a. Prune p% of smallest weights: mask = Prune(|θ_T|, p)
   b. Rewind: θ ← mask ⊙ θ_rewind
   c. Retrain from rewind point: θ_T = Train(f, θ, D, T-k)
   d. Update current_sparsity
```

## PyTorch Implementation

```python
import torch
import torch.nn as nn
import copy
from typing import Dict, Tuple, Optional


class IterativeMagnitudePruner:
    """
    Iterative Magnitude Pruning with optional weight rewinding.
    
    Implements the Lottery Ticket Hypothesis methodology:
    1. Train network
    2. Prune smallest weights
    3. Reset to initial/rewound weights
    4. Repeat until target sparsity
    """
    
    def __init__(self,
                 model: nn.Module,
                 prune_rate: float = 0.2,
                 rewind_epoch: Optional[int] = None):
        """
        Args:
            model: Neural network model
            prune_rate: Fraction of remaining weights to prune each iteration
            rewind_epoch: Epoch to rewind to (None = use initial weights)
        """
        self.model = model
        self.prune_rate = prune_rate
        self.rewind_epoch = rewind_epoch
        
        # Store initial weights
        self.initial_state = copy.deepcopy(model.state_dict())
        self.rewind_state = None
        
        # Pruning masks (all ones initially)
        self.masks = {}
        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() > 1:
                self.masks[name] = torch.ones_like(param.data)
    
    def save_rewind_checkpoint(self):
        """Save current weights as rewind checkpoint."""
        self.rewind_state = copy.deepcopy(self.model.state_dict())
    
    def get_current_sparsity(self) -> float:
        """Calculate current overall sparsity."""
        total_params = 0
        total_zeros = 0
        for name, mask in self.masks.items():
            total_params += mask.numel()
            total_zeros += (mask == 0).sum().item()
        return total_zeros / total_params if total_params > 0 else 0.0
    
    def prune_iteration(self) -> float:
        """
        Execute one pruning iteration.
        
        1. Identify weights to prune based on magnitude
        2. Update masks
        3. Return new sparsity level
        """
        # Collect all weight magnitudes (considering current masks)
        all_weights = []
        weight_info = []
        
        for name, param in self.model.named_parameters():
            if name in self.masks:
                # Get active (non-pruned) weights
                active_mask = self.masks[name]
                active_weights = param.data.abs() * active_mask
                
                # Only consider non-zero (active) weights
                active_values = active_weights[active_mask.bool()]
                all_weights.append(active_values.flatten())
                weight_info.append((name, param, active_mask))
        
        if not all_weights:
            return self.get_current_sparsity()
        
        all_weights = torch.cat(all_weights)
        
        # Determine threshold for pruning
        num_to_prune = int(len(all_weights) * self.prune_rate)
        if num_to_prune == 0:
            return self.get_current_sparsity()
        
        threshold = torch.kthvalue(all_weights, num_to_prune)[0]
        
        # Update masks
        for name, param, _ in weight_info:
            importance = param.data.abs()
            # Prune weights below threshold (but keep already-pruned weights pruned)
            new_prune = (importance < threshold) & self.masks[name].bool()
            self.masks[name][new_prune] = 0
        
        return self.get_current_sparsity()
    
    def apply_masks(self):
        """Apply current masks to model weights."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.masks:
                    param.data *= self.masks[name]
    
    def reset_weights(self, use_rewind: bool = True):
        """
        Reset weights to initial/rewind checkpoint (masked weights stay zero).
        
        Args:
            use_rewind: If True and rewind_state exists, use rewind checkpoint
        """
        if use_rewind and self.rewind_state is not None:
            checkpoint = self.rewind_state
        else:
            checkpoint = self.initial_state
        
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.data.copy_(checkpoint[name])
                if name in self.masks:
                    param.data *= self.masks[name]
    
    def get_mask_sparsity_report(self) -> Dict[str, float]:
        """Get per-layer sparsity from masks."""
        report = {}
        for name, mask in self.masks.items():
            sparsity = (mask == 0).sum().item() / mask.numel()
            report[name] = sparsity
        return report


def train_with_imp(model: nn.Module,
                   train_loader: torch.utils.data.DataLoader,
                   test_loader: torch.utils.data.DataLoader,
                   target_sparsity: float = 0.9,
                   prune_rate: float = 0.2,
                   epochs_per_round: int = 10,
                   rewind_epoch: Optional[int] = 5,
                   lr: float = 1e-3,
                   device: str = 'cpu') -> Tuple[nn.Module, Dict]:
    """
    Train model using Iterative Magnitude Pruning.
    
    Args:
        model: Model to prune
        train_loader: Training data
        test_loader: Test data
        target_sparsity: Final sparsity target
        prune_rate: Fraction to prune each iteration
        epochs_per_round: Training epochs per pruning round
        rewind_epoch: Epoch to save rewind checkpoint (None for no rewinding)
        lr: Learning rate
        device: Training device
        
    Returns:
        Pruned model, training history
    """
    model = model.to(device)
    pruner = IterativeMagnitudePruner(model, prune_rate, rewind_epoch)
    
    criterion = nn.CrossEntropyLoss()
    history = {'sparsity': [], 'accuracy': [], 'round': []}
    
    round_num = 0
    
    while pruner.get_current_sparsity() < target_sparsity:
        round_num += 1
        print(f"\n{'='*60}")
        print(f"Pruning Round {round_num}")
        print(f"Current Sparsity: {pruner.get_current_sparsity()*100:.1f}%")
        print(f"{'='*60}")
        
        # Reset to initial/rewind weights
        if round_num > 1:
            pruner.reset_weights(use_rewind=True)
        
        # Create optimizer (fresh for each round)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs_per_round
        )
        
        # Training loop
        for epoch in range(epochs_per_round):
            model.train()
            
            # Save rewind checkpoint at specified epoch
            if rewind_epoch is not None and epoch == rewind_epoch and round_num == 1:
                pruner.save_rewind_checkpoint()
                print(f"Saved rewind checkpoint at epoch {epoch}")
            
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                # Re-apply masks after optimizer step
                pruner.apply_masks()
            
            scheduler.step()
        
        # Evaluate
        accuracy = evaluate_accuracy(model, test_loader, device)
        
        # Prune
        new_sparsity = pruner.prune_iteration()
        
        # Log
        history['sparsity'].append(new_sparsity)
        history['accuracy'].append(accuracy)
        history['round'].append(round_num)
        
        print(f"Round {round_num} complete: "
              f"Sparsity={new_sparsity*100:.1f}%, "
              f"Accuracy={accuracy*100:.2f}%")
        
        # Early stopping if accuracy drops too much
        if round_num > 1 and accuracy < 0.5 * history['accuracy'][0]:
            print("Warning: Accuracy dropped significantly. Stopping early.")
            break
    
    return model, history


def evaluate_accuracy(model: nn.Module,
                     test_loader: torch.utils.data.DataLoader,
                     device: str) -> float:
    """Evaluate model accuracy."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    return correct / total
```

## Finding Winning Tickets

### Verification Protocol

To verify that a sparse network is a "winning ticket":

```python
def verify_winning_ticket(dense_model: nn.Module,
                          mask: Dict[str, torch.Tensor],
                          initial_weights: Dict[str, torch.Tensor],
                          train_loader: torch.utils.data.DataLoader,
                          test_loader: torch.utils.data.DataLoader,
                          epochs: int = 50,
                          device: str = 'cpu') -> Dict:
    """
    Verify if a sparse subnetwork is a winning ticket.
    
    A winning ticket should:
    1. Achieve comparable accuracy to the dense network
    2. Train successfully from the matched initial weights
    3. Fail if randomly reinitialized (control experiment)
    """
    results = {}
    
    # 1. Train dense network (baseline)
    dense_copy = copy.deepcopy(dense_model)
    dense_copy.load_state_dict(initial_weights)
    dense_acc = train_and_evaluate(dense_copy, train_loader, test_loader, epochs, device)
    results['dense_accuracy'] = dense_acc
    
    # 2. Train sparse network with matched initialization (ticket)
    sparse_ticket = copy.deepcopy(dense_model)
    sparse_ticket.load_state_dict(initial_weights)
    apply_mask(sparse_ticket, mask)
    ticket_acc = train_and_evaluate(sparse_ticket, train_loader, test_loader, epochs, device,
                                    mask=mask)
    results['ticket_accuracy'] = ticket_acc
    
    # 3. Train sparse network with random initialization (control)
    sparse_random = copy.deepcopy(dense_model)
    # Random initialization
    for name, param in sparse_random.named_parameters():
        if 'weight' in name:
            nn.init.kaiming_normal_(param)
    apply_mask(sparse_random, mask)
    random_acc = train_and_evaluate(sparse_random, train_loader, test_loader, epochs, device,
                                    mask=mask)
    results['random_init_accuracy'] = random_acc
    
    # Compute sparsity
    total, zeros = 0, 0
    for m in mask.values():
        total += m.numel()
        zeros += (m == 0).sum().item()
    results['sparsity'] = zeros / total
    
    # Is it a winning ticket?
    results['is_winning_ticket'] = (
        ticket_acc >= dense_acc - 0.02 and  # Within 2% of dense
        ticket_acc > random_acc + 0.02       # Better than random init by >2%
    )
    
    return results
```

## Late Resetting (Learning Rate Rewinding)

Renda et al. (2020) found that rewinding the learning rate schedule (not just weights) can match or exceed the Lottery Ticket results with simpler implementation:

```python
def train_with_learning_rate_rewinding(model: nn.Module,
                                       train_loader: torch.utils.data.DataLoader,
                                       test_loader: torch.utils.data.DataLoader,
                                       target_sparsity: float = 0.9,
                                       prune_rate: float = 0.2,
                                       total_epochs: int = 100,
                                       rewind_epochs: int = 10,
                                       initial_lr: float = 0.1,
                                       device: str = 'cpu') -> nn.Module:
    """
    Iterative pruning with learning rate rewinding.
    
    Instead of rewinding weights, rewind the learning rate schedule
    after each pruning step. Simpler and often equally effective.
    """
    model = model.to(device)
    masks = {}
    
    # Initialize masks
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() > 1:
            masks[name] = torch.ones_like(param.data)
    
    criterion = nn.CrossEntropyLoss()
    epochs_trained = 0
    
    while get_sparsity(masks) < target_sparsity:
        # Create optimizer with rewound learning rate
        optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr, 
                                    momentum=0.9, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[int(rewind_epochs*0.5), int(rewind_epochs*0.8)]
        )
        
        # Train for rewind_epochs
        for epoch in range(rewind_epochs):
            model.train()
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                # Maintain sparsity
                apply_masks_to_model(model, masks)
            
            scheduler.step()
        
        epochs_trained += rewind_epochs
        
        # Prune
        masks = prune_global(model, masks, prune_rate)
        
        # Evaluate
        acc = evaluate_accuracy(model, test_loader, device)
        sparsity = get_sparsity(masks)
        print(f"Epochs: {epochs_trained}, Sparsity: {sparsity*100:.1f}%, Acc: {acc*100:.2f}%")
    
    return model
```

## Practical Considerations

### Pruning Schedule

The rate at which sparsity increases affects final accuracy:

| Strategy | Description | Best For |
|----------|-------------|----------|
| One-shot | Single prune step | Quick experiments |
| Linear | Fixed absolute amount per round | Moderate compression |
| Exponential | Fixed percentage per round | High compression |
| Cubic | Gradual start, aggressive end | Maximum accuracy |

```python
def get_pruning_schedule(target_sparsity: float,
                         num_rounds: int,
                         schedule: str = 'exponential') -> list:
    """Generate pruning schedule."""
    if schedule == 'one_shot':
        return [target_sparsity]
    
    elif schedule == 'linear':
        return [target_sparsity * (i+1) / num_rounds 
                for i in range(num_rounds)]
    
    elif schedule == 'exponential':
        # Prune same fraction of remaining weights each round
        sparsities = []
        remaining = 1.0
        prune_rate = 1 - (1 - target_sparsity) ** (1 / num_rounds)
        for _ in range(num_rounds):
            remaining *= (1 - prune_rate)
            sparsities.append(1 - remaining)
        return sparsities
    
    elif schedule == 'cubic':
        # Slow start, fast end
        return [target_sparsity * (1 - (1 - (i+1)/num_rounds) ** 3)
                for i in range(num_rounds)]
```

### Computational Cost

IMP is computationally expensive:

| Method | Training Cost | Typical Quality |
|--------|--------------|-----------------|
| One-shot pruning | 2× baseline | Good |
| IMP (5 rounds) | 6× baseline | Very good |
| IMP (10+ rounds) | 11× baseline | Excellent |

For practical use, consider:
1. Fewer pruning rounds with larger prune rates
2. Learning rate rewinding instead of weight rewinding
3. Structured pruning for actual speedup

## References

1. Frankle, J. & Carlin, M. "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks." ICLR 2019.
2. Frankle, J., et al. "Stabilizing the Lottery Ticket Hypothesis." arXiv 2019.
3. Renda, A., et al. "Comparing Rewinding and Fine-tuning in Neural Network Pruning." ICLR 2020.
4. Malach, E., et al. "Proving the Lottery Ticket Hypothesis: Pruning is All You Need." ICML 2020.
5. Chen, T., et al. "The Lottery Ticket Hypothesis for Pre-trained BERT Networks." NeurIPS 2020.
