# Advanced Activation Techniques

## Overview

This section covers advanced topics in activation functions including gradient flow analysis, the dead neuron problem, adaptive activations, and initialization strategies. Understanding these concepts is essential for training deep networks successfully.

## Learning Objectives

By the end of this section, you will understand:

1. Gradient flow analysis in deep networks
2. The dead ReLU problem and solutions
3. Parametric and adaptive activations
4. Proper initialization for different activations
5. Activation debugging and monitoring

---

## Gradient Flow Analysis

### Understanding Gradient Flow

In backpropagation, gradients flow from the loss back through each layer. The gradient at layer $l$ depends on all subsequent layers:

$$\frac{\partial L}{\partial W_l} = \frac{\partial L}{\partial z_L} \cdot \prod_{k=l}^{L-1} \frac{\partial z_{k+1}}{\partial z_k} \cdot \frac{\partial z_l}{\partial W_l}$$

If any $\frac{\partial z_{k+1}}{\partial z_k}$ is very small (vanishing) or very large (exploding), training fails.

### Measuring Gradient Norms

```python
import torch
import torch.nn as nn

def analyze_gradient_flow(model, test_input, test_target, criterion):
    """Analyze gradient norms at each layer."""
    # Forward pass
    output = model(test_input)
    loss = criterion(output, test_target)
    
    # Backward pass
    loss.backward()
    
    # Collect gradient statistics
    gradient_stats = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad.data
            stats = {
                'name': name,
                'mean': grad.mean().item(),
                'std': grad.std().item(),
                'norm': grad.norm().item(),
                'max': grad.max().item(),
                'min': grad.min().item(),
            }
            gradient_stats.append(stats)
            print(f"{name:30s} | norm: {stats['norm']:.2e} | "
                  f"mean: {stats['mean']:.2e} | std: {stats['std']:.2e}")
    
    return gradient_stats

# Example usage
model = nn.Sequential(
    nn.Linear(64, 64), nn.ReLU(),
    nn.Linear(64, 64), nn.ReLU(),
    nn.Linear(64, 64), nn.ReLU(),
    nn.Linear(64, 10)
)
x = torch.randn(32, 64)
y = torch.randint(0, 10, (32,))
analyze_gradient_flow(model, x, y, nn.CrossEntropyLoss())
```

### Comparing Activation Gradient Flow

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def compare_activation_gradients(num_layers=10, hidden_dim=64):
    """Compare gradient flow with different activations in deep networks."""
    
    activations = {
        'ReLU': nn.ReLU,
        'Sigmoid': nn.Sigmoid,
        'Tanh': nn.Tanh,
        'GELU': nn.GELU,
        'LeakyReLU': lambda: nn.LeakyReLU(0.1),
    }
    
    results = {}
    
    for name, activation_fn in activations.items():
        # Build deep network
        layers = []
        for _ in range(num_layers):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), activation_fn()])
        model = nn.Sequential(*layers)
        
        # Forward and backward
        x = torch.randn(32, hidden_dim, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        # Measure gradient at input
        grad_norm = x.grad.norm().item()
        results[name] = grad_norm
        
        print(f"{name:15s}: Input gradient norm = {grad_norm:.6f}")
    
    return results

# Compare
results = compare_activation_gradients(num_layers=10)
```

---

## The Dead ReLU Problem

### Understanding Dead Neurons

A "dead" ReLU neuron produces zero output for all inputs in the dataset:

1. **Cause**: Large negative bias or weights push pre-activation permanently negative
2. **Effect**: Since ReLU(x) = 0 for x < 0, gradient is also 0
3. **Result**: Weights never update; neuron is permanently inactive

### Detection

```python
import torch
import torch.nn as nn

def detect_dead_neurons(model, dataloader, threshold=0.0):
    """Detect neurons that are always inactive."""
    model.eval()
    
    # Track activations for each ReLU layer
    activation_max = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            if name not in activation_max:
                activation_max[name] = output.detach().max(dim=0)[0]
            else:
                # Keep running max
                activation_max[name] = torch.max(
                    activation_max[name], 
                    output.detach().max(dim=0)[0]
                )
        return hook
    
    # Register hooks
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU):
            handles.append(module.register_forward_hook(hook_fn(name)))
    
    # Process entire dataset
    with torch.no_grad():
        for batch in dataloader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            model(x)
    
    # Remove hooks
    for h in handles:
        h.remove()
    
    # Analyze results
    total_dead = 0
    total_neurons = 0
    
    for name, max_acts in activation_max.items():
        dead = (max_acts <= threshold).sum().item()
        total = max_acts.numel()
        total_dead += dead
        total_neurons += total
        print(f"{name}: {dead}/{total} dead neurons ({100*dead/total:.1f}%)")
    
    print(f"\nTotal: {total_dead}/{total_neurons} dead "
          f"({100*total_dead/total_neurons:.1f}%)")
    
    return total_dead, total_neurons
```

### Prevention Strategies

#### 1. Use Leaky ReLU or ELU

```python
# Instead of ReLU
model = nn.Sequential(
    nn.Linear(64, 128),
    nn.LeakyReLU(0.1),  # Allows gradient flow for negative inputs
    nn.Linear(128, 64),
    nn.LeakyReLU(0.1),
    nn.Linear(64, 10)
)
```

#### 2. Proper Initialization

```python
def init_relu_network(model):
    """Initialize network for ReLU activations."""
    for module in model.modules():
        if isinstance(module, nn.Linear):
            # He initialization for ReLU
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                # Small positive bias helps prevent dead neurons
                nn.init.constant_(module.bias, 0.01)

model = nn.Sequential(
    nn.Linear(64, 128), nn.ReLU(),
    nn.Linear(128, 64), nn.ReLU(),
    nn.Linear(64, 10)
)
init_relu_network(model)
```

#### 3. Lower Learning Rate

```python
# Use lower learning rate to prevent drastic weight updates
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # Lower than default 1e-3
```

#### 4. Batch Normalization

```python
class StableBlock(nn.Module):
    """Block with BatchNorm to maintain healthy activations."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # BN normalizes pre-activations, reducing dead neuron risk
        return self.relu(self.bn(self.fc(x)))
```

---

## PReLU: Learnable Negative Slopes

### How PReLU Works

PReLU learns the optimal negative slope during training:

$$\text{PReLU}(x) = \begin{cases} x & \text{if } x > 0 \\ a \cdot x & \text{if } x \leq 0 \end{cases}$$

where $a$ is a learnable parameter.

### Implementation and Usage

```python
import torch
import torch.nn as nn

class PReLUNetwork(nn.Module):
    """Network demonstrating PReLU usage."""
    
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # One learnable parameter per channel
        self.prelu1 = nn.PReLU(num_parameters=hidden_dim)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.prelu2 = nn.PReLU(num_parameters=hidden_dim)
        
        self.fc3 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = self.prelu1(self.fc1(x))
        x = self.prelu2(self.fc2(x))
        return self.fc3(x)
    
    def print_prelu_weights(self):
        """Display learned negative slopes."""
        print("PReLU weights (negative slopes):")
        print(f"  Layer 1: mean={self.prelu1.weight.mean():.4f}, "
              f"std={self.prelu1.weight.std():.4f}")
        print(f"  Layer 2: mean={self.prelu2.weight.mean():.4f}, "
              f"std={self.prelu2.weight.std():.4f}")

# Training example
model = PReLUNetwork(784, 256, 10)
print("Before training:")
model.print_prelu_weights()

# ... train the model ...

print("\nAfter training:")
model.print_prelu_weights()
```

### PReLU Options

```python
# Single shared parameter for all channels
prelu_shared = nn.PReLU(num_parameters=1)

# One parameter per channel (recommended)
prelu_per_channel = nn.PReLU(num_parameters=64)

# Custom initialization
prelu_custom = nn.PReLU(num_parameters=64, init=0.1)  # Start with 0.1
```

---

## Adaptive Activation Functions

### Mixture of Activations

Learn to combine multiple activation functions:

```python
class AdaptiveActivation(nn.Module):
    """
    Learns a weighted combination of multiple activations.
    Weights are optimized during training.
    """
    
    def __init__(self, activations=None):
        super().__init__()
        
        if activations is None:
            self.activations = [
                ('relu', lambda x: torch.relu(x)),
                ('tanh', lambda x: torch.tanh(x)),
                ('sigmoid', lambda x: torch.sigmoid(x)),
                ('identity', lambda x: x),
            ]
        else:
            self.activations = activations
        
        # Learnable weights for mixing
        num_acts = len(self.activations)
        self.weights = nn.Parameter(torch.ones(num_acts) / num_acts)
    
    def forward(self, x):
        # Softmax ensures weights sum to 1
        w = torch.softmax(self.weights, dim=0)
        
        # Compute weighted sum of activations
        result = torch.zeros_like(x)
        for i, (name, act_fn) in enumerate(self.activations):
            result = result + w[i] * act_fn(x)
        
        return result
    
    def get_mixture_weights(self):
        """Return current mixture weights as dictionary."""
        w = torch.softmax(self.weights, dim=0)
        return {name: w[i].item() for i, (name, _) in enumerate(self.activations)}
```

### Shape-Learning Activation

```python
class ShapeLearningActivation(nn.Module):
    """
    Learns the activation shape using a small neural network.
    Can approximate any continuous activation function.
    """
    
    def __init__(self, hidden_dim=16):
        super().__init__()
        
        # Small network to learn the activation shape
        self.shape_net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        
        # Initialize to approximate ReLU
        self._init_as_relu()
    
    def _init_as_relu(self):
        """Initialize to approximate ReLU."""
        # This is a simplified initialization
        # A proper implementation would fit to ReLU outputs
        pass
    
    def forward(self, x):
        # Flatten, apply shape network, reshape
        original_shape = x.shape
        x_flat = x.view(-1, 1)
        out_flat = self.shape_net(x_flat)
        return out_flat.view(original_shape)
```

---

## Initialization Strategies

### He (Kaiming) Initialization

For ReLU and variants:

$$W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{\text{in}}}}\right)$$

```python
def he_init(module):
    """He initialization for ReLU networks."""
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)

# Apply to model
model.apply(he_init)
```

### Xavier (Glorot) Initialization

For tanh and sigmoid:

$$W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{\text{in}} + n_{\text{out}}}}\right)$$

```python
def xavier_init(module):
    """Xavier initialization for tanh/sigmoid networks."""
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
```

### Activation-Specific Initialization

```python
def init_for_activation(model, activation_type='relu'):
    """Initialize model based on activation type."""
    
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            if activation_type in ['relu', 'leaky_relu', 'elu']:
                nn.init.kaiming_normal_(
                    module.weight, 
                    mode='fan_in',
                    nonlinearity=activation_type.replace('_', '')
                )
            elif activation_type in ['tanh', 'sigmoid']:
                nn.init.xavier_normal_(module.weight)
            elif activation_type == 'selu':
                # LeCun normal for SELU
                nn.init.normal_(module.weight, std=1.0 / module.weight.shape[1]**0.5)
            
            if module.bias is not None:
                nn.init.zeros_(module.bias)
```

---

## Monitoring Activations

### Activation Statistics Hook

```python
class ActivationMonitor:
    """Monitor activation statistics during training."""
    
    def __init__(self, model):
        self.model = model
        self.stats = {}
        self.handles = []
        
        # Register hooks
        for name, module in model.named_modules():
            if isinstance(module, (nn.ReLU, nn.LeakyReLU, nn.GELU, nn.SiLU)):
                handle = module.register_forward_hook(self._create_hook(name))
                self.handles.append(handle)
    
    def _create_hook(self, name):
        def hook(module, input, output):
            self.stats[name] = {
                'mean': output.detach().mean().item(),
                'std': output.detach().std().item(),
                'sparsity': (output.detach() == 0).float().mean().item(),
                'max': output.detach().max().item(),
                'min': output.detach().min().item(),
            }
        return hook
    
    def print_stats(self):
        """Print current activation statistics."""
        for name, stat in self.stats.items():
            print(f"{name}: mean={stat['mean']:.4f}, std={stat['std']:.4f}, "
                  f"sparsity={stat['sparsity']:.2%}")
    
    def remove_hooks(self):
        """Clean up hooks."""
        for h in self.handles:
            h.remove()

# Usage
model = nn.Sequential(
    nn.Linear(64, 128), nn.ReLU(),
    nn.Linear(128, 64), nn.ReLU(),
    nn.Linear(64, 10)
)
monitor = ActivationMonitor(model)

x = torch.randn(32, 64)
_ = model(x)
monitor.print_stats()

monitor.remove_hooks()  # Clean up
```

---

## Summary

| Technique | When to Use |
|-----------|-------------|
| **Gradient analysis** | Debugging training issues |
| **Dead neuron detection** | When loss plateaus unexpectedly |
| **PReLU** | When ReLU causes dead neurons |
| **Adaptive activations** | Research, task-specific tuning |
| **He initialization** | ReLU family |
| **Xavier initialization** | Tanh, Sigmoid |
| **Activation monitoring** | During development, debugging |

!!! tip "Best Practices"
    1. Always use appropriate initialization for your activation
    2. Monitor for dead neurons in early training
    3. Use Leaky ReLU or PReLU if dead neuron rate is high
    4. Add BatchNorm for additional stability
    5. Consider lower learning rates for deep networks
