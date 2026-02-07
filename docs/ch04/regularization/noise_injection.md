# Noise Injection

## Overview

Noise injection is a regularization technique that adds random perturbations to inputs, weights, or activations during training. By exposing the model to noisy data, it learns more robust representations that generalize better to test data, which may contain natural variations or perturbations.

## Types of Noise Injection

### Input Noise

Adding noise directly to input features:

$$
\tilde{x} = x + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2 I)
$$

### Weight Noise

Adding noise to model weights during forward pass:

$$
\tilde{w} = w + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2 I)
$$

### Gradient Noise

Adding noise to gradients during optimization:

$$
\tilde{g} = g + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma_t^2 I)
$$

### Activation Noise

Adding noise to hidden layer activations:

$$
\tilde{h} = h + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2 I)
$$

## Theoretical Foundation

### Regularization Effect

For linear regression with input noise, adding Gaussian noise $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$ to inputs is equivalent to L2 regularization:

$$
\mathbb{E}_\epsilon[\|y - (x + \epsilon)^T w\|^2] = \|y - x^T w\|^2 + \sigma^2 \|w\|^2
$$

This shows that input noise implicitly penalizes large weights.

### Robustness Interpretation

Noise injection creates a smoothed loss landscape:

$$
\mathcal{L}_{\text{smooth}}(w) = \mathbb{E}_\epsilon[\mathcal{L}(w + \epsilon)]
$$

The model learns to minimize loss not just at a single point but over a neighborhood, leading to flatter minima that generalize better.

## PyTorch Implementation

### Input Noise

```python
import torch
import torch.nn as nn

class GaussianNoise(nn.Module):
    """Add Gaussian noise to inputs during training."""
    
    def __init__(self, std: float = 0.1, relative: bool = False):
        """
        Args:
            std: Standard deviation of noise
            relative: If True, std is relative to input magnitude
        """
        super().__init__()
        self.std = std
        self.relative = relative
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.std > 0:
            if self.relative:
                noise_std = self.std * torch.abs(x)
            else:
                noise_std = self.std
            noise = torch.randn_like(x) * noise_std
            return x + noise
        return x


class UniformNoise(nn.Module):
    """Add uniform noise to inputs during training."""
    
    def __init__(self, low: float = -0.1, high: float = 0.1):
        super().__init__()
        self.low = low
        self.high = high
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            noise = torch.empty_like(x).uniform_(self.low, self.high)
            return x + noise
        return x


class SaltAndPepperNoise(nn.Module):
    """Salt and pepper noise for images."""
    
    def __init__(self, prob: float = 0.05):
        super().__init__()
        self.prob = prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.prob > 0:
            mask = torch.rand_like(x)
            salt = mask < self.prob / 2
            pepper = mask > (1 - self.prob / 2)
            
            x = x.clone()
            x[salt] = 1.0
            x[pepper] = 0.0
        return x
```

### Weight Noise

```python
class NoisyLinear(nn.Module):
    """Linear layer with noise injection on weights."""
    
    def __init__(self, in_features: int, out_features: int, 
                 noise_std: float = 0.1, bias: bool = True):
        super().__init__()
        self.noise_std = noise_std
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features) / 
                                   (in_features ** 0.5))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.noise_std > 0:
            weight_noise = torch.randn_like(self.weight) * self.noise_std
            noisy_weight = self.weight + weight_noise
            
            if self.bias is not None:
                bias_noise = torch.randn_like(self.bias) * self.noise_std
                noisy_bias = self.bias + bias_noise
            else:
                noisy_bias = None
            
            return nn.functional.linear(x, noisy_weight, noisy_bias)
        
        return nn.functional.linear(x, self.weight, self.bias)
```

### Gradient Noise

```python
class GradientNoiseCallback:
    """
    Add noise to gradients during training.
    
    The noise schedule typically decays: sigma_t^2 = eta / (1 + t)^gamma
    """
    
    def __init__(self, eta: float = 0.01, gamma: float = 0.55):
        self.eta = eta
        self.gamma = gamma
        self.step = 0
    
    def get_noise_std(self) -> float:
        """Compute current noise standard deviation."""
        variance = self.eta / ((1 + self.step) ** self.gamma)
        return variance ** 0.5
    
    def add_gradient_noise(self, model: nn.Module):
        """Add noise to all gradients."""
        std = self.get_noise_std()
        
        with torch.no_grad():
            for param in model.parameters():
                if param.grad is not None:
                    noise = torch.randn_like(param.grad) * std
                    param.grad.add_(noise)
        
        self.step += 1
```

### Activation Noise

```python
class ActivationNoise(nn.Module):
    """Add noise to layer activations."""
    
    def __init__(self, std: float = 0.1, additive: bool = True):
        """
        Args:
            std: Noise standard deviation
            additive: If True, add noise; if False, multiply
        """
        super().__init__()
        self.std = std
        self.additive = additive
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.std > 0:
            if self.additive:
                noise = torch.randn_like(x) * self.std
                return x + noise
            else:
                noise = 1 + torch.randn_like(x) * self.std
                return x * noise
        return x


class NetworkWithActivationNoise(nn.Module):
    """Example network with activation noise."""
    
    def __init__(self, input_dim, hidden_dims, output_dim, noise_std=0.1):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                ActivationNoise(std=noise_std)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
```

## Advanced Techniques

### Scheduled Noise

```python
class ScheduledNoise(nn.Module):
    """Noise with magnitude that changes during training."""
    
    def __init__(self, initial_std: float = 0.2, final_std: float = 0.01,
                 decay_steps: int = 10000):
        super().__init__()
        self.initial_std = initial_std
        self.final_std = final_std
        self.decay_steps = decay_steps
        self.current_step = 0
    
    @property
    def current_std(self) -> float:
        if self.current_step >= self.decay_steps:
            return self.final_std
        
        progress = self.current_step / self.decay_steps
        return self.initial_std + (self.final_std - self.initial_std) * progress
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            noise = torch.randn_like(x) * self.current_std
            self.current_step += 1
            return x + noise
        return x
```

### Variational Layer (Learnable Noise)

```python
class VariationalLayer(nn.Module):
    """Layer that learns appropriate noise levels."""
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        
        self.w_mean = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.w_log_var = nn.Parameter(torch.full((out_features, in_features), -10.0))
        
        self.b_mean = nn.Parameter(torch.zeros(out_features))
        self.b_log_var = nn.Parameter(torch.full((out_features,), -10.0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            w_std = torch.exp(0.5 * self.w_log_var)
            w = self.w_mean + w_std * torch.randn_like(self.w_mean)
            
            b_std = torch.exp(0.5 * self.b_log_var)
            b = self.b_mean + b_std * torch.randn_like(self.b_mean)
        else:
            w = self.w_mean
            b = self.b_mean
        
        return nn.functional.linear(x, w, b)
    
    def kl_divergence(self) -> torch.Tensor:
        """KL divergence from prior (standard normal)."""
        kl_w = -0.5 * torch.sum(1 + self.w_log_var - self.w_mean.pow(2) - 
                                 self.w_log_var.exp())
        kl_b = -0.5 * torch.sum(1 + self.b_log_var - self.b_mean.pow(2) - 
                                 self.b_log_var.exp())
        return kl_w + kl_b
```

## Complete Training Example

```python
import torch.optim as optim
from torch.utils.data import DataLoader

def train_with_noise_injection(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    input_noise_std: float = 0.1,
    gradient_noise_eta: float = 0.01,
    epochs: int = 100
) -> dict:
    """Train model with multiple types of noise injection."""
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    input_noise = GaussianNoise(std=input_noise_std)
    gradient_noise = GradientNoiseCallback(eta=gradient_noise_eta)
    
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            
            # Add input noise
            X_noisy = input_noise(X_batch)
            
            outputs = model(X_noisy)
            loss = criterion(outputs, y_batch)
            loss.backward()
            
            # Add gradient noise
            gradient_noise.add_gradient_noise(model)
            
            optimizer.step()
            train_loss += loss.item()
        
        # Validation (no noise)
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += y_batch.size(0)
                val_correct += predicted.eq(y_batch).sum().item()
        
        history['train_loss'].append(train_loss / len(train_loader))
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_acc'].append(val_correct / val_total)
    
    return history
```

## Comparison of Noise Types

| Type | Location | Effect | Use Case |
|------|----------|--------|----------|
| Input noise | Data | Data augmentation, robustness | Limited data, noisy inputs |
| Weight noise | Parameters | Approximate Bayesian inference | Uncertainty estimation |
| Gradient noise | Optimization | Escape local minima | Deep networks, non-convex loss |
| Activation noise | Hidden layers | Similar to dropout | General regularization |

## Practical Guidelines

### Choosing Noise Type

1. **Input noise**: When test data may have natural variations
2. **Weight noise**: When uncertainty quantification is needed
3. **Gradient noise**: For very deep networks or difficult optimization
4. **Activation noise**: General-purpose regularization

### Noise Magnitude Selection

- **Too small**: Little regularization effect
- **Too large**: Prevents learning, destroys signal
- **Guidelines**:
  - Input noise: 1-10% of input standard deviation
  - Weight noise: 0.01-0.1 relative to weight magnitude
  - Gradient noise: Start with eta=0.01, decay over training

### Combining with Other Techniques

Noise injection can complement:
- **Dropout**: Different mechanisms, often synergistic
- **L2 regularization**: Noise provides implicit L2-like effect
- **Data augmentation**: Noise is a form of continuous augmentation

## References

1. Bishop, C. M. (1995). Training with Noise is Equivalent to Tikhonov Regularization. *Neural Computation*, 7(1), 108-116.
2. Neelakantan, A., et al. (2015). Adding Gradient Noise Improves Learning for Very Deep Networks. *arXiv*.
3. Fortunato, M., et al. (2018). Noisy Networks for Exploration. *ICLR*.
4. An, G. (1996). The Effects of Adding Noise During Backpropagation Training. *Neural Computation*, 8(3), 643-674.
